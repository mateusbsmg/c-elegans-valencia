"""Cérebro A: os 302 neurônios com as ligações reais (Cook et al. 2019) movendo os 95 músculos.

Modelo de cada neurônio: potencial graduado (a maioria dos neurônios do C. elegans não dispara "spikes"):
    tau dV/dt = -V + viés + sum(W_quim * a_pre) + sum(G_junção * (V_viz - V)) + sensores + propriocepção
    a = max(0, tanh(V))          (atividade de 0 a 1)
Músculo = soma das sinapses neuromusculares (ACh contrai, GABA relaxa), cortada entre 0 e 1.

O que vem da biologia (o "instinto", fixo): QUEM liga com QUEM, quantas sinapses, e se excita ou inibe.
O que o aprendizado por reforço ajusta: um multiplicador para cada ligação (θ=0 é o conectoma original)
e o viés de cada neurônio. Assim a dor e o prazer ficam "enraizados" nos pesos.

Simplificações assumidas (ver referencias/02_o_modelo_e_as_simplificacoes.md):
- propriocepção: os motoneurônios B (frente) sentem a curva do trecho anterior e os A (ré) do posterior
  (Wen et al. 2012) — é isso que faz a onda viajar pelo corpo;
- um oscilador na cabeça (para frente) e outro na cauda (para trás), ligados pelos comandos AVB e AVA;
- neuromodulação resumida: octopamina (RIC, fome) acelera; serotonina (NSM/ADF, comendo) e dopamina
  (CEP/ADE/PDE, sobre a comida) desaceleram.
"""
import json
import re
from pathlib import Path

import numpy as np

from mundo import NSEG, DT

DADOS = Path(__file__).resolve().parent.parent / "dados" / "conectoma.json"
TAU_NEURONIO = 0.1
TAU_MUSCULO = 0.05
W_QUIMICA = 0.8
RAIO_ESPECTRAL = 0.9   # realimentação da rede: < 1 para não "convulsionar" (tudo saturado)
RUIDO = 0.3            # ruído neural (desvio por raiz de segundo)
GAP_MAX_SOMA = 3.0     # soma das junções do neurônio mais conectado (estabilidade numérica)
SUBPASSOS = 2          # passos neurais por passo do corpo
K_PROPRIO = 0.6          # corrente por unidade de curvatura (1/mm)
A_OSC = 1.6
LIMIAR_VENTRAL, LIMIAR_DORSAL = -0.4, -0.8
DELTA_PROPRIO = 1        # fileiras além da própria área muscular que o motoneurônio "sente"

SENSORES = {  # neurônio: (sinal do mundo, ganho)
    "AWCL": ("cheiro_piora", 3.0), "AWCR": ("cheiro_piora", 3.0),      # AWC: ativa quando o cheiro some
    "AWAL": ("cheiro_melhora", 3.0), "AWAR": ("cheiro_melhora", 3.0),  # AWA: ativa quando o cheiro aumenta
    "ASEL": ("cheiro_melhora", 2.0), "ASER": ("cheiro_piora", 2.0),    # ASE: ASEL liga, ASER desliga
    "ASHL": ("toque_nariz", 3.0), "ASHR": ("toque_nariz", 3.0),        # nociceptor polimodal: dor
    "FLPL": ("toque_nariz", 2.0), "FLPR": ("toque_nariz", 2.0),
    "ALML": ("toque_anterior", 2.0), "ALMR": ("toque_anterior", 2.0), "AVM": ("toque_anterior", 2.0),
    "PLML": ("toque_posterior", 2.0), "PLMR": ("toque_posterior", 2.0), "PVM": ("toque_posterior", 2.0),
    "RICL": ("fome", 2.0), "RICR": ("fome", 2.0),                      # octopamina: fome
    "NSML": ("comendo", 2.0), "NSMR": ("comendo", 2.0),                # serotonina: comida engolida
    "ADFL": ("comendo", 1.5), "ADFR": ("comendo", 1.5),
    "CEPDL": ("na_comida", 2.0), "CEPDR": ("na_comida", 2.0), "CEPVL": ("na_comida", 2.0),
    "CEPVR": ("na_comida", 2.0), "ADEL": ("na_comida", 2.0), "ADER": ("na_comida", 2.0),
    "PDEL": ("na_comida", 2.0), "PDER": ("na_comida", 2.0),            # dopamina: bactérias sob o corpo
}
CABECA_DORSAL = ["SMDDL", "SMDDR", "RMDDL", "RMDDR", "SMBDL", "SMBDR", "RMED"]
CABECA_VENTRAL = ["SMDVL", "SMDVR", "RMDVL", "RMDVR", "SMBVL", "SMBVR", "RMEV"]
CAUDA_DORSAL = ["DA8", "DA9"]
CAUDA_VENTRAL = ["VA11", "VA12"]
# frente é o padrão (AVB ligado); ré (AVA) precisa de estímulo: dor no nariz, cheiro piorando...
VIES_INICIAL = {"AVBL": 1.0, "AVBR": 1.0, "PVCL": 0.3, "PVCR": 0.3, "AVAL": -0.6, "AVAR": -0.6}


def sinal_do_neuronio(nts):
    if "GABA" in nts:
        return -1.0
    if "Acetylcholine" in nts or "Glutamate" in nts:
        return 1.0
    if any("Tyramine" in n for n in nts):
        return -1.0                  # RIM: tiramina inibe AVB (Pirri et al. 2009)
    return 0.3                       # só monoamina/peptídeo: efeito lento e fraco


class Conectoma:
    def __init__(self):
        d = json.loads(DADOS.read_text(encoding="utf-8"))
        self.neuronios = [n["nome"] for n in d["neuronios"]]
        self.categoria = [n["categoria"] for n in d["neuronios"]]
        self.musculos = d["musculos"]
        self.N, self.M = len(self.neuronios), len(self.musculos)
        ix = {n: i for i, n in enumerate(self.neuronios)}
        mx = {m: i for i, m in enumerate(self.musculos)}
        self.ix = ix
        sinal = np.array([sinal_do_neuronio(n["neurotransmissor"]) for n in d["neuronios"]])

        # químicas: pós <- pré, normalizado pela entrada total de cada neurônio
        q = np.array([[ix[a], ix[b], c] for a, b, c in d["quimicas"]])
        self.q_pre, self.q_pos = q[:, 0], q[:, 1]
        entrada = np.bincount(self.q_pos, weights=q[:, 2], minlength=self.N)
        self.q_base = sinal[self.q_pre] * W_QUIMICA * q[:, 2] / np.sqrt(np.maximum(entrada[self.q_pos], 1))
        awc_aiy = np.isin(self.q_pre, [ix["AWCL"], ix["AWCR"]]) & np.isin(self.q_pos, [ix["AIYL"], ix["AIYR"]])
        self.q_base[awc_aiy] *= -1   # AWC (glutamato) inibe AIY (Chalasani et al. 2007)
        W = np.zeros((self.N, self.N))
        np.add.at(W, (self.q_pos, self.q_pre), self.q_base)
        self.q_base *= RAIO_ESPECTRAL / np.linalg.eigvals(W).real.max()

        g = np.array([[ix[a], ix[b], c] for a, b, c in d["juncoes"]])
        self.g_a, self.g_b = g[:, 0], g[:, 1]
        self.g_base = np.sqrt(g[:, 2])
        tot = np.bincount(self.g_a, self.g_base, self.N) + np.bincount(self.g_b, self.g_base, self.N)
        self.g_base *= GAP_MAX_SOMA / tot.max()   # o neurônio mais conectado (AVA) soma GAP_MAX_SOMA

        m = np.array([[ix[a], mx[b], c] for a, b, c in d["neuronio_musculo"]])
        self.m_pre, self.m_mus = m[:, 0], m[:, 1]
        exc = np.bincount(self.m_mus, weights=m[:, 2] * (sinal[self.m_pre] > 0), minlength=self.M)
        self.m_base = sinal[self.m_pre] * m[:, 2] / (0.35 * np.maximum(exc[self.m_mus], 1))

        # músculo -> (lado, fileira)
        self.mus_dorsal = np.array([mm.startswith("d") for mm in self.musculos])
        self.mus_fileira = np.array([int(re.search(r"(\d+)$", mm).group(1)) - 1 for mm in self.musculos])

        # posição de cada motoneurônio = fileira média dos músculos que ele inerva
        fileira = np.full(self.N, -1)
        self.fileira_min = np.full(self.N, -1)
        self.fileira_max = np.full(self.N, -1)
        for i in range(self.N):
            sel = self.m_pre == i
            if sel.any():
                fs = self.mus_fileira[self.m_mus[sel]]
                fileira[i] = int(round(np.average(fs, weights=m[sel, 2])))
                self.fileira_min[i], self.fileira_max[i] = fs.min(), fs.max()
        self.fileira = fileira

        def grupo(padrao):
            return np.array([i for i, n in enumerate(self.neuronios) if re.match(padrao, n) and fileira[i] >= 0], int)
        self.B_v, self.B_d = grupo(r"^VB\d+$"), grupo(r"^DB\d+$")
        self.A_v, self.A_d = grupo(r"^VA\d+$"), grupo(r"^DA\d+$")
        frente_de = lambda idx: np.clip(self.fileira_min[idx] - DELTA_PROPRIO, 0, NSEG - 1)
        atras_de = lambda idx: np.clip(self.fileira_max[idx] + DELTA_PROPRIO, 0, NSEG - 1)
        self.sente_B_v, self.sente_B_d = frente_de(self.B_v), frente_de(self.B_d)
        self.sente_A_v, self.sente_A_d = atras_de(self.A_v), atras_de(self.A_d)
        self.cab_d = np.array([ix[n] for n in CABECA_DORSAL])
        self.cab_v = np.array([ix[n] for n in CABECA_VENTRAL])
        self.cau_d = np.array([ix[n] for n in CAUDA_DORSAL])
        self.cau_v = np.array([ix[n] for n in CAUDA_VENTRAL])
        self.sens_idx = np.array([ix[n] for n in SENSORES])
        self.sens_sinal = [SENSORES[n][0] for n in SENSORES]
        self.sens_ganho = np.array([SENSORES[n][1] for n in SENSORES])
        self.avb = np.array([ix["AVBL"], ix["AVBR"]])
        self.ava = np.array([ix["AVAL"], ix["AVAR"]])
        self.ric = np.array([ix["RICL"], ix["RICR"]])
        self.ser = np.array([ix[n] for n in ("NSML", "NSMR", "ADFL", "ADFR")])
        self.dop = np.array([ix[n] for n in ("CEPDL", "CEPDR", "CEPVL", "CEPVR", "ADEL", "ADER", "PDEL", "PDER")])
        self.vies_base = np.zeros(self.N)
        for n, v in VIES_INICIAL.items():
            self.vies_base[ix[n]] = v
        # limiar dos motoneurônios A e B: a curvatura (propriocepção) decide quem contrai;
        # o dorsal é mais alto porque recebe mais entrada tônica no conectoma
        self.vies_base[np.r_[self.B_v, self.A_v]] = LIMIAR_VENTRAL
        self.vies_base[np.r_[self.B_d, self.A_d]] = LIMIAR_DORSAL

        # vetor de parâmetros que o aprendizado ajusta
        self.fatias = {}
        ini = 0
        for nome, tam in (("quimicas", len(self.q_base)), ("juncoes", len(self.g_base)),
                          ("neuromuscular", len(self.m_base)), ("vies", self.N)):
            self.fatias[nome] = slice(ini, ini + tam)
            ini += tam
        self.n_param = ini

        # o que o aprendizado pode mudar: sinapses que chegam em neurônios sensoriais, interneurônios e da
        # cabeça, e o viés deles. O circuito que gera a onda no corpo (A, B, D, AS, VC), as junções
        # neuromusculares e a faringe ficam fixos: é o "instinto" da marcha, como no verme real.
        cat = np.array(self.categoria)
        corpo = np.array([bool(re.match(r"^(VA|VB|VD|VC|DA|DB|DD|AS)\d+$", n)) for n in self.neuronios])
        fixo = corpo | (cat == "faringe")
        self.aprende = np.zeros(self.n_param, bool)
        self.aprende[self.fatias["quimicas"]] = ~fixo[self.q_pos]
        self.aprende[self.fatias["juncoes"]] = ~fixo[self.g_a] & ~fixo[self.g_b]
        self.aprende[self.fatias["vies"]] = ~fixo

    def resumo(self):
        return (f"{self.N} neurônios, {self.M} músculos, {len(self.q_base)} ligações químicas, "
                f"{len(self.g_base)} junções comunicantes, {len(self.m_base)} ligações neurônio→músculo, "
                f"{int(self.aprende.sum())} parâmetros que o aprendizado ajusta")


class CerebroConectoma:
    """P cérebros (um por verme), cada um com seu vetor de parâmetros theta (P, n_param)."""

    def __init__(self, con: Conectoma, theta, seed=0, mapa_ruido=None):
        """mapa_ruido[i] = qual sorteio de ruído o verme i usa (os pares +/- do treino usam o mesmo)."""
        self.rng = np.random.default_rng(seed)
        self.c = c = con
        theta = np.atleast_2d(theta).astype(np.float32)
        self.P = P = theta.shape[0]
        N, M = c.N, c.M
        f = c.fatias
        self.W = np.zeros((P, N, N), np.float32)
        self.W[:, c.q_pos, c.q_pre] = c.q_base * (1 + theta[:, f["quimicas"]])
        gg = c.g_base * np.clip(1 + theta[:, f["juncoes"]], 0, 2)     # junção: nem negativa nem instável
        self.G = np.zeros((P, N, N), np.float32)
        self.G[:, c.g_a, c.g_b] = gg
        self.G[:, c.g_b, c.g_a] = gg
        self.Gsoma = self.G.sum(2)
        self.NM = np.zeros((P, M, N), np.float32)
        np.add.at(self.NM, (slice(None), c.m_mus, c.m_pre), (c.m_base * (1 + theta[:, f["neuromuscular"]])))
        self.vies = (c.vies_base + theta[:, f["vies"]]).astype(np.float32)
        self.mapa = np.arange(P) if mapa_ruido is None else np.asarray(mapa_ruido)
        self.n_ruido = int(self.mapa.max()) + 1
        self.reset()

    def reset(self):
        P = self.P
        self.V = np.zeros((P, self.c.N), np.float32)
        self.a = np.zeros_like(self.V)
        self.musc = np.zeros((P, self.c.M), np.float32)
        self.fase_cab = np.zeros(P)
        self.fase_cau = np.zeros(P)

    def passo(self, s):
        """s: sensores do mundo. Devolve (músculos dorsais (P,24), ventrais (P,24))."""
        c, P = self.c, self.P
        dc = s["dcheiro"]
        sinais = {"cheiro_melhora": np.clip(dc, 0, 1), "cheiro_piora": np.clip(-dc, 0, 1)}
        I = np.zeros((P, c.N), np.float32)
        for k, (idx, nome) in enumerate(zip(c.sens_idx, c.sens_sinal)):
            I[:, idx] += c.sens_ganho[k] * (sinais[nome] if nome in sinais else s[nome])

        # propriocepção: B sente o trecho anterior, A o posterior (kappa > 0 = curva ventral)
        kap = s["kappa"]
        # (cada um sente o trecho logo antes/depois da área que ele mesmo contrai — sem laço consigo)
        # o ganho depende do comando: B só propaga a onda com AVB ligado, A só com AVA ligado
        a = self.a
        frente, re_ = a[:, c.avb].mean(1), a[:, c.ava].mean(1)
        gb, ga = (K_PROPRIO * frente)[:, None], (K_PROPRIO * re_)[:, None]
        I[:, c.B_v] += gb * kap[:, c.sente_B_v]
        I[:, c.B_d] -= gb * kap[:, c.sente_B_d]
        I[:, c.A_v] += ga * kap[:, c.sente_A_v]
        I[:, c.A_d] -= ga * kap[:, c.sente_A_d]

        # neuromodulação e osciladores (cabeça: para frente; cauda: para trás)
        mod = (1 + 0.6 * a[:, c.ric].mean(1)) * (1 - 0.5 * a[:, c.ser].mean(1)) * (1 - 0.4 * a[:, c.dop].mean(1))
        self.fase_cab += 2 * np.pi * (0.25 + 0.9 * frente) * mod * DT
        self.fase_cau += 2 * np.pi * (0.25 + 0.9 * re_) * mod * DT
        oc = (A_OSC * frente * mod * np.sin(self.fase_cab))[:, None]
        ot = (A_OSC * re_ * mod * np.sin(self.fase_cau))[:, None]
        I[:, c.cab_d] += oc
        I[:, c.cab_v] -= oc
        I[:, c.cau_d] += ot
        I[:, c.cau_v] -= ot

        h = DT / SUBPASSOS / TAU_NEURONIO
        for _ in range(SUBPASSOS):
            quim = np.matmul(self.W, self.a[..., None])[..., 0]
            junc = np.matmul(self.G, self.V[..., None])[..., 0] - self.Gsoma * self.V
            self.V += h * (-self.V + self.vies + quim + junc + I)                 + (RUIDO * np.sqrt(h)) * self.rng.standard_normal((self.n_ruido, c.N), np.float32)[self.mapa]
            self.V = np.clip(self.V, -5, 5)
            self.a = np.maximum(0, np.tanh(self.V))

        alvo = np.clip(np.matmul(self.NM, self.a[..., None])[..., 0], 0, 1)
        self.musc += (alvo - self.musc) * (DT / TAU_MUSCULO)
        dors = np.zeros((P, NSEG), np.float32)
        vent = np.zeros((P, NSEG), np.float32)
        nd = np.zeros(NSEG)
        nv = np.zeros(NSEG)
        np.add.at(dors.T, c.mus_fileira[c.mus_dorsal], self.musc[:, c.mus_dorsal].T)
        np.add.at(vent.T, c.mus_fileira[~c.mus_dorsal], self.musc[:, ~c.mus_dorsal].T)
        np.add.at(nd, c.mus_fileira[c.mus_dorsal], 1)
        np.add.at(nv, c.mus_fileira[~c.mus_dorsal], 1)
        return dors / nd, vent / nv
