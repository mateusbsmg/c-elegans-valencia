"""Cérebro B: o Laya lê um texto com o que o verme sente e escolhe uma ação; um gerador de ritmo
(CPG, "central pattern generator") transforma a ação na onda de contração dos músculos.

- Decide 1 vez por segundo do corpo (meio dia de vida).
- Aprendizado por reforço (REINFORCE com linha de base do grupo, parecido com o GRPO): cada verme vive
  o episódio sorteando ações pelas chances do Laya; depois, ações seguidas de mais prazer e menos dor
  ganham chance, as outras perdem. Não há professor: só a experiência.
"""
import glob
import os
from pathlib import Path

import numpy as np

from mundo import NSEG, DT, SEGUNDOS_POR_DIA, DIAS_FOME_MORTE, ESPECIES

NOMES_BACT = [e["nome"] for e in ESPECIES]

os.environ.setdefault("HF_HOME", r"D:\Mateus\.hf-cache")

INTERVALO_DECISAO = 1.0                                  # s do corpo
PASSOS_DECISAO = int(round(INTERVALO_DECISAO / DT))
# o Laya "pensa" enquanto o corpo continua se mexendo; a ação vale 0,25 s depois de ler os sentidos
# (como o tempo de reação do verme). Assim o vídeo não para a cada decisão.
PASSOS_LATENCIA = int(round(0.25 / DT))
# coordenação motora fina: virar levemente ou forte para cada lado (7 ações)
ACOES = ["Crawl forward",
         "Turn slightly toward the dorsal side",
         "Turn sharply toward the dorsal side",
         "Turn slightly toward the ventral side",
         "Turn sharply toward the ventral side",
         "Reverse and turn around (pirouette)",
         "Stop and feed here"]
ACOES_PT = ["seguir em frente", "virar levemente p/ dorsal", "virar forte p/ dorsal",
            "virar levemente p/ ventral", "virar forte p/ ventral", "dar ré e girar (pirueta)", "parar e comer"]
FRENTE, PIRUETA, PARAR = 0, 5, 6
INSTRUCOES = ("You control a C. elegans worm. Choose the next movement that brings more pleasure "
              "(eating, food smell getting stronger) and less pain (hitting walls, hunger).")


# ---------------- gerador de ritmo (motor), com parâmetros que o treino ajusta ----------------
def _musculos():
    import json
    d = json.loads((Path(__file__).resolve().parent.parent / "dados" / "conectoma.json").read_text(encoding="utf-8"))
    return d["musculos"]


MUSCULOS = _musculos()                                      # 95 nomes, mesma ordem do cérebro A
MUS_DORSAL = np.array([m.startswith("d") for m in MUSCULOS])
MUS_FILEIRA = np.array([int("".join(ch for ch in m[5:] if ch.isdigit())) - 1 for m in MUSCULOS])
N_ACOES = len(ACOES)
# parâmetros motores (o vetor que o treino ajusta; tudo zero = o gerador de ritmo feito à mão):
#   por ação (7 x 4): log da frequência, log da força da onda, log do comprimento de onda, curva da cabeça
#   por músculo (95): log do ganho (quanto aquele músculo contrai com o mesmo comando)
#   por fileira (24): atraso de fase (em radianos) — o "tempo" de cada trecho do corpo na onda
FATIA_ACAO = slice(0, N_ACOES * 4)
FATIA_GANHO = slice(FATIA_ACAO.stop, FATIA_ACAO.stop + len(MUSCULOS))
FATIA_FASE = slice(FATIA_GANHO.stop, FATIA_GANHO.stop + NSEG)
N_MOTOR = FATIA_FASE.stop
BASE_FREQ = np.array([0.8, 0.8, 0.75, 0.8, 0.75, 0.9, 0.15])
BASE_AMP = np.array([0.5, 0.5, 0.45, 0.5, 0.45, 0.5, 0.12])
BASE_CURVA = np.array([0.0, -0.12, -0.35, 0.12, 0.35, 0.0, 0.0])


def converter_genoma_antigo(th):
    """Genomas de 139 genes (5 ações) -> 147 genes (7 ações): as viradas antigas viram leve e forte."""
    th = np.asarray(th, float)
    if th.shape[-1] == N_MOTOR:
        return th
    velho = th[..., :20].reshape(th.shape[:-1] + (5, 4))
    novo = velho[..., [0, 1, 1, 2, 2, 3, 4], :].reshape(th.shape[:-1] + (N_ACOES * 4,))
    return np.concatenate([novo, th[..., 20:]], -1)


class MotorCPG:
    """Transforma a ação escolhida em ativação de cada um dos 95 músculos."""
    S = np.arange(NSEG) / NSEG
    # a curva de direção é mais forte no nariz e se desfaz suavemente pelo pescoço: a cabeça "aponta" a direção
    CABECA = np.clip(1 - S / 0.4, 0, 1) ** 1.5
    # a cabeça balança menos que o corpo (60% da onda no nariz): menos "chacoalhar", direção mais firme
    PERFIL_ONDA = 0.6 + 0.4 * np.clip(S / 0.3, 0, 1)

    TAU_TRANSICAO = 0.25         # s (0,5 atrapalhava a virada: oráculo 18->3 de 32): frequência, força e curva mudam aos poucos entre ações (não freia de uma vez)

    def __init__(self, P, theta=None, rng=None):
        self.P = P
        rng = rng or np.random.default_rng()
        self.fase = rng.uniform(0, 2 * np.pi, P)   # cada verme começa num ponto diferente da onda
        self.v_fase = np.zeros(P)                  # frequência atual (com sinal: negativa = ré)
        self.amp_atual = np.zeros(P)
        self.lam_atual = np.full(P, 0.65)
        self.curva_atual = np.zeros(P)
        self.omega_atual = np.zeros(P)
        self.reflexo = np.zeros(P)                 # s restantes do reflexo de fuga (ré automática)
        self.acao = np.zeros(P, int)
        self.t_acao = np.zeros(P)
        self.salto = np.zeros(P)
        self.brusco = np.zeros(P)
        self.definir_parametros(np.zeros(N_MOTOR) if theta is None else converter_genoma_antigo(theta))
        nd = np.bincount(MUS_FILEIRA[MUS_DORSAL], minlength=NSEG)
        nv = np.bincount(MUS_FILEIRA[~MUS_DORSAL], minlength=NSEG)
        self.nd, self.nv = nd, nv

    def definir_parametros(self, theta):
        th = np.broadcast_to(np.asarray(theta, float), (self.P, N_MOTOR))
        pa = th[:, FATIA_ACAO].reshape(self.P, N_ACOES, 4)
        self.freq = BASE_FREQ * np.exp(np.clip(pa[..., 0], -1.5, 1.5))
        self.amp = BASE_AMP * np.exp(np.clip(pa[..., 1], -1.5, 1.0))
        self.lam = 0.65 * np.exp(np.clip(pa[..., 2], -0.7, 0.7))
        self.curva = BASE_CURVA + np.clip(pa[..., 3], -0.5, 0.5)
        self.ganho = np.exp(np.clip(th[:, FATIA_GANHO], -1.5, 1.0))
        self.defasagem = th[:, FATIA_FASE]

    def _comando(self, a):
        """O comando voluntário de cada ação: (velocidade da onda com sinal, curva da cabeça, curva ômega)."""
        i = np.arange(self.P)
        pir = a == PIRUETA
        return np.stack([np.where(pir, -self.freq[i, a], self.freq[i, a]), self.curva[i, a], pir * 0.35], -1)

    def definir(self, acoes):
        acoes = np.asarray(acoes, int)
        # quão brusca é a troca: diferença entre o comando novo e o anterior (curva pesa mais: é o que desalinha)
        dif = np.abs(self._comando(acoes) - self._comando(self.acao)) * np.array([1.0, 3.0, 3.0])
        self.salto = dif.sum(-1)
        self.acao = acoes
        self.t_acao[:] = 0

    def passo(self, s=None):
        """s: sensores do mundo, para os reflexos que não passam pelo Laya (ver referencias/02)."""
        a, i = self.acao, np.arange(self.P)
        freq, amp, lam, curva = self.freq[i, a], self.amp[i, a], self.lam[i, a], self.curva[i, a]
        sentido = np.where((a == PIRUETA) & (self.t_acao < 0.6), -1.0, 1.0)   # pirueta: ré e depois vira
        if s is not None:
            na, h = s["na_comida"], s["fome"]
            # reflexo de fuga: nariz bateu -> ré por 0,8 s (ASH/FLP -> AVA; Kaplan & Horvitz 1993)
            novo = (s["toque_nariz"] > 0) & (self.reflexo <= 0)
            self.reflexo = np.where(novo, 0.8, np.maximum(0, self.reflexo - DT))
            sentido = np.where(self.reflexo > 0, -1.0, sentido)
            # desaceleração sobre as bactérias (dopamina), maior com fome (serotonina; Sawin et al. 2000)
            devagar = np.clip(1 - na * (0.45 + 0.45 * h), 0.1, 1)
            # com fome e longe da comida, explora mais rápido (octopamina)
            explora = 1 + 0.4 * h * (1 - na)
            freq = freq * devagar * explora
            amp = amp * (0.5 + 0.5 * devagar)
        omega = np.where((a == PIRUETA) & (self.t_acao >= 0.6), 0.35, 0.0)  # curva forte em "ômega"
        # transição suave: cada grandeza vai do valor atual ao da nova ação em ~TAU_TRANSICAO
        k = DT / self.TAU_TRANSICAO
        self.v_fase += (freq * sentido - self.v_fase) * k
        self.amp_atual += (amp - self.amp_atual) * k
        self.lam_atual += (lam - self.lam_atual) * k
        self.curva_atual += (curva - self.curva_atual) * k
        self.omega_atual += (omega - self.omega_atual) * k
        self.fase += 2 * np.pi * self.v_fase * DT
        self.t_acao += DT
        # salto de comando voluntário (calculado em definir): cobrado uma vez, no passo da troca
        self.brusco, self.salto = self.salto, np.zeros(self.P)
        onda = self.amp_atual[:, None] * self.PERFIL_ONDA[None] * np.sin(
            self.fase[:, None] - 2 * np.pi * self.S[None] / self.lam_atual[:, None] + self.defasagem)
        alvo = onda + self.curva_atual[:, None] * self.CABECA[None] + self.omega_atual[:, None]   # > 0 = ventral
        d_fil, v_fil = np.clip(0.5 - alvo, 0, 1), np.clip(0.5 + alvo, 0, 1)
        # cada músculo recebe o comando da sua fileira e lado, multiplicado pelo seu ganho aprendido
        cmd = np.where(MUS_DORSAL[None], d_fil[:, MUS_FILEIRA], v_fil[:, MUS_FILEIRA])
        self.musc = np.clip(cmd * self.ganho, 0, 1)
        dors = np.zeros((self.P, NSEG))
        vent = np.zeros((self.P, NSEG))
        np.add.at(dors.T, MUS_FILEIRA[MUS_DORSAL], self.musc[:, MUS_DORSAL].T)
        np.add.at(vent.T, MUS_FILEIRA[~MUS_DORSAL], self.musc[:, ~MUS_DORSAL].T)
        self.ultima = (dors / self.nd, vent / self.nv)
        return self.ultima


# ---------------- sentidos em texto ----------------
# genes sensoriais: cada verme pode expressar ou não cada sentido; só os expressos entram no texto do Laya.
# Cada sentido expresso custa um pouco de energia (neurônios e receptores custam caro na biologia).
SENTIDOS = ["fome", "cheiro", "lado", "toque", "paladar", "memoria", "feromonio"]
SENTIDOS_PT = {"fome": "fome (RIC)", "cheiro": "nível do cheiro (AWA)", "lado": "lado do cheiro (klinotaxia, AIY)",
               "toque": "toque (ASH/ALM)", "paladar": "está comendo (NSM)", "memoria": "memória da última ação (AIB)",
               "feromonio": "feromônio: lotação e escassez (ASK/ADL)"}
CUSTO_SENTIDO = 0.0004        # valência negativa por passo para cada sentido expresso (~1 por episódio de 30 dias)
def _nivel(c):
    for lim, nome in ((0.05, "almost absent"), (0.2, "very faint"), (0.4, "faint"), (0.6, "moderate"),
                      (0.85, "strong")):
        if c < lim:
            return nome
    return "very strong"


def _fome(h, dias):
    for lim, nome in ((0.02, "none"), (0.1, "mild"), (0.3, "growing"), (0.6, "strong")):
        if h < lim:
            return f"{nome} ({dias:.0f} days without food)"
    return f"desperate ({dias:.0f} days without food, death at day {DIAS_FOME_MORTE:.0f})"


class Sentidos:
    """Junta o que aconteceu desde a última decisão e escreve o texto para o Laya."""

    def __init__(self, P):
        self.P = P
        self.reset()

    def reset(self):
        P = self.P
        self.c_ini = None
        self.soma_v = np.zeros(P)
        self.n_v = np.zeros(P)
        self.soma_d = np.zeros(P)
        self.n_d = np.zeros(P)
        self.toque = np.zeros(P)
        self.comeu = np.zeros(P)
        self.esp_comida = np.full(P, -1)
        self.doente = np.zeros(P)
        self.ultima = np.full(P, -1)
        # lado do cheiro POR ESPÉCIE (o verme real distingue os odores de cada bactéria)
        self.esp_v = np.zeros((P, len(NOMES_BACT)))
        self.esp_d = np.zeros((P, len(NOMES_BACT)))
        # memória de infecção: dura o episódio todo (aversão aprendida à Pseudomonas; serotonina em ADF)
        self.lembra_infeccao = np.zeros(P, bool)

    def acumular(self, s):
        c = s["cheiro"]
        if self.c_ini is None:
            self.c_ini = c.copy()
        ventral = s["kappa"][:, 2] > 0                                          # cabeça virada para o ventral
        self.soma_v += np.where(ventral, c, 0)
        self.n_v += ventral
        self.soma_d += np.where(~ventral, c, 0)
        self.n_d += ~ventral
        self.toque = np.maximum(self.toque, s["toque_nariz"] + 0.5 * s["toque_anterior"])
        self.comeu = np.maximum(self.comeu, s["comendo"])
        self.esp_comida = np.where(s["especie_comida"] >= 0, s["especie_comida"], self.esp_comida)
        self.doente = np.maximum(self.doente, s["doente"])
        self.lembra_infeccao |= s["doente"] > 0.2
        ce = s["cheiro_esp"]
        self.esp_v += np.where(ventral[:, None], ce, 0)
        self.esp_d += np.where(~ventral[:, None], ce, 0)

    def textos(self, s, dia, genes=None):
        """genes: (P, len(SENTIDOS)) booleano — quais sentidos cada verme expressa (só esses entram no texto)."""
        c = s["cheiro"]
        ini = self.c_ini if self.c_ini is not None else c
        var = (c - ini) / np.maximum(ini, 1e-3)
        lado = (self.soma_v / np.maximum(self.n_v, 1) - self.soma_d / np.maximum(self.n_d, 1)) / np.maximum(c, 1e-3)
        out = []
        for i in range(self.P):
            tem = dict(zip(SENTIDOS, genes[i])) if genes is not None else {k: True for k in SENTIDOS}
            # lado do cheiro de cada espécie presente: "Comamonas smell is stronger on the ventral side; ..."
            frases = []
            if self.n_v[i] and self.n_d[i]:
                mv, md = self.esp_v[i] / self.n_v[i], self.esp_d[i] / self.n_d[i]
                for e, nome in enumerate(NOMES_BACT):
                    nivel = (mv[e] + md[e]) / 2
                    if nivel < 0.03:
                        continue
                    rel = (mv[e] - md[e]) / nivel
                    frases.append(f"{nome} smell is stronger on the {'ventral' if rel > 0 else 'dorsal'} side"
                                  if abs(rel) > 0.01 else f"{nome} smell is the same on both sides")
            lat = ("; ".join(frases) + ".") if frases else "The smell is the same on both sides."
            toque = ("The nose is pressing against a wall (pain)." if self.toque[i] >= 1 else
                     "The body is touching a wall." if self.toque[i] > 0 else "Nothing touches the body.")
            come = (f"It just ate {NOMES_BACT[self.esp_comida[i]]} bacteria (pleasure)." if self.comeu[i]
                    else "It is not eating.")
            if self.doente[i] > 0.2:
                come += " It has an intestinal infection from eating Pseudomonas (pain)."
            ant = ""
            if self.ultima[i] >= 0:
                efeito = "rose" if var[i] > 0.01 else "fell" if var[i] < -0.01 else "did not change"
                ant = f" Previous action: {ACOES[self.ultima[i]].lower()}; the smell then {efeito}."
            partes = [f"C. elegans worm, day {dia:.0f} of life."]
            if tem["fome"]:
                partes.append(f"Hunger: {_fome(s['fome'][i], s['dias_sem_comer'][i])}.")
            if tem["cheiro"]:
                esp = s["especie_cheiro"][i]
                partes.append(f"Food smell: {_nivel(c[i])}" + (f", mostly {NOMES_BACT[esp]}." if esp >= 0 else "."))
            if tem["lado"]:
                partes.append(lat)
            if tem["toque"]:
                partes.append(toque)
            if tem["paladar"]:
                partes.append(come)
            if tem.get("feromonio"):
                f, n = s["feromonio"][i], int(s["comida_perto"][i])
                lot = ("No other worms are nearby." if f < 0.3 else "A few other worms are nearby (weak pheromone)."
                       if f < 1.5 else "Many other worms are nearby, competing for food (strong pheromone).")
                esc = ("There is no food nearby (food is scarce here)." if n == 0 else
                       f"Food around here is scarce ({n} {'colony' if n == 1 else 'colonies'} nearby)." if n <= 2 else
                       f"There is plenty of food nearby ({n} colonies).")
                partes.append(lot + " " + esc)
            if tem["memoria"] and self.lembra_infeccao[i]:
                partes.append("It remembers getting sick after eating Pseudomonas.")
            if tem["memoria"] and ant:
                partes.append(ant.strip())
            out.append(" ".join(partes))
        return out

    def nova_decisao(self, acoes, s):
        self.ultima = np.asarray(acoes, int)
        self.c_ini = s["cheiro"].copy()
        for a in (self.soma_v, self.n_v, self.soma_d, self.n_d, self.toque, self.comeu, self.doente,
                  self.esp_v, self.esp_d):
            a[:] = 0
        self.esp_comida[:] = -1


# ---------------- a política (Laya) ----------------
def snapshot_laya():
    return Path(glob.glob(r"D:\Mateus\.hf-cache\hub\models--convaiinnovations--laya\snapshots\*")[0])


class PoliticaLaya:
    def __init__(self, modelo=None, device="cuda", treinar=False):
        import torch
        import laya
        from laya.common import collate_items
        self.torch, self.collate = torch, collate_items
        self.dev = torch.device(device)
        self.modelo = str(modelo or snapshot_laya())
        self.agent = laya.load(self.modelo, device=device)
        self.model = self.agent.model
        self.pad = self.agent.tok.pad_token_id
        self.q = self.agent._to_internal({"type": "choice", "instructions": INSTRUCOES, "criteria": ACOES})
        if treinar:
            self.model.float().to(self.dev)
            if hasattr(self.model.encoder, "gradient_checkpointing_enable"):
                self.model.encoder.gradient_checkpointing_enable()
            self.model.head_checkpointing = True

    def codificar(self, textos):
        return [self.agent._encode_state(t, ["action"], {"action": self.q})[0] for t in textos]

    def logits(self, itens, grad=False):
        torch = self.torch
        lote = self.collate([[it] for it in itens], self.pad)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            lg, _ = self.model(lote["input_ids"].to(self.dev), lote["attention_mask"].to(self.dev),
                               lote["marker_pos"].to(self.dev), lote["marker_mask"].to(self.dev),
                               lote["qtype"].to(self.dev))
        return lg.float()[:, :len(ACOES)]

    def escolher(self, textos, rng, amostrar=True):
        itens = self.codificar(textos)
        lg = self.logits(itens)
        p = self.torch.softmax(lg, -1).cpu().numpy().astype(float)
        p /= p.sum(1, keepdims=True)
        acoes = np.array([rng.choice(len(ACOES), p=pi) for pi in p]) if amostrar else p.argmax(1)
        return acoes, p, itens


class Decisor:
    """Roda o Laya numa linha de execução separada: pede no instante k, usa no instante k + latência."""

    def __init__(self, pol, rng, amostrar=True):
        from concurrent.futures import ThreadPoolExecutor
        self.pol, self.rng, self.amostrar = pol, rng, amostrar
        self.pool = ThreadPoolExecutor(1)
        self.fut = None

    def pedir(self, textos):
        self.fut = self.pool.submit(self.pol.escolher, textos, self.rng, self.amostrar)

    def pegar(self):
        """Devolve (ações, chances, itens, segundos esperando)."""
        import time
        t = time.perf_counter()
        a, p, it = self.fut.result()
        self.fut = None
        return a, p, it, time.perf_counter() - t
