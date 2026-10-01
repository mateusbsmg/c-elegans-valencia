"""Treinos por reforço (ao vivo) e a "vida" de um verme, publicando o que acontece para a página.

- Cérebro A (conectoma): Estratégias Evolutivas (OpenAI-ES, Salimans et al. 2017), um tipo de aprendizado
  por reforço sem gradiente: a cada geração, 32 variações do cérebro vivem o mesmo episódio; as que
  somaram mais prazer e menos dor puxam os pesos na sua direção. Roda na CPU (não esquenta a placa).
- Cérebro B (Laya): REINFORCE com linha de base do grupo, na placa de vídeo (com pausas para esfriar).

Nada aqui começa sozinho: o servidor só chama estas funções quando você aperta o botão na página.
"""
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

import mundo as M
from cerebro_conectoma import Conectoma, CerebroConectoma
from cerebro_laya import ACOES, ACOES_PT, CUSTO_SENTIDO, converter_genoma_antigo, SENTIDOS, SENTIDOS_PT, N_MOTOR, PASSOS_DECISAO, PASSOS_LATENCIA, Decisor, MotorCPG, PoliticaLaya, Sentidos, snapshot_laya

RAIZ = Path(__file__).resolve().parent.parent
RUNS = RAIZ / "runs"
PASTA_A = RUNS / "conectoma_es"
PASTA_B = RUNS / "laya_rl"
CUSTO_BRUSCO = 0.08   # valência negativa por unidade de "salto" de comando (troca brusca de curva/sentido)
CUSTO_ATP = 0.0025     # valência negativa por passo por unidade de contração média dos 95 músculos (energia)


class Parar(Exception):
    pass


# ---------------- utilidades ----------------
def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class Relogio:
    """Segura a simulação na velocidade escolhida (1x = tempo real do corpo) e decide quando publicar."""

    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.t0 = time.perf_counter()
        self.sim0 = 0.0
        self.ultima_pub = 0.0
        self.vel = None

    def tique(self, sim_t):
        vel = self.ctrl.velocidade
        if vel != self.vel:                       # mudou a velocidade: recomeça a contagem
            self.vel, self.t0, self.sim0 = vel, time.perf_counter(), sim_t
        if self.ctrl.parar.is_set():
            raise Parar()
        while self.ctrl.pausado.is_set() and not self.ctrl.parar.is_set():
            time.sleep(0.1)
            self.t0, self.sim0 = time.perf_counter(), sim_t
        if vel and vel > 0:
            alvo = self.t0 + (sim_t - self.sim0) / vel
            espera = alvo - time.perf_counter()
            if espera > 0:
                time.sleep(espera)
        agora = time.perf_counter()
        if vel and vel > 0:                       # um quadro a cada 1/30 s de vídeo, medido no tempo do corpo
            if sim_t >= getattr(self, "prox", -1) or sim_t < self.sim0:
                self.prox = sim_t + vel / 30
                return True
            return False
        if agora - self.ultima_pub >= 1 / 30:
            self.ultima_pub = agora
            return True
        return False


def verme_json(w, i, passo=1, marca=None):
    return {"p": np.round(w.pontos[i, ::passo], 3).tolist(), "vivo": bool(w.vivo[i]),
            "comendo": bool(w.sensores["comendo"][i] > 0), **(marca or {})}


def arena_json(w, idx, destaque, titulo, extra, marcas=None):
    s = w.sensores
    d = destaque
    return {"titulo": titulo, "largura": M.LARGURA, "altura": M.ALTURA,
            "especies": [e["nome"] for e in M.ESPECIES], "bacterias": w.bacterias_json(d),
            "vermes": [verme_json(w, i, 2 if i != d else 1, marcas[i] if marcas else None) for i in idx],
            "destaque": idx.index(d),
            "doente": float(w.doente[d]), "comidas_esp": w.comidas[d].tolist(),
            "dia": w.dia, "dias_sem_comer": float(w.dias_sem_comer[d]), "fome": float(s["fome"][d]),
            "vivo": bool(w.vivo[d]), "cheiro": float(s["cheiro"][d]), "comeu": float(w.comeu_total[d]),
            "manchas": int(w.comidas[d].sum()), **extra}


def cor_tribo(fundador):
    """Cor de cada tribo (linhagem): ângulo de ouro, para tribos vizinhas terem cores bem diferentes."""
    return int((int(fundador) * 137.508) % 360) if isinstance(fundador, int) else None


def media_movel(v, n=10):
    return float(np.mean(v[-n:])) if v else None


# ======================= Cérebro A: estratégias evolutivas =======================
class TreinoConectoma:
    def __init__(self, ctrl, pares=16, sigma=0.08, lr=0.02, dias=30, pasta=PASTA_A):
        self.ctrl, self.K, self.sigma, self.lr, self.dias = ctrl, pares, sigma, lr, dias
        self.pasta = Path(pasta)
        self.con = Conectoma()
        n = self.con.n_param
        self.theta = np.zeros(n)
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0
        self.geracao, self.hist = 0, []
        self.rng = np.random.default_rng(1234)
        est = self.pasta / "estado.npz"
        if est.exists():
            d = np.load(est)
            self.theta, self.m, self.v = d["theta"], d["m"], d["v"]
            self.t, self.geracao = int(d["t"]), int(d["geracao"])
            self.rng = np.random.default_rng(1234 + self.geracao)
            self.hist = json.loads((self.pasta / "historico.json").read_text(encoding="utf-8"))

    def salvar(self):
        self.pasta.mkdir(parents=True, exist_ok=True)
        tmp = self.pasta / "estado.tmp.npz"
        np.savez(tmp, theta=self.theta, m=self.m, v=self.v, t=self.t, geracao=self.geracao)
        os.replace(tmp, self.pasta / "estado.npz")
        (self.pasta / "historico.json").write_text(json.dumps(self.hist), encoding="utf-8")
        np.save(self.pasta / "theta_atual.npy", self.theta)
        mm = [h["media"] for h in self.hist]
        if len(mm) >= 3 and media_movel(mm, 5) >= max(media_movel(mm[:i], 5) for i in range(3, len(mm) + 1)):
            np.save(self.pasta / "theta_melhor.npy", self.theta)

    def geracao_uma(self, publicar):
        K, P = self.K, 2 * self.K + 1
        eps = self.rng.standard_normal((K, self.con.n_param)) * self.con.aprende
        thetas = np.concatenate([self.theta[None], self.theta + self.sigma * eps, self.theta - self.sigma * eps])
        # mesmo começo (posição, direção, fome inicial) e mesmo ruído para cada par +/-: comparação justa
        par = np.r_[0, np.arange(1, K + 1), np.arange(1, K + 1)]
        x = self.rng.uniform(0.9, 2.0, K + 1)[par]
        y = self.rng.uniform(1.2, M.ALTURA - 1.2, K + 1)[par]
        ang = self.rng.uniform(-np.pi, np.pi, K + 1)[par]
        fome0 = self.rng.uniform(0, 60, K + 1)[par]
        seed = int(self.rng.integers(1 << 30))
        w = M.Mundo(P, seed)
        s = w.reset((x, y, ang))
        w.dias_sem_comer[:] = fome0
        b = CerebroConectoma(self.con, thetas, seed=seed, mapa_ruido=par)
        R, PR, DR = np.zeros(P), np.zeros(P), np.zeros(P)
        rel = Relogio(self.ctrl)
        passos = M.PASSOS_POR_DIA * self.dias
        idx = list(range(P))
        for k in range(passos):
            d, v = b.passo(s)
            s, r, pr, dr, _ = w.passo_fisica(d, v)
            R += r
            PR += pr
            DR += dr
            if rel.tique(k * M.DT):
                publicar(self._snap(w, b, idx, pr, dr, R, k, passos))
            if not w.vivo.any():
                break
        # atualização (fitness em ranking centralizado, Adam)
        F = R[1:]
        rk = np.empty(2 * K)
        rk[np.argsort(F)] = np.arange(2 * K)
        rk = rk / (2 * K - 1) - 0.5
        grad = ((rk[:K] - rk[K:])[:, None] * eps).sum(0) / (2 * K * self.sigma)
        grad = grad * self.con.aprende - 0.005 * self.theta                         # puxa de leve para o instinto (conectoma original)
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad ** 2
        mh, vh = self.m / (1 - 0.9 ** self.t), self.v / (1 - 0.999 ** self.t)
        self.theta = self.theta + self.lr * mh / (np.sqrt(vh) + 1e-8)
        self.geracao += 1
        self.hist.append({"g": self.geracao, "media": float(F.mean()), "melhor": float(F.max()),
                          "centro": float(R[0]), "chegaram": float(np.isfinite(w.chegou_dia[1:]).mean()),
                          "prazer": float(PR[1:].mean()), "dor": float(DR[1:].mean()),
                          "mudanca": float(np.abs(self.theta).mean())})
        self.salvar()

    def _snap(self, w, b, idx, pr, dr, R, k, passos):
        d = 0
        extra = {"neuronios": np.round(b.a[d], 2).tolist(), "musculos": np.round(b.musc[d], 2).tolist(),
                 "prazer": float(pr[d]), "dor": float(dr[d]), "recompensa": float(R[d]),
                 "placar": [{"id": "atual" if i == 0 else f"var {i}", "comidas": w.comidas[i].tolist(),
                             "vivo": bool(w.vivo[i]), "elite": i == 0} for i in idx],
                 "legenda_destaque": "cérebro atual (sem variação)", "chegaram": int(np.isfinite(w.chegou_dia).sum()),
                 "total": w.P}
        return {"modo": "treino", "cerebro": "A", "geracao": self.geracao + 1, "progresso": k / passos,
                "t": k * M.DT, "ep": f"A{self.geracao}",
                "historico": self.hist[-300:], "status": f"Geração {self.geracao + 1}: {w.P} vermes vivendo "
                f"{self.dias} dias (1 sem variação + {self.K} pares de variações)",
                "arenas": [arena_json(w, idx, d, "Cérebro A: conectoma de 302 neurônios", extra)]}

    def rodar(self, publicar, n=100):
        """Roda n gerações e para sozinho (cada geração já fica salva)."""
        for self.feitos in range(n):
            self.geracao_uma(publicar)
        self.feitos = n


# ======================= Cérebro B: Laya + REINFORCE =======================
class TreinoLaya:
    """Laya (decisões, por reforço) + genes musculares e sensoriais (algoritmo genético em ilhas).

    - Âncora (penalidade KL): o Laya treinado é puxado de volta para o Laya original, para não esquecer o que já
      sabia (ele já virava para o lado do cheiro). Incentivo a variar (entropia) bem pequeno.
    - Só as 8 camadas de cima do codificador e a cabeça de decisão treinam; as de baixo ("entendem o inglês")
      ficam congeladas — cabe a cópia do Laya original (a âncora) na placa.
    - 24 vermes em 3 ilhas de 8 que evoluem separadas; a cada 5 gerações o melhor de cada ilha migra para a próxima.
    """

    def __init__(self, ctrl, grupo=24, ilhas=3, dias=30, lr_enc=4e-6, lr_cab=2e-5, gama=0.9, entropia=0.005,
                 beta_kl=0.1, camadas_treinadas=8, temp_alvo=80, temp_retomar=72, teto_gb=10.0, pasta=PASTA_B,
                 publicar_status=None):
        import torch
        self.torch = torch
        self.ctrl, self.G, self.dias, self.gama, self.ent, self.beta_kl = ctrl, grupo, dias, gama, entropia, beta_kl
        self.temp_alvo, self.temp_retomar = temp_alvo, temp_retomar
        self.pasta = Path(pasta)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, teto_gb * 2**30 / total), 0)
        self.pol = PoliticaLaya(device="cuda", treinar=True)
        mdl = self.pol.model
        # congela as camadas de baixo do codificador
        n_camadas = 1 + max(int(m.group(1)) for n, _ in mdl.named_parameters()
                            if (m := re.match(r"encoder\.layers\.(\d+)\.", n)))
        livres = set(range(n_camadas - camadas_treinadas, n_camadas))
        for n, p_ in mdl.named_parameters():
            if n.startswith("encoder."):
                m = re.match(r"encoder\.layers\.(\d+)\.", n)
                p_.requires_grad = bool(m and int(m.group(1)) in livres) or n.startswith("encoder.final_norm")
        enc = [p_ for n, p_ in mdl.named_parameters() if n.startswith("encoder.") and p_.requires_grad]
        cab = [p_ for n, p_ in mdl.named_parameters() if not n.startswith("encoder.")]
        self.params = enc + cab
        self.n_treinaveis = sum(p_.numel() for p_ in self.params)
        self.opt = torch.optim.AdamW([{"params": enc, "lr": lr_enc}, {"params": cab, "lr": lr_cab}],
                                     weight_decay=0.0, fused=True)
        # a âncora: o Laya original, congelado, em meia precisão
        self.ref = PoliticaLaya(device="cuda", treinar=False)
        self.ref.model.to(torch.bfloat16).eval()
        for p_ in self.ref.model.parameters():
            p_.requires_grad = False
        # continua sempre do modelo mais recente salvo; se não houver, começa do Laya original
        self.hist = []
        try:
            self.hist = json.loads((self.pasta / "historico.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        self.iteracao = len(self.hist)
        atual = self.pasta / "atual" / "model.safetensors"
        self.reinicio_laya = not atual.exists()
        if atual.exists():
            from safetensors.torch import load_file
            mdl.load_state_dict(load_file(str(atual), device="cuda"))
            ret = self.pasta / "retomar.pt"          # o otimizador só vem junto se o ponto for o mesmo
            if ret.exists():
                st = torch.load(ret, map_location="cpu", weights_only=False)   # na RAM: não estoura a placa
                if st["iteracao"] == self.iteracao:
                    try:
                        self.opt.load_state_dict(st["otimizador"])
                    except (ValueError, KeyError):
                        pass
        self.rng = np.random.default_rng(99 + self.iteracao)
        self.sonda_ref = self.sondar(self.ref)
        # genes: ALGORITMO GENÉTICO EM ILHAS (cada verme tem 147 genes musculares + 6 sensoriais)
        self.n_ilhas, self.por_ilha = ilhas, grupo // ilhas
        self.n_elite, self.torneio, self.p_mutacao, self.sigma_mutacao = 2, 3, 0.25, 0.06
        self.migrar_cada = 5
        self.gen_genes = 0
        arq = self.pasta / "genes.npz"
        if arq.exists():
            d = np.load(arq, allow_pickle=False)
            genomas = converter_genoma_antigo(d["genomas"])
            self.gen_genes = int(d["geracao"])
            sentidos = d["sentidos"] if "sentidos" in d.files else np.ones((len(genomas), len(SENTIDOS)), bool)
            if sentidos.shape[1] < len(SENTIDOS):      # sentido novo (feromônio): sorteado, 50% de chance
                extra = self.rng.random((len(sentidos), len(SENTIDOS) - sentidos.shape[1])) < 0.5
                sentidos = np.concatenate([sentidos, extra], 1)
            fichas = json.loads((self.pasta / "linhagem.json").read_text(encoding="utf-8"))
            if len(genomas) != self.G or any("ilha" not in f for f in fichas):
                genomas, sentidos, fichas = self._em_ilhas(genomas, sentidos, fichas)
            self.genomas, self.sentidos, self.fichas = genomas, sentidos, fichas
            ints = [f["fundador"] for f in self.fichas if isinstance(f["fundador"], int)]
            self.prox_tribo = max(ints + [15]) + 1
            tribo_s = None
            for f in self.fichas:                      # descendentes de selvagens (marcados "selvagem") = uma tribo
                if not isinstance(f["fundador"], int):
                    if tribo_s is None:
                        tribo_s, self.prox_tribo = self.prox_tribo, self.prox_tribo + 1
                    f["selvagem"] = f["id"].startswith("S")
                    f["fundador"] = tribo_s
        else:
            self.prox_tribo = self.G
            ruido = self.rng.standard_normal((self.G, N_MOTOR)) * 0.1
            ruido[0] = 0
            self.genomas = ruido
            self.sentidos = self.rng.random((self.G, len(SENTIDOS))) < 0.7
            self.sentidos[0] = True
            self.fichas = [{"id": f"F{i}", "pais": [], "nasceu": 0, "fundador": i, "vitorias": 0,
                            "ilha": i // self.por_ilha} for i in range(self.G)]
        self.idx_campeao = 0
        # hall da fama: o melhor de todos os tempos (média de >= 3 vidas) volta se a população ficar pior que ele
        self.hall = None
        arq_h = self.pasta / "hall_da_fama.npz"
        if arq_h.exists():
            d = np.load(arq_h, allow_pickle=False)
            g = converter_genoma_antigo(d["genoma"])
            sen = d["sentidos"]
            if len(sen) < len(SENTIDOS):
                sen = np.concatenate([sen, np.zeros(len(SENTIDOS) - len(sen), bool)])
            self.hall = {"genoma": g, "sentidos": sen,
                         "ficha": json.loads((self.pasta / "hall_da_fama.json").read_text(encoding="utf-8"))}

    def _em_ilhas(self, genomas, sentidos, fichas):
        """Reparte a população salva em ilhas de por_ilha vermes, completando com cópias mutadas."""
        novos_g, novos_s, novas_f = [], [], []
        n = len(genomas)
        for k in range(self.n_ilhas):
            membros = list(range(k, n, self.n_ilhas))
            j = 0
            while len(membros) < self.por_ilha:
                membros.append(membros[j % max(1, len(membros))])
                j += 1
            for pos, i in enumerate(membros[:self.por_ilha]):
                f = dict(fichas[i], ilha=k)
                g = genomas[i].copy()
                if pos >= len(range(k, n, self.n_ilhas)):             # cópia extra: com mutação e nome novo
                    g = g + (self.rng.random(len(g)) < self.p_mutacao) * self.rng.normal(0, self.sigma_mutacao, len(g))
                    f = {"id": f"{fichas[i]['id']}m{pos}", "pais": [fichas[i]["id"]], "nasceu": self.gen_genes,
                         "fundador": fichas[i]["fundador"], "vitorias": 0, "notas": [], "ilha": k}
                novos_g.append(g)
                novos_s.append(sentidos[i].copy())
                novas_f.append(f)
        return np.array(novos_g), np.array(novos_s), novas_f

    # teste de discernimento: as mesmas situações, mudando um fator por vez
    SONDAS = {
        "cheiro mais forte do lado dorsal": "The smell is stronger when the head swings to the dorsal side. Nothing touches the body. It is not eating.",
        "cheiro mais forte do lado ventral": "The smell is stronger when the head swings to the ventral side. Nothing touches the body. It is not eating.",
        "Comamonas à ventral, Pseudomonas à dorsal": "Comamonas smell is stronger on the ventral side; Pseudomonas smell is stronger on the dorsal side. Nothing touches the body. It is not eating. It remembers getting sick after eating Pseudomonas.",
        "Pseudomonas à ventral, Comamonas à dorsal": "Pseudomonas smell is stronger on the ventral side; Comamonas smell is stronger on the dorsal side. Nothing touches the body. It is not eating. It remembers getting sick after eating Pseudomonas.",
        "nariz batendo na parede": "The smell is the same on both sides. The nose is pressing against a wall (pain). It is not eating.",
        "acabou de comer Comamonas": "The smell is the same on both sides. Nothing touches the body. It just ate Comamonas bacteria (pleasure).",
        "infecção por Pseudomonas": "The smell is the same on both sides. Nothing touches the body. It just ate Pseudomonas bacteria (pleasure). It has an intestinal infection from eating Pseudomonas (pain).",
        "cheiro piorou depois de seguir em frente": "The smell is the same on both sides. Nothing touches the body. It is not eating. Previous action: crawl forward; the smell then fell.",
        "cheiro melhorou depois de seguir em frente": "The smell is the same on both sides. Nothing touches the body. It is not eating. Previous action: crawl forward; the smell then rose.",
    }

    def sondar(self, pol=None):
        pol = pol or self.pol
        base = "C. elegans worm, day 12 of life. Hunger: growing (40 days without food). Food smell: faint, mostly Comamonas. "
        itens = pol.codificar([base + t for t in self.SONDAS.values()])
        with self.torch.no_grad():
            p = self.torch.softmax(pol.logits(itens).float(), -1).cpu().numpy()
        return {"diferenciacao": float(p.std(0).mean() * 100),
                "respostas": {k: np.round(v, 3).tolist() for k, v in zip(self.SONDAS, p)}}

    def esfriar(self, publicar, snap):
        t = gpu_temp()
        if t is None or t <= self.temp_alvo:
            return
        while t is not None and t > self.temp_retomar:
            if self.ctrl.parar.is_set():
                raise Parar()
            snap["status"] = f"Pausa para esfriar a placa de vídeo: {t} °C (volta abaixo de {self.temp_retomar} °C)"
            publicar(snap)
            time.sleep(5)
            t = gpu_temp()

    def episodio(self, publicar):
        G = self.G
        # mesma região, direção parecida e mesma fome para todos (comparação justa entre os genes),
        # mas cada um num ponto um pouco diferente: assim não andam "em bloco"
        x0, y0, a0 = self.rng.uniform(1.1, 1.8), self.rng.uniform(1.6, M.ALTURA - 1.6), self.rng.uniform(-np.pi, np.pi)
        inicio = (x0 + self.rng.uniform(-0.4, 0.4, G), y0 + self.rng.uniform(-0.6, 0.6, G),
                  a0 + self.rng.uniform(-0.6, 0.6, G))
        w = M.Mundo(G, int(self.rng.integers(1 << 30)), compartilhado=True)   # mesma placa: disputam a comida
        s = w.reset(inicio)
        w.dias_sem_comer[:] = self.rng.uniform(0, 60)
        motor, sent = MotorCPG(G, self.genomas, self.rng), Sentidos(G)
        custo_sentidos = CUSTO_SENTIDO * self.sentidos.sum(1)
        cab_ant = w.pontos[:, 0].copy()
        self.caminho = np.zeros(G)
        self.energia = np.zeros(G)
        itens, acoes, recomp = [], [], []          # por decisão: listas de G
        r_janela = np.zeros(G)
        R, probs, textos = np.zeros(G), np.full((G, len(ACOES)), 1 / len(ACOES)), [""] * G
        rel = Relogio(self.ctrl)
        passos = M.PASSOS_POR_DIA * self.dias
        pr = dr = np.zeros(G)
        dec = Decisor(self.pol, self.rng, amostrar=True)
        for k in range(passos):
            if k % PASSOS_DECISAO == 0:
                textos = sent.textos(s, w.dia, self.sentidos)
                dec.pedir(textos)                                    # o Laya pensa enquanto o corpo anda
            if k % PASSOS_DECISAO == PASSOS_LATENCIA and dec.fut is not None:
                a, probs, it, espera = dec.pegar()
                if acoes:
                    recomp.append(r_janela.copy())                   # prazer - dor que veio depois da ação anterior
                r_janela[:] = 0
                itens.append(it)
                acoes.append(a)
                motor.definir(a)
                sent.nova_decisao(a, s)
                if espera > 0.5:
                    rel.t0, rel.sim0 = time.perf_counter(), k * M.DT     # espera longa: não "corre" para compensar
            d, v = motor.passo(s)
            s, r, pr, dr, _ = w.passo_fisica(d, v)
            gasto = motor.musc.mean(1) * w.vivo                        # contração média dos 95 músculos
            # movimento brusco: troca repentina de comando (virar forte para um lado e logo para o outro, ré...)
            brusco = motor.brusco * w.vivo
            r = r - CUSTO_BRUSCO * brusco
            w.comp["movimentos bruscos"] += -CUSTO_BRUSCO * brusco
            self.energia += gasto * M.DT
            r = r - custo_sentidos * w.vivo - CUSTO_ATP * gasto           # sentidos e contração custam energia (ATP)
            w.comp["energia (ATP)"] += -CUSTO_ATP * gasto
            w.comp["custo dos sentidos"] += -custo_sentidos * w.vivo
            self.caminho += np.linalg.norm(w.pontos[:, 0] - cab_ant, axis=1)
            cab_ant = w.pontos[:, 0].copy()
            sent.acumular(s)
            r_janela += r
            R += r
            if rel.tique(k * M.DT):
                publicar(self._snap(w, motor, probs, textos, pr, dr, R, k, passos, "Vivendo o episódio"))
        if dec.fut is not None:
            dec.pegar()
        recomp.append(r_janela.copy())
        return w, itens, np.array(acoes), np.array(recomp), R, (motor, probs, textos, pr, dr)

    def iteracao_uma(self, publicar):
        torch = self.torch
        w, itens, acoes, recomp, R, extra = self.episodio(publicar)
        T = len(itens)
        ret = np.zeros_like(recomp)
        acc = np.zeros(self.G)
        for t in range(T - 1, -1, -1):
            acc = recomp[t] + self.gama * acc
            ret[t] = acc
        vant = ret - ret.mean(1, keepdims=True)                     # linha de base: média do grupo no mesmo instante
        vant = vant / (vant.std() + 1e-6)
        todos = [(itens[t][g], int(acoes[t, g]), float(vant[t, g])) for t in range(T) for g in range(self.G)]
        self.rng.shuffle(todos)
        mb = 32
        acum = math.ceil(len(todos) / mb)                           # 1 passo do otimizador por episódio (estável)
        self.pol.model.train()
        motor, probs, textos, pr, dr = extra
        snap = self._snap(w, motor, probs, textos, pr, dr, R, 1, 1, "")
        snap.pop("t")                                               # fase de aprendizado: mostra na hora
        perdas, kls = [], []
        self.opt.zero_grad(set_to_none=True)
        for j in range(0, len(todos), mb):
            lote = todos[j:j + mb]
            lg = self.pol.logits([b[0] for b in lote], grad=True)
            logp = torch.log_softmax(lg, -1)
            with torch.no_grad():                                   # âncora: o que o Laya original faria
                logp_ref = torch.log_softmax(self.ref.logits([b[0] for b in lote]).float(), -1)
            a = torch.tensor([b[1] for b in lote], device=lg.device)
            adv = torch.tensor([b[2] for b in lote], device=lg.device)
            ent = -(logp.exp() * logp).sum(-1)
            kl = (logp.exp() * (logp - logp_ref)).sum(-1)
            kls.append(kl.mean().item())
            loss = (-(adv * logp.gather(1, a[:, None])[:, 0]) - self.ent * ent + self.beta_kl * kl).mean() / acum
            loss.backward()
            perdas.append(loss.item() * acum)
            if (j // mb + 1) % acum == 0 or j + mb >= len(todos):
                torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
            snap["status"] = (f"Aprendendo com o episódio {self.iteracao + 1}: ajustando os pesos do Laya "
                              f"({min(j + mb, len(todos))}/{len(todos)} decisões). O próximo episódio começa no dia 0.")
            snap["fase"] = "aprendendo"
            snap["progresso"] = min(j + mb, len(todos)) / len(todos)
            publicar(snap)
            if self.ctrl.parar.is_set():
                raise Parar()
            self.esfriar(publicar, snap)
        self.pol.model.eval()
        eficiencia = float(np.mean(self.caminho / np.maximum(self.energia, 1e-6)))   # mm por unidade de contração
        comp = {k: float(v.mean()) for k, v in w.comp.items()}
        sonda = self.sondar() if (self.iteracao % 10 == 0 or self.reinicio_laya) else None
        linhagens = len({str(f["fundador"]) for f in self.fichas if not f.get("selvagem")})
        campeao = self.evoluir(R)
        self.iteracao += 1
        self.hist.append({"g": self.iteracao, "media": float(R.mean()), "melhor": float(R.max()),
                          "chegaram": float(np.isfinite(w.chegou_dia).mean()), "perda": float(np.mean(perdas)),
                          "acoes": np.bincount(acoes.ravel(), minlength=len(ACOES)).tolist(),
                          "velocidade": float(self.caminho.mean() / (self.dias * M.SEGUNDOS_POR_DIA)),
                          "motor_mudanca": float(np.abs(self.genomas).mean()),
                          "campeao": campeao, "geracao_genes": self.gen_genes,
                          "sentidos_freq": {k: float(v) for k, v in zip(SENTIDOS, self.sentidos.mean(0))},
                          "eficiencia": eficiencia,
                          "comidas_esp": w.comidas.sum(0).tolist(),
                          "componentes": comp, "kl": float(np.mean(kls)), "linhagens": linhagens,
                          **({"sonda": sonda} if sonda else {}),
                          **({"reinicio_laya": True} if self.reinicio_laya else {})})
        self.reinicio_laya = False
        self.salvar()

    def evoluir(self, R):
        """Uma geração do algoritmo genético em ilhas. Devolve a ficha do campeão (de todas as ilhas)."""
        rng = self.rng
        # cada genoma é julgado pela média das suas vidas (até 5): sorte numa vida só não basta
        for i, f in enumerate(self.fichas):
            f["notas"] = (f.get("notas", []) + [float(R[i])])[-5:]
        nota = np.array([np.mean(f["notas"]) for f in self.fichas])
        melhor = int(np.argmax(nota))
        campeao = dict(self.fichas[melhor], nota=float(nota[melhor]), vidas=len(self.fichas[melhor]["notas"]),
                       sentidos=[k for k, t in zip(SENTIDOS, self.sentidos[melhor]) if t])
        self.fichas[melhor]["vitorias"] = self.fichas[melhor].get("vitorias", 0) + 1
        if self.hall is not None:                    # o campeão do hall vivo de novo: a nota dele é a média atual
            for i, f in enumerate(self.fichas):
                if f["id"] == self.hall["ficha"]["id"] and len(f["notas"]) >= 3:
                    self.hall["ficha"]["nota"] = float(nota[i])
        experientes = [i for i, f in enumerate(self.fichas) if len(f["notas"]) >= 3]
        if experientes:
            i = max(experientes, key=lambda j: nota[j])
            if self.hall is None or nota[i] > self.hall["ficha"]["nota"]:
                self.hall = {"genoma": self.genomas[i].copy(), "sentidos": self.sentidos[i].copy(),
                             "ficha": dict(self.fichas[i], nota=float(nota[i]), entrou_no_hall=self.gen_genes + 1)}
        self.gen_genes += 1
        novos, sent_novos, fichas = [], [], []
        for ilha in range(self.n_ilhas):
            membros = np.array([i for i, f in enumerate(self.fichas) if f.get("ilha", 0) == ilha])
            ordem = membros[np.argsort(-nota[membros])]
            for i in ordem[:self.n_elite]:                     # elite da ilha: sobrevive inteira
                novos.append(self.genomas[i].copy())
                sent_novos.append(self.sentidos[i].copy())
                fichas.append(dict(self.fichas[i], nota=float(nota[i]), ilha=ilha))
                if i == melhor:
                    self.idx_campeao = len(novos) - 1

            def torneio():
                c = rng.choice(membros, min(self.torneio, len(membros)), replace=False)
                return c[np.argmax(nota[c])]
            k = 0
            while len(novos) < (ilha + 1) * self.por_ilha - 1:
                a, b = torneio(), torneio()
                tent = 0
                while b == a and tent < 10:
                    b, tent = torneio(), tent + 1
                filho = np.where(rng.random(N_MOTOR) < 0.5, self.genomas[a], self.genomas[b])
                filho = filho + (rng.random(N_MOTOR) < self.p_mutacao) * rng.normal(0, self.sigma_mutacao, N_MOTOR)
                sf = np.where(rng.random(len(SENTIDOS)) < 0.5, self.sentidos[a], self.sentidos[b])
                sf = sf ^ (rng.random(len(SENTIDOS)) < 0.08)
                novos.append(filho)
                sent_novos.append(sf)
                fa, fb = self.fichas[a], self.fichas[b]
                fichas.append({"id": f"G{self.gen_genes}.{'ABC'[ilha]}{k}", "pais": [fa["id"], fb["id"]],
                               "nasceu": self.gen_genes, "vitorias": 0, "notas": [], "ilha": ilha,
                               "fundador": fa["fundador"] if nota[a] >= nota[b] else fb["fundador"]})
                k += 1
            # 1 selvagem por ilha: sem herança nenhuma (diversidade)
            novos.append(rng.standard_normal(N_MOTOR) * 0.1)
            sent_novos.append(rng.random(len(SENTIDOS)) < 0.6)
            fichas.append({"id": f"S{self.gen_genes}.{'ABC'[ilha]}", "pais": [], "nasceu": self.gen_genes,
                           "fundador": self.prox_tribo, "selvagem": True, "vitorias": 0, "notas": [], "ilha": ilha})
            self.prox_tribo += 1
        # migração: a cada 5 gerações, o melhor de cada ilha vai para a próxima (no lugar do último filho)
        if self.gen_genes % self.migrar_cada == 0:
            for ilha in range(self.n_ilhas):
                de = ilha * self.por_ilha                                    # 1ª elite da ilha
                para = ((ilha + 1) % self.n_ilhas + 1) * self.por_ilha - 2   # último filho da ilha seguinte
                novos[para] = novos[de].copy()
                sent_novos[para] = sent_novos[de].copy()
                fichas[para] = dict(fichas[de], ilha=(ilha + 1) % self.n_ilhas, migrou=True, notas=[])
        if self.hall is not None:
            ids = {f["id"] for f in fichas}
            melhor_elite = max(f.get("nota", -1e9) for i, f in enumerate(fichas) if i % self.por_ilha < self.n_elite)
            if self.hall["ficha"]["id"] not in ids and melhor_elite < self.hall["ficha"]["nota"]:
                ilha = self.hall["ficha"].get("ilha", 0)
                i = (ilha + 1) * self.por_ilha - 2                       # no lugar do último filho da ilha dele
                novos[i] = self.hall["genoma"].copy()
                sent_novos[i] = self.hall["sentidos"].copy()
                fichas[i] = dict(self.hall["ficha"], ilha=ilha, retorno=True, notas=self.hall["ficha"].get("notas", []))
                campeao["retorno_do_hall"] = self.hall["ficha"]["id"]
        tribos = {f["fundador"] for f in fichas if not f.get("selvagem")}
        if len(tribos) <= 1:
            filhos = [i for i, f in enumerate(fichas)
                      if i % self.por_ilha >= self.n_elite and not f.get("selvagem") and not f.get("migrou")]
            for j, i in enumerate(filhos[-2:]):
                novos[i] = rng.standard_normal(N_MOTOR) * 0.1
                sent_novos[i] = rng.random(len(SENTIDOS)) < 0.7
                fichas[i] = {"id": f"T{self.prox_tribo}", "pais": [], "nasceu": self.gen_genes,
                             "fundador": self.prox_tribo, "vitorias": 0, "notas": [], "ilha": fichas[i]["ilha"],
                             "tribo_nova": True}
                self.prox_tribo += 1
            campeao["tribos_novas"] = 2
        self.genomas, self.fichas, self.sentidos = np.array(novos), fichas, np.array(sent_novos)
        return campeao

    def salvar(self):
        import shutil
        from safetensors.torch import save_file
        torch = self.torch
        self.pasta.mkdir(parents=True, exist_ok=True)
        (self.pasta / "historico.json").write_text(json.dumps(self.hist), encoding="utf-8")
        np.savez(self.pasta / "genes.npz", genomas=self.genomas, geracao=self.gen_genes, sentidos=self.sentidos)
        if self.hall is not None:
            np.savez(self.pasta / "hall_da_fama.npz", genoma=self.hall["genoma"], sentidos=self.hall["sentidos"])
            (self.pasta / "hall_da_fama.json").write_text(json.dumps(self.hall["ficha"], ensure_ascii=False),
                                                          encoding="utf-8")
        (self.pasta / "linhagem.json").write_text(json.dumps(self.fichas, ensure_ascii=False), encoding="utf-8")
        mm = [h["media"] for h in self.hist]
        melhor = len(mm) >= 3 and media_movel(mm, 5) >= max(media_movel(mm[:i], 5) for i in range(3, len(mm) + 1))
        pastas = [self.pasta / "atual"] + ([self.pasta / "melhor"] if melhor else [])
        snap = snapshot_laya()
        for p in pastas:
            p.mkdir(parents=True, exist_ok=True)
            save_file({k: v.detach().to("cpu").contiguous() for k, v in self.pol.model.state_dict().items()},
                      str(p / "model.safetensors"))
            for d in ("tokenizer", "encoder"):
                if (snap / d).exists():
                    shutil.copytree(snap / d, p / d, dirs_exist_ok=True)
            cfg = json.loads((snap / "rl_agent_config.json").read_text(encoding="utf-8"))
            cfg["temperature"], cfg["temperature_by_options"] = [1.0, 1.0, 1.0], {}
            cfg["c_elegans_rl"] = {"iteracao": self.iteracao}
            (p / "rl_agent_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
            c = self.idx_campeao
            np.save(p / "motor.npy", self.genomas[c])        # o genoma do campeão vai junto
            (p / "sentidos.json").write_text(json.dumps([k for k, t in zip(SENTIDOS, self.sentidos[c]) if t]))
        if self.iteracao % 10 == 0:            # ponto de retomada (grande: ~5 GB com o otimizador)
            tmp = self.pasta / "retomar.pt.tmp"
            torch.save({"modelo": self.pol.model.state_dict(), "otimizador": self.opt.state_dict(),
                        "iteracao": self.iteracao, "hist": self.hist}, tmp)
            os.replace(tmp, self.pasta / "retomar.pt")

    def _snap(self, w, motor, probs, textos, pr, dr, R, k, passos, status):
        d = self.idx_campeao                        # o campeão da geração anterior (de todas as ilhas)
        f0 = self.fichas[d]
        legenda = (f"campeão da geração anterior ({f0['id']}, {f0.get('vitorias', 0)} vitórias)"
                   if self.gen_genes else "fundador F0 (gerador de ritmo original)")
        extra = {"musculos": motor_musculos(motor, d), "prazer": float(pr[d]), "dor": float(dr[d]),
                 "recompensa": float(R[d]), "texto": textos[d], "probs": np.round(probs[d], 3).tolist(),
                 "acoes": ACOES_PT, "acao": int(motor.acao[d]), "legenda_destaque": legenda,
                 "sentidos": [SENTIDOS_PT[k] for k, t in zip(SENTIDOS, self.sentidos[d]) if t],
                 "sentidos_pop": {SENTIDOS_PT[k]: int(v) for k, v in zip(SENTIDOS, self.sentidos.sum(0))},
                 "chegaram": int(np.isfinite(w.chegou_dia).sum()), "total": w.P}
        extra["placar"] = [{"id": f["id"], "comidas": w.comidas[i].tolist(), "vivo": bool(w.vivo[i]),
                            "elite": i % self.por_ilha < self.n_elite and self.gen_genes > 0,
                            "selvagem": bool(f.get("selvagem")), "ilha": "ABC"[f.get("ilha", 0)],
                            "cor": cor_tribo(f["fundador"]),
                            "vitorias": f.get("vitorias", 0), "retorno": bool(f.get("retorno"))}
                           for i, f in enumerate(self.fichas)]
        ultima_sonda = next((h["sonda"] for h in reversed(self.hist) if "sonda" in h), None)
        extra["sonda"] = {"atual": ultima_sonda, "original": self.sonda_ref, "acoes": ACOES_PT,
                          "episodio": next((h["g"] for h in reversed(self.hist) if "sonda" in h), None)}
        marcas = [{"cor": cor_tribo(f["fundador"]),
                   "n_sentidos": int(self.sentidos[i].sum()),
                   "selvagem": bool(f.get("selvagem")), "elite": i % self.por_ilha < self.n_elite and self.gen_genes > 0,
                   "id": f["id"]} for i, f in enumerate(self.fichas)]
        return {"modo": "treino", "cerebro": "B", "geracao": self.iteracao + 1, "progresso": k / passos,
                "t": k * M.DT, "ep": f"B{self.iteracao}",
                "historico": self.hist[-300:],
                "status": status and f"Episódio {self.iteracao + 1} · geração genética {self.gen_genes + 1}: "
                                     f"{self.n_ilhas} ilhas × ({self.n_elite} elites + {self.por_ilha - self.n_elite - 1} filhos + "
                                     f"1 selvagem); cor = tribo",
                "arenas": [arena_json(w, list(range(w.P)), d, "Cérebro B: Laya + genes musculares", extra, marcas)]}

    def rodar(self, publicar, n=100):
        """Roda n episódios e para sozinho (cada episódio já fica salvo)."""
        for self.feitos in range(n):
            self.iteracao_uma(publicar)
        self.feitos = n


def motor_musculos(motor, i):
    """Para a página: a ativação de cada um dos 95 músculos (mesma ordem do cérebro A)."""
    m = getattr(motor, "musc", None)
    return np.round(m[i], 2).tolist() if m is not None else [0.0] * 95


# ======================= Uma vida (ou duas, lado a lado) =======================
def vida(ctrl, publicar, cerebros, dias=M.DIAS_FOME_MORTE, seed=None):
    """cerebros: lista com "A-instinto", "A-treinado", "B-original" ou "B-treinado". Todos começam igual."""
    rng = np.random.default_rng(seed)
    inicio = (rng.uniform(1.0, 1.6), rng.uniform(1.5, M.ALTURA - 1.5), rng.uniform(-np.pi, np.pi))
    semente = int(rng.integers(1 << 30))
    mundos, agentes = [], []
    for nome in cerebros:
        w = M.Mundo(1, semente)
        s = w.reset(inicio)
        if nome.startswith("A"):
            con = Conectoma()
            arq = PASTA_A / "theta_melhor.npy"
            if not arq.exists():
                arq = PASTA_A / "theta_atual.npy"
            theta = np.load(arq) if nome == "A-treinado" and arq.exists() else np.zeros(con.n_param)
            titulo = "A: conectoma " + ("treinado" if nome == "A-treinado" and arq.exists() else "original (instinto)")
            agentes.append({"tipo": "A", "b": CerebroConectoma(con, theta[None], seed=semente), "s": s, "titulo": titulo})
        else:
            pasta = PASTA_B / "melhor"
            if not pasta.exists():
                pasta = PASTA_B / "atual"
            treinado = nome == "B-treinado" and pasta.exists()
            publicar({"modo": "vida", "status": "Carregando o Laya na placa de vídeo...", "arenas": []})
            pol = PoliticaLaya(modelo=pasta if treinado else None, device="cuda")
            mot = pasta / "motor.npy"
            theta_m = np.load(mot) if treinado and mot.exists() else None
            arq_s = pasta / "sentidos.json"
            sent_v = json.loads(arq_s.read_text()) if treinado and arq_s.exists() else SENTIDOS
            genes_s = np.array([[k in sent_v for k in SENTIDOS]])
            agentes.append({"tipo": "B", "pol": pol, "m": MotorCPG(1, theta_m), "se": Sentidos(1), "s": s,
                            "genes_s": genes_s,
                            "probs": np.full((1, len(ACOES)), 0.2), "textos": [""],
                            "titulo": "B: Laya " + ("treinado" if treinado else "original (sem treino)")})
        mundos.append(w)
    rel = Relogio(ctrl)
    R = [0.0] * len(mundos)
    k = 0
    while any(w.vivo[0] for w in mundos) and k < dias * M.PASSOS_POR_DIA:
        arenas = []
        for j, (w, ag) in enumerate(zip(mundos, agentes)):
            s = ag["s"]
            if not w.vivo[0]:
                pr = dr = np.zeros(1)
            else:
                if ag["tipo"] == "A":
                    d, v = ag["b"].passo(s)
                else:
                    if "dec" not in ag:
                        ag["dec"] = Decisor(ag["pol"], rng, amostrar=False)
                    if k % PASSOS_DECISAO == 0:
                        ag["textos"] = ag["se"].textos(s, w.dia, ag["genes_s"])
                        ag["dec"].pedir(ag["textos"])
                    if k % PASSOS_DECISAO == PASSOS_LATENCIA and ag["dec"].fut is not None:
                        a, ag["probs"], _, espera = ag["dec"].pegar()
                        ag["m"].definir(a)
                        ag["se"].nova_decisao(a, s)
                        if espera > 0.5:
                            rel.t0, rel.sim0 = time.perf_counter(), k * M.DT
                    d, v = ag["m"].passo(s)
                    ag["se"].acumular(s)
                s, r, pr, dr, _ = w.passo_fisica(d, v)
                ag["s"] = s
                R[j] += float(r[0])
                ag["pr"], ag["dr"] = pr, dr
            if ag["tipo"] == "A":
                extra = {"neuronios": np.round(ag["b"].a[0], 2).tolist(), "musculos": np.round(ag["b"].musc[0], 2).tolist()}
            else:
                extra = {"musculos": motor_musculos(ag["m"], 0), "texto": ag["textos"][0],
                         "probs": np.round(ag["probs"][0], 3).tolist(), "acoes": ACOES_PT, "acao": int(ag["m"].acao[0])}
            extra.update({"prazer": float(ag.get("pr", np.zeros(1))[0]), "dor": float(ag.get("dr", np.zeros(1))[0]),
                          "recompensa": R[j], "legenda_destaque": "", "chegaram": int(np.isfinite(w.chegou_dia[0])),
                          "total": 1})
            arenas.append(arena_json(w, [0], 0, ag["titulo"], extra))
        k += 1
        if rel.tique(k * M.DT):
            publicar({"modo": "vida", "status": "Uma vida: 1 dia = 2 s (a 1x, 90 dias ≈ 3 min)", "arenas": arenas,
                      "t": k * M.DT, "ep": f"vida{semente}"})
    publicar({"modo": "vida", "status": "Fim da vida.", "arenas": arenas, "fim": True})
