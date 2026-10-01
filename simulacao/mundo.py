"""Mundo do verme: corpo (física), arena, bactérias (comida), cheiro e os impulsos (dor, prazer, fome).

Tudo é vetorizado em numpy para P vermes ao mesmo tempo (a população do treino anda junta).
Referências: referencias/01_biologia_do_c_elegans.md e referencias/02_o_modelo_e_as_simplificacoes.md.

Unidades: comprimento em mm (o verme adulto tem ~1 mm), tempo em "segundos do corpo".
Relógio da vida: 1 dia = 2 s do corpo, então 90 dias = 180 s (~3 min assistindo em 1x).
"""
import numpy as np

# ---- corpo ----
NSEG = 24                  # 24 fileiras de músculos ao longo do corpo (Cook 2019: dBWML1..24 etc.)
COMPRIMENTO = 1.0          # mm
DS = COMPRIMENTO / NSEG
C_TANGENTE = 1.0           # arrasto ao longo do corpo
C_NORMAL = 20.0            # arrasto de lado: no ágar é 20 a 40x maior (é o que faz a onda empurrar o verme)
TAU_CORPO = 0.12           # s: quanto a curvatura demora a seguir os músculos (músculo + elasticidade)
KAPPA_MAX = 11.0           # 1/mm: curvatura com um lado todo contraído e o outro relaxado
DIFUSAO_KAPPA = 0.15       # suaviza a curvatura (a cutícula é elástica)

# ---- tempo ----
DT = 0.025                 # s por passo
SEGUNDOS_POR_DIA = 2.0
PASSOS_POR_DIA = int(round(SEGUNDOS_POR_DIA / DT))
DIAS_FOME_MORTE = 90.0     # morre de fome com 90 dias sem comer (pedido do projeto; dauer aguenta ~4 meses)
TAU_FOME = 15.0            # dias: escala do crescimento exponencial da fome

# ---- arena ----
LARGURA, ALTURA = 8.0, 5.0             # mm
MARGEM_TOQUE = 0.03

# ---- bactérias: microcolônias espalhadas que se movem devagar, somem ao ser comidas e renascem em outro lugar ----
# Comamonas: o melhor alimento (acelera o desenvolvimento; Watson et al. 2014).
# Bacillus subtilis: bom alimento, cheiro mais fraco.
# Pseudomonas (tipo P. aeruginosa): cheiro muito atraente, nutre pouco e deixa o verme doente — o verme real
#   aprende a evitá-la (Zhang, Lu & Bargmann 2005). Com fome extrema, ainda compensa comer.
ESPECIES = [
    {"nome": "Comamonas", "nutricao": 6.0, "cheiro": 1.0, "doenca": 0.0, "chance": 0.40, "vel": 0.015},
    {"nome": "Bacillus", "nutricao": 4.0, "cheiro": 0.6, "doenca": 0.0, "chance": 0.35, "vel": 0.008},
    {"nome": "Pseudomonas", "nutricao": 3.0, "cheiro": 1.5, "doenca": 1.0, "chance": 0.25, "vel": 0.03},
]
NUTRICAO = np.array([e["nutricao"] for e in ESPECIES])      # dias de fome que cada microcolônia "devolve"
FORCA_CHEIRO = np.array([e["cheiro"] for e in ESPECIES])
DOENCA = np.array([e["doenca"] for e in ESPECIES])
CHANCE = np.array([e["chance"] for e in ESPECIES])
VEL_BACT = np.array([e["vel"] for e in ESPECIES])           # mm/s (o verme anda ~0,2 mm/s)
N_BACT_MIN, N_BACT_MAX = 30, 50                            # quantidade sorteada em cada mundo
SIGMA_CHEIRO = 0.6                                          # mm: alcance do cheiro de cada microcolônia
RAIO_COMER = 0.1                                            # mm: o nariz precisa chegar a esta distância
RAIO_PERTO = 0.35                                           # mm: sente as bactérias sob o corpo (dopamina)
RENASCE = (2.0, 6.0)                                        # s até outra aparecer em lugar sorteado
TAU_DOENCA = 4.0                                            # s: o mal-estar passa devagar
SIGMA_FEROMONIO = 0.8                                       # mm: alcance do feromônio de cada verme
RAIO_ESCASSEZ = 1.2                                         # mm: "comida por perto"
DOR_DOENCA = 0.03                                           # valência negativa por passo com mal-estar 1
DOR_MORTE = 30.0      # a morte é a valência negativa MÁXIMA do modelo: maior que qualquer outro evento
                      # (comer a melhor bactéria com fome extrema dá no máximo ~24)
PESO_CHEIRO = 60.0                     # prazer por chegar mais perto do cheiro (e dor leve por se afastar)
DOR_BATIDA = 0.4                       # dor no instante em que o nariz bate na parede (0,1 se for o corpo)
PESO_PAREDE = 0.005                    # dor leve por passo continuando encostado


def fome(dias_sem_comer):
    """0 logo depois de comer, cresce exponencialmente e chega a 1 no dia 90 sem comida (morte)."""
    d = np.clip(dias_sem_comer, 0, DIAS_FOME_MORTE)
    return (np.exp(d / TAU_FOME) - 1.0) / (np.exp(DIAS_FOME_MORTE / TAU_FOME) - 1.0)


def forma(kappa):
    """Curvatura por segmento (P, NSEG) -> pontos (P, NSEG+1, 2), meios dos segmentos e tangentes,
    no referencial do corpo (centro no ponto médio, ângulo médio zero). Ponto 0 = nariz."""
    theta = np.cumsum(kappa * DS, axis=1)
    theta = theta - theta.mean(axis=1, keepdims=True)
    t = np.stack([np.cos(theta), np.sin(theta)], -1)                     # tangente de cada segmento
    pts = np.concatenate([np.zeros_like(t[:, :1]), np.cumsum(t * DS, axis=1)], axis=1)
    pts = pts - pts.mean(axis=1, keepdims=True)
    meio = 0.5 * (pts[:, 1:] + pts[:, :-1])
    # "para frente" é na direção do nariz: a tangente aponta da cabeça para a cauda
    return pts, meio, t


def rot(v, ang):
    """Gira vetores v (P, 2) ou (P, N, 2) pelos ângulos ang (P)."""
    c, s = np.cos(ang), np.sin(ang)
    if v.ndim == 3:
        c, s = c[:, None], s[:, None]
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


class Mundo:
    """P vermes com bactérias espalhadas.
    compartilhado=True: todos na mesma placa, disputando as mesmas bactérias (o que um come some para todos).
    compartilhado=False: cada verme tem a sua cópia da placa (mesmas bactérias no começo)."""

    def __init__(self, P, seed=0, compartilhado=False):
        self.P = P
        self.rng = np.random.default_rng(seed)
        self.compartilhado = compartilhado
        self.reset()

    # ---------- início ----------
    def reset(self, inicio=None):
        P, rng = self.P, self.rng
        self.kappa = np.zeros((P, NSEG))
        if inicio is None:
            x = rng.uniform(1.0, 1.8, P)
            y = rng.uniform(1.4, ALTURA - 1.4, P)
            ang = rng.uniform(-np.pi, np.pi, P)
        else:
            x, y, ang = (np.full(P, v, dtype=float) for v in inicio)
        self.pos = np.stack([x, y], -1)
        self.phi = ang + np.pi        # phi é o ângulo da tangente (cabeça->cauda); o nariz aponta para "ang"
        self._criar_bacterias()
        self.passo = 0
        self.dias_sem_comer = np.zeros(P)
        self.vivo = np.ones(P, bool)
        self.comeu_total = np.zeros(P)
        self.chegou_dia = np.full(P, np.nan)          # primeiro dia em que comeu
        self.comidas = np.zeros((P, len(ESPECIES)), int)   # quantas microcolônias de cada espécie comeu
        self.doente = np.zeros(P)
        self.especie_cheiro = np.full(P, -1)
        # de onde vem a valência (somas no episódio, por verme): para analisar o treino
        self.comp = {k: np.zeros(P) for k in ("comer", "cheiro melhorando", "cheiro piorando", "parede", "fome",
                                                "infecção", "morte", "energia (ATP)", "custo dos sentidos",
                                                "movimentos bruscos")}
        pts, _, _ = forma(self.kappa)
        self.pontos = self.pos[:, None] + rot(pts, self.phi)
        self.c_lento = self.cheiro(self.pontos[:, 0])
        self.c_rapido = self.c_lento.copy()
        self.c_ant = self.c_lento.copy()
        self.toque_ant = (np.zeros(P), np.zeros(P))
        self.dc = np.zeros(P)
        self.sensores = self._sensores(np.zeros(P), np.zeros(P), np.full(P, -1))
        return self.sensores

    def _criar_bacterias(self):
        rng, nb = self.rng, 1 if self.compartilhado else self.P
        self.banco = np.zeros(self.P, int) if self.compartilhado else np.arange(self.P)
        n = rng.integers(N_BACT_MIN, N_BACT_MAX + 1)
        pos = np.stack([rng.uniform(0.3, LARGURA - 0.3, N_BACT_MAX), rng.uniform(0.3, ALTURA - 0.3, N_BACT_MAX)], -1)
        esp = rng.choice(len(ESPECIES), N_BACT_MAX, p=CHANCE)
        ativa = np.arange(N_BACT_MAX) < n
        # (sem compartilhar: todos começam com a mesma placa, para a comparação ser justa)
        self.b_pos = np.tile(pos, (nb, 1, 1))
        self.b_esp = np.tile(esp, (nb, 1))
        self.b_ativa = np.tile(ativa, (nb, 1))
        self.b_volta = np.where(self.b_ativa, 0.0, np.inf)        # as que não nasceram ficam fora
        ang = rng.uniform(-np.pi, np.pi, (nb, N_BACT_MAX))
        self.b_vel = np.stack([np.cos(ang), np.sin(ang)], -1) * VEL_BACT[self.b_esp][..., None]

    def _mover_bacterias(self):
        rng = self.rng
        # passeio aleatório lento, refletindo nas paredes
        ang = rng.normal(0, 0.6 * np.sqrt(DT), self.b_vel.shape[:2])
        c, s_ = np.cos(ang), np.sin(ang)
        vx, vy = self.b_vel[..., 0], self.b_vel[..., 1]
        self.b_vel = np.stack([c * vx - s_ * vy, s_ * vx + c * vy], -1)
        self.b_pos += self.b_vel * DT
        for eixo, lim in ((0, LARGURA), (1, ALTURA)):
            fora = (self.b_pos[..., eixo] < 0.2) | (self.b_pos[..., eixo] > lim - 0.2)
            self.b_vel[..., eixo] = np.where(fora, -self.b_vel[..., eixo], self.b_vel[..., eixo])
            self.b_pos[..., eixo] = np.clip(self.b_pos[..., eixo], 0.2, lim - 0.2)
        # as comidas renascem em outro lugar, de espécie sorteada
        self.b_volta = np.where(self.b_ativa, 0.0, self.b_volta - DT)
        nasce = (~self.b_ativa) & (self.b_volta <= 0)
        if nasce.any():
            k = int(nasce.sum())
            self.b_pos[nasce] = np.stack([rng.uniform(0.3, LARGURA - 0.3, k), rng.uniform(0.3, ALTURA - 0.3, k)], -1)
            self.b_esp[nasce] = rng.choice(len(ESPECIES), k, p=CHANCE)
            ang = rng.uniform(-np.pi, np.pi, k)
            self.b_vel[nasce] = np.stack([np.cos(ang), np.sin(ang)], -1) * VEL_BACT[self.b_esp[nasce]][:, None]
            self.b_ativa[nasce] = True

    @property
    def dia(self):
        return self.passo / PASSOS_POR_DIA

    def cheiro(self, p):
        """Cheiro no ponto p de cada verme (P, 2): soma das microcolônias, saturando em 1.
        Guarda também qual espécie domina o cheiro ali."""
        bp, ba, be = self.b_pos[self.banco], self.b_ativa[self.banco], self.b_esp[self.banco]
        d2 = ((bp - p[:, None]) ** 2).sum(-1)
        contrib = np.where(ba, FORCA_CHEIRO[be] * np.exp(-d2 / (2 * SIGMA_CHEIRO ** 2)), 0.0)
        por_esp = np.stack([np.where(be == e, contrib, 0).sum(1) for e in range(len(ESPECIES))], 1)
        self.especie_cheiro = np.where(por_esp.sum(1) > 0.02, por_esp.argmax(1), -1)
        self.cheiro_esp = por_esp                  # cheiro de cada espécie (AWA/AWC distinguem os odores)
        S = contrib.sum(1)
        return S / (S + 1.0)

    # ---------- física ----------
    def _mover(self, kappa_novo):
        _, meio_a, _ = forma(self.kappa)
        pts, meio_n, t = forma(kappa_novo)
        u = (meio_n - meio_a) / DT                                            # velocidade da mudança de forma
        tt = t[..., :, None] * t[..., None, :]                                # (P, N, 2, 2)
        eye = np.eye(2)
        D = C_TANGENTE * tt + C_NORMAL * (eye - tt)
        Jm = np.stack([-meio_n[..., 1], meio_n[..., 0]], -1)                  # z x m
        A = np.zeros((self.P, 3, 3))
        A[:, :2, :2] = D.sum(1)
        A[:, :2, 2] = np.einsum("pnij,pnj->pi", D, Jm)
        A[:, 2, :2] = np.einsum("pni,pnij->pj", Jm, D)
        A[:, 2, 2] = np.einsum("pni,pnij,pnj->p", Jm, D, Jm)
        Du = np.einsum("pnij,pnj->pni", D, u)
        b = -np.concatenate([Du.sum(1), np.einsum("pni,pni->p", Jm, Du)[:, None]], -1)
        sol = np.linalg.solve(A, b[..., None])[..., 0]                        # (Vx, Vy, omega) no referencial do corpo
        vivo = self.vivo[:, None]
        self.pos = self.pos + np.where(vivo, rot(sol[:, :2], self.phi) * DT, 0)
        self.phi = self.phi + np.where(self.vivo, sol[:, 2] * DT, 0)
        self.kappa = np.where(vivo, kappa_novo, self.kappa)
        self.pontos = self.pos[:, None] + rot(pts, self.phi)
        # paredes: empurra o corpo inteiro para dentro e marca onde encostou
        lo = self.pontos.min(1)
        hi = self.pontos.max(1)
        empurra = np.maximum(0, -lo) - np.maximum(0, hi - np.array([LARGURA, ALTURA]))
        self.pos = self.pos + empurra
        self.pontos = self.pontos + empurra[:, None]

    def _toques(self):
        p = self.pontos
        dist = np.minimum.reduce([p[..., 0], LARGURA - p[..., 0], p[..., 1], ALTURA - p[..., 1]])   # (P, N+1)
        toca = dist < MARGEM_TOQUE
        nariz = toca[:, :2].any(1)
        anterior = toca[:, 2:NSEG // 2].any(1)
        posterior = toca[:, NSEG // 2:].any(1)
        return nariz, anterior, posterior

    def _populacao(self):
        """Feromônio (ascarosídeos, sentidos por ASK/ADL): soma do cheiro dos OUTROS vermes no nariz — só existe
        quando dividem a mesma placa. E quantas microcolônias há por perto (escassez de comida)."""
        nariz = self.pontos[:, 0]
        if self.compartilhado and self.P > 1:
            d2 = ((nariz[:, None] - self.pos[None]) ** 2).sum(-1)
            fer = np.exp(-d2 / (2 * SIGMA_FEROMONIO ** 2)) * self.vivo[None]
            np.fill_diagonal(fer, 0)
            fer = fer.sum(1)
        else:
            fer = np.zeros(self.P)
        bp, ba = self.b_pos[self.banco], self.b_ativa[self.banco]
        perto = (ba & (((bp - nariz[:, None]) ** 2).sum(-1) < RAIO_ESCASSEZ ** 2)).sum(1)
        return fer, perto

    def _sensores(self, comendo, na_comida, especie_comida):
        nariz, anterior, posterior = self._toques()
        h = fome(self.dias_sem_comer)
        feromonio, comida_perto = self._populacao()
        return {"cheiro": self.c_rapido.copy(), "dcheiro": self.dc.copy(), "toque_nariz": nariz.astype(float),
                "toque_anterior": anterior.astype(float), "toque_posterior": posterior.astype(float),
                "na_comida": np.asarray(na_comida, float), "comendo": np.asarray(comendo, float), "fome": h,
                "dias_sem_comer": self.dias_sem_comer.copy(), "especie_comida": np.asarray(especie_comida),
                "especie_cheiro": self.especie_cheiro.copy(), "doente": self.doente.copy(),
                "feromonio": feromonio, "comida_perto": comida_perto, "cheiro_esp": self.cheiro_esp.copy(),
                "kappa": self.kappa.copy()}

    def passo_fisica(self, musculo_dorsal, musculo_ventral):
        """Recebe a ativação (0..1) dos músculos dorsais e ventrais de cada fileira (P, 24).
        Devolve (sensores, recompensa, prazer, dor, morreu_agora)."""
        alvo = KAPPA_MAX * (np.clip(musculo_ventral, 0, 1) - np.clip(musculo_dorsal, 0, 1))
        k = self.kappa + (alvo - self.kappa) * (DT / TAU_CORPO)
        lap = np.zeros_like(k)
        lap[:, 1:-1] = k[:, 2:] - 2 * k[:, 1:-1] + k[:, :-2]
        k = k + DIFUSAO_KAPPA * lap
        self._mover(k)
        self._mover_bacterias()
        self.passo += 1

        # cheiro no nariz: nível rápido e lento (adaptação); a diferença é "está melhorando ou piorando"
        c = self.cheiro(self.pontos[:, 0])
        self.c_rapido += (c - self.c_rapido) * (DT / 0.3)
        self.c_lento += (c - self.c_lento) * (DT / 2.0)
        self.dc = (self.c_rapido - self.c_lento) * 20.0

        # comer: o nariz chega numa microcolônia -> ela some (e outra renasce depois em outro lugar)
        nariz = self.pontos[:, 0]
        bp, ba, be = self.b_pos[self.banco], self.b_ativa[self.banco], self.b_esp[self.banco]
        d2 = ((bp - nariz[:, None]) ** 2).sum(-1)
        alcanca = ba & (d2 < RAIO_COMER ** 2) & self.vivo[:, None]
        na_comida = np.minimum(1.0, (ba & (d2 < RAIO_PERTO ** 2)).sum(1) / 2.0) * self.vivo
        come = np.where(alcanca, NUTRICAO[be], 0.0).sum(1)
        doenca = np.where(alcanca, DOENCA[be], 0.0).sum(1)
        especie_comida = np.where(alcanca.any(1), np.where(alcanca, be, -1).max(1), -1)
        vi, bi = np.nonzero(alcanca)
        if len(vi):
            np.add.at(self.comidas, (vi, be[vi, bi]), 1)
            banco = self.banco[vi]
            self.b_ativa[banco, bi] = False
            self.b_volta[banco, bi] = self.rng.uniform(*RENASCE, len(vi))
        comendo = come > 0
        dias_dt = DT / SEGUNDOS_POR_DIA
        h_antes = fome(self.dias_sem_comer)
        self.dias_sem_comer = np.where(self.vivo, np.maximum(0.0, self.dias_sem_comer + dias_dt - come), self.dias_sem_comer)
        self.comeu_total += come
        primeira = comendo & np.isnan(self.chegou_dia)
        self.chegou_dia[primeira] = self.dia
        self.doente = self.doente * (1 - DT / TAU_DOENCA) + doenca

        h = fome(self.dias_sem_comer)
        sens = self._sensores(comendo, na_comida, especie_comida)
        morreu = self.vivo & (self.dias_sem_comer >= DIAS_FOME_MORTE)
        self.vivo &= ~morreu

        # ---- os impulsos primordiais ----
        # prazer: comer (vale mais quanto maior a fome) + um pouco de "o cheiro está melhorando" (antecipação)
        # (o do cheiro é a diferença do cheiro entre passos: ir e voltar não dá lucro, só chegar mais perto)
        dcheiro = (c - self.c_ant) * (0.3 + h) * PESO_CHEIRO
        self.c_ant = c
        prazer = come * (1.0 + 3.0 * h_antes) + np.clip(dcheiro, 0, None)
        # dor: encostar na parede (nociceptores ASH/toque) + o incômodo da fome, que cresce com ela
        parede = sens["toque_nariz"] * 1.0 + 0.3 * (sens["toque_anterior"] + sens["toque_posterior"])
        corpo = np.maximum(sens["toque_anterior"], sens["toque_posterior"])
        batida = DOR_BATIDA * np.clip(sens["toque_nariz"] - self.toque_ant[0], 0, 1) \
            + 0.25 * DOR_BATIDA * np.clip(corpo - self.toque_ant[1], 0, 1)
        self.toque_ant = (sens["toque_nariz"], corpo)
        # mal-estar depois de comer Pseudomonas (patógeno)
        dor = np.clip(-dcheiro, 0, None) + batida + parede * PESO_PAREDE + h * 0.02 + morreu * DOR_MORTE \
            + DOR_DOENCA * np.minimum(self.doente, 2.0)
        vivo_antes = self.vivo | morreu
        prazer, dor = prazer * vivo_antes, dor * vivo_antes
        for k, v in (("comer", come * (1.0 + 3.0 * h_antes)), ("cheiro melhorando", np.clip(dcheiro, 0, None)),
                     ("cheiro piorando", -np.clip(-dcheiro, 0, None)), ("parede", -(batida + parede * PESO_PAREDE)),
                     ("fome", -h * 0.02), ("infecção", -DOR_DOENCA * np.minimum(self.doente, 2.0)),
                     ("morte", -DOR_MORTE * morreu)):
            self.comp[k] += v * vivo_antes
        self.sensores = sens
        return sens, prazer - dor, prazer, dor, morreu

    # ---------- para a página ----------
    def bacterias_json(self, i):
        b = self.banco[i]
        m = self.b_ativa[b]
        idx = np.nonzero(m)[0]                     # o índice é a identidade fixa de cada microcolônia
        return [[round(float(self.b_pos[b, j, 0]), 3), round(float(self.b_pos[b, j, 1]), 3), int(self.b_esp[b, j]), int(j)]
                for j in idx]
