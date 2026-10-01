"""Servidor da página do C. elegans: mostra ao vivo o treino por reforço e a vida dos vermes.

Nada começa sozinho: o servidor abre parado e só treina quando você aperta o botão na página.

Uso:
    D:\\Mateus\\Laya\\.venv\\Scripts\\python.exe servidor.py
    abrir http://127.0.0.1:8770
"""
import collections
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "simulacao"))

import treino as T  # noqa: E402
from cerebro_conectoma import Conectoma  # noqa: E402


class Controle:
    def __init__(self):
        self.velocidade = 4.0          # 1 = tempo real do corpo; 0 = o mais rápido possível
        self.parar = threading.Event()
        self.pausado = threading.Event()
        self.tarefa = None
        self.nome = "parado"
        self.snap = {"modo": "parado", "status": "Parado. Escolha um treino ou uma vida.", "arenas": []}
        self.lock = threading.Lock()
        self.quadros = collections.deque(maxlen=400)     # (seq, quadro): a página toca em ritmo constante
        self.seq = 0
        self.treinadores = {}          # o Laya fica carregado entre um treino e outro

    def publicar(self, snap):
        with self.lock:
            self.snap = snap
            self.seq += 1
            self.quadros.append((self.seq, json.dumps(snap, ensure_ascii=False)))

    def iniciar(self, nome, alvo):
        self.encerrar()
        self.parar.clear()
        self.pausado.clear()
        self.nome = nome

        def rodar():
            try:
                alvo()
            except T.Parar:
                self.publicar({**self.snap, "status": "Parado. O progresso do treino fica salvo; dá para continuar."})
            except Exception as e:
                traceback.print_exc()
                self.publicar({**self.snap, "status": f"Erro: {type(e).__name__}: {e}"})
            self.nome = "parado"
        self.tarefa = threading.Thread(target=rodar, daemon=True)
        self.tarefa.start()

    def encerrar(self):
        if self.tarefa and self.tarefa.is_alive():
            self.parar.set()
            self.tarefa.join(timeout=60)


CTRL = Controle()
CON = Conectoma()


def treinar(cerebro, n=100, dias=30):
    n = max(1, int(n))
    dias = min(90, max(5, int(dias)))

    def alvo():
        if cerebro == "A":
            CTRL.publicar({"modo": "treino", "cerebro": "A", "status": "Preparando o treino do conectoma...", "arenas": []})
            tr = T.TreinoConectoma(CTRL, dias=dias)
        else:
            CTRL.publicar({"modo": "treino", "cerebro": "B", "status": "Carregando o Laya na placa de vídeo (~30 s)...",
                           "arenas": []})
            tr = CTRL.treinadores.get("B") or T.TreinoLaya(CTRL)
            CTRL.treinadores["B"] = tr
            tr.dias = dias
        tr.feitos = 0
        total0 = len(tr.hist)

        def pub(snap):
            CTRL.publicar({**snap, "sessao": [min(tr.feitos + 1, n), n], "total_salvo": len(tr.hist)})
        tr.rodar(pub, n)
        nome = "gerações" if cerebro == "A" else "episódios"
        CTRL.publicar({**CTRL.snap, "sessao": [n, n], "total_salvo": len(tr.hist), "progresso": 1,
                       "status": f"Concluído: {len(tr.hist) - total0} {nome} nesta sessão. Total salvo: "
                                 f"{len(tr.hist)}. Da próxima vez continua daqui."})
    CTRL.iniciar(f"treino {cerebro}", alvo)


def viver(cerebros):
    if any(c.startswith("B") for c in cerebros):
        CTRL.treinadores.pop("B", None)      # libera a placa se havia um treino do Laya carregado
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    CTRL.iniciar("vida", lambda: T.vida(CTRL, CTRL.publicar, cerebros))


def info():
    def hist(p):
        try:
            return json.loads((p / "historico.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
    ha, hb = hist(T.PASTA_A), hist(T.PASTA_B)
    return {"tarefa": CTRL.nome, "velocidade": CTRL.velocidade, "pausado": CTRL.pausado.is_set(),
            "conectoma": CON.resumo(), "geracoes_A": len(ha), "iteracoes_B": len(hb),
            "historico_A": ha[-300:], "historico_B": hb[-300:],
            "neuronios": CON.neuronios, "categorias": CON.categoria, "musculos": CON.musculos}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def enviar(self, code, corpo, tipo="application/json"):
        if not isinstance(corpo, bytes):
            corpo = json.dumps(corpo, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        caminho = self.path.split("?", 1)[0]
        if caminho in ("/", "/index.html"):
            self.enviar(200, (AQUI / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif caminho == "/api/estado":
            with CTRL.lock:
                snap = CTRL.snap
            self.enviar(200, {**snap, "tarefa": CTRL.nome, "velocidade": CTRL.velocidade,
                              "pausado": CTRL.pausado.is_set()})
        elif caminho == "/api/quadros":
            q = self.path.split("desde=", 1)
            desde = int(q[1]) if len(q) > 1 and q[1].isdigit() else 0
            with CTRL.lock:
                novos = [j for n, j in CTRL.quadros if n > desde][-150:]
                if desde == 0 and not novos:
                    novos = [json.dumps(CTRL.snap, ensure_ascii=False)]
                seq = CTRL.seq
            corpo = ('{"seq": %d, "velocidade": %s, "pausado": %s, "tarefa": %s, "quadros": [%s]}' % (
                seq, json.dumps(CTRL.velocidade), json.dumps(CTRL.pausado.is_set()), json.dumps(CTRL.nome),
                ",".join(novos))).encode("utf-8")
            self.enviar(200, corpo)
        elif caminho == "/api/info":
            self.enviar(200, info())
        else:
            self.enviar(404, {"erro": "não encontrado"})

    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            acao = d.get("acao")
            if acao == "treinar":
                treinar(d["cerebro"], d.get("n", 100), d.get("dias", 30))
            elif acao == "vida":
                viver(d["cerebros"])
            elif acao == "parar":
                CTRL.encerrar()
            elif acao == "pausar":
                (CTRL.pausado.clear if CTRL.pausado.is_set() else CTRL.pausado.set)()
            elif acao == "velocidade":
                CTRL.velocidade = float(d["valor"])
            else:
                raise ValueError(f"ação desconhecida: {acao}")
            self.enviar(200, {"ok": True})
        except Exception as e:
            self.enviar(400, {"erro": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    print(f"C. elegans em http://127.0.0.1:8770  ({CON.resumo()})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8770), Handler).serve_forever()
