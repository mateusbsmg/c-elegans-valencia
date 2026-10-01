"""Gera dados/conectoma.json a partir das planilhas originais (rodar uma vez só).

Fontes (baixadas de github.com/openworm/c302/tree/master/c302/data):
- SI_5_Connectome_adjacency_matrices.xlsx: Cook et al. 2019 (Nature 571:63). Hermafrodita:
  sinapses químicas e junções comunicantes (gap junctions) de TODAS as células, com os 95 músculos.
- CElegansNeuronTables.xls: Varshney et al. 2011 + neurotransmissor de cada neurônio (tabela do OpenWorm).

Uso (precisa de pandas/xlrd/openpyxl só aqui; cache no D:):
    set UV_CACHE_DIR=D:\\Mateus\\.uv-cache
    uv run --no-project --python 3.12 --with pandas --with xlrd --with openpyxl python dados\\preparar_conectoma.py
"""
import json
import re
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
COOK = AQUI / "SI_5_Connectome_adjacency_matrices.xlsx"
TABELAS = AQUI / "CElegansNeuronTables.xls"

# Seções da matriz do Cook 2019 que são neurônios (o resto é músculo, glia, células de órgão...)
SECOES_NEURONIOS = {"PHARYNX": "faringe", "SENSORY NEURONS": "sensorial", "INTERNEURONS": "interneuronio",
                    "MOTOR NEURONS": "motor", "SEX SPECIFIC": "sexo", "HEAD MOTOR NEURONS": "motor",
                    "SUBLATERAL MOTOR NEURONS": "motor", "VENTRAL CORD MOTOR NEURONS": "motor",
                    "BODY MOTOR NEURONS": "motor", "UNKNOWN": "desconhecido"}
RE_MUSCULO = re.compile(r"^([dv])BWM([LR])(\d+)$")


def normal(nome):
    """AS03 -> AS3, DA01 -> DA1 (a tabela do OpenWorm não usa zero à esquerda)."""
    nome = str(nome).strip()
    return re.sub(r"^([A-Z]+)0(\d)$", r"\1\2", nome)


def ler_matriz(aba):
    df = pd.read_excel(COOK, sheet_name=aba, header=None)
    colunas = [normal(c) if isinstance(c, str) else None for c in df.iloc[2]]
    secao, linhas = None, []
    for i in range(3, len(df)):
        if isinstance(df.iat[i, 0], str) and df.iat[i, 0].strip():
            secao = df.iat[i, 0].strip()
        nome = df.iat[i, 2]
        if isinstance(nome, str) and nome.strip():
            linhas.append((i, normal(nome), secao))
    return df, colunas, linhas


def main():
    df, colunas, linhas = ler_matriz("hermaphrodite chemical")
    secoes = sorted({s for _, _, s in linhas})
    print("seções encontradas:", secoes)
    neuronios, categoria = [], {}
    for _, nome, secao in linhas:
        cat = next((v for k, v in SECOES_NEURONIOS.items() if secao and secao.upper().startswith(k)), None)
        if cat and nome not in categoria:
            neuronios.append(nome)
            categoria[nome] = cat
    # CANL/CANR aparecem como "outras células" no Cook, mas são neurônios (contados nos 302)
    for extra in ("CANL", "CANR"):
        if extra not in categoria:
            neuronios.append(extra)
            categoria[extra] = "interneuronio"
    musculos = sorted({c for c in colunas if c and RE_MUSCULO.match(c)},
                      key=lambda m: (m[0], m[4], int(RE_MUSCULO.match(m).group(3))))

    quimicas, neuro_musculo = [], []
    idx_col = {c: j for j, c in enumerate(colunas) if c}
    for i, origem, _ in linhas:
        if origem not in categoria:
            continue
        for alvo, j in idx_col.items():
            v = df.iat[i, j]
            if pd.notna(v) and float(v) > 0:
                if alvo in categoria:
                    quimicas.append([origem, alvo, int(v)])
                elif alvo in musculos:
                    neuro_musculo.append([origem, alvo, int(v)])

    dg, colg, linhasg = ler_matriz("herm gap jn symmetric")
    idxg = {c: j for j, c in enumerate(colg) if c}
    gaps, vistos = [], set()
    for i, a, _ in linhasg:
        if a not in categoria:
            continue
        for b, j in idxg.items():
            v = dg.iat[i, j]
            if b in categoria and a != b and pd.notna(v) and float(v) > 0 and (b, a) not in vistos:
                vistos.add((a, b))
                gaps.append([a, b, int(v)])

    # neurotransmissor: tabela do OpenWorm (coluna "Neurotransmitter" das conexões que o neurônio envia)
    tab = pd.read_excel(TABELAS, sheet_name=None)
    nt = {}
    for aba in ("Connectome", "NeuronsToMuscle"):
        d = tab[aba]
        col_origem = "Origin" if "Origin" in d.columns else "Neuron"
        for _, r in d.iterrows():
            t = r.get("Neurotransmitter")
            if isinstance(t, str) and t != "Generic_GJ":
                nt.setdefault(normal(r[col_origem]), set()).update(x.strip() for x in t.split("_"))
    sensor = tab["Sensory"]
    funcao = {normal(r["Neuron"]): str(r["Function"]) for _, r in sensor.iterrows() if pd.notna(r["Function"])}

    saida = {
        "fonte": "Cook et al. 2019 (SI 5, hermafrodita) + neurotransmissores da tabela do OpenWorm (Varshney 2011)",
        "neuronios": [{"nome": n, "categoria": categoria[n], "neurotransmissor": sorted(nt.get(n, [])),
                       "funcao": funcao.get(n, "")} for n in neuronios],
        "musculos": musculos,
        "quimicas": quimicas,            # [origem, alvo, nº de sinapses]
        "juncoes": gaps,                 # [a, b, nº] (simétrico)
        "neuronio_musculo": neuro_musculo,
    }
    (AQUI / "conectoma.json").write_text(json.dumps(saida, ensure_ascii=False), encoding="utf-8")
    cats = {}
    for n in neuronios:
        cats[categoria[n]] = cats.get(categoria[n], 0) + 1
    print(f"neurônios {len(neuronios)} {cats} | músculos {len(musculos)} | químicas {len(quimicas)} | "
          f"junções {len(gaps)} | neurônio->músculo {len(neuro_musculo)} | sem neurotransmissor "
          f"{sum(1 for n in neuronios if n not in nt)}")


if __name__ == "__main__":
    main()
