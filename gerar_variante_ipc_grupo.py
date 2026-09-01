#!/usr/bin/env python3
"""Acrescenta a variante `ipc_grupo`: so as descricoes de grupo e subgrupo.

Motivacao: em `ipc_hierarquia` mais da metade do texto sao os niveis de cima
(secao, classe, subclasse), identicos em milhares de documentos — nao
discriminam e diluem o resumo no BM25. Esta variante mantem apenas os niveis
com barra no simbolo (grupo principal e subgrupo), que sao os que carregam
conteudo tecnico especifico.

Entrada/saida: corpus_piloto_ipc.tsv (acrescenta 2 colunas, in place).
"""
import csv, re, sys
from pathlib import Path
import pandas as pd

ARQ = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus_piloto_ipc.tsv")
INICIO = re.compile(r"(?:^|; )([A-H]\d{2}[A-Z]\s?\d+/\d+):\s")

def niveis_grupo(hierarquia: str) -> str:
    """Extrai, de cada cadeia, so os niveis cujo simbolo tem barra."""
    if not hierarquia:
        return ""
    marcas = list(INICIO.finditer(hierarquia))
    trechos = []
    for i, m in enumerate(marcas):
        fim = marcas[i + 1].start() if i + 1 < len(marcas) else len(hierarquia)
        trechos.append(hierarquia[m.end():fim])
    if not trechos:                      # formato inesperado: nao inventa nada
        return ""
    vistos, saida = set(), []
    for trecho in trechos:
        for nivel in trecho.split(" > "):
            nivel = nivel.strip().rstrip(";").strip()
            codigo = nivel.split(" - ")[0]
            if "/" in codigo and nivel not in vistos:
                vistos.add(nivel)
                saida.append(nivel)
    return "; ".join(saida)

d = pd.read_csv(ARQ, sep="\t", dtype=str, low_memory=False,
                quoting=csv.QUOTE_NONE, escapechar="\\").fillna("")
d["ipc_grupo_descricao_pt"] = d.ipc_hierarquia_descricao_pt.map(niveis_grupo)
d["texto_para_embedding_ipc_grupo_pt"] = [
    (t + " Classificacao IPC: " + g).strip() if g else t
    for t, g in zip(d.texto_para_embedding, d.ipc_grupo_descricao_pt)
]
d.to_csv(ARQ, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")

vazios = int((d.ipc_grupo_descricao_pt == "").sum())
print(f"{ARQ.name}: coluna gerada | sem descricao de grupo: {vazios}/{len(d)}")
for c in ["texto_para_embedding", "texto_para_embedding_ipc_grupo_pt",
          "texto_para_embedding_ipc_pt", "texto_para_embedding_ipc_hierarquia_pt"]:
    print(f"  {c:42s} mediana {int(d[c].str.split().str.len().median()):4d} palavras")
