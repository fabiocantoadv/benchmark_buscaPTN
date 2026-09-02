#!/usr/bin/env python3
"""Monta o pool de um par de queries (tecnica + parafrase natural).

Duas fontes, união:
  1. pooling automatico: top-k de cada variante de texto, para AS DUAS queries
     do par (uma recupera o que a outra nao alcanca);
  2. corrida manual: documentos que casam simultaneamente com um padrao de
     termos tecnicos e com o IPC do tema.

A corrida manual e essencial. Um pool so de BM25 deixa de fora justamente os
documentos que a query em linguagem natural deveria recuperar mas nao recupera
por falta de sobreposicao lexical -- ou seja, enviesa o gabarito contra os
sistemas que resolvem a lacuna de vocabulario.
"""
import csv, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_bm25 as bm

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
CORPUS = DADOS / "corpus_piloto_ipc.tsv"
VARIANTES = ["texto_para_embedding", "texto_para_embedding_ipc_grupo_pt",
             "texto_para_embedding_ipc_pt", "texto_para_embedding_ipc_hierarquia_pt"]

def montar(par: tuple[str, str], termos: str, ipc: str, prof: int = 8) -> pd.DataFrame:
    d = pd.read_csv(CORPUS, sep="\t", dtype=str, low_memory=False,
                    quoting=csv.QUOTE_NONE, escapechar="\\").fillna("")
    q = pd.read_csv(DADOS / "queries_piloto.tsv", sep="\t", dtype=str,
                    quoting=csv.QUOTE_NONE, escapechar="\\").fillna("")
    q[q.query_id.isin(par)].to_csv("/tmp/_par.tsv", sep="\t", index=False,
                                   quoting=csv.QUOTE_NONE, escapechar="\\")
    origem: dict[str, set] = {}
    for col in VARIANTES:
        r = bm.buscar_bm25(CORPUS, "/tmp/_par.tsv", coluna_texto=col, top_k=prof)
        for i, qid in enumerate(r["query_ids"]):
            for doc in r["doc_ids_ranking"][i]:
                origem.setdefault(str(doc), set()).add(f"bm25:{str(qid)}")
    txt = (d.titulo + " " + d.resumo).str.lower()
    manual = d[txt.str.contains(termos, regex=True, na=False)
               & d.ipc.str.contains(ipc, regex=True, na=False)]
    for doc in manual.num_pedido_normalizado:
        origem.setdefault(str(doc), set()).add("manual")
    pool = pd.DataFrame([{"num_pedido_normalizado": k,
                          "origem": "manual" if o == {"manual"} else
                                    ("pooling+manual" if "manual" in o else "pooling"),
                          "recuperado_por": "|".join(sorted(o))}
                         for k, o in origem.items()])
    return pool.merge(d[["num_pedido_normalizado", "titulo", "resumo", "ipc"]],
                      on="num_pedido_normalizado", how="left")

if __name__ == "__main__":
    print(__doc__)
