#!/usr/bin/env python3
"""Confere o corpus reexportado do Postgres com IPC e o promove a corpus oficial.

Rodar DEPOIS de:
  1) psql ... -f export_corpus_piloto_ipc.sql        -> /tmp/corpus_piloto_bruto.tsv
  2) python3 enriquecer_ipc_json_pt.py --entrada corpus_piloto_bruto.tsv
                                       --saida corpus_piloto_ipc.tsv

Verifica cobertura, garante que nenhum documento julgado se perdeu e regrava
qrels_piloto.tsv sobre o corpus final.
"""
import csv, re, sys
from pathlib import Path
import pandas as pd

DADOS = Path(__file__).resolve().parent.parent / "dados"
ENTRADA = Path(sys.argv[1]) if len(sys.argv) > 1 else DADOS / "corpus_piloto_ipc.tsv"

def ler(p):
    return pd.read_csv(p, sep="\t", dtype=str, low_memory=False,
                       quoting=csv.QUOTE_NONE, escapechar="\\")

novo = ler(ENTRADA)
# lista esperada: os 1.000 numeros congelados do corpus piloto
esperados_raw = [l.strip() for l in (DADOS / "numeros_corpus_piloto.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
pool = ler(DADOS / "pool_piloto_gabarito.tsv").fillna("")

esperados = {"BR" + re.sub(r"[^0-9]", "", n) for n in esperados_raw}
obtidos = set(novo.num_pedido_normalizado.dropna())
perdidos = esperados - obtidos
julgados = set(pool.num_pedido_normalizado)

print(f"esperados: {len(esperados)} | no Postgres: {len(obtidos)} | perdidos: {len(perdidos)}")
if perdidos:
    print("  exemplos:", sorted(perdidos)[:10])
julg_perdidos = julgados & perdidos
if julg_perdidos:
    print(f"ATENCAO: {len(julg_perdidos)} documentos JULGADOS nao voltaram do Postgres:")
    print(" ", sorted(julg_perdidos))
    print("  -> mantenha o texto do xlsx para esses, ou remova-os do gabarito.")

for col in ("ipc", "texto_para_embedding_ipc_pt", "texto_para_embedding_ipc_hierarquia_pt"):
    if col in novo.columns:
        print(f"{col}: {int((novo[col].fillna('') != '').sum())}/{len(novo)} preenchidos")
    else:
        print(f"{col}: AUSENTE — rode o enriquecedor antes")

sem_resumo = int(novo.get("resumo", pd.Series(dtype=str)).fillna("").eq("").sum())
print("sem resumo:", sem_resumo)

# --- regrava o qrels sobre o corpus final ------------------------------------
nota = pool.relevancia_final.where(pool.relevancia_final.str.strip() != "",
                                   pool.relevancia_llm)
pool = pool.assign(relevance=pd.to_numeric(nota, errors="coerce").fillna(0).astype(int))
linhas = []
for qid in sorted(pool.query_id.unique()):
    g = pool[pool.query_id == qid]
    notas = dict(zip(g.num_pedido_normalizado, g.relevance))
    for doc in novo.num_pedido_normalizado.dropna():
        linhas.append((qid, doc, notas.get(doc, 0),
                       "julgado" if doc in notas else "presumido"))
pd.DataFrame(linhas, columns=["query_id", "num_pedido_normalizado", "relevance",
                              "origem_julgamento"]).to_csv(
    DADOS / "qrels_piloto.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
print("qrels_piloto.tsv regravado sobre", ENTRADA.name)
