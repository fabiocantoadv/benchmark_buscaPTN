#!/usr/bin/env python3
"""Pool piloto para gabarito qualificado (poucas queries, ~30 docs cada).

Sistemas do pool (sem GPU): BM25 sobre as 3 variantes de texto.
Quando os embeddings densos existirem, basta acrescentar os rankings ao dict.
"""
# NOTA: este script construiu o corpus piloto a partir da amostra original
# de 1.000 e do amostra_50000.xlsx, arquivos que nao estao mais no
# repositorio (abordagem por regras, removida). O sorteio dos distratores
# esta congelado em dados/numeros_corpus_piloto.txt; para reproduzir o
# corpus use src/rodar_export_ipc.sh. Mantido como registro do metodo.

import csv, sys
import pandas as pd
import baseline_bm25 as bm

QUERIES_PILOTO = ["QF002", "QA006", "QG001"]
PROF = 12          # top-k de cada sistema que entra no pool
N_ALVO = 30        # candidatos por query
SEED = 20260901

def ler(p):
    return pd.read_csv(p, sep="\t", dtype=str, low_memory=False,
                       quoting=csv.QUOTE_NONE, escapechar="\\")

docs = ler("patentes_benchmark_amostra_1000_ipc_pt.tsv")
queries = ler("queries_benchmark_patentes.tsv")
queries = queries[queries.query_id.isin(QUERIES_PILOTO)].reset_index(drop=True)
queries.to_csv("/tmp/queries_piloto.tsv", sep="\t", index=False,
               quoting=csv.QUOTE_NONE, escapechar="\\")

sistemas = {
    "bm25_tr": "texto_para_embedding",
    "bm25_ipc": "texto_para_embedding_ipc_pt",
    "bm25_hier": "texto_para_embedding_ipc_hierarquia_pt",
}

registros = {}
for nome, col in sistemas.items():
    r = bm.buscar_bm25("patentes_benchmark_amostra_1000_ipc_pt.tsv",
                       "/tmp/queries_piloto.tsv", coluna_texto=col, top_k=PROF)
    for i, qid in enumerate(r["query_ids"]):
        for pos, did in enumerate(r["doc_ids_ranking"][i][:PROF], 1):
            k = (str(qid), str(did))
            reg = registros.setdefault(k, {"query_id": str(qid),
                                           "num_pedido_normalizado": str(did),
                                           "sistemas": set(), "melhor_posicao": pos})
            reg["sistemas"].add(nome)
            reg["melhor_posicao"] = min(reg["melhor_posicao"], pos)

pool = pd.DataFrame([{**v, "sistemas": "|".join(sorted(v["sistemas"])),
                      "n_sistemas": len(v["sistemas"]),
                      "origem": "pooling"} for v in registros.values()])

# completa ate N_ALVO com negativos aleatorios (fora do pool) -> ancoram o zero
partes = []
for qid, g in pool.groupby("query_id"):
    g = g.sort_values(["n_sistemas", "melhor_posicao"], ascending=[False, True])
    falta = N_ALVO - len(g)
    if falta > 0:
        fora = docs[~docs.num_pedido_normalizado.isin(g.num_pedido_normalizado)]
        extra = fora.sample(falta, random_state=SEED + hash(qid) % 1000)
        extra = pd.DataFrame({"query_id": qid,
                              "num_pedido_normalizado": extra.num_pedido_normalizado.values,
                              "sistemas": "", "melhor_posicao": 999,
                              "n_sistemas": 0, "origem": "aleatorio"})
        g = pd.concat([g, extra], ignore_index=True)
    partes.append(g.head(N_ALVO))
pool = pd.concat(partes, ignore_index=True)

cols_doc = ["num_pedido_normalizado", "num_publicacao", "titulo", "resumo", "ipc"]
out = pool.merge(docs[cols_doc], on="num_pedido_normalizado", how="left")
out = out.merge(queries[["query_id", "tema", "tipo_query", "query_text",
                         "criterio_relevancia_alta", "observacoes_negativos_dificeis"]],
                on="query_id", how="left")
out["relevancia_llm"] = ""
out["justificativa_llm"] = ""
out["relevancia_final"] = ""
out["revisor"] = ""
out = out.sort_values(["query_id", "n_sistemas", "melhor_posicao"],
                      ascending=[True, False, True])
out.to_csv("pool_piloto_gabarito.tsv", sep="\t", index=False,
           quoting=csv.QUOTE_NONE, escapechar="\\")
print(out.groupby(["query_id", "origem"]).size())
print("total", len(out))
