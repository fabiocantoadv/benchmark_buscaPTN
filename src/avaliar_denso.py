#!/usr/bin/env python3
"""Compara EmbeddingGemma x BM25 nas 4 variantes de texto, por tipo de query.

Requer os embeddings ja gerados em embeddings/ (veja docs/embeddings-gemma300.md).
Uso:  python3 src/avaliar_denso.py
"""
import csv, sys
from pathlib import Path
import numpy as np
import pandas as pd

# 26 documentos do corpus nao tem resumo (pedidos renumerados BR122...): mediana
# de 14 palavras contra 145 dos demais. Textos degenerados produzem embeddings
# proximos de quase tudo no espaco vetorial, e o Gemma os colocava em 78 das 180
# posicoes do top-10 (o BM25, em zero, porque nao ha termo para casar). Sao
# excluidos da avaliacao: uma patente sem resumo nao tem como ser representada,
# e mante-los mede ruido de dados, nao qualidade de recuperacao.
EXCLUIR_SEM_RESUMO = True

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import avaliar_benchmark as ab
import baseline_bm25 as bm

DADOS = RAIZ / "dados"
EMB = RAIZ / "embeddings"
CORPUS = DADOS / "corpus_piloto_ipc.tsv"
QUERIES = DADOS / "queries_piloto.tsv"

VARIANTES = {
    "tr":             ("texto_para_embedding",                    "gemma300_tr_docs"),
    "ipc_grupo":      ("texto_para_embedding_ipc_grupo_pt",       "gemma300_tr_ipc_grupo_pt_docs"),
    "ipc_direto":     ("texto_para_embedding_ipc_pt",             "gemma300_tr_ipc_direto_pt_docs"),
    "ipc_hierarquia": ("texto_para_embedding_ipc_hierarquia_pt",  "gemma300_tr_ipc_hierarquia_pt_docs"),
}
METRICA = "nDCG@10"

def main() -> int:
    qrels = ab.carregar_qrels(DADOS / "qrels_piloto.tsv")
    corpus_df = pd.read_csv(CORPUS, sep="\t", dtype=str, low_memory=False,
                            quoting=csv.QUOTE_NONE, escapechar="\\").fillna("")
    sem_resumo = set()
    if EXCLUIR_SEM_RESUMO:
        sem_resumo = set(corpus_df[corpus_df.resumo.str.strip() == ""].num_pedido_normalizado)
        corpus_df = corpus_df[~corpus_df.num_pedido_normalizado.isin(sem_resumo)]
        qrels = qrels[~qrels.num_pedido_normalizado.isin(sem_resumo)]
        corpus_uso = RAIZ / "resultados" / "_corpus_avaliacao.tsv"
        corpus_uso.parent.mkdir(exist_ok=True)
        corpus_df.to_csv(corpus_uso, sep="\t", index=False,
                         quoting=csv.QUOTE_NONE, escapechar="\\")
        print(f"excluidos {len(sem_resumo)} documentos sem resumo; corpus de avaliacao: {len(corpus_df)}")
    else:
        corpus_uso = CORPUS
    qs = pd.read_csv(QUERIES, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                     escapechar="\\").fillna("").set_index("query_id")

    faltando = [n for _, (_, n) in VARIANTES.items() if not (EMB / n).exists()]
    if not (EMB / "gemma300_queries").exists():
        faltando.append("gemma300_queries")
    if faltando:
        print("Faltam colecoes de embeddings:", ", ".join(faltando))
        print("Gere com:  python3 src/gerar_embeddings_gemma300_benchmark.py --variant <nome>")
        return 1

    emb_q = ab.carregar_colecao(EMB / "gemma300_queries", "query_id")
    por_query = {}
    for nome, (coluna, pasta) in VARIANTES.items():
        emb_d = ab.carregar_colecao(EMB / pasta, "num_pedido_normalizado")
        if sem_resumo:
            keep = np.array([i for i, x in enumerate(emb_d["ids"]) if str(x) not in sem_resumo])
            emb_d = {**emb_d, "ids": np.asarray(emb_d["ids"])[keep],
                     "vetores": emb_d["vetores"][keep]}
        r = ab.buscar(emb_q, emb_d, top_k=100)
        por_query[f"gemma_{nome}"] = ab.avaliar(r, qrels, ks=(10,))["por_query"] \
                                       .set_index("query_id")[METRICA]
        rb = bm.buscar_bm25(corpus_uso, QUERIES, coluna_texto=coluna, top_k=100)
        por_query[f"bm25_{nome}"] = ab.avaliar(rb, qrels, ks=(10,))["por_query"] \
                                      .set_index("query_id")[METRICA]

    M = pd.DataFrame(por_query)
    M.insert(0, "tipo", [qs.loc[i, "tipo_query"] for i in M.index])

    print(f"\n=== {METRICA} por query ===")
    print(M.round(3).sort_values("tipo").to_string())

    print(f"\n=== media por tipo de query ===")
    med = M.groupby("tipo").mean(numeric_only=True).round(3)
    print(med.to_string())

    print("\n=== o teste central: denso x lexico na lacuna de vocabulario ===")
    nat = [i for i in M.index if qs.loc[i, "tipo_query"] == "natural"]
    tec = [i for i in M.index if qs.loc[i, "tipo_query"] == "tecnica"]
    linhas = []
    for nome in VARIANTES:
        linhas.append({
            "variante": nome,
            "bm25_tecnica": round(M.loc[tec, f"bm25_{nome}"].mean(), 3),
            "gemma_tecnica": round(M.loc[tec, f"gemma_{nome}"].mean(), 3),
            "bm25_natural": round(M.loc[nat, f"bm25_{nome}"].mean(), 3),
            "gemma_natural": round(M.loc[nat, f"gemma_{nome}"].mean(), 3),
            "ganho_denso_natural": round(M.loc[nat, f"gemma_{nome}"].mean()
                                         - M.loc[nat, f"bm25_{nome}"].mean(), 3),
        })
    print(pd.DataFrame(linhas).to_string(index=False))

    print("\n=== pares tecnica/natural: queda do BM25 e do Gemma ===")
    PARES = [("QP001","QP002"),("QD001","QD002"),("QE001","QE002"),
             ("QC001","QC002"),("QM001","QM002"),("QB001","QB002")]
    linhas = []
    for tq, nq in PARES:
        linhas.append({"par": f"{tq}/{nq}",
                       "bm25_tec": round(M.loc[tq, "bm25_tr"], 3),
                       "bm25_nat": round(M.loc[nq, "bm25_tr"], 3),
                       "queda_bm25": round(M.loc[nq, "bm25_tr"] - M.loc[tq, "bm25_tr"], 3),
                       "gemma_tec": round(M.loc[tq, "gemma_tr"], 3),
                       "gemma_nat": round(M.loc[nq, "gemma_tr"], 3),
                       "queda_gemma": round(M.loc[nq, "gemma_tr"] - M.loc[tq, "gemma_tr"], 3)})
    d = pd.DataFrame(linhas)
    print(d.to_string(index=False))
    print(f"\nqueda media do BM25: {d.queda_bm25.mean():+.3f}  |  do Gemma: {d.queda_gemma.mean():+.3f}")
    print("Se a queda do Gemma for menor, ele fecha parte da lacuna de vocabulario.")

    saida = RAIZ / "resultados"; saida.mkdir(exist_ok=True)
    carimbo = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    M.to_csv(saida / f"denso_x_bm25_{carimbo}.csv")
    print(f"\ngravado: resultados/denso_x_bm25_{carimbo}.csv")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
