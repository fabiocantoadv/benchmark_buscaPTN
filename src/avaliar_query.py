#!/usr/bin/env python3
"""Avalia UMA query contra um gabarito simples (fase 2).

Gabarito: TSV com query_id, query_text, num_pedido, titulo, resumo e a coluna
de relevancia (0/1/2). A coluna usada e, nesta ordem: relevancia_humana,
relevancia, relevancia_llm — ou seja, o julgamento revisado manda quando
existe. --relevancia força uma coluna especifica.
Tudo que esta no corpus e nao esta no gabarito entra como relevancia 0.

    python3 src/avaliar_query.py dados/gabaritos/QN003.tsv

Gemma entra automaticamente se houver embedding da query em
embeddings/gemma300_queries_fase2 (gere no Mac, com o modelo baixado):

    python3 src/gerar_embeddings_gemma300_benchmark.py --kind queries

Cada configuracao de instrucao presente em embeddings/ vira uma linha da
tabela: "gemma" (instrucao PT nos dois lados), "gemma_si" (sem instrucao,
veja src/gerar_sem_instrucao.sh) e "gemma_qipc" (so a instrucao da query
muda, pedindo correspondencia de classificacao):

    python3 src/gerar_embeddings_gemma300_benchmark.py --kind queries \
        --query-instruction ipc --overwrite
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np, pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import avaliar_benchmark as ab
import baseline_bm25 as bm

CORPUS = RAIZ / "dados" / "corpus_piloto_ipc.tsv"
EMB = RAIZ / "embeddings"
EMB_QUERIES = EMB / "gemma300_queries_fase2"
# Cada configuracao do Gemma: rotulo -> (sufixo da colecao de queries,
# sufixo das colecoes de documento). Separar os dois permite trocar so a
# instrucao da query, que custa 2 embeddings, contra 974 x 4 do lado do
# documento. Configuracoes ausentes em embeddings/ sao ignoradas.
CONFIGS_GEMMA = {
    "gemma":      ("", ""),                             # instrucao PT nos dois lados
    "gemma_si":   ("_sem_instrucao", "_sem_instrucao"),  # sem instrucao nenhuma
    "gemma_qipc": ("_ipc", ""),                          # query pede correspondencia de CIP
}
VARIANTES = {
    "tr":             ("texto_para_embedding",                   "gemma300_tr_docs"),
    "ipc_grupo":      ("texto_para_embedding_ipc_grupo_pt",      "gemma300_tr_ipc_grupo_pt_docs"),
    "ipc_direto":     ("texto_para_embedding_ipc_pt",            "gemma300_tr_ipc_direto_pt_docs"),
    "ipc_hierarquia": ("texto_para_embedding_ipc_hierarquia_pt", "gemma300_tr_ipc_hierarquia_pt_docs"),
}
# Mesma exclusao de avaliar_denso.py: 26 pedidos renumerados sem resumo.
EXCLUIR_SEM_RESUMO = True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("gabarito", type=Path)
    p.add_argument("--ks", default="5,10,20")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--saida", type=Path, default=None)
    p.add_argument("--relevancia", default=None,
                   help="Coluna de relevancia a usar. Padrao: relevancia_humana, "
                        "senao relevancia, senao relevancia_llm.")
    args = p.parse_args()
    ks = tuple(int(k) for k in args.ks.split(","))

    gab = pd.read_csv(args.gabarito, sep="\t", dtype=str).fillna("")
    gab = gab.drop_duplicates(subset=["query_id", "num_pedido"], keep="first")
    col = args.relevancia
    if col is None:
        for c in ("relevancia_humana", "relevancia", "relevancia_llm"):
            if c in gab.columns and (gab[c].str.strip() != "").all():
                col = c
                break
    if col is None or col not in gab.columns:
        raise SystemExit(f"{args.gabarito}: nenhuma coluna de relevancia utilizavel")
    gab[col] = gab[col].astype(int)
    qid = gab.query_id.iloc[0]
    qtext = gab.query_text.iloc[0]
    julg = dict(zip(gab.num_pedido, gab[col]))

    outra = "relevancia_llm" if col == "relevancia_humana" else None
    if outra and outra in gab.columns and (gab[outra].str.strip() != "").all():
        b = gab[outra].astype(int)
        iguais = int((b == gab[col]).sum())
        print(f"julgamento: {col} (LLM concorda em {iguais}/{len(gab)}; "
              f"o LLM da relevancia maior em {int((b > gab[col]).sum())}, "
              f"menor em {int((b < gab[col]).sum())})")
    else:
        print(f"julgamento: {col}")

    corpus = pd.read_csv(CORPUS, sep="\t", dtype=str, low_memory=False,
                         quoting=csv.QUOTE_NONE, escapechar="\\").fillna("")
    if EXCLUIR_SEM_RESUMO:
        corpus = corpus[corpus.resumo.str.strip() != ""]
    ids = set(corpus.num_pedido_normalizado)
    fora = sorted(n for n in julg if n not in ids)
    if fora:
        print(f"aviso: {len(fora)} julgados fora do corpus de avaliacao: {fora[:5]}")

    tmp = RAIZ / "resultados" / "_fase2"
    tmp.mkdir(parents=True, exist_ok=True)
    corpus_uso = tmp / "corpus.tsv"
    corpus.to_csv(corpus_uso, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    q_tsv = tmp / f"{qid}_query.tsv"
    pd.DataFrame([{"query_id": qid, "query_text": qtext}]).to_csv(
        q_tsv, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    qrels_tsv = tmp / f"{qid}_qrels.tsv"
    pd.DataFrame({"query_id": qid,
                  "num_pedido_normalizado": corpus.num_pedido_normalizado,
                  "relevance": [julg.get(n, 0) for n in corpus.num_pedido_normalizado],
                  "origem_julgamento": ["julgado" if n in julg else "presumido"
                                        for n in corpus.num_pedido_normalizado]}).to_csv(
        qrels_tsv, sep="\t", index=False)
    qrels = ab.carregar_qrels(qrels_tsv)

    n_rel = sum(1 for v in julg.values() if v > 0)
    print(f"{qid}: {qtext!r}")
    print(f"corpus de avaliacao: {len(corpus)} | julgados: {len(julg)} | "
          f"relevantes: {n_rel} (rel=2: {sum(1 for v in julg.values() if v == 2)})\n")

    emb_q = {}
    for rotulo, (sufixo_q, _) in CONFIGS_GEMMA.items():
        pasta_q = EMB_QUERIES.with_name(EMB_QUERIES.name + sufixo_q)
        if not pasta_q.exists():
            continue
        c = ab.carregar_colecao(pasta_q, "query_id")
        sel = [i for i, x in enumerate(c["ids"]) if str(x) == qid]
        if sel:
            emb_q[rotulo] = {**c, "ids": np.asarray(c["ids"])[sel],
                             "vetores": c["vetores"][sel]}
    if not emb_q:
        print(f"sem embedding de {qid} em {EMB_QUERIES.name}: so BM25 nesta rodada.\n")
    else:
        print("configuracoes do Gemma nesta rodada:", ", ".join(emb_q), "\n")

    linhas = []
    for nome, (coluna, pasta) in VARIANTES.items():
        r = bm.buscar_bm25(corpus_uso, q_tsv, coluna_texto=coluna, top_k=args.top_k)
        m = ab.avaliar(r, qrels, ks=ks)["por_query"].iloc[0].to_dict()
        linhas.append({"sistema": "bm25", "variante": nome, **m})
        for rotulo, q_vec in emb_q.items():
            pasta_d = EMB / (pasta + CONFIGS_GEMMA[rotulo][1])
            if not pasta_d.exists():
                continue
            emb_d = ab.carregar_colecao(pasta_d, "num_pedido_normalizado")
            keep = np.array([i for i, x in enumerate(emb_d["ids"]) if str(x) in ids])
            emb_d = {**emb_d, "ids": np.asarray(emb_d["ids"])[keep],
                     "vetores": emb_d["vetores"][keep]}
            rd = ab.buscar(q_vec, emb_d, top_k=args.top_k)
            m = ab.avaliar(rd, qrels, ks=ks)["por_query"].iloc[0].to_dict()
            linhas.append({"sistema": rotulo, "variante": nome, **m})

    M = pd.DataFrame(linhas).drop(columns=["query_id"])
    cols = ["sistema", "variante", "nDCG@10", "R-Precision", "MRR", "P@10"]
    cols = [c for c in cols if c in M.columns] + [c for c in M.columns if c not in cols]
    M = M[cols]
    pd.set_option("display.width", 160, "display.max_columns", 40)
    print(M.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    saida = args.saida or (RAIZ / "resultados" / f"{qid}_bm25_gemma.csv")
    M.to_csv(saida, index=False)
    print(f"\n-> {saida.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
