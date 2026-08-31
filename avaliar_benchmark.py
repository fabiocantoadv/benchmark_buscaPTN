#!/usr/bin/env python3
"""Busca semantica e metricas de avaliacao para o benchmark de patentes.

Funciona tanto no Colab quanto localmente. Nao depende de GPU nem de
sentence-transformers: opera sobre os arrays .npy ja gerados.

Uso tipico:

    import avaliar_benchmark as ab

    docs = ab.carregar_colecao("embeddings/gemma300_tr_docs")
    queries = ab.carregar_colecao("embeddings/gemma300_queries")
    qrels = ab.carregar_qrels("qrels_candidatos_queries_benchmark.tsv")

    ranking = ab.buscar(queries, docs, top_k=100)
    resultado = ab.avaliar(ranking, qrels)
    print(resultado["agregado"])
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


CHAVE_DOC_PADRAO = "num_pedido_normalizado"
CHAVE_QUERY_PADRAO = "query_id"


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def carregar_colecao(pasta: str | Path, coluna_chave: str | None = None) -> dict:
    """Le uma pasta de embeddings (blocos .npy + metadata .tsv) gerada pelos
    scripts do projeto e devolve {"ids": np.ndarray, "vetores": np.ndarray,
    "metadata": DataFrame, "config": dict}.

    Os blocos sao concatenados na ordem do indice, que e a mesma ordem das
    linhas do TSV de entrada.
    """
    pasta = Path(pasta)
    if not pasta.exists():
        raise FileNotFoundError(f"Pasta de embeddings nao encontrada: {pasta}")

    blocos_npy = sorted(pasta.glob("embeddings_bloco_*.npy"))
    blocos_tsv = sorted(pasta.glob("metadata_bloco_*.tsv"))
    if not blocos_npy:
        # compatibilidade com o nome usado no script do Colab antigo
        blocos_npy = sorted(pasta.glob("embeddings_*bloco_*.npy"))
    if not blocos_npy:
        raise FileNotFoundError(f"Nenhum bloco .npy encontrado em {pasta}")
    if len(blocos_npy) != len(blocos_tsv):
        raise ValueError(
            f"Blocos inconsistentes em {pasta}: "
            f"{len(blocos_npy)} .npy vs {len(blocos_tsv)} .tsv"
        )

    vetores = np.concatenate([np.load(p) for p in blocos_npy], axis=0)
    metadata = pd.concat(
        [pd.read_csv(p, sep="\t", dtype=str, quoting=csv.QUOTE_NONE) for p in blocos_tsv],
        ignore_index=True,
    )

    if len(metadata) != len(vetores):
        raise ValueError(
            f"Metadata ({len(metadata)}) e embeddings ({len(vetores)}) "
            f"tem tamanhos diferentes em {pasta}"
        )

    config = {}
    caminho_config = pasta / "config.json"
    if caminho_config.exists():
        config = json.loads(caminho_config.read_text(encoding="utf-8"))

    if coluna_chave is None:
        for candidata in (CHAVE_DOC_PADRAO, CHAVE_QUERY_PADRAO, "num_pedido", "id"):
            if candidata in metadata.columns:
                coluna_chave = candidata
                break
    if coluna_chave is None or coluna_chave not in metadata.columns:
        raise ValueError(
            f"Coluna de chave nao encontrada em {pasta}. "
            f"Colunas disponiveis: {list(metadata.columns)}"
        )

    ids = metadata[coluna_chave].astype(str).to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError(f"Ha ids duplicados na coluna {coluna_chave} em {pasta}")

    return {
        "ids": ids,
        "vetores": normalizar(vetores),
        "metadata": metadata,
        "config": config,
        "coluna_chave": coluna_chave,
        "pasta": str(pasta),
    }


def normalizar(matriz: np.ndarray) -> np.ndarray:
    """Normaliza L2 as linhas. Idempotente para vetores ja normalizados."""
    matriz = np.asarray(matriz, dtype=np.float32)
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    return matriz / normas


def carregar_qrels(caminho: str | Path) -> pd.DataFrame:
    """Le o TSV de qrels (query_id, num_pedido_normalizado, relevance)."""
    qrels = pd.read_csv(
        caminho,
        sep="\t",
        dtype={"query_id": str, CHAVE_DOC_PADRAO: str, "relevance": int},
        quoting=csv.QUOTE_NONE,
    )
    faltando = {"query_id", CHAVE_DOC_PADRAO, "relevance"} - set(qrels.columns)
    if faltando:
        raise ValueError(f"Colunas ausentes no qrels: {sorted(faltando)}")
    return qrels


# ---------------------------------------------------------------------------
# Busca
# ---------------------------------------------------------------------------

def buscar(queries: dict, docs: dict, top_k: int = 100) -> dict:
    """Similaridade de cosseno entre todas as queries e todos os documentos.

    Com vetores normalizados, o produto interno e a similaridade de cosseno.
    Devolve {"query_ids", "doc_ids_ranking", "scores_ranking"} onde as duas
    ultimas sao matrizes (n_queries, top_k).
    """
    top_k = min(top_k, len(docs["ids"]))
    similaridades = queries["vetores"] @ docs["vetores"].T

    # argpartition para o top-k, depois ordena so essa fatia
    parciais = np.argpartition(-similaridades, kth=top_k - 1, axis=1)[:, :top_k]
    scores_parciais = np.take_along_axis(similaridades, parciais, axis=1)
    ordem = np.argsort(-scores_parciais, axis=1)
    indices = np.take_along_axis(parciais, ordem, axis=1)
    scores = np.take_along_axis(scores_parciais, ordem, axis=1)

    return {
        "query_ids": queries["ids"],
        "doc_ids_ranking": docs["ids"][indices],
        "scores_ranking": scores,
        "similaridades": similaridades,
    }


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def _dcg(ganhos: Sequence[float]) -> float:
    ganhos = np.asarray(ganhos, dtype=float)
    if ganhos.size == 0:
        return 0.0
    descontos = np.log2(np.arange(2, ganhos.size + 2))
    return float(np.sum((2 ** ganhos - 1) / descontos))


def ndcg_at_k(relevancias_ranking: Sequence[int], relevancias_ideais: Sequence[int], k: int) -> float:
    dcg = _dcg(relevancias_ranking[:k])
    idcg = _dcg(sorted(relevancias_ideais, reverse=True)[:k])
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(relevancias_ranking: Sequence[int], total_relevantes: int, k: int, limiar: int = 1) -> float:
    if total_relevantes == 0:
        return float("nan")
    recuperados = sum(1 for r in relevancias_ranking[:k] if r >= limiar)
    return recuperados / total_relevantes


def precision_at_k(relevancias_ranking: Sequence[int], k: int, limiar: int = 1) -> float:
    if k == 0:
        return 0.0
    return sum(1 for r in relevancias_ranking[:k] if r >= limiar) / k


def reciprocal_rank(relevancias_ranking: Sequence[int], limiar: int = 1) -> float:
    for posicao, rel in enumerate(relevancias_ranking, start=1):
        if rel >= limiar:
            return 1.0 / posicao
    return 0.0


def avaliar(
    ranking: dict,
    qrels: pd.DataFrame,
    ks: Iterable[int] = (5, 10, 20, 50, 100),
    limiar_relevante: int = 1,
) -> dict:
    """Calcula nDCG@k, Recall@k, P@k e MRR por query e agregado (media macro).

    Documentos ausentes do qrels sao tratados como relevance 0.
    """
    ks = sorted(ks)
    mapa_qrels: dict[str, dict[str, int]] = {}
    for query_id, grupo in qrels.groupby("query_id"):
        mapa_qrels[str(query_id)] = dict(
            zip(grupo[CHAVE_DOC_PADRAO].astype(str), grupo["relevance"].astype(int))
        )

    linhas = []
    for i, query_id in enumerate(ranking["query_ids"]):
        query_id = str(query_id)
        julgamentos = mapa_qrels.get(query_id, {})
        if not julgamentos:
            continue

        docs_ranqueados = [str(d) for d in ranking["doc_ids_ranking"][i]]
        rels = [julgamentos.get(d, 0) for d in docs_ranqueados]
        rels_ideais = list(julgamentos.values())
        total_relevantes = sum(1 for r in rels_ideais if r >= limiar_relevante)
        total_altamente = sum(1 for r in rels_ideais if r >= 2)

        linha = {
            "query_id": query_id,
            "relevantes_no_qrels": total_relevantes,
            "altamente_relevantes": total_altamente,
            "MRR": reciprocal_rank(rels, limiar_relevante),
            "MRR_rel2": reciprocal_rank(rels, 2),
        }
        for k in ks:
            linha[f"nDCG@{k}"] = ndcg_at_k(rels, rels_ideais, k)
            linha[f"Recall@{k}"] = recall_at_k(rels, total_relevantes, k, limiar_relevante)
            linha[f"P@{k}"] = precision_at_k(rels, k, limiar_relevante)
        linhas.append(linha)

    por_query = pd.DataFrame(linhas)
    if por_query.empty:
        raise ValueError("Nenhuma query do ranking foi encontrada no qrels.")

    colunas_metricas = [c for c in por_query.columns if c not in {"query_id", "relevantes_no_qrels", "altamente_relevantes"}]
    agregado = por_query[colunas_metricas].mean().to_dict()

    return {"por_query": por_query, "agregado": agregado}


def comparar_variantes(resultados: dict[str, dict], metricas: Sequence[str] | None = None) -> pd.DataFrame:
    """Monta uma tabela variante x metrica a partir de {nome: resultado_avaliar}."""
    if metricas is None:
        metricas = ["nDCG@10", "nDCG@20", "MRR", "Recall@10", "Recall@50", "P@10"]
    linhas = []
    for nome, resultado in resultados.items():
        linha = {"variante": nome}
        linha.update({m: resultado["agregado"].get(m, float("nan")) for m in metricas})
        linhas.append(linha)
    return pd.DataFrame(linhas).set_index("variante").round(4)


def avaliar_por_faceta(
    resultado: dict,
    queries_df: pd.DataFrame,
    faceta: str,
    metricas: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Quebra as metricas por uma coluna do TSV de queries (tema, tipo_query...)."""
    if metricas is None:
        metricas = ["nDCG@10", "MRR", "Recall@10", "Recall@50"]
    juncao = resultado["por_query"].merge(
        queries_df[["query_id", faceta]].astype(str), on="query_id", how="left"
    )
    return juncao.groupby(faceta)[list(metricas)].mean().round(4)


def inspecionar_query(
    ranking: dict,
    query_id: str,
    docs_metadata: pd.DataFrame,
    qrels: pd.DataFrame,
    top_n: int = 10,
    coluna_titulo: str = "titulo",
) -> pd.DataFrame:
    """Mostra os top_n resultados de uma query com titulo, IPC e relevancia."""
    indices = np.where(np.asarray([str(q) for q in ranking["query_ids"]]) == str(query_id))[0]
    if indices.size == 0:
        raise ValueError(f"query_id nao encontrada no ranking: {query_id}")
    i = int(indices[0])

    docs = [str(d) for d in ranking["doc_ids_ranking"][i][:top_n]]
    scores = ranking["scores_ranking"][i][:top_n]

    julgamentos = dict(
        zip(
            qrels.loc[qrels["query_id"].astype(str) == str(query_id), CHAVE_DOC_PADRAO].astype(str),
            qrels.loc[qrels["query_id"].astype(str) == str(query_id), "relevance"].astype(int),
        )
    )

    meta = docs_metadata.set_index(docs_metadata[CHAVE_DOC_PADRAO].astype(str))
    linhas = []
    for posicao, (doc_id, score) in enumerate(zip(docs, scores), start=1):
        registro = {
            "posicao": posicao,
            "score": round(float(score), 4),
            "relevance": julgamentos.get(doc_id, 0),
            CHAVE_DOC_PADRAO: doc_id,
        }
        if doc_id in meta.index:
            for coluna in (coluna_titulo, "ipc"):
                if coluna in meta.columns:
                    valor = meta.loc[doc_id, coluna]
                    registro[coluna] = str(valor)[:120]
        linhas.append(registro)
    return pd.DataFrame(linhas)
