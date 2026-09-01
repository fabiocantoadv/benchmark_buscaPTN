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


def r_precision(relevancias_ranking, total_relevantes: int, limiar: int = 1) -> float:
    """Precisao no corte R, onde R = numero de relevantes daquela query.

    Diferente de P@10, o corte se adapta ao tamanho do conjunto relevante, o
    que a torna comparavel entre queries com R muito diferente — exatamente o
    caso deste benchmark, onde R varia de dezenas a centenas.
    """
    if total_relevantes == 0:
        return float("nan")
    corte = min(total_relevantes, len(relevancias_ranking))
    if corte == 0:
        return 0.0
    return sum(1 for r in relevancias_ranking[:corte] if r >= limiar) / total_relevantes


def recall_teto(total_relevantes: int, k: int) -> float:
    """Maior Recall@k alcancavel: min(1, k/R).

    Quando k < R, Recall@k nao pode passar de k/R por construcao. Nesse regime
    a metrica mede o tamanho do conjunto relevante, nao a qualidade do
    ranqueamento, e comparar Recall@k entre queries com R diferente nao faz
    sentido.
    """
    if total_relevantes == 0:
        return float("nan")
    return min(1.0, k / total_relevantes)


def recall_normalizado(relevancias_ranking, total_relevantes: int, k: int, limiar: int = 1) -> float:
    """Recall@k dividido pelo seu teto: 1.0 = todos os k primeiros relevantes.

    Nota: quando k < R, isto e algebricamente identico a P@k — o teto k/R
    cancela o R do denominador do recall. Fica aqui porque no regime k >= R
    os dois divergem, mas neste benchmark (R mediano de 172) espere ver
    RecallNorm@10 e P@10 iguais; nao sao duas evidencias independentes.
    """
    teto = recall_teto(total_relevantes, k)
    if not teto or np.isnan(teto):
        return float("nan")
    return recall_at_k(relevancias_ranking, total_relevantes, k, limiar) / teto


def diagnosticar_qrels(qrels: pd.DataFrame, n_docs: int | None = None,
                       ks: Iterable[int] = (10, 20)) -> pd.DataFrame:
    """Verifica se o gabarito consegue discriminar sistemas.

    Um gabarito em que uma fracao grande do corpus e relevante torna P@k e MRR
    quase insensiveis a qualidade: o baseline aleatorio ja fica alto e o teto
    do Recall@k desaba. Rode isto ANTES de interpretar qualquer metrica.
    """
    por_query = qrels[qrels["relevance"] >= 1].groupby("query_id").size()
    altos = qrels[qrels["relevance"] >= 2].groupby("query_id").size()
    if n_docs is None:
        n_docs = qrels[CHAVE_DOC_PADRAO].nunique()

    linhas = [
        {"metrica": "documentos no corpus", "valor": n_docs},
        {"metrica": "queries", "valor": qrels["query_id"].nunique()},
        {"metrica": "R (rel>=1) minimo", "valor": int(por_query.min())},
        {"metrica": "R (rel>=1) mediano", "valor": float(por_query.median())},
        {"metrica": "R (rel>=1) maximo", "valor": int(por_query.max())},
        {"metrica": "% do corpus relevante (mediano)", "valor": round(100 * por_query.median() / n_docs, 1)},
        {"metrica": "R (rel=2) mediano", "valor": float(altos.median()) if len(altos) else 0.0},
    ]
    for k in ks:
        linhas.append({
            "metrica": f"P@{k} de um ranqueador aleatorio",
            "valor": round(float((por_query / n_docs).mean()), 3),
        })
        linhas.append({
            "metrica": f"teto mediano de Recall@{k}",
            "valor": round(float(np.median([recall_teto(r, k) for r in por_query])), 4),
        })
    return pd.DataFrame(linhas)


def comparar_pareado(resultados: dict[str, dict], metrica: str = "nDCG@10") -> pd.DataFrame:
    """Teste pareado de Wilcoxon entre todos os pares de sistemas.

    Uma diferenca de media entre dois sistemas nao diz se ela e consistente
    entre as queries. O teste pareado diz. Atencao a multiplicidade: com n
    sistemas sao n(n-1)/2 comparacoes, e um p de 0,04 isolado nesse conjunto
    nao e evidencia forte.
    """
    try:
        from scipy import stats
    except ImportError:
        raise SystemExit("Instale scipy: pip install scipy")

    nomes = list(resultados)
    linhas = []
    for i, a in enumerate(nomes):
        for b in nomes[i + 1:]:
            x = resultados[a]["por_query"].set_index("query_id")[metrica]
            y = resultados[b]["por_query"].set_index("query_id")[metrica].reindex(x.index)
            dif = y - x
            teste = stats.wilcoxon(x, y) if dif.abs().sum() > 0 else None
            linhas.append({
                "sistema_A": a,
                "sistema_B": b,
                f"delta_{metrica}": round(float(dif.mean()), 4),
                "B_vence": int((dif > 0).sum()),
                "A_vence": int((dif < 0).sum()),
                "empates": int((dif == 0).sum()),
                "p_wilcoxon": round(float(teste.pvalue), 4) if teste else float("nan"),
            })
    tabela = pd.DataFrame(linhas)
    tabela.attrs["aviso"] = (
        f"{len(linhas)} comparacoes sem correcao para multiplos testes; "
        "considere Bonferroni (p < 0.05/n) antes de afirmar diferenca."
    )
    return tabela


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
    saturadas: set[int] = set()
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
            "R-Precision": r_precision(rels, total_relevantes, limiar_relevante),
        }
        for k in ks:
            linha[f"nDCG@{k}"] = ndcg_at_k(rels, rels_ideais, k)
            linha[f"Recall@{k}"] = recall_at_k(rels, total_relevantes, k, limiar_relevante)
            linha[f"RecallNorm@{k}"] = recall_normalizado(rels, total_relevantes, k, limiar_relevante)
            linha[f"P@{k}"] = precision_at_k(rels, k, limiar_relevante)
            if k < total_relevantes:
                saturadas.add(k)
        linhas.append(linha)

    por_query = pd.DataFrame(linhas)
    if por_query.empty:
        raise ValueError("Nenhuma query do ranking foi encontrada no qrels.")

    colunas_metricas = [c for c in por_query.columns if c not in {"query_id", "relevantes_no_qrels", "altamente_relevantes"}]
    agregado = por_query[colunas_metricas].mean().to_dict()

    avisos = []
    if saturadas:
        ks_sat = ", ".join(f"Recall@{k}" for k in sorted(saturadas))
        avisos.append(
            f"ATENCAO: {ks_sat} esta(o) saturado(s) em ao menos uma query (k < R). "
            "Nesse regime Recall@k e limitado a k/R e mede o tamanho do conjunto "
            "relevante, nao a qualidade do ranqueamento. Use R-Precision, "
            "RecallNorm@k ou nDCG para comparar sistemas."
        )
    for aviso in avisos:
        print(aviso)

    return {"por_query": por_query, "agregado": agregado, "avisos": avisos}


def comparar_variantes(resultados: dict[str, dict], metricas: Sequence[str] | None = None) -> pd.DataFrame:
    """Monta uma tabela variante x metrica a partir de {nome: resultado_avaliar}."""
    if metricas is None:
        # R-Precision e nDCG lideram porque nao saturam quando R e grande.
        # Recall@k cru foi removido do padrao justamente por saturar.
        metricas = ["nDCG@10", "nDCG@100", "R-Precision", "MRR", "MRR_rel2",
                    "RecallNorm@10", "P@10"]
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
        metricas = ["nDCG@10", "R-Precision", "MRR", "MRR_rel2"]
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
