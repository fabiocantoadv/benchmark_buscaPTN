#!/usr/bin/env python3
"""Monta o pool de julgamento para reconstruir o gabarito manualmente.

Por que pooling e nao regra: o qrels gerado por regras de IPC + termos marca
cerca de 20% do corpus como relevante, o que faz P@k e MRR saturarem e
impede o benchmark de distinguir sistemas. Alem disso, avaliar variantes
enriquecidas com IPC contra um gabarito derivado de IPC e circular.

O metodo padrao em recuperacao de informacao (usado no TREC) e o pooling:
juntam-se os top-k de varios sistemas diferentes, julga-se manualmente so
essa uniao, e tudo que ficou de fora e considerado nao relevante. Isso
concentra o esforco humano nos documentos que algum sistema considerou bons,
que sao justamente os que decidem as metricas.

Uso:

    import avaliar_benchmark as ab, baseline_bm25 as bm, gerar_pool_revisao as gp

    rankings = {
        "tr": ab.buscar(q, docs_tr, top_k=100),
        "ipc_hierarquia": ab.buscar(q, docs_ipc, top_k=100),
        "bm25": bm.buscar_bm25(...),
    }
    pool = gp.montar_pool(rankings, profundidade=20)
    gp.exportar_para_revisao(pool, docs_meta, queries_df, "pool_revisao.tsv",
                             revisores=("fabio", "colega"))
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

CHAVE_DOC = "num_pedido_normalizado"


def montar_pool(rankings: dict[str, dict], profundidade: int = 20) -> pd.DataFrame:
    """Une os top-`profundidade` de cada sistema, por query.

    Devolve um DataFrame com uma linha por par (query, documento) unico,
    registrando quais sistemas o recuperaram e em que melhor posicao. Um
    documento trazido por varios sistemas independentes tende a ser mais
    relevante — e util para ordenar a fila de revisao.
    """
    registros: dict[tuple[str, str], dict] = {}

    for nome_sistema, ranking in rankings.items():
        for i, query_id in enumerate(ranking["query_ids"]):
            query_id = str(query_id)
            docs = ranking["doc_ids_ranking"][i][:profundidade]
            for posicao, doc_id in enumerate(docs, start=1):
                chave = (query_id, str(doc_id))
                registro = registros.setdefault(chave, {
                    "query_id": query_id,
                    CHAVE_DOC: str(doc_id),
                    "sistemas": set(),
                    "melhor_posicao": posicao,
                })
                registro["sistemas"].add(nome_sistema)
                registro["melhor_posicao"] = min(registro["melhor_posicao"], posicao)

    linhas = []
    for registro in registros.values():
        linhas.append({
            "query_id": registro["query_id"],
            CHAVE_DOC: registro[CHAVE_DOC],
            "n_sistemas": len(registro["sistemas"]),
            "sistemas": ",".join(sorted(registro["sistemas"])),
            "melhor_posicao": registro["melhor_posicao"],
        })

    pool = pd.DataFrame(linhas)
    # ordem de revisao: primeiro o que muitos sistemas trouxeram no topo
    return pool.sort_values(
        ["query_id", "n_sistemas", "melhor_posicao"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def exportar_para_revisao(
    pool: pd.DataFrame,
    docs_metadata: pd.DataFrame,
    queries_df: pd.DataFrame,
    caminho_saida: Path | str,
    qrels_regra: pd.DataFrame | None = None,
    revisores: tuple[str, ...] = ("revisor_a", "revisor_b"),
    fracao_sobreposicao: float = 0.15,
    semente: int = 42,
) -> pd.DataFrame:
    """Gera o TSV de julgamento, com contexto e divisao entre revisores.

    Uma fracao das linhas e atribuida a TODOS os revisores (sobreposicao), para
    que se possa medir concordancia entre eles — sem isso nao ha como saber se
    o gabarito e reprodutivel ou reflete o criterio pessoal de cada um.

    Colunas a preencher na revisao:
      relevance_humana : 0 (irrelevante), 1 (relevante), 2 (altamente relevante)
      nota_revisor     : texto livre, para casos duvidosos
    """
    saida = pool.copy()

    colunas_doc = [c for c in (CHAVE_DOC, "titulo", "ipc", "num_publicacao")
                   if c in docs_metadata.columns]
    saida = saida.merge(
        docs_metadata[colunas_doc].astype(str).drop_duplicates(CHAVE_DOC),
        on=CHAVE_DOC, how="left",
    )

    colunas_query = [c for c in ("query_id", "tema", "tipo_query", "query_text",
                                 "criterio_relevancia_alta")
                     if c in queries_df.columns]
    saida = saida.merge(queries_df[colunas_query].astype(str), on="query_id", how="left")

    if qrels_regra is not None:
        saida = saida.merge(
            qrels_regra.rename(columns={"relevance": "relevance_regra"}),
            on=["query_id", CHAVE_DOC], how="left",
        )
        saida["relevance_regra"] = saida["relevance_regra"].fillna(0).astype(int)

    # divisao entre revisores, estratificada por query para que ninguem receba
    # um tema inteiro e desenvolva um criterio proprio para ele
    rng = np.random.default_rng(semente)
    atribuicoes = []
    for _, grupo in saida.groupby("query_id", sort=False):
        n = len(grupo)
        n_sobrepostas = max(1, int(round(n * fracao_sobreposicao)))
        indices = rng.permutation(n)
        sobrepostas = set(indices[:n_sobrepostas])
        for j, _ in enumerate(grupo.index):
            if j in sobrepostas:
                atribuicoes.append("TODOS")
            else:
                atribuicoes.append(revisores[j % len(revisores)])
    saida["revisor"] = atribuicoes

    saida["relevance_humana"] = ""
    saida["nota_revisor"] = ""

    ordem = ["query_id", "tema", "tipo_query", "query_text", "revisor",
             "relevance_humana", "nota_revisor", CHAVE_DOC, "titulo", "ipc",
             "n_sistemas", "sistemas", "melhor_posicao"]
    if "relevance_regra" in saida.columns:
        ordem.insert(ordem.index("relevance_humana") + 1, "relevance_regra")
    if "criterio_relevancia_alta" in saida.columns:
        ordem.append("criterio_relevancia_alta")
    ordem = [c for c in ordem if c in saida.columns]
    saida = saida[ordem + [c for c in saida.columns if c not in ordem]]

    caminho_saida = Path(caminho_saida)
    saida.to_csv(caminho_saida, sep="\t", index=False,
                 quoting=csv.QUOTE_MINIMAL)

    print(f"Pool exportado: {caminho_saida}")
    print(f"  {len(saida)} pares (query, documento) para julgar")
    print(f"  {saida['query_id'].nunique()} queries | "
          f"mediana de {saida.groupby('query_id').size().median():.0f} candidatos por query")
    print("\nDistribuicao por revisor:")
    print(saida["revisor"].value_counts().to_string())
    if "relevance_regra" in saida.columns:
        print("\nRotulo da regra no pool (para comparar depois com o humano):")
        print(saida["relevance_regra"].value_counts().sort_index().to_string())
    return saida


def consolidar_qrels_revisado(
    caminho_revisado: Path | str,
    caminho_saida: Path | str,
    todos_os_docs: list[str] | None = None,
) -> pd.DataFrame:
    """Converte o TSV revisado em um qrels utilizavel.

    Linhas sem `relevance_humana` preenchida sao descartadas com aviso. Se
    `todos_os_docs` for informado, os documentos fora do pool sao gravados
    explicitamente como 0 (convencao do pooling).
    """
    revisado = pd.read_csv(caminho_revisado, sep="\t", dtype=str)
    revisado["relevance_humana"] = pd.to_numeric(
        revisado["relevance_humana"], errors="coerce"
    )

    faltando = revisado["relevance_humana"].isna().sum()
    if faltando:
        print(f"AVISO: {faltando} linhas sem julgamento; serao ignoradas.")
    julgadas = revisado.dropna(subset=["relevance_humana"]).copy()
    julgadas["relevance_humana"] = julgadas["relevance_humana"].astype(int)

    # nas linhas de sobreposicao pode haver mais de um julgamento: usa o maximo
    # (criterio conservador, mantem o documento no conjunto relevante)
    qrels = (
        julgadas.groupby(["query_id", CHAVE_DOC])["relevance_humana"]
        .max().reset_index().rename(columns={"relevance_humana": "relevance"})
    )

    if todos_os_docs is not None:
        completo = []
        for query_id in qrels["query_id"].unique():
            julgados = set(qrels.loc[qrels["query_id"] == query_id, CHAVE_DOC])
            for doc_id in todos_os_docs:
                if doc_id not in julgados:
                    completo.append({"query_id": query_id, CHAVE_DOC: doc_id,
                                     "relevance": 0})
        if completo:
            qrels = pd.concat([qrels, pd.DataFrame(completo)], ignore_index=True)

    qrels = qrels.sort_values(["query_id", CHAVE_DOC])
    qrels.to_csv(caminho_saida, sep="\t", index=False, quoting=csv.QUOTE_NONE)

    relevantes = qrels[qrels["relevance"] >= 1].groupby("query_id").size()
    print(f"\nqrels revisado: {caminho_saida}")
    print(f"  R mediano por query: {relevantes.median():.0f} "
          f"(antes da revisao era ~200)")
    return qrels


# ---------------------------------------------------------------------------
# Pool diferencial: julga so o que decide a comparacao entre variantes
# ---------------------------------------------------------------------------

def montar_pool_diferencial(
    rankings: dict[str, dict],
    profundidade: int = 10,
    minimo_sistemas_ausentes: int = 1,
) -> pd.DataFrame:
    """Diferenca simetrica dos top-k entre os sistemas comparados.

    Onde todos os sistemas ranqueiam o mesmo documento na mesma faixa, o
    julgamento daquele documento e irrelevante para a comparacao: ele entra
    igual nas metricas de todos. So os documentos que ALGUM sistema trouxe e
    outro NAO decidem quem ganha.

    Isso reduz o esforco de julgamento em uma ordem de grandeza em relacao ao
    pool completo, ao custo de produzir um gabarito parcial — bom para
    comparar sistemas entre si, insuficiente para reportar nDCG absoluto (veja
    a nota em avaliar_diferencial).

    `minimo_sistemas_ausentes` = 1 mantem todo documento que falte em ao menos
    um sistema. Aumente para focar nas discordancias mais fortes.
    """
    if len(rankings) < 2:
        raise ValueError("Sao necessarios ao menos dois sistemas para comparar")

    nomes = list(rankings)
    # indexa: (query, doc) -> {sistema: posicao}
    posicoes: dict[tuple[str, str], dict[str, int]] = {}
    queries_vistas: list[str] = []

    for nome in nomes:
        ranking = rankings[nome]
        for i, query_id in enumerate(ranking["query_ids"]):
            query_id = str(query_id)
            if query_id not in queries_vistas:
                queries_vistas.append(query_id)
            for posicao, doc_id in enumerate(ranking["doc_ids_ranking"][i][:profundidade], start=1):
                posicoes.setdefault((query_id, str(doc_id)), {})[nome] = posicao

    linhas = []
    consensuais = 0
    for (query_id, doc_id), por_sistema in posicoes.items():
        ausentes = [n for n in nomes if n not in por_sistema]
        if len(ausentes) < minimo_sistemas_ausentes:
            consensuais += 1
            continue
        linhas.append({
            "query_id": query_id,
            CHAVE_DOC: doc_id,
            "n_sistemas": len(por_sistema),
            "sistemas": ",".join(sorted(por_sistema)),
            "ausente_em": ",".join(sorted(ausentes)),
            "melhor_posicao": min(por_sistema.values()),
            **{f"pos_{n}": por_sistema.get(n, "") for n in nomes},
        })

    pool = pd.DataFrame(linhas)
    total = len(posicoes)
    print(f"Pool diferencial (top-{profundidade}, {len(nomes)} sistemas):")
    print(f"  {total} pares (query, doc) distintos no total")
    print(f"  {consensuais} consensuais (todos os sistemas trouxeram) -> nao precisam julgamento")
    print(f"  {len(pool)} DISCORDANTES -> estes decidem a comparacao")
    if total:
        print(f"  reducao de esforco: {100 * consensuais / total:.0f}%")
    if not pool.empty:
        print(f"  mediana de {pool.groupby('query_id').size().median():.0f} pares por query")

    return pool.sort_values(
        ["query_id", "melhor_posicao"], ascending=[True, True]
    ).reset_index(drop=True)


def avaliar_diferencial(
    rankings: dict[str, dict],
    qrels_parcial: pd.DataFrame,
    profundidade: int = 10,
) -> pd.DataFrame:
    """Compara sistemas usando SO os pares julgados no pool diferencial.

    Metrica: precisao no top-`profundidade` calculada apenas sobre os
    documentos que tem julgamento humano, com o denominador ajustado por
    query. Documentos sem julgamento sao excluidos do calculo, nao tratados
    como zero — tratar como zero penalizaria arbitrariamente o sistema que
    trouxe documentos que ninguem julgou.

    NAO REPORTE estes valores como nDCG ou precisao absoluta do sistema: o
    gabarito e parcial por construcao e cobre so a regiao de discordancia.
    O uso legitimo e a comparacao RELATIVA entre as variantes.
    """
    julgados = {
        (str(q), str(d)): int(v)
        for q, d, v in zip(
            qrels_parcial["query_id"],
            qrels_parcial[CHAVE_DOC],
            qrels_parcial["relevance"],
        )
    }
    if not julgados:
        raise ValueError("qrels_parcial esta vazio")

    linhas = []
    for nome, ranking in rankings.items():
        acertos = coberturas = 0
        por_query = []
        for i, query_id in enumerate(ranking["query_ids"]):
            query_id = str(query_id)
            docs = [str(d) for d in ranking["doc_ids_ranking"][i][:profundidade]]
            com_julgamento = [(d, julgados[(query_id, d)]) for d in docs
                              if (query_id, d) in julgados]
            if not com_julgamento:
                continue
            rel = sum(1 for _, v in com_julgamento if v >= 1)
            altos = sum(1 for _, v in com_julgamento if v >= 2)
            por_query.append({
                "query_id": query_id,
                "julgados_no_topo": len(com_julgamento),
                "precisao_julgada": rel / len(com_julgamento),
                "precisao_alta_julgada": altos / len(com_julgamento),
            })
            acertos += rel
            coberturas += len(com_julgamento)

        detalhe = pd.DataFrame(por_query)
        linhas.append({
            "sistema": nome,
            "queries_avaliadas": len(detalhe),
            "docs_julgados_no_topo": coberturas,
            "precisao_julgada": round(detalhe["precisao_julgada"].mean(), 4),
            "precisao_alta_julgada": round(detalhe["precisao_alta_julgada"].mean(), 4),
        })

    tabela = pd.DataFrame(linhas).set_index("sistema")
    tabela.attrs["aviso"] = (
        "Gabarito parcial (so a regiao de discordancia). Use para comparacao "
        "relativa entre sistemas; nao reporte como metrica absoluta."
    )
    print(tabela.attrs["aviso"])
    return tabela
