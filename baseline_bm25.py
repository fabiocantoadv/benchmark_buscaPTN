#!/usr/bin/env python3
"""Baseline lexical BM25 para o benchmark de patentes.

Serve de ponto de referencia para as variantes densas: sem um baseline, um
nDCG de 0,70 nao tem interpretacao — nao se sabe se o modelo semantico ganha
ou perde de uma busca por palavra-chave.

Implementacao propria (Okapi BM25), sem dependencias alem de numpy/pandas,
para nao exigir instalacao extra no Colab e para manter a tokenizacao em
portugues sob controle.

Uso:

    import baseline_bm25 as bm
    import avaliar_benchmark as ab

    ranking = bm.buscar_bm25(
        docs_tsv="patentes_benchmark_amostra_1000.tsv",
        queries_tsv="queries_benchmark_patentes.tsv",
        coluna_texto="texto_para_embedding",
    )
    resultado = ab.avaliar(ranking, ab.carregar_qrels("qrels_...tsv"))
"""

from __future__ import annotations

import csv
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# Stopwords do portugues + termos genericos onipresentes em patentes, que
# so adicionam ruido ao ranking lexical.
STOPWORDS_PT = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as",
    "ate", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles",
    "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre",
    "era", "eram", "essa", "essas", "esse", "esses", "esta", "estas", "este",
    "estes", "eu", "foi", "foram", "ha", "isso", "isto", "ja", "la", "lhe",
    "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas",
    "muito", "na", "nas", "nao", "nem", "no", "nos", "nossa", "nossas",
    "nosso", "nossos", "num", "numa", "o", "os", "ou", "para", "pela",
    "pelas", "pelo", "pelos", "per", "por", "qual", "quando", "que", "quem",
    "se", "sem", "sendo", "ser", "seu", "seus", "so", "sob", "sobre", "sua",
    "suas", "tambem", "te", "tem", "tendo", "ter", "teu", "teus", "tua",
    "tuas", "um", "uma", "umas", "uns", "voce", "voces",
}

STOPWORDS_PATENTES = {
    "invencao", "presente", "refere", "trata", "compreende", "compreendendo",
    "caracterizado", "fato", "pelo", "resumo", "patente", "pedido", "dito",
    "dita", "ditos", "ditas", "referido", "referida", "modalidade",
    "modalidades", "reivindicacao", "reivindicacoes", "figura", "figuras",
}

STOPWORDS = STOPWORDS_PT | STOPWORDS_PATENTES

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalizar_texto(texto: str) -> str:
    """Minusculas e remocao de acentos, para casar 'água' com 'agua'."""
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def tokenizar(texto: str, min_len: int = 3) -> list[str]:
    return [
        t for t in _TOKEN_RE.findall(normalizar_texto(texto))
        if len(t) >= min_len and t not in STOPWORDS
    ]


class BM25:
    """Okapi BM25 classico.

    k1 controla a saturacao da frequencia do termo; b, a normalizacao pelo
    tamanho do documento. Os valores padrao (1.5 / 0.75) sao os usuais na
    literatura e funcionam bem para textos de tamanho heterogeneo como
    resumos de patente.
    """

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        if self.n_docs == 0:
            raise ValueError("Corpus vazio")

        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) or 1.0

        self.freqs: list[Counter] = [Counter(d) for d in corpus_tokens]

        df: Counter = Counter()
        for doc in self.freqs:
            df.update(doc.keys())

        # IDF de Robertson com piso, para nao produzir peso negativo em
        # termos presentes em mais da metade dos documentos.
        self.idf = {
            termo: max(
                0.0,
                math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0),
            )
            for termo, n in df.items()
        }

        # indice invertido: termo -> (indices dos docs, frequencias)
        self.indice: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        posicoes: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(self.freqs):
            for termo, f in doc.items():
                posicoes.setdefault(termo, []).append((i, f))
        for termo, pares in posicoes.items():
            idxs, fs = zip(*pares)
            self.indice[termo] = (np.array(idxs, dtype=np.int32),
                                  np.array(fs, dtype=np.float32))

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        denom_len = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for termo in query_tokens:
            if termo not in self.indice:
                continue
            idxs, fs = self.indice[termo]
            contrib = self.idf[termo] * (fs * (self.k1 + 1)) / (fs + denom_len[idxs])
            scores[idxs] += contrib
        return scores


def _ler_tsv(caminho: Path | str) -> pd.DataFrame:
    return pd.read_csv(
        caminho, sep="\t", dtype=str, low_memory=False,
        quoting=csv.QUOTE_NONE, escapechar="\\",
    )


def buscar_bm25(
    docs_tsv: Path | str,
    queries_tsv: Path | str,
    coluna_texto: str = "texto_para_embedding",
    coluna_query: str = "query_text",
    chave_doc: str = "num_pedido_normalizado",
    chave_query: str = "query_id",
    top_k: int = 100,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict:
    """Devolve um ranking no mesmo formato de avaliar_benchmark.buscar()."""
    docs = _ler_tsv(docs_tsv)
    queries = _ler_tsv(queries_tsv)

    for coluna, df, nome in ((coluna_texto, docs, "docs"), (coluna_query, queries, "queries")):
        if coluna not in df.columns:
            raise ValueError(f"Coluna {coluna} ausente em {nome}")

    corpus = [tokenizar(t) for t in docs[coluna_texto].fillna("")]
    modelo = BM25(corpus, k1=k1, b=b)

    doc_ids = docs[chave_doc].astype(str).to_numpy()
    query_ids = queries[chave_query].astype(str).to_numpy()
    top_k = min(top_k, len(doc_ids))

    ranking_ids, ranking_scores = [], []
    for texto in queries[coluna_query].fillna(""):
        s = modelo.scores(tokenizar(texto))
        # desempate estavel: ordena por (-score, indice)
        ordem = np.lexsort((np.arange(len(s)), -s))[:top_k]
        ranking_ids.append(doc_ids[ordem])
        ranking_scores.append(s[ordem])

    return {
        "query_ids": query_ids,
        "doc_ids_ranking": np.array(ranking_ids),
        "scores_ranking": np.array(ranking_scores, dtype=np.float32),
        "modelo": f"BM25(k1={k1}, b={b})",
    }
