#!/usr/bin/env python3
"""Monta o corpus (1.000 docs) e o qrels do benchmark piloto.

Fonte unica de texto: amostra_50000.xlsx (45.087 patentes unicas), que ja contem
os 1.000 da amostra original. Usar uma so fonte evita misturar duas
normalizacoes de texto no mesmo corpus.

Composicao do corpus, deterministica e independente de quantos julgamentos
existam:

  - os documentos julgados que vieram da amostra original de 1.000 (semente);
  - distratores sorteados FORA da amostra original, ate completar 1.000.

A amostra original foi montada em torno dos 3 temas das queries, entao sortear
dela traria relevantes nao julgados. Sorteando dos outros ~44 mil a prevalencia
a priori e baixa, e tratar os nao julgados como relevance=0 e defensavel.

Sem coluna de IPC no xlsx, so a variante `tr` (titulo + resumo) existe neste
corpus; a comparacao entre variantes de texto continua restrita ao corpus de
1.000 enriquecido.
"""
# NOTA: este script construiu o corpus piloto a partir da amostra original
# de 1.000 e do amostra_50000.xlsx, arquivos que nao estao mais no
# repositorio (abordagem por regras, removida). O sorteio dos distratores
# esta congelado em dados/numeros_corpus_piloto.txt; para reproduzir o
# corpus use src/rodar_export_ipc.sh. Mantido como registro do metodo.

import csv
import pandas as pd

N_CORPUS = 1000
SEED = 20260901

def ler_tsv(p):
    return pd.read_csv(p, sep="\t", dtype=str, low_memory=False,
                       quoting=csv.QUOTE_NONE, escapechar="\\")

def limpar(s):
    return (s.fillna("").astype(str)
             .str.replace(r"[\t\r\n]+", " ", regex=True)
             .str.replace(r"\\", "/", regex=True)
             .str.replace(r"\s+", " ", regex=True).str.strip())

base = pd.read_excel("amostra_50000.xlsx", dtype=str)
base["num_pedido_normalizado"] = ("BR" + base.numero_inpi.astype(str)
                                  .str.replace(r"[^0-9]", "", regex=True))
base["titulo"] = limpar(base.titulo)
base["resumo"] = limpar(base.resumo)
base = base.drop_duplicates("num_pedido_normalizado")
base["texto_para_embedding"] = (base.titulo + ". " + base.resumo).str.strip()

pool = ler_tsv("pool_piloto_gabarito.tsv").fillna("")
amostra_original = set(ler_tsv("patentes_benchmark_amostra_1000_ipc_pt.tsv")
                       .num_pedido_normalizado)
# semente = julgados que vieram da amostra original; nao cresce quando o pool cresce
semente = set(pool.num_pedido_normalizado) & amostra_original

fora = base[~base.num_pedido_normalizado.isin(amostra_original)]
distratores = fora.sample(N_CORPUS - len(semente), random_state=SEED)
corpus = pd.concat([base[base.num_pedido_normalizado.isin(semente)], distratores])
corpus = corpus[["num_pedido_normalizado", "numero_inpi", "titulo", "resumo",
                 "texto_para_embedding"]].drop_duplicates("num_pedido_normalizado")
corpus.to_csv("corpus_piloto.tsv", sep="\t", index=False,
              quoting=csv.QUOTE_NONE, escapechar="\\")

# --- qrels: nota final quando revisada, senao a do LLM; nao julgado = 0 -------
nota = pool.relevancia_final.where(pool.relevancia_final.str.strip() != "",
                                   pool.relevancia_llm)
pool = pool.assign(relevance=pd.to_numeric(nota, errors="coerce").fillna(0).astype(int))

linhas = []
for qid in sorted(pool.query_id.unique()):
    g = pool[pool.query_id == qid]
    notas = dict(zip(g.num_pedido_normalizado, g.relevance))
    for doc in corpus.num_pedido_normalizado:
        linhas.append((qid, doc, notas.get(doc, 0),
                       "julgado" if doc in notas else "presumido"))
qrels = pd.DataFrame(linhas, columns=["query_id", "num_pedido_normalizado",
                                      "relevance", "origem_julgamento"])
qrels.to_csv("qrels_piloto.tsv", sep="\t", index=False,
             quoting=csv.QUOTE_NONE, escapechar="\\")

fora_corpus = set(pool.num_pedido_normalizado) - set(corpus.num_pedido_normalizado)
print("corpus:", len(corpus), "| semente:", len(semente),
      "| julgados fora do corpus:", len(fora_corpus))
print(qrels[qrels.relevance > 0].groupby(["query_id", "relevance"]).size().to_string())
