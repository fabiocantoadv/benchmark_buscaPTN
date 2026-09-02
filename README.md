# Benchmark de busca semântica em patentes

Avalia recuperação sobre patentes do INPI comparando **BM25** e
**[EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m)**, e
mede o efeito de enriquecer o texto do documento com descrições de CIP/IPC em
português.

**→ [`docs/resultados.md`](docs/resultados.md) — a análise.**

## O resultado em uma tabela

nDCG@10 médio por tipo de consulta, 18 queries, corpus de 974 documentos:

| tipo | n | BM25 | Gemma |
|---|---|---|---|
| técnica | 7 | **0,761** | 0,725 |
| específica | 3 | **0,871** | 0,826 |
| curta | 1 | **0,757** | 0,622 |
| natural | 7 | 0,177 | **0,546** |

Seis dos nove temas têm um par de queries — uma técnica e uma paráfrase em
linguagem natural — apontando para o **mesmo conjunto de documentos relevantes**.
Dentro do par só o vocabulário muda. O BM25 perde 0,60 de nDCG@10 nessa
travessia; o Gemma perde 0,19, em 6 de 6 pares.

> Os 622 julgamentos são pré-classificação por LLM, ainda sem revisão humana.
> Nada é reportável antes disso.

## Estrutura

```
dados/     corpus, queries, gabarito e qrels
src/       avaliação, embeddings, construção do corpus e do pool
docs/      resultados e instruções dos embeddings
notebooks/ pipeline no Colab
```

Arquivos centrais: `dados/corpus_piloto_ipc.tsv` (974 documentos úteis, 4
variantes de texto), `dados/queries_piloto.tsv` (18 queries com critério de
relevância e negativos difíceis declarados), `dados/qrels_piloto.tsv`,
`dados/pool_piloto_gabarito.tsv` (a planilha de revisão).

## Rodar

```bash
python3 src/avaliar_denso.py     # Gemma × BM25, requer embeddings/
```

Só BM25, sem GPU nem token:

```python
import sys; sys.path.insert(0, "src")
import avaliar_benchmark as ab, baseline_bm25 as bm
qrels = ab.carregar_qrels("dados/qrels_piloto.tsv")
r = bm.buscar_bm25("dados/corpus_piloto_ipc.tsv", "dados/queries_piloto.tsv",
                   coluna_texto="texto_para_embedding")
ab.avaliar(r, qrels)["agregado"]
```

Gerar os embeddings: [`docs/embeddings-gemma300.md`](docs/embeddings-gemma300.md).

## Trabalho em dupla

Cada pessoa roda a própria cópia e envia mudanças por commit. `embeddings/` e
`resultados/` não são versionados.
