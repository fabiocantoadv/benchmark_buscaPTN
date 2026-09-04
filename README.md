# Benchmark de busca semântica em patentes

Avalia recuperação sobre patentes do INPI comparando **BM25** e
**[EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m)**, e
mede o efeito de enriquecer o texto do documento com descrições de CIP/IPC em
português.

Corpus: amostra de **1.000 pedidos** (`dados/corpus_piloto_ipc.tsv`), 974 com
resumo utilizável, em 4 variantes de texto.

## Fase 2 — queries por humano + LLM

Cada query é elaborada em conjunto (humano define o tema e o critério, LLM
propõe a formulação) e recebe um gabarito próprio, simples:

```
dados/queries_fase2.tsv     uma linha por query
dados/gabaritos/<qid>.tsv   query_id, query_text, num_pedido, titulo, resumo, relevancia
```

`relevancia` é 0, 1 ou 2, atribuída por LLM sobre os candidatos recuperados.
Documentos do corpus fora do gabarito entram na avaliação como 0. Os
julgamentos ainda **não passaram por revisão humana** — nada é reportável
antes disso.

### QN003 — primeira query da fase

`Terapias baseadas em anticorpos para tratamento de câncer` (natural),
45 documentos julgados, 18 relevantes, nDCG@10:

| variante do texto do documento | BM25 | Gemma | Gemma sem instrução |
|---|---|---|---|
| tr (título + resumo) | 0,217 | **0,736** | 0,619 |
| ipc_grupo | 0,349 | 0,694 | 0,621 |
| ipc_direto | 0,217 | 0,667 | 0,638 |
| ipc_hierarquia | **0,415** | 0,698 | 0,524 |

O enriquecimento por IPC ajuda o BM25 e atrapalha o Gemma: a descrição da
classificação em português dá ao ranking lexical o vocabulário que a query
natural não compartilha com o resumo, enquanto o modelo denso já resolve esse
salto sozinho e trata o texto da CIP como diluição. O Gemma acerta o primeiro
colocado em todas as variantes (MRR 1,000).

A terceira coluna testa a instrução em português prefixada a documento e
query, que substitui os prompts nativos do EmbeddingGemma (`title: none |
text:` e `task: search result | query:`). Ela ganha nas quatro variantes, e
ganha mais justamente onde o texto da CIP é maior — 0,17 de nDCG@10 no
`ipc_hierarquia` contra 0,03 no `ipc_direto`. Ou seja: a instrução não
conflita com o enriquecimento por IPC, ela protege contra a diluição que ele
causa. Fica como está.

## Rodar

Uma query, BM25 nas 4 variantes (não precisa de GPU nem de modelo):

```bash
python3 src/avaliar_query.py dados/gabaritos/QN003.tsv
```

Com o Gemma, gerando antes o embedding das queries:

```bash
python3 src/gerar_embeddings_gemma300_benchmark.py --kind queries
python3 src/avaliar_query.py dados/gabaritos/QN003.tsv
```

Embeddings dos documentos: [`docs/embeddings-gemma300.md`](docs/embeddings-gemma300.md).

## Estrutura

```
dados/gabaritos/  um gabarito por query
dados/            corpus e queries
src/              avaliação, embeddings e construção do corpus
notebooks/        pipeline no Colab
```

`embeddings/` e `resultados/` não são versionados.

## Trabalho em dupla

Cada pessoa roda a própria cópia e envia mudanças por commit.
