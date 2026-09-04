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

### As duas primeiras queries

nDCG@10, corpus de 974, gabarito com revisão humana quando existe:

**QN003** — `Terapias baseadas em anticorpos para tratamento de câncer`
(45 julgados, 18 relevantes, julgamento só por LLM)

| variante | BM25 | Gemma | Gemma s/ instrução |
|---|---|---|---|
| tr | 0,217 | **0,736** | 0,619 |
| ipc_grupo | 0,349 | 0,694 | 0,621 |
| ipc_direto | 0,217 | 0,667 | 0,638 |
| ipc_hierarquia | 0,415 | 0,698 | 0,524 |

**QN001** — `Tecnologias e processos para remover ou reduzir contaminantes
químicos presentes em água e efluentes` (21 julgados, 13 relevantes, revisto
por humano; o LLM concordava em 16 dos 21)

| variante | BM25 | Gemma | Gemma s/ instrução |
|---|---|---|---|
| tr | 0,116 | 0,697 | 0,715 |
| ipc_grupo | 0,170 | 0,795 | 0,821 |
| ipc_direto | 0,142 | 0,802 | **0,834** |
| ipc_hierarquia | 0,196 | 0,658 | 0,756 |

**O que se sustenta nas duas:** o Gemma ganha do BM25 em toda variante, por
margens de 0,3 a 0,7 de nDCG@10, e acerta o primeiro colocado em quase todas
(MRR 1,000). Em queries naturais o baseline lexical não compete.

**O que não se sustenta:** as duas conclusões que a QN003 sugeria se invertem
na QN001. Lá o texto puro era a melhor variante do Gemma e o IPC diluía;
aqui o IPC melhora o Gemma (0,697 → 0,802 no `ipc_direto`). Lá a instrução em
português ganhava nas quatro variantes; aqui ela perde nas quatro. Nenhum dos
dois efeitos tem sinal estável com duas queries — a variação entre queries é
maior que a variação entre configurações. Só o que vale por enquanto é a
comparação Gemma × BM25.

Para o IPC ser decidido são necessárias mais queries, com temas de perfis de
classificação diferentes. A instrução vs. prompt nativo do EmbeddingGemma
(`title: none | text:` / `task: search result | query:`) idem.

**Instrução da query pedindo correspondência de CIP** (`gemma_qipc`, só o
lado query muda, contra as mesmas coleções de documento): sem efeito. A
diferença para a instrução atual fica entre -0,047 e +0,079 de nDCG@10, com
sinal trocado entre as duas queries em metade das variantes, e média de
+0,013 nas oito medidas. O único sinal consistente é no `ipc_direto`, onde
ganha nas duas (+0,022 e +0,079) — com duas queries, isso é uma pista para
reexaminar depois, não um resultado. Redação de instrução parece render
pouco em relação ao que varia entre queries; o esforço rende mais em número
de queries e na limpeza do texto de CIP.

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
