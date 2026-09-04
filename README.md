# Benchmark de busca semântica em patentes

Avalia recuperação sobre patentes do INPI comparando **BM25**,
**[EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m)** e
**[BGE-M3](https://huggingface.co/BAAI/bge-m3)** (denso, esparso e híbrido), e
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

### BGE-M3

Segundo modelo denso, para separar "ganho da abordagem densa" de "ganho deste
modelo". Não usa instrução — a documentação do BGE-M3 diz que, ao contrário
dos BGE anteriores, ele "não requer mais adicionar instruções às queries" — e
tem contexto de 8192 tokens contra 2048 do Gemma, o que elimina o
truncamento nas variantes com IPC. Só nas variantes `tr` e `ipc_grupo`.

nDCG@10:

| | BM25 | M3 denso | M3 esparso | M3 híbrido (0,3) | Gemma |
|---|---|---|---|---|---|
| QN001 tr | 0,116 | 0,449 | 0,556 | 0,526 | **0,697** |
| QN001 ipc_grupo | 0,170 | 0,454 | 0,665 | 0,570 | **0,795** |
| QN003 tr | 0,217 | 0,521 | 0,548 | 0,519 | **0,736** |
| QN003 ipc_grupo | 0,349 | 0,629 | 0,594 | 0,600 | **0,694** |

**Primeiro efeito que não troca de sinal entre as queries:** BM25 < M3 denso
< Gemma nas quatro células. Depois de três testes de instrução que
inverteram, uma ordenação estável — e é sobre modelo, que é a pergunta do
benchmark. O M3 tem perfil de erro próprio: acerta o primeiro colocado mas
degrada entre as posições 5 e 20 (nDCG@5 de 0,301 contra Recall@20 de 0,692
na `tr` da QN001). Encontra os relevantes e não sabe ordená-los — que é o
defeito que um reranqueador, ou a cabeça ColBERT do proprio M3, conserta.

**A cabeça esparsa é o melhor componente do M3**, ganhando do denso em 3 das
4 células e do BM25 por uma margem enorme (0,556 contra 0,116 na `tr` da
QN001). Os dois são lexicais; a diferença é que o esparso do M3 expande
vocabulário. Parte do que se costuma atribuir a "semântica" é, na verdade, a
incapacidade do BM25 de expandir termos.

**O peso 0,3 do artigo do M3 não serve aqui.** O denso é cosseno, em
[-1, 1]; o escore lexical vive em outra faixa, e com 0,3 o denso domina a
soma — o híbrido fica pior que o esparso sozinho. Varrendo o peso, o híbrido
estabiliza entre 3 e 5 e chega a 0,709 na `ipc_grupo` da QN001, acima de
qualquer componente isolado:

| peso | tr QN001 | tr QN003 | ipc_grupo QN001 | ipc_grupo QN003 |
|---|---|---|---|---|
| 0,3 | 0,526 | 0,519 | 0,570 | 0,600 |
| 1 | 0,646 | 0,521 | 0,568 | 0,519 |
| 3 | 0,643 | 0,607 | 0,709 | 0,604 |
| 5 | 0,624 | 0,605 | 0,710 | 0,604 |
| 10 | 0,610 | 0,601 | 0,675 | 0,601 |

Esse 0,709 **não é reportável**: varrer o peso em duas queries e escolher o
melhor é ajustar hiperparâmetro no conjunto de teste. O padrão no código
continua 0,3, o valor do artigo; `--peso-esparso` muda. Para o híbrido
entrar em qualquer resultado, o peso precisa ser escolhido em queries fora
da avaliação. E mesmo ajustado, o híbrido não alcança o Gemma sozinho.

### O estado da fase 2, em uma frase

Duas queries, três modelos, quatro variantes de texto e três configurações
de instrução. A única coisa estável é a ordem entre modelos; todo efeito de
configuração medido até aqui trocou de sinal entre as duas queries. O
gargalo é número de queries, não número de configurações.

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

BGE-M3, denso e esparso (requer FlagEmbedding para a cabeça esparsa):

```bash
python3 src/gerar_embeddings_bgem3.py --kind queries --sparse
python3 src/gerar_embeddings_bgem3.py --kind docs --variant tr --sparse
python3 src/gerar_embeddings_bgem3.py --kind docs --variant ipc_grupo --sparse
```

Os pesos lexicais vão para `sparse_bloco_00000.npz` ao lado do `.npy` denso, e
a avaliação os lê com numpy puro — testar pesos de fusão não exige o modelo.

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
