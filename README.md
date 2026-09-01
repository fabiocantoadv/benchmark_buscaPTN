# Benchmark de busca semântica em patentes

Avalia recuperação semântica sobre patentes do INPI, comparando BM25 e
[EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) e
medindo o efeito de enriquecer o texto do documento com descrições de CIP/IPC
em português.

## Estado atual

Benchmark **piloto**: 3 queries, corpus de 1.000 documentos, gabarito julgado
manualmente (escala 0–3). O gabarito anterior, gerado por regras de IPC, foi
abandonado — marcava 17% do corpus como relevante, o que fazia um ranqueador
aleatório alcançar P@10 ≈ 0,21 e tornava a avaliação de variantes com IPC
circular. Ele continua no histórico do git, até o commit `f3e68eb`.

| | gabarito por regras | gabarito piloto |
|---|---|---|
| R mediano (relevantes por query) | 172 / 1.000 | 28 / 1.000 |
| P@10 de um ranqueador aleatório | 0,21 | 0,025 |

**Pendências, em ordem:** revisar os 135 julgamentos pré-classificados por LLM ·
rodar os embeddings no Colab · refazer o pool com os rankings densos · escalar
para 9–12 queries.

Detalhes do método, limitações e como revisar: **[`docs/gabarito-piloto.md`](docs/gabarito-piloto.md)**.

## Estrutura

```
dados/     corpus, queries, gabarito e qrels
src/       módulos de avaliação, geração de embeddings e montagem do corpus
docs/      guia do gabarito e comandos operacionais
notebooks/ pipeline completo no Colab
```

### `dados/`

| arquivo | conteúdo |
|---|---|
| `corpus_piloto_ipc.tsv` | **o corpus** — 1.000 patentes com IPC, descrições PT e hierarquia; 4 variantes de texto |
| `queries_piloto.tsv` | as 3 queries do piloto |
| `qrels_piloto.tsv` | julgamentos usados na avaliação (3.000 linhas; `origem_julgamento` distingue julgado de presumido) |
| `pool_piloto_gabarito.tsv` | **planilha de revisão** — 135 candidatos com nota do LLM, justificativa e colunas em branco para o revisor |
| `queries_benchmark_patentes.tsv` | as 45 queries do desenho completo, para quando escalar |
| `numeros_corpus_piloto.txt` | os 1.000 números que definem o corpus (congela o sorteio dos distratores) |
| `*_resumo.tsv`, `ipc_simbolos_sem_descricao_*` | estatísticas de cobertura |

### Variantes de texto no corpus

| coluna | conteúdo | mediana |
|---|---|---|
| `texto_para_embedding` | título + resumo | 143 palavras |
| `texto_para_embedding_ipc_pt` | + descrição dos símbolos IPC da patente | 178 |
| `texto_para_embedding_ipc_grupo_pt` | + descrição de grupo e subgrupo | 208 |
| `texto_para_embedding_ipc_hierarquia_pt` | + cadeia hierárquica completa | 322 |

### `src/`

| arquivo | função |
|---|---|
| `avaliar_benchmark.py` | busca por similaridade e métricas — nDCG, R-Precision, MRR, P@k, diagnóstico do gabarito, teste pareado |
| `baseline_bm25.py` | baseline lexical BM25 em português, sem dependências extras |
| `gerar_embeddings_gemma300_benchmark.py` | gera os embeddings (Mac/MPS ou Colab/GPU) |
| `gerar_pool_revisao.py` | pool de julgamento completo e diferencial |
| `rodar_export_ipc.sh` + `export_corpus_piloto_ipc.sql` | reexportam o corpus do Postgres com IPC |
| `enriquecer_ipc_json_pt.py` | acrescenta descrições IPC em PT e hierarquia |
| `gerar_variante_ipc_grupo.py` | gera a variante `ipc_grupo` |
| `consolidar_corpus_ipc.py` | confere cobertura e regrava o qrels |
| `gerar_pool_piloto.py`, `montar_corpus_piloto.py` | como o corpus e o pool foram construídos (histórico; entradas não estão mais no repo) |

## Como rodar

**Colab** — abra `notebooks/benchmark_patentes_colab.ipynb`. Ele clona o repo,
gera os embeddings das 4 variantes e calcula as métricas. Requer GPU, aceitar a
licença do `google/embeddinggemma-300m` e um `HF_TOKEN` nos Secrets do Colab
(cada pessoa cadastra o seu).

**Local, só BM25** — não precisa de GPU nem de token:

```python
import sys; sys.path.insert(0, "src")
import avaliar_benchmark as ab, baseline_bm25 as bm
qrels = ab.carregar_qrels("dados/qrels_piloto.tsv")
r = bm.buscar_bm25("dados/corpus_piloto_ipc.tsv", "dados/queries_piloto.tsv",
                   coluna_texto="texto_para_embedding")
ab.avaliar(r, qrels)["agregado"]
```

Rode `ab.diagnosticar_qrels(qrels)` antes de interpretar qualquer métrica.

## Trabalho em dupla

Cada pessoa roda a própria cópia do notebook e envia mudanças por commit. Os
arquivos em `resultados/` recebem carimbo de data e hora para que duas execuções
não se sobrescrevam. `embeddings/` e `resultados/` não são versionados.
