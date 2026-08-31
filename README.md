# Benchmark de busca semântica em patentes

Benchmark para avaliar recuperação semântica sobre uma amostra de 1.000 pedidos
de patente do INPI, com 45 queries em português e embeddings do
[EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m).

## Como rodar no Colab

Abra `notebooks/benchmark_patentes_colab.ipynb` no Google Colab. O notebook
clona este repositório, gera os embeddings, roda a busca e calcula as métricas
de ponta a ponta em poucos minutos numa GPU T4.

**Pré-requisitos por pessoa:**

1. Ambiente de execução com GPU (`Ambiente de execução > Alterar o tipo de ambiente de execução > GPU`).
2. O modelo `google/embeddinggemma-300m` é *gated*: aceitar a licença em
   https://huggingface.co/google/embeddinggemma-300m e criar um token em
   https://huggingface.co/settings/tokens.
3. Cadastrar o token nos Secrets do Colab com o nome `HF_TOKEN`. Secrets não são
   compartilhados entre contas — cada pessoa cadastra o seu.

## Estrutura

### Corpus

| arquivo | conteúdo |
|---|---|
| `patentes_benchmark_amostra_1000.tsv` | 1.000 pedidos com título, resumo, IPC e `texto_para_embedding` |
| `patentes_benchmark_amostra_1000_ipc_pt.tsv` | o mesmo, enriquecido com descrições IPC em português e a hierarquia completa |
| `ipc_simbolos_sem_descricao_json_pt.tsv` | 358 símbolos IPC sem descrição em PT encontrados na amostra |
| `*_resumo.tsv` | estatísticas de cobertura das extrações |

Cobertura: 1.000/1.000 pedidos encontrados, 985 com resumo, 82% dos símbolos IPC
com descrição em PT (96% considerando a hierarquia).

### Queries e gabaritos

| arquivo | conteúdo |
|---|---|
| `queries_benchmark_patentes.tsv` | 45 queries — 3 temas (câncer/fármacos, 5G, purificação de água) × 3 tipos (curta, técnica, linguagem natural) × 5 |
| `qrels_candidatos_queries_benchmark.tsv` | 45.000 julgamentos (45 × 1.000), relevância 0/1/2 |
| `gabaritos_candidatos_revisao.tsv` | 90 candidatos por query selecionados para revisão manual |

> **Atenção:** o qrels atual é gerado por regras (padrões de IPC + termos) e o
> script que o produz se descreve como "intencionalmente fraco". As métricas
> medem concordância com a regra, não acerto semântico. A revisão manual do
> gabarito é o passo pendente mais importante do projeto.

### Código

| arquivo | função |
|---|---|
| `export_amostra_benchmark_tsv.sql` | extrai a amostra do banco Postgres do INPI |
| `enriquecer_ipc_json_pt.py` | acrescenta descrições IPC em português e hierarquia |
| `gerar_gabaritos_candidatos.py` | gera o qrels por regras e a amostra para revisão |
| `gerar_embeddings_gemma300_benchmark.py` | gera embeddings localmente (Mac, MPS) |
| `avaliar_benchmark.py` | busca por similaridade e métricas — nDCG@k, MRR, Recall@k, P@k |
| `notebooks/benchmark_patentes_colab.ipynb` | pipeline completo no Colab |

## Variantes de texto

O benchmark compara três formas de representar cada patente:

| variante | texto |
|---|---|
| `tr` | título + resumo |
| `ipc_direto` | título + resumo + descrições IPC em PT |
| `ipc_hierarquia` | título + resumo + hierarquia IPC completa em PT |

## Notas técnicas

- Os embeddings são gravados normalizados (L2), então a similaridade de cosseno
  é o produto interno. Com 45 queries × 1.000 documentos a busca é uma
  multiplicação de matrizes em numpy — não há necessidade de FAISS nessa escala.
- O EmbeddingGemma tem prompts nativos distintos para documento e consulta
  (`encode_document` / `encode_query`). O notebook usa os prompts nativos por
  padrão; o toggle `USAR_PROMPTS_NATIVOS = False` reproduz a abordagem de
  instrução manual em PT do script local, para comparação. Não misture as duas
  abordagens entre documentos e queries — isso desalinha os vetores.
- A pasta `embeddings/` não é versionada: são ~3 MB por coleção e a regeração
  leva poucos minutos.

## Trabalho em dupla

Cada pessoa roda a própria cópia do notebook no Colab e envia mudanças por
commit. Os arquivos em `resultados/` recebem carimbo de data e hora para que
duas execuções não se sobrescrevam.
