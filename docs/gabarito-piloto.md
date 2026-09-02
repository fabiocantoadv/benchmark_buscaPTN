# Gabarito qualificado — piloto (3 queries, corpus de 1.000)

Substitui o qrels por regras (17% do corpus relevante, baseline aleatório
P@10 ≈ 0,21) por um gabarito julgado sobre um corpus recomposto a partir de
`amostra_50000.xlsx`.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `src/gerar_pool_piloto.py` | monta o pool inicial sobre a amostra enriquecida de 1.000 (união dos top-12 de 3 variantes BM25 + sorteio) |
| `dados/pool_piloto_gabarito.tsv` | 135 julgamentos com `relevancia_llm`, `justificativa_llm` e colunas em branco para revisão |
| `src/montar_corpus_piloto.py` | monta `corpus_piloto.tsv` (1.000 docs) e `dados/qrels_piloto.tsv` a partir do pool |
| `corpus_piloto.tsv` | 1.000 documentos, texto do xlsx (só a variante `tr`) |
| `src/rodar_export_ipc.sh` | reexporta esses mesmos 1.000 do Postgres com IPC |
| `dados/corpus_piloto_ipc.tsv` | **corpus oficial** — os 1.000 com IPC, descrições PT e hierarquia |
| `src/consolidar_corpus_ipc.py` | confere cobertura e regrava o qrels sobre o corpus oficial |
| `dados/qrels_piloto.tsv` | 3.000 julgamentos, com `origem_julgamento` = julgado \| presumido |

Queries: `QF002` (técnica, câncer), `QA006` (natural, água), `QG001` (curta, 5G).

## Composição do corpus

Fonte única de texto: `amostra_50000.xlsx` — 47.601 linhas, 45.087 patentes
únicas, contendo os 1.000 da amostra original. Usar uma fonte só evita misturar
duas normalizações de texto no mesmo corpus (o xlsx traz resumo em minúsculas,
com o título prefixado e sem o boilerplate "Resumo"; 75% dos resumos são
idênticos aos do TSV depois de normalizar).

Os 911 distratores são sorteados **fora** da amostra original de 1.000, que foi
montada em torno dos 3 temas das queries e portanto traria relevantes não
julgados. A semente do corpus são os 89 julgados da rodada inicial, de modo que
o corpus não muda quando o pool cresce.

O xlsx não tem coluna de IPC. Por isso os mesmos 1.000 foram reexportados do
Postgres (`src/rodar_export_ipc.sh` → `src/enriquecer_ipc_json_pt.py` →
`src/consolidar_corpus_ipc.py`), o que devolveu título, resumo e IPC na mesma
normalização da amostra original e recuperou as três variantes de texto.
Cobertura: 1.000/1.000 encontrados, IPC em 1.000, descrição PT em 950 documentos
(666 com todos os símbolos descritos), 26 sem resumo — um deles julgado
(`BR122020017793`). **O corpus oficial é `dados/corpus_piloto_ipc.tsv`.**

## Escala de relevância (0–3)

| nota | critério |
|---|---|
| 3 | atende plenamente ao `criterio_relevancia_alta` da query |
| 2 | mesmo objeto técnico, mas falta um elemento do critério |
| 1 | tangencial ou negativo difícil (parece relevante, não é) |
| 0 | não relevante |

| query | 3 | 2 | 1 | 0 |
|---|---|---|---|---|
| QF002 | 7 | 6 | 15 | 16 |
| QG001 | 3 | 10 | 16 | 16 |
| QA006 | 0 | 7 | 10 | 29 |

## Efeito no diagnóstico

| | qrels por regras | gabarito piloto |
|---|---|---|
| corpus | 1.000 (enriquecido) | 1.000 (sorteado dos 45 mil) |
| R mediano (rel≥1) | 172 | 28 |
| % do corpus relevante | 17% | 2,8% |
| P@10 de um ranqueador aleatório | 0,21 | 0,025 |

BM25 sobre `dados/corpus_piloto_ipc.tsv`:

| variante | nDCG@10 | R-Prec | QF002 | QA006 | QG001 |
|---|---|---|---|---|---|
| `tr` | 0,669 | 0,648 | 0,869 | 0,382 | 0,757 |
| `ipc_direto` | 0,782 | 0,706 | 0,869 | 0,802 | 0,674 |
| `ipc_hierarquia` | 0,777 | 0,753 | 0,845 | 0,853 | 0,631 |

Agora que o gabarito não é derivado de IPC, o efeito do enriquecimento deixa de
ser uniforme e passa a ter sinal: ele **resgata** QA006 (0,38 → 0,85, a query
cujo vocabulário natural — "metais tóxicos", "limpar água" — não casa com o
texto das patentes, mas casa com as descrições de C02F) e **prejudica** QG001
(0,76 → 0,63, onde a query já é técnica e a descrição de IPC só dilui). É a
hipótese de diluição do README aparecendo em dados. Com 3 queries isso é
indício, não conclusão — e ainda carrega viés de pool (o pool inicial incluiu o
top-12 das variantes com IPC).

## Como revisar

1. Abrir `dados/pool_piloto_gabarito.tsv` (separador tab). O `titulo`/`resumo` ali é o
   mesmo texto que os sistemas veem.
2. Ler contra `query_text` e `criterio_relevancia_alta`.
3. Preencher `relevancia_final` e `revisor`. Em branco aceita a nota do LLM.
4. Rodar `python3 montar_corpus_piloto.py` — ele usa `relevancia_final` quando
   preenchida e cai para `relevancia_llm` quando vazia.

Casos que pedem atenção humana:

- `BR112022009672` e `BR112022002391` (QF002) são quase duplicatas.
- **QA006 não tem nenhum documento nota 3** e é a query com pior desempenho
  (nDCG@10 = 0,38). Ou o corpus não contém patente de remoção de metais tóxicos
  de água, ou nenhum sistema do pool a alcançou. Antes de reportar essa query,
  vale uma busca dirigida no Postgres por `C02F 1/28`, `C02F 1/62`, `C02F 101/20`.

## Limitações a declarar no artigo

1. **Viés de pool.** O pool veio só de variantes BM25, então o BM25 joga em casa
   e seus números não devem ser reportados como desempenho absoluto. A troca do
   corpus já mostrou o mecanismo: ao ampliar a base de sorteio, o BM25 trouxe 45
   candidatos inéditos — entre eles um ADC anti-HER3 com ligante explícito em 1º
   lugar para QF002, que teria contado como irrelevante. Quando os embeddings
   densos estiverem gerados, acrescente os rankings ao dicionário `sistemas` em
   `src/gerar_pool_piloto.py`, refaça o pool sobre `corpus_piloto.tsv` e julgue os
   documentos novos antes de comparar denso × léxico.
2. **Distratores presumidos irrelevantes.** Os 911 sorteados não foram julgados.
   Sorteá-los fora da amostra temática reduz o risco, mas não o elimina.
3. **Pré-classificação por LLM.** As notas foram atribuídas por um modelo lendo
   título + resumo, com justificativa por documento. Precisa de revisão humana e
   de nota metodológica; o ideal é reportar a concordância (Cohen's κ) entre LLM
   e revisor sobre os 135 itens.
4. **3 queries.** Valida o protocolo, não sustenta conclusão estatística.
   Escalar para 9–12 queries antes de reportar.

## Quatro variantes de texto

| variante | conteúdo | mediana |
|---|---|---|
| `tr` | título + resumo | 143 palavras |
| `ipc_grupo` | + descrição do grupo principal e do subgrupo (sem seção/classe/subclasse) | 208 |
| `ipc_direto` | + descrição só dos símbolos listados na patente | 178 |
| `ipc_hierarquia` | + cadeia hierárquica completa | 322 |

`ipc_grupo` (gerada por `src/gerar_variante_ipc_grupo.py`) recupera o grupo
principal que o `ipc_direto` perde: quando a patente traz só `A61K 47/68`, o
`ipc_direto` mostra "o agente de modificação sendo um anticorpo…" sem o
referente, enquanto o `ipc_grupo` traz junto `A61K 47/00 - Preparações
medicinais caracterizadas pelos ingredientes não ativos…`.

### BM25 sobre as quatro (corpus piloto, gabarito julgado)

| variante | nDCG@10 | R-Prec | QF002 | QA006 | QG001 |
|---|---|---|---|---|---|
| `tr` | 0,669 | 0,648 | 0,869 | 0,382 | 0,757 |
| `ipc_grupo` | 0,735 | 0,734 | 0,837 | 0,880 | 0,486 |
| `ipc_direto` | 0,782 | 0,706 | 0,869 | 0,802 | 0,674 |
| `ipc_hierarquia` | 0,777 | 0,753 | 0,845 | 0,853 | 0,631 |

**Não é diluição por comprimento — é injeção de vocabulário.** `ipc_grupo` tem
208 palavras contra 322 da `ipc_hierarquia` e vai *pior* em QG001 (0,486 contra
0,631); se o problema fosse volume de texto, seria o contrário. O que ocorre é
que as descrições de IPC injetam nos documentos exatamente as palavras da query:
25 documentos do corpus ganham "aloca…" e 33 ganham "recurso" só pela
classificação. No top-10 de QG001 sob `ipc_grupo` entram dois documentos nota 0
(`BR112022007557`, `BR112019021401`), ambos H04W 72/04 — irmãos de classe que
herdaram o vocabulário sem tratarem do assunto. A `ipc_hierarquia` sofre menos
porque o texto mais longo reduz o peso de cada termo na normalização do BM25.

Leitura: o enriquecimento **ajuda** quando o vocabulário da query não está no
texto da patente e a descrição faz a ponte (QA006: "metais tóxicos" →
C02F 101/20), e **atrapalha** quando o vocabulário da query já é o vocabulário da
classificação (QG001: "alocação de recursos" é o título de H04W 72). Com 3
queries e pool ainda enviesado, é hipótese com mecanismo identificado.

## Pendente: a parte densa

Tudo acima é **BM25**. Os embeddings do EmbeddingGemma não foram gerados nesta
rodada — o modelo é *gated* e exige GPU e token HF. `src/gerar_embeddings_gemma300_benchmark.py`
e `notebooks/benchmark_patentes_colab.ipynb` já estão apontados para
`dados/corpus_piloto_ipc.tsv`, `dados/queries_piloto.tsv` e `dados/qrels_piloto.tsv`, com a
variante `ipc_grupo` incluída — basta rodar o notebook no Colab.

Ao comparar denso × léxico, lembre que o pool foi construído só com BM25: refaça
o pool com os rankings densos e julgue os documentos novos antes de reportar.

## Queries derivadas (subconjunto estrito)

Três queries mais específicas, cada uma um subconjunto de uma das originais.
A relevância é **aninhada**: todo documento relevante para a filha também é
relevante para a mãe (verificado — 11/11, 7/7 e 10/10). Isso permite medir se o
sistema distingue o específico dentro do geral.

| filha | mãe | query | R (rel≥1) |
|---|---|---|---|
| `QF002E` | QF002 | conjugado anticorpo-fármaco direcionado a HER2 para tumor sólido | 11 |
| `QA006E` | QA006 | remoção de metais pesados de lixiviado de aterro sanitário | 7 |
| `QG001E` | QG001 | concessão configurada de uplink e escalonamento semi-persistente em 5G NR | 10 |

A coluna `query_derivada_de` em `dados/queries_piloto.tsv` registra a filiação.
Dois documentos entraram no pool à mão (`BR102021004721` em QA006E,
`BR112019001900` em QG001E): nenhuma variante os recuperou, e sem isso contariam
como irrelevantes. Três documentos trazidos pelas filhas foram julgados também
nas mães, para fechar o aninhamento (`origem = herdado_da_derivada`).

### nDCG@10 por query — BM25

| variante | QF002 | QF002E | QA006 | QA006E | QG001 | QG001E |
|---|---|---|---|---|---|---|
| `tr` | 0,869 | **1,000** | 0,382 | 0,822 | 0,757 | 0,905 |
| `ipc_grupo` | 0,837 | 0,980 | 0,880 | 0,839 | 0,486 | 0,922 |
| `ipc_direto` | 0,869 | 0,980 | 0,802 | 0,841 | 0,674 | 0,898 |
| `ipc_hierarquia` | 0,845 | 0,980 | 0,853 | 0,783 | 0,631 | 0,916 |

**Nas queries específicas o efeito do IPC desaparece.** Amplitude entre variantes:
0,02 em QF002E, 0,06 em QA006E, 0,02 em QG001E — contra 0,50 em QA006 e 0,27 em
QG001. E todas as quatro ficam acima de 0,78.

Isso reforça o mecanismo de injeção de vocabulário. Uma query específica carrega
termos raros que estão no resumo e não na classificação — "HER2", "lixiviado",
"semi-persistente" — então o BM25 já resolve com o texto puro, e acrescentar
descrição de IPC não muda o ranking nem para melhor nem para pior. O
enriquecimento por IPC só tem efeito material onde a query é genérica ou em
linguagem natural, isto é, onde o vocabulário da consulta e o do documento não
se encontram.

Ressalvas: R entre 7 e 11, com efeito de teto (QF002E chega a 1,000 no `tr`);
pool construído só com BM25, agora com as 4 variantes; notas ainda do LLM.
