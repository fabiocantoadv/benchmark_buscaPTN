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

## Expansão para 18 queries em 9 temas

Seis temas novos, cada um com um **par**: uma query técnica e uma paráfrase em
linguagem natural que aponta para o **mesmo conjunto de documentos**. Os
julgamentos são compartilhados dentro do par, então qualquer diferença de
métrica é atribuível só ao vocabulário da consulta.

| tema | técnica | natural |
|---|---|---|
| perfuração de poços | QP001 | QP002 |
| defensivos agrícolas | QD001 | QD002 |
| embalagens | QE001 | QE002 |
| instrumentos cirúrgicos | QC001 | QC002 |
| materiais compósitos | QM001 | QM002 |
| produção microbiana | QB001 | QB002 |

522 julgamentos, R mediano 17,5 em 1.000 (1,8% do corpus), P@10 aleatório 0,017.

### Correção do viés de pool

Um pool feito só com BM25 deixa de fora justamente os documentos que a query em
linguagem natural deveria recuperar e não recupera — ou seja, enviesa o gabarito
**contra** os sistemas que resolvem a lacuna de vocabulário. O pool passou a ter
duas fontes (`src/montar_pool_tema.py`): o pooling automático das 4 variantes
para as duas queries do par, mais uma **corrida manual** por termo técnico + IPC
do tema. A corrida manual trouxe 35 dos 163 candidatos (21%) que o BM25 nunca
alcançou.

Caso exemplar: para QD002 existe no corpus a `BR0920002`, "MÉTODO PARA REDUZIR A
DERIVA DE PULVERIZAÇÃO DURANTE A APLICAÇÃO DE UM PESTICIDA". Nenhuma das quatro
variantes a recuperou. A query fala "agrotóxico", "vento", "espalhar"; a patente
fala "deriva", "pulverização", "formulação".

### Resultado: nDCG@10 médio por tipo de query

| tipo | n | `tr` | `ipc_grupo` | `ipc_direto` | `ipc_hierarquia` |
|---|---|---|---|---|---|
| técnica | 7 | 0,765 | 0,788 | 0,764 | 0,793 |
| específica | 3 | 0,909 | 0,913 | 0,906 | 0,893 |
| natural | 7 | **0,178** | **0,317** | 0,295 | 0,316 |

**O achado robusto é o nível, não o ganho.** O BM25 desaba em consultas em
linguagem natural: 0,178 contra 0,765 nas técnicas — um fator de 4, com o
conjunto relevante idêntico dentro de cada par. Em QB002 ("usar bactérias
modificadas para fabricar substâncias dentro de um tanque") o `tr` marca
**0,000**: nenhum relevante no top-10.

Sobre as naturais, o `tr` **nunca vence**: 0 de 29 subconjuntos de 5 ou mais
queries naturais. Esse é o primeiro resultado deste benchmark que sobrevive ao
jackknife.

### O que ainda NÃO está estabelecido

O ganho do IPC dentro de cada par, medido como `ipc_grupo − tr`:

| tema | ganho na técnica | ganho na natural |
|---|---|---|
| perfuração | +0,027 | +0,045 |
| defensivos | −0,109 | −0,139 |
| embalagens | +0,098 | +0,103 |
| cirúrgicos | +0,008 | +0,173 |
| compósitos | +0,125 | +0,278 |
| microbiana | +0,047 | +0,014 |
| **média** | **+0,033** | **+0,079** |
| desvio | 0,082 | 0,143 |

A direção é a esperada — o enriquecimento rende mais onde há lacuna de
vocabulário — mas o efeito ajuda em 5 dos 6 pares nos **dois** tipos, e os
desvios são maiores que a diferença entre as médias. Com n = 6 pares, a
**interação** entre tipo de query e enriquecimento não está demonstrada.

Qual variante de IPC usar também segue indefinido: sobre as naturais,
`ipc_grupo` vence 12 de 29 subconjuntos e `ipc_hierarquia` 11.

E o par de defensivos agrícolas contraria a tendência nos dois tipos (−0,109 e
−0,139), o que merece inspeção antes de qualquer generalização.

## EmbeddingGemma × BM25 (primeira execução)

### Um defeito de dados encontrado no caminho

26 documentos do corpus (2,6%) não têm resumo — são pedidos renumerados
(`BR122…`), com mediana de **14 palavras** contra 145 dos demais. Textos assim
produzem embeddings próximos de quase tudo no espaço vetorial, e o Gemma os
colocava em **78 das 180 posições** do top-10 na variante `tr`. O BM25 colocava
zero, porque não há termo para casar.

Isso é ruído de dados amplificado por uma propriedade da similaridade de
cosseno, não uma falha do modelo. Os 26 são excluídos da avaliação
(`EXCLUIR_SEM_RESUMO` em `src/avaliar_denso.py`); o custo foram 4 julgamentos,
um deles relevante. O efeito no `gemma_tr` é grande: natural 0,336 → 0,494,
técnica 0,574 → 0,696, e a query curta QG001 de 0,191 → 0,608.

### nDCG@10 médio por tipo de query (corpus de 974)

| tipo | n | `bm25_tr` | `gemma_tr` | `bm25_ipc_grupo` | `gemma_ipc_grupo` |
|---|---|---|---|---|---|
| técnica | 7 | **0,765** | 0,696 | **0,788** | 0,721 |
| específica | 3 | **0,909** | 0,768 | **0,913** | 0,736 |
| curta | 1 | **0,757** | 0,608 | 0,492 | **0,787** |
| natural | 7 | 0,178 | **0,494** | 0,326 | **0,526** |

### O resultado central

| par | queda BM25 | queda Gemma |
|---|---|---|
| perfuração | −0,483 | −0,191 |
| defensivos | −0,416 | −0,084 |
| embalagens | −0,720 | −0,147 |
| cirúrgicos | −0,776 | −0,203 |
| compósitos | −0,442 | −0,186 |
| microbiana | −0,781 | −0,473 |
| **média** | **−0,603** | **−0,214** |

Dentro de cada par o conjunto relevante é idêntico e só o vocabulário da
consulta muda. O BM25 perde 0,60 de nDCG@10; o Gemma, 0,21 — **um terço da
queda, em 6 de 6 pares**. Nas consultas em linguagem natural o Gemma quase
triplica o BM25 (0,494 contra 0,178 na variante `tr`).

O quadro não é "denso vence": o BM25 é melhor nas consultas técnicas (0,765
contra 0,696) e nas específicas (0,909 contra 0,768). Cada um ganha onde sua
premissa vale — o léxico quando a consulta traz os termos do documento, o denso
quando não traz. Isso sugere que a recomendação prática é híbrida, não a
substituição de um pelo outro.

### O que ainda contamina estes números

O pool foi construído com BM25 mais a corrida manual, então **164 documentos
inéditos aparecem no top-10 do Gemma e contam como irrelevantes por não terem
sido julgados** — em média 5,3 dos 10 na variante `tr`, contra 0,78 do BM25.
O déficit do Gemma nas consultas técnicas e específicas é, em parte
indeterminada, esse artefato. O resultado central sobrevive porque o viés
empurra contra ele: mesmo penalizado, o denso vence onde a hipótese previa.

Os candidatos inéditos foram julgados — ver a seção seguinte.

## Pool fechado com os rankings densos

Dos 164 candidatos inéditos do top-10 do Gemma, 62 eram documentos sem resumo
(excluídos). Os 102 restantes foram julgados. O pool passou de 522 para 622
julgamentos e a cobertura ficou praticamente completa: **2 de 180** posições não
julgadas no top-10 do `gemma_tr`, contra 78 na primeira medição.

### nDCG@10 médio por tipo de query — corpus de 974, pool fechado

| tipo | n | `bm25_tr` | `gemma_tr` | `bm25_ipc_grupo` | `gemma_ipc_grupo` |
|---|---|---|---|---|---|
| técnica | 7 | **0,761** | 0,725 | **0,781** | 0,757 |
| específica | 3 | **0,871** | 0,826 | **0,874** | 0,809 |
| curta | 1 | **0,757** | 0,622 | 0,492 | **0,800** |
| natural | 7 | 0,177 | **0,546** | 0,323 | **0,592** |

Fechar o pool praticamente eliminou o déficit do Gemma nas consultas técnicas:
de −0,069 para **−0,036**. Ou seja, boa parte do que parecia inferioridade do
modelo denso era gabarito incompleto. Nas naturais o ganho subiu de +0,315 para
**+0,369**.

### O resultado

| par | queda BM25 | queda Gemma |
|---|---|---|
| perfuração | −0,483 | −0,162 |
| defensivos | −0,416 | −0,084 |
| embalagens | −0,720 | −0,145 |
| cirúrgicos | −0,756 | −0,152 |
| compósitos | −0,444 | −0,129 |
| microbiana | −0,781 | −0,447 |
| **média** | **−0,600** | **−0,186** |

Dentro de cada par o conjunto relevante é idêntico e só o vocabulário muda. O
BM25 perde 0,60 de nDCG@10; o Gemma, 0,19 — **31% da queda, em 6 de 6 pares**.

Ganho do Gemma sobre o BM25 nas 7 naturais: média +0,369, desvio 0,255,
positivo em 6 de 7 (a exceção é QP002, −0,049). E o denso vence em **29 de 29**
subconjuntos de 5 ou mais queries naturais. É o resultado mais estável do
projeto.

Na direção contrária, o BM25 continua à frente nas técnicas e nas específicas,
por margens pequenas (0,036 e 0,045). Cada abordagem ganha onde sua premissa
vale — o léxico quando a consulta traz os termos do documento, o denso quando
não traz. A recomendação prática que sai daqui é híbrida, não substitutiva.

### Ressalvas que permanecem

1. Os 622 julgamentos ainda são pré-classificação por LLM, sem revisão humana e
   sem κ medido. **Nada disso é reportável antes dessa revisão.**
2. O pool está fechado para BM25 e Gemma no top-10, não para outros sistemas nem
   para k maior.
3. 18 queries em 9 temas; o efeito nas naturais é grande e estável, mas as
   diferenças pequenas (variantes de IPC, déficit nas técnicas) continuam dentro
   do ruído.

## O que o enriquecimento por IPC faz ao espaço vetorial

### Métricas do Gemma entre as quatro variantes

| tipo | `tr` | `ipc_grupo` | `ipc_direto` | `ipc_hierarquia` |
|---|---|---|---|---|
| técnica | 0,725 | **0,757** | 0,748 | 0,754 |
| natural | 0,546 | 0,592 | **0,593** | 0,539 |
| específica | **0,826** | 0,809 | 0,799 | 0,755 |
| curta | 0,622 | **0,800** | 0,693 | 0,554 |

Ganho sobre `tr`: `ipc_grupo` +0,033 nas técnicas e +0,047 nas naturais;
`ipc_direto` +0,023 e +0,047; `ipc_hierarquia` +0,029 e **−0,007** (positivo em
apenas 1 das 7 naturais). Sobre subconjuntos de 14 ou mais das 18 queries,
`ipc_grupo` vence em 3.810 de 4.048 e o `tr` em nenhum — para o modelo denso,
enriquecer ajuda; a questão é com quanto.

### A geometria

Similaridade de cosseno média entre todos os pares do corpus (974 documentos):

| variante | sim. média | mesma subclasse IPC | outra subclasse | separação | deslocamento vs `tr` |
|---|---|---|---|---|---|
| `tr` | 0,578 | 0,642 | 0,576 | 0,066 | — |
| `ipc_direto` | 0,583 | 0,662 | 0,581 | 0,081 | 0,955 |
| `ipc_grupo` | 0,589 | 0,683 | 0,586 | 0,097 | 0,942 |
| `ipc_hierarquia` | 0,611 | 0,728 | 0,607 | **0,121** | 0,900 |

O enriquecimento faz exatamente o que se esperaria: **aproxima entre si os
documentos da mesma classe**. A separação intra-classe menos inter-classe quase
dobra da `tr` para a `ipc_hierarquia` (0,066 → 0,121), e é a hierarquia que mais
desloca os vetores (cosseno 0,900 com o vetor original do mesmo documento).

Isso é útil quando a consulta mira uma classe inteira e nocivo quando ela precisa
distinguir dentro da classe — e explica por que a `ipc_hierarquia` é a pior das
três enriquecidas nas consultas naturais: ela agrupa por classe a ponto de
apagar a distinção interna.

### A dissociação

Margem de separação do ponto de vista da consulta, isto é
`sim(query, relevantes) − sim(query, irrelevantes)`:

| tipo | `tr` | `ipc_grupo` | `ipc_direto` | `ipc_hierarquia` |
|---|---|---|---|---|
| técnica | 0,1139 | 0,1146 | 0,1139 | 0,1125 |
| natural | 0,1024 | 0,1043 | 0,1031 | 0,1023 |
| específica | 0,1484 | 0,1452 | 0,1486 | 0,1375 |

**As margens são praticamente idênticas — diferenças na terceira casa decimal.**
O enriquecimento move os documentos uns em relação aos outros de forma
substancial, mas quase não altera o quanto a consulta separa relevante de
irrelevante. O ganho de nDCG das variantes com IPC (+0,03 a +0,05) vem de
reordenar empates próximos, não de melhorar o sinal consulta-documento.

Isso qualifica bastante a recomendação: o enriquecimento por IPC no modelo denso
é um efeito de reordenação marginal, e o custo é 2,1× em tokens na
`ipc_hierarquia`.

### Por que o denso resiste à paráfrase

A margem do Gemma cai pouco entre consulta técnica e natural: 0,114 → 0,102,
menos de 10%. O nDCG cai de 0,725 para 0,546 (25%), e o do BM25 despenca de
0,761 para 0,177 (77%). O sinal semântico que o modelo captura sobrevive à
mudança de vocabulário; o que se degrada é a ordenação na vizinhança do topo,
não a capacidade de reconhecer o documento certo.
