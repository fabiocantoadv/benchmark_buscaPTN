# Resultados — EmbeddingGemma × BM25

Última rodada: 18 queries, corpus de 974 documentos, 622 julgamentos.
Métrica principal: nDCG@10. Reprodução: `python3 src/avaliar_denso.py`.

> **Atenção.** Os 622 julgamentos são pré-classificação por um modelo de
> linguagem, sem revisão humana e sem κ medido. Nenhum número abaixo é
> reportável antes dessa revisão.

## 1. O desenho que sustenta a leitura

Seis dos nove temas têm um **par** de queries: uma com vocabulário técnico e
uma paráfrase em linguagem natural, ambas apontando para o **mesmo conjunto de
documentos relevantes**. Dentro do par, portanto, a única coisa que muda é o
vocabulário da consulta — e qualquer diferença de métrica se atribui a ele.

| tipo | n | descrição |
|---|---|---|
| técnica | 7 | vocabulário do domínio, presente nos resumos |
| natural | 7 | paráfrase leiga da técnica correspondente |
| específica | 3 | subconjunto estrito de uma query geral |
| curta | 1 | consulta de poucas palavras |

## 2. Resultado principal

nDCG@10 médio por tipo de consulta, variante `tr` (título + resumo):

| tipo | BM25 | Gemma |
|---|---|---|
| técnica | **0,761** | 0,725 |
| específica | **0,871** | 0,826 |
| curta | **0,757** | 0,622 |
| natural | 0,177 | **0,546** |

A queda de cada sistema quando a mesma necessidade é expressa em linguagem
natural, par a par:

| par | queda BM25 | queda Gemma |
|---|---|---|
| perfuração de poços | −0,483 | −0,162 |
| defensivos agrícolas | −0,416 | −0,084 |
| embalagens | −0,720 | −0,145 |
| instrumentos cirúrgicos | −0,756 | −0,152 |
| materiais compósitos | −0,444 | −0,129 |
| produção microbiana | −0,781 | −0,447 |
| **média** | **−0,600** | **−0,186** |

**O BM25 perde 0,60 de nDCG@10; o Gemma, 0,19 — 31% da queda, em 6 de 6 pares.**
Nas naturais o ganho do denso é +0,369 em média (desvio 0,255), positivo em 6 de
7, e o denso vence em **29 de 29** subconjuntos de 5 ou mais queries naturais.
É o resultado que sobrevive ao jackknife.

Na direção contrária, o BM25 fica à frente nas técnicas e nas específicas, por
margens pequenas (0,036 e 0,045). Cada abordagem ganha onde sua premissa vale.

## 3. Por que o denso resiste à paráfrase

Margem de separação da consulta, `sim(query, relevantes) − sim(query, irrelevantes)`:

| tipo | `tr` | `ipc_grupo` | `ipc_direto` | `ipc_hierarquia` |
|---|---|---|---|---|
| técnica | 0,1139 | 0,1146 | 0,1139 | 0,1125 |
| natural | 0,1024 | 0,1043 | 0,1031 | 0,1023 |
| específica | 0,1484 | 0,1452 | 0,1486 | 0,1375 |

A margem do Gemma cai menos de 10% da consulta técnica para a natural (0,114 →
0,102). O nDCG cai 25%; o do BM25 despenca 77%. O sinal semântico sobrevive à
mudança de vocabulário — o que se degrada é a ordenação na vizinhança do topo,
não a capacidade de reconhecer o documento certo.

## 4. O enriquecimento por IPC

Quatro variantes de texto, com mediana de palavras:

| variante | conteúdo | mediana |
|---|---|---|
| `tr` | título + resumo | 143 |
| `ipc_direto` | + descrição dos símbolos IPC da patente | 178 |
| `ipc_grupo` | + descrição de grupo principal e subgrupo | 208 |
| `ipc_hierarquia` | + cadeia hierárquica completa | 322 |

nDCG@10 do Gemma:

| tipo | `tr` | `ipc_grupo` | `ipc_direto` | `ipc_hierarquia` |
|---|---|---|---|---|
| técnica | 0,725 | **0,757** | 0,748 | 0,754 |
| natural | 0,546 | 0,592 | **0,593** | 0,539 |
| específica | **0,826** | 0,809 | 0,799 | 0,755 |

Sobre subconjuntos de 14 ou mais das 18 queries, `ipc_grupo` vence em 3.810 de
4.048 e o `tr` em nenhum.

### O que o enriquecimento faz ao espaço vetorial

Similaridade de cosseno média entre todos os pares do corpus:

| variante | sim. média | mesma subclasse IPC | outra subclasse | separação | deslocamento vs `tr` |
|---|---|---|---|---|---|
| `tr` | 0,578 | 0,642 | 0,576 | 0,066 | — |
| `ipc_direto` | 0,583 | 0,662 | 0,581 | 0,081 | 0,955 |
| `ipc_grupo` | 0,589 | 0,683 | 0,586 | 0,097 | 0,942 |
| `ipc_hierarquia` | 0,611 | 0,728 | 0,607 | **0,121** | 0,900 |

O enriquecimento aproxima entre si os documentos da mesma classe — a separação
intra menos inter-classe quase dobra da `tr` para a `hierarquia`, que também é a
que mais desloca os vetores. Isso ajuda quando a consulta mira a classe inteira e
atrapalha quando é preciso distinguir dentro dela, o que explica a `hierarquia`
ser a pior das três enriquecidas nas naturais (−0,007 sobre `tr`, positiva em
apenas 1 das 7).

**Mas as margens da seção 3 são praticamente idênticas entre as variantes**, com
diferenças na terceira casa decimal. O enriquecimento move os documentos uns em
relação aos outros de forma substancial e quase não altera o quanto a consulta
separa relevante de irrelevante: o ganho de +0,03 a +0,05 vem de reordenar
empates próximos, não de melhorar o sinal. Custo: 1,45× em tokens na
`ipc_grupo`, 2,1× na `ipc_hierarquia`.

## 5. O híbrido ingênuo não funciona

Fusão recíproca de posto (RRF, k = 60) entre BM25 e Gemma, variante `tr`:

| tipo | BM25 | Gemma | RRF |
|---|---|---|---|
| técnica | 0,761 | 0,725 | **0,778** |
| curta | 0,757 | 0,622 | **0,780** |
| específica | **0,871** | 0,826 | 0,864 |
| natural | 0,177 | **0,546** | 0,335 |
| **geral (18 queries)** | 0,552 | **0,666** | 0,620 |

A fusão ganha onde os dois já são competentes e destrói o ganho onde ele importa:
nas naturais, derruba de 0,546 para 0,335 (em QD002, de 0,826 para 0,240). O RRF
é simétrico e dá o mesmo peso a um sistema que ali marca 0,177.

A recomendação híbrida só se sustenta com **peso adaptativo ao tipo de consulta**,
ou com a combinação aprendida dentro do modelo, em vez de fundida por posto
depois do fato.

## 6. Um achado colateral sobre dados

26 documentos do corpus original (2,6%) não tinham resumo — pedidos renumerados
`BR122…`, com mediana de 14 palavras contra 145. Textos degenerados produzem
embeddings próximos de quase tudo: o Gemma colocava esses 26 documentos em **78
das 180 posições** do top-10; o BM25, em zero. Excluí-los muda o `gemma_tr` de
0,336 para 0,494 nas naturais e de 0,574 para 0,696 nas técnicas.

Vale como alerta para qualquer trabalho de recuperação densa sobre o dump do
INPI: metadados incompletos não são ruído neutro em espaços vetoriais.

## Apêndice — como o gabarito foi construído

O necessário para julgar a validade dos números acima; o restante está no
histórico do git.

**Corpus.** 1.000 patentes do INPI com título, resumo, IPC e descrições de IPC
em português; composição congelada em `dados/numeros_corpus_piloto.txt`. Os
distratores foram sorteados de uma base de 45 mil, **fora** da amostra temática
que originou as primeiras queries — sortear dela traria relevantes não julgados.
Na avaliação ficam 974 (seção 6).

**Gabarito.** 622 julgamentos em escala 0–3 (`dados/pool_piloto_gabarito.tsv`),
com nota do LLM, justificativa e as colunas `relevancia_final` e `revisor` em
branco. `src/consolidar_corpus_ipc.py` regrava o qrels usando a nota humana
quando preenchida — dá para revisar aos poucos. R mediano 21,5 em 974 (2,2% do
corpus); P@10 de um ranqueador aleatório 0,021.

**Pool, três fontes.** Pooling automático das 4 variantes; corrida manual por
termo técnico + IPC do tema; e o top-10 de cada coleção densa, julgado depois.
A corrida manual não é opcional: um pool só de BM25 deixa de fora justamente os
documentos que a consulta natural deveria recuperar e não recupera, enviesando o
gabarito **contra** os sistemas que resolvem a lacuna de vocabulário. A
`BR0920002` — alvo exato da QD002 — não foi recuperada por nenhuma variante
lexical. O mesmo se repetiu com o denso: 164 candidatos inéditos, 102 julgados;
a cobertura do top-10 denso passou de 5,3 posições não julgadas em 10 para 2 em
180. **Todo sistema novo exige nova rodada.**

**Métricas.** O MRR saturou em 1,000 em todas as combinações — o primeiro
colocado é sempre relevante. Recall@k é limitado por `k/R`; use nDCG ou
R-Precision. O tokenizador do BM25 descarta tokens com menos de 3 caracteres,
então "5G" não entra nas queries de telecomunicações.

**Reproduzir.** `python3 src/avaliar_denso.py` refaz a tabela da seção 2 a
partir do que está versionado, desde que os embeddings existam em `embeddings/`
(ver `docs/embeddings-gemma300.md`). Reconstruir o corpus do zero exige o
Postgres do INPI: `src/rodar_export_ipc.sh` → `src/enriquecer_ipc_json_pt.py` →
`src/gerar_variante_ipc_grupo.py` → `src/consolidar_corpus_ipc.py`.

## 7. Limitações

1. **Julgamentos sem revisão humana.** Bloqueante para publicação.
2. **Pool fechado só para BM25 e Gemma, no top-10.** Qualquer sistema novo exige
   nova rodada de julgamento — com o Gemma foram 164 candidatos inéditos.
3. **18 queries em 9 temas.** O efeito nas naturais é grande e estável; as
   diferenças pequenas (entre variantes de IPC, e o déficit do denso nas
   técnicas) continuam dentro do ruído.
4. **Um único modelo denso**, em uma única configuração de prompts.
