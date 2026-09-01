# Recuperar IPC do corpus piloto no Postgres

O corpus vem de `amostra_50000.xlsx`, que não tem IPC. Para ter as três
variantes de texto (`tr`, `ipc_direto`, `ipc_hierarquia`) no corpus de 1.000, é
mais limpo reexportar os 1.000 do Postgres do que costurar IPC no texto do xlsx
— assim título, resumo e IPC voltam todos da mesma fonte e da mesma
normalização usada na amostra original.

A lista de números já está pronta: **`numeros_corpus_piloto.txt`** (1.000
linhas, coluna `numero_inpi` do corpus — 981 no formato `112020004359` e 19 no
formato `PI0812436`; o SQL tira os não-dígitos e casa pelos dígitos).

```bash
cd ~/Downloads/dados_patentes/benchmark_patentes_semantica

# 1. lista de números para dentro do container
docker cp numeros_corpus_piloto.txt patentes-postgres:/tmp/numeros_corpus_piloto.txt

# 2. export com IPC (mesmo SQL da amostra original, só com os caminhos trocados)
docker cp export_corpus_piloto_ipc.sql patentes-postgres:/tmp/
docker exec -u postgres patentes-postgres \
  psql -d <banco> -f /tmp/export_corpus_piloto_ipc.sql

# 3. trazer de volta
docker cp patentes-postgres:/tmp/corpus_piloto_bruto.tsv .
docker cp patentes-postgres:/tmp/corpus_piloto_bruto_nao_encontradas.tsv .

# 4. descrições IPC em PT + hierarquia
python3 enriquecer_ipc_json_pt.py --entrada corpus_piloto_bruto.tsv \
                                  --saida corpus_piloto_ipc.tsv

# 5. conferir cobertura e regravar o qrels
python3 consolidar_corpus_ipc.py corpus_piloto_ipc.tsv
```

O passo 5 avisa se algum dos 135 documentos julgados não voltou do Postgres —
esse é o único caso que exige decisão sua (manter o texto do xlsx para ele ou
tirá-lo do gabarito).

Depois disso, aponte `avaliar_benchmark` e `baseline_bm25` para
`corpus_piloto_ipc.tsv` e as colunas `texto_para_embedding`,
`texto_para_embedding_ipc_pt`, `texto_para_embedding_ipc_hierarquia_pt`.

Notas:

- `enriquecer_ipc_json_pt.py` agora aceita `--entrada/--saida/--faltantes/--resumo/--ipc-json`;
  sem argumentos ele mantém o comportamento antigo sobre a amostra de 1.000.
- `export_corpus_piloto_ipc.sql` é cópia de `export_amostra_benchmark_tsv.sql`
  com os caminhos trocados: lê `/tmp/numeros_corpus_piloto.txt` e escreve
  `/tmp/corpus_piloto_bruto*.tsv`.
- O JSON de IPC em PT está fixado em
  `~/Documents/ipc_net_beta/dist/ipc/ipc_titles_pt_flat_20260101.json`; use
  `--ipc-json` se mudar de lugar.
