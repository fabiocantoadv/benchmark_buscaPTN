#!/usr/bin/env bash
# Exporta os 1.000 do corpus piloto com IPC, a partir do Postgres em Docker.
# Uso:  bash rodar_export_ipc.sh [container] [usuario] [banco]
set -euo pipefail
cd "$(dirname "$0")"

CONTAINER="${1:-patentes-postgres}"
PGUSER_IN="${2:-}"
PGDB_IN="${3:-}"

echo "== container: $CONTAINER"
docker inspect -f '{{.State.Status}}' "$CONTAINER" >/dev/null 2>&1 || {
  echo "Container nao encontrado. Rodando agora:"; docker ps --format '  {{.Names}}'; exit 1; }

# usuario: argumento > POSTGRES_USER do container > tentativa por eliminacao
PGUSER="$PGUSER_IN"
if [ -z "$PGUSER" ]; then
  PGUSER=$(docker exec "$CONTAINER" printenv POSTGRES_USER 2>/dev/null || true)
fi
if [ -z "$PGUSER" ]; then
  for cand in postgres patentes inpi admin "$USER"; do
    if docker exec "$CONTAINER" psql -U "$cand" -Atc "SELECT 1" postgres >/dev/null 2>&1; then
      PGUSER="$cand"; break
    fi
  done
fi
[ -n "$PGUSER" ] || { echo "Nao descobri o usuario do banco."
  echo "Veja com:  docker exec $CONTAINER env | grep -i postgres"
  echo "e rode:    bash rodar_export_ipc.sh $CONTAINER <usuario> <banco>"; exit 1; }
echo "== usuario: $PGUSER"

# banco: argumento > POSTGRES_DB do container > primeiro banco nao-template
PGDB="$PGDB_IN"
if [ -z "$PGDB" ]; then
  PGDB=$(docker exec "$CONTAINER" printenv POSTGRES_DB 2>/dev/null || true)
fi
if [ -z "$PGDB" ]; then
  PGDB=$(docker exec "$CONTAINER" psql -U "$PGUSER" -d postgres -Atc \
    "SELECT datname FROM pg_database WHERE datistemplate = false AND datname <> 'postgres' LIMIT 1;" 2>/dev/null || true)
fi
[ -n "$PGDB" ] || { echo "Nao descobri o banco. Liste com:  docker exec $CONTAINER psql -U $PGUSER -l"; exit 1; }
echo "== banco: $PGDB"

# confere que o schema esperado existe
docker exec "$CONTAINER" psql -U "$PGUSER" -d "$PGDB" -Atc \
  "SELECT count(*) FROM information_schema.tables WHERE table_schema='dbptn';" \
  | { read n; echo "== tabelas em dbptn: $n"; [ "$n" -gt 0 ] || {
        echo "Schema dbptn vazio neste banco. Bancos disponiveis:"
        docker exec "$CONTAINER" psql -U "$PGUSER" -d "$PGDB" -Atc \
          "SELECT datname FROM pg_database WHERE datistemplate=false;" | sed 's/^/  /'
        exit 1; }; }

echo "== enviando lista e SQL"
docker cp numeros_corpus_piloto.txt "$CONTAINER":/tmp/numeros_corpus_piloto.txt
docker cp export_corpus_piloto_ipc.sql "$CONTAINER":/tmp/export_corpus_piloto_ipc.sql

echo "== rodando o export"
docker exec "$CONTAINER" psql -U "$PGUSER" -d "$PGDB" -v ON_ERROR_STOP=1 \
  -f /tmp/export_corpus_piloto_ipc.sql

echo "== trazendo os arquivos"
docker cp "$CONTAINER":/tmp/corpus_piloto_bruto.tsv .
docker cp "$CONTAINER":/tmp/corpus_piloto_bruto_nao_encontradas.tsv . 2>/dev/null || true

echo
echo "OK. Registros em corpus_piloto_bruto.tsv: $(( $(wc -l < corpus_piloto_bruto.tsv) - 1 ))"
echo "Pode avisar o Claude — ele faz o enriquecimento de IPC e a consolidacao daqui."
