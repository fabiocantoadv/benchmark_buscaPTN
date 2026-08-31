\set ON_ERROR_STOP on

CREATE TEMP TABLE tmp_amostra AS
WITH file_text AS (
  SELECT pg_read_file('/tmp/numeros_amostra_1000_patentes.txt') AS content
),
linhas AS (
  SELECT regexp_split_to_table(content, E'\n') AS line
  FROM file_text
)
SELECT
  row_number() OVER () AS ordem_arquivo,
  trim(line) AS num_original,
  regexp_replace(trim(line), '[^0-9]', '', 'g') AS num_digits,
  'BR' || regexp_replace(trim(line), '[^0-9]', '', 'g') AS num_pedido_normalizado
FROM linhas
WHERE trim(line) <> '';

CREATE TEMP TABLE tmp_matches AS
WITH matches AS (
  SELECT
    a.ordem_arquivo,
    p.cod_pedido,
    'num_pedido' AS origem_match,
    1 AS prioridade
  FROM tmp_amostra a
  JOIN dbptn.ptn_pedido p
    ON regexp_replace(p.num_pedido, '[^0-9]', '', 'g') = a.num_digits

  UNION ALL

  SELECT
    a.ordem_arquivo,
    p.cod_pedido,
    'renumeracao_origem' AS origem_match,
    2 AS prioridade
  FROM tmp_amostra a
  JOIN dbptn.ptn_renumeracao rn
    ON regexp_replace(rn.no_pedido_origem, '[^0-9]', '', 'g') = a.num_digits
  JOIN dbptn.ptn_pedido p
    ON p.cod_pedido = rn.cd_pedido_derivad
)
SELECT DISTINCT ON (ordem_arquivo)
  ordem_arquivo,
  cod_pedido,
  origem_match
FROM matches
ORDER BY ordem_arquivo, prioridade, cod_pedido;

CREATE TEMP TABLE tmp_resumos AS
SELECT
  cod_pedido,
  string_agg(
    DISTINCT NULLIF(trim(regexp_replace(resumo, '[[:space:]]+', ' ', 'g')), ''),
    ' '
  ) AS resumo
FROM dbptn.ptn_resumo_pedido
GROUP BY cod_pedido;

CREATE TEMP TABLE tmp_ipc AS
SELECT
  cp.cod_pedido,
  string_agg(
    DISTINCT trim(c.cod_internacional),
    '; ' ORDER BY trim(c.cod_internacional)
  ) AS ipc,
  string_agg(
    DISTINCT cp.cod_classificacao::text,
    '; ' ORDER BY cp.cod_classificacao::text
  ) AS ipc_cod_classificacao
FROM dbptn.ptn_classif_pedido cp
JOIN dbptn.ptn_classif c
  ON c.cod_classificacao = cp.cod_classificacao
GROUP BY cp.cod_pedido;

CREATE TEMP TABLE tmp_pct AS
SELECT
  cod_pedido,
  string_agg(
    DISTINCT trim(num_pct),
    '; ' ORDER BY trim(num_pct)
  ) AS num_pct,
  string_agg(
    DISTINCT CASE
      WHEN NULLIF(trim(cd_ompi), '') IS NULL THEN NULL
      ELSE 'WO' || trim(cd_ompi)
    END,
    '; ' ORDER BY CASE
      WHEN NULLIF(trim(cd_ompi), '') IS NULL THEN NULL
      ELSE 'WO' || trim(cd_ompi)
    END
  ) AS num_publicacao_pct,
  string_agg(
    DISTINCT dt_pct::date::text,
    '; ' ORDER BY dt_pct::date::text
  ) AS dt_pct,
  string_agg(
    DISTINCT dt_ompi::date::text,
    '; ' ORDER BY dt_ompi::date::text
  ) AS dt_publicacao_pct
FROM dbptn.ptn_pct
GROUP BY cod_pedido;

COPY (
  SELECT
    a.ordem_arquivo,
    a.num_original,
    a.num_pedido_normalizado,
    (p.cod_pedido IS NOT NULL) AS encontrado,
    m.origem_match,
    p.cod_pedido,
    CASE
      WHEN p.num_pedido IS NULL THEN NULL
      ELSE 'BR' || regexp_replace(p.num_pedido, '[^0-9]', '', 'g')
    END AS num_pedido_db_normalizado,
    p.dt_deposito,
    p.dt_publicacao,
    COALESCE(
      pct.num_publicacao_pct,
      CASE
        WHEN p.dt_publicacao IS NULL OR p.num_pedido IS NULL THEN NULL
        ELSE 'BR' || regexp_replace(p.num_pedido, '[^0-9]', '', 'g')
      END
    ) AS num_publicacao,
    pct.num_publicacao_pct,
    pct.dt_publicacao_pct,
    pct.num_pct,
    pct.dt_pct,
    NULLIF(trim(regexp_replace(p.titulo, '[[:space:]]+', ' ', 'g')), '') AS titulo,
    r.resumo,
    ipc.ipc,
    ipc.ipc_cod_classificacao,
    concat_ws(
      ' ',
      CASE
        WHEN NULLIF(trim(regexp_replace(p.titulo, '[[:space:]]+', ' ', 'g')), '') IS NULL THEN NULL
        ELSE 'Titulo: ' || NULLIF(trim(regexp_replace(p.titulo, '[[:space:]]+', ' ', 'g')), '')
      END,
      CASE
        WHEN r.resumo IS NULL THEN NULL
        ELSE 'Resumo: ' || r.resumo
      END
    ) AS texto_para_embedding
  FROM tmp_amostra a
  LEFT JOIN tmp_matches m
    ON m.ordem_arquivo = a.ordem_arquivo
  LEFT JOIN dbptn.ptn_pedido p
    ON p.cod_pedido = m.cod_pedido
  LEFT JOIN tmp_resumos r
    ON r.cod_pedido = p.cod_pedido
  LEFT JOIN tmp_ipc ipc
    ON ipc.cod_pedido = p.cod_pedido
  LEFT JOIN tmp_pct pct
    ON pct.cod_pedido = p.cod_pedido
  ORDER BY a.ordem_arquivo, p.cod_pedido
) TO '/tmp/patentes_benchmark_amostra_1000.tsv'
WITH (FORMAT text, DELIMITER E'\t', HEADER true, ENCODING 'UTF8');

COPY (
  SELECT a.*
  FROM tmp_amostra a
  LEFT JOIN tmp_matches m
    ON m.ordem_arquivo = a.ordem_arquivo
  WHERE m.cod_pedido IS NULL
  ORDER BY a.ordem_arquivo
) TO '/tmp/patentes_benchmark_amostra_1000_nao_encontradas.tsv'
WITH (FORMAT text, DELIMITER E'\t', HEADER true, ENCODING 'UTF8');

COPY (
  WITH resultado AS (
    SELECT
      a.ordem_arquivo,
      p.cod_pedido,
      p.titulo,
      r.resumo,
      ipc.ipc,
      COALESCE(
        pct.num_publicacao_pct,
        CASE
          WHEN p.dt_publicacao IS NULL OR p.num_pedido IS NULL THEN NULL
          ELSE 'BR' || regexp_replace(p.num_pedido, '[^0-9]', '', 'g')
        END
      ) AS num_publicacao
    FROM tmp_amostra a
    LEFT JOIN tmp_matches m
      ON m.ordem_arquivo = a.ordem_arquivo
    LEFT JOIN dbptn.ptn_pedido p
      ON p.cod_pedido = m.cod_pedido
    LEFT JOIN tmp_resumos r
      ON r.cod_pedido = p.cod_pedido
    LEFT JOIN tmp_ipc ipc
      ON ipc.cod_pedido = p.cod_pedido
    LEFT JOIN tmp_pct pct
      ON pct.cod_pedido = p.cod_pedido
  )
  SELECT 'linhas_arquivo' AS metrica, count(*)::text AS valor FROM tmp_amostra
  UNION ALL
  SELECT 'linhas_exportadas', count(*)::text FROM resultado
  UNION ALL
  SELECT 'pedidos_encontrados', count(cod_pedido)::text FROM resultado
  UNION ALL
  SELECT 'pedidos_nao_encontrados', count(*)::text FROM resultado WHERE cod_pedido IS NULL
  UNION ALL
  SELECT 'com_titulo', count(*)::text FROM resultado WHERE NULLIF(trim(titulo), '') IS NOT NULL
  UNION ALL
  SELECT 'com_resumo', count(*)::text FROM resultado WHERE NULLIF(trim(resumo), '') IS NOT NULL
  UNION ALL
  SELECT 'com_ipc', count(*)::text FROM resultado WHERE NULLIF(trim(ipc), '') IS NOT NULL
  UNION ALL
  SELECT 'com_num_publicacao', count(*)::text FROM resultado WHERE NULLIF(trim(num_publicacao), '') IS NOT NULL
) TO '/tmp/patentes_benchmark_amostra_1000_resumo.tsv'
WITH (FORMAT text, DELIMITER E'\t', HEADER true, ENCODING 'UTF8');
