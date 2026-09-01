# Gemma 300M - comandos iniciais

Instalar ou atualizar dependencias, se necessario:

```bash
python3 -m pip install -U sentence-transformers torch pandas numpy
```

Smoke test com 50 documentos, titulo + resumo:

```bash
cd /Users/fabiocanto/Downloads/dados_patentes/benchmark_patentes_semantica
python3 gerar_embeddings_gemma300_benchmark.py --variant tr --limit 50 --output-dir embeddings/gemma300_tr_docs_smoke50 --overwrite
```

Primeira colecao completa, titulo + resumo:

```bash
cd /Users/fabiocanto/Downloads/dados_patentes/benchmark_patentes_semantica
python3 gerar_embeddings_gemma300_benchmark.py --variant tr
```

Segunda colecao completa, titulo + resumo + IPC hierarquica em portugues:

```bash
cd /Users/fabiocanto/Downloads/dados_patentes/benchmark_patentes_semantica
python3 gerar_embeddings_gemma300_benchmark.py --variant ipc_hierarquia
```

Embeddings das queries textuais:

```bash
cd /Users/fabiocanto/Downloads/dados_patentes/benchmark_patentes_semantica
python3 gerar_embeddings_gemma300_benchmark.py --kind queries
```

Saidas padrao:

```text
embeddings/gemma300_tr_docs/
embeddings/gemma300_tr_ipc_hierarquia_pt_docs/
embeddings/gemma300_queries/
```

Cada pasta contem:

```text
config.json
manifest.jsonl
embeddings_bloco_00000.npy
metadata_bloco_00000.tsv
```
