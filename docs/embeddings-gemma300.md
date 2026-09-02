# Rodar o EmbeddingGemma no benchmark

O modelo `google/embeddinggemma-300m` é *gated*: é preciso aceitar a licença em
https://huggingface.co/google/embeddinggemma-300m e estar autenticado. Se o
smoke test já rodou nesta máquina, o modelo está em cache e não é preciso
repetir nada disso.

## Caminho local (Mac, MPS)

```bash
cd ~/Downloads/dados_patentes/benchmark_patentes_semantica
python3 -m pip install -U sentence-transformers torch pandas numpy   # se necessário

python3 src/gerar_embeddings_gemma300_benchmark.py --kind queries
python3 src/gerar_embeddings_gemma300_benchmark.py --variant tr
python3 src/gerar_embeddings_gemma300_benchmark.py --variant ipc_grupo
python3 src/gerar_embeddings_gemma300_benchmark.py --variant ipc_direto
python3 src/gerar_embeddings_gemma300_benchmark.py --variant ipc_hierarquia

python3 src/avaliar_denso.py
```

São 5 coleções: 18 queries e 4 × 1.000 documentos. Cada coleção grava em
`embeddings/<nome>/` e o script pula blocos já feitos, então dá para
interromper e retomar. Use `--overwrite` para regerar.

Antes de rodar tudo, um teste de 50 documentos para conferir que o ambiente está
de pé:

```bash
python3 src/gerar_embeddings_gemma300_benchmark.py --variant tr --limit 50 \
  --output-dir embeddings/_smoke50 --overwrite
```

Se der erro de memória no MPS, reduza: `--batch-size 8 --block-size 128`.
Para forçar CPU (mais lento, sempre funciona): `--device cpu`.

## Caminho Colab (GPU)

`notebooks/benchmark_patentes_colab.ipynb` clona o repositório e faz o mesmo de
ponta a ponta. Requer GPU no ambiente de execução e o `HF_TOKEN` cadastrado nos
Secrets do Colab (cada pessoa cadastra o seu). **Dê `git push` antes** — o
notebook usa a versão que está no GitHub, não a sua cópia local.

## O que `avaliar_denso.py` responde

Ele imprime, além do nDCG@10 por query e por tipo:

- **denso × léxico separados por tipo de query** — a comparação que interessa;
- **a queda dentro de cada par técnica/natural**, para BM25 e para o Gemma.

O conjunto relevante é idêntico dentro de cada par, então a queda mede só o
efeito do vocabulário da consulta. O BM25 cai de 0,765 para 0,178. Se a queda do
Gemma for menor, ele fecha parte da lacuna — e é essa a hipótese central.

Resultados vão para `resultados/` com carimbo de data e hora.
