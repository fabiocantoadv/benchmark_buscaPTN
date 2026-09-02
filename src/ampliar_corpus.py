#!/usr/bin/env python3
"""Sorteia os documentos adicionais do corpus ampliado e grava a lista de numeros.

O corpus de 1.000 e mantido integralmente (todos os julgamentos ja feitos
continuam validos) e recebe distratores novos sorteados do amostra_50000.xlsx,
sempre FORA da amostra tematica original de 1.000 -- que foi montada em torno
dos 3 primeiros temas e traria relevantes nao julgados.

Saida: dados/numeros_corpus_ampliado.txt, para o rodar_export_ipc.sh.
"""
import csv, re, sys
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
N_TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SEED = 20260902

base = pd.read_excel(RAIZ / "amostra_50000.xlsx", dtype=str)
base["num_pedido_normalizado"] = ("BR" + base.numero_inpi.astype(str)
                                  .str.replace(r"[^0-9]", "", regex=True))
base = base.drop_duplicates("num_pedido_normalizado")

atual = pd.read_csv(DADOS / "corpus_piloto_ipc.tsv", sep="\t", dtype=str, low_memory=False,
                    quoting=csv.QUOTE_NONE, escapechar="\\")
manter = set(atual.num_pedido_normalizado)

# a amostra tematica original: reconstituida a partir da lista congelada do piloto
piloto = {("BR" + re.sub(r"[^0-9]", "", l)).strip()
          for l in (DADOS / "numeros_corpus_piloto.txt").read_text(encoding="utf-8").split()}

fora = base[~base.num_pedido_normalizado.isin(manter | piloto)]
novos = fora.sample(N_TOTAL - len(manter), random_state=SEED)

lista = pd.concat([atual[["num_pedido_normalizado"]].assign(
                       numero_inpi=atual.num_pedido_normalizado.str[2:]),
                   novos[["num_pedido_normalizado", "numero_inpi"]]])
lista.numero_inpi.to_csv(DADOS / "numeros_corpus_ampliado.txt", index=False, header=False)
print(f"corpus ampliado: {len(lista)} documentos "
      f"({len(manter)} mantidos + {len(novos)} novos)")
print("lista:", DADOS / "numeros_corpus_ampliado.txt")
