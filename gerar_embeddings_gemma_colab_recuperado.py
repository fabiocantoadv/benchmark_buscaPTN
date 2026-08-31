"""Gera embeddings do EmbeddingGemma no Google Colab com retomada pelo Drive.

Uso no Colab (apos copiar este arquivo e o TSV para o Google Drive):

    !pip -q install -U sentence-transformers
    %run /content/drive/MyDrive/dados_patentes/gerar_embeddings_gemma_colab.py

Os caminhos podem ser alterados por variaveis de ambiente, por exemplo:

    %env PATENTEIA_DRIVE_DIR=/content/drive/MyDrive/minha_pasta
"""

import csv
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_NAME = "google/embeddinggemma-300m"
COLUNA_TEXTO = "text"
DIMENSAO_EMBEDDING_ESPERADA = 768

# Configuracao solicitada para o Colab.
BLOCO_SIZE = int(os.getenv("BLOCO_SIZE", "10000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
MAX_BLOCOS_POR_EXECUCAO = int(os.getenv("MAX_BLOCOS_POR_EXECUCAO", "0"))

DRIVE_DIR = Path(
    os.getenv("PATENTEIA_DRIVE_DIR", "/content/drive/MyDrive/dados_patentes")
)
ARQUIVO_ENTRADA = Path(
    os.getenv(
        "PATENTEIA_EMBEDDINGS_INPUT_TSV",
        str(DRIVE_DIR / "03_patentes_embeddings_dataset_normalizado.tsv"),
    )
)
PASTA_EMBEDDINGS = Path(
    os.getenv(
        "PATENTEIA_EMBEDDINGS_OUTPUT_DIR",
        str(DRIVE_DIR / "output" / "patentes_embeddinggemma_300m_colab_10k"),
    )
)
MANIFEST_PATH = PASTA_EMBEDDINGS / "manifest_gemma300m_colab_10k.jsonl"


def montar_drive() -> None:
    try:
        from google.colab import drive
    except ImportError:
        return

    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")


def limpar_memoria() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def contar_linhas_dados(path: Path) -> int:
    total = 0
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["num_pedido"],
        dtype=str,
        low_memory=False,
        chunksize=100_000,
        quoting=csv.QUOTE_NONE,
    ):
        total += len(chunk)
    return total


def caminho_bloco(indice: int) -> Path:
    return PASTA_EMBEDDINGS / f"embeddings_gemma300m_bloco_{indice:05d}.npy"


def bloco_existente_valido(indice: int, linhas_esperadas: int) -> bool:
    destino = caminho_bloco(indice)
    if not destino.exists():
        return False
    try:
        array = np.load(destino, mmap_mode="r")
        valido = (
            array.shape == (linhas_esperadas, DIMENSAO_EMBEDDING_ESPERADA)
            and np.isfinite(array[: min(10, linhas_esperadas)]).all()
        )
        shape = array.shape
        del array
    except Exception as exc:
        print(f"Bloco {indice:05d} nao passou na validacao: {exc}")
        return False
    if valido:
        print(f"Bloco {indice:05d} ja concluido ({shape}); pulando.")
    return valido


def batches_para_tentativa() -> list[int]:
    batches = [BATCH_SIZE]
    while batches[-1] > 1:
        batches.append(max(1, batches[-1] // 2))
    return batches


def erro_de_memoria(exc: Exception) -> bool:
    mensagem = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in mensagem


def salvar_bloco(
    model: SentenceTransformer,
    df: pd.DataFrame,
    indice: int,
    inicio: int,
) -> int:
    if COLUNA_TEXTO not in df.columns:
        raise ValueError(f"Coluna obrigatoria ausente: {COLUNA_TEXTO}")

    textos = df[COLUNA_TEXTO].fillna("").astype(str).str.strip().tolist()
    vazios = sum(not texto for texto in textos)
    if vazios:
        raise ValueError(f"Bloco {indice:05d} contem {vazios} textos vazios")

    embeddings = None
    batch_usado = BATCH_SIZE
    for batch_tentativa in batches_para_tentativa():
        try:
            print(f"Gerando bloco {indice:05d} com batch {batch_tentativa}...")
            embeddings = model.encode_document(
                textos,
                batch_size=batch_tentativa,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            batch_usado = batch_tentativa
            break
        except Exception as exc:
            if not erro_de_memoria(exc) or batch_tentativa == 1:
                raise
            print(f"Memoria insuficiente com batch {batch_tentativa}; reduzindo.")
            limpar_memoria()

    array = np.asarray(embeddings, dtype=np.float32)
    shape_esperado = (len(textos), DIMENSAO_EMBEDDING_ESPERADA)
    if array.shape != shape_esperado:
        raise ValueError(f"Shape inesperado: {array.shape}; esperado: {shape_esperado}")

    destino = caminho_bloco(indice)
    temporario = destino.with_suffix(".tmp.npy")
    np.save(temporario, array)
    os.replace(temporario, destino)

    registro = {
        "indice_bloco": indice,
        "arquivo": str(destino),
        "inicio": inicio,
        "fim": inicio + len(textos) - 1,
        "num_registros": len(textos),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "modelo": MODEL_NAME,
        "batch_size_configurado": BATCH_SIZE,
        "batch_size_usado": batch_usado,
        "max_seq_length": MAX_SEQ_LENGTH,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with MANIFEST_PATH.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")

    print(f"Salvo: {destino} | shape={array.shape}")
    del array, embeddings, textos
    limpar_memoria()
    return batch_usado


def main() -> None:
    montar_drive()
    PASTA_EMBEDDINGS.mkdir(parents=True, exist_ok=True)

    if not ARQUIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"TSV nao encontrado: {ARQUIVO_ENTRADA}\n"
            "Copie-o para o Drive ou defina PATENTEIA_EMBEDDINGS_INPUT_TSV."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Ative uma GPU no Colab: Ambiente de execucao > Alterar tipo > GPU")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Entrada: {ARQUIVO_ENTRADA}")
    print(f"Saida: {PASTA_EMBEDDINGS}")
    print(f"Bloco: {BLOCO_SIZE} documentos | batch size: {BATCH_SIZE}")

    total = contar_linhas_dados(ARQUIVO_ENTRADA)
    total_blocos = (total + BLOCO_SIZE - 1) // BLOCO_SIZE
    print(f"Total: {total} documentos em {total_blocos} blocos")

    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model.max_seq_length = MAX_SEQ_LENGTH

    leitor = pd.read_csv(
        ARQUIVO_ENTRADA,
        sep="\t",
        dtype=str,
        low_memory=False,
        chunksize=BLOCO_SIZE,
        quoting=csv.QUOTE_NONE,
    )
    gerados = 0
    inicio_execucao = time.time()
    for indice, df in enumerate(leitor):
        if bloco_existente_valido(indice, len(df)):
            continue
        salvar_bloco(model, df, indice, indice * BLOCO_SIZE)
        gerados += 1
        del df
        limpar_memoria()
        if MAX_BLOCOS_POR_EXECUCAO and gerados >= MAX_BLOCOS_POR_EXECUCAO:
            print("Limite de blocos desta execucao atingido.")
            break

    minutos = (time.time() - inicio_execucao) / 60
    print(f"Execucao encerrada: {gerados} blocos novos em {minutos:.1f} min")
    print("Pode executar novamente: os blocos validos serao pulados automaticamente.")


if __name__ == "__main__":
    main()
