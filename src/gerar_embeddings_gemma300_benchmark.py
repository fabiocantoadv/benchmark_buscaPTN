#!/usr/bin/env python3
"""Gera embeddings Gemma 300M para o benchmark de patentes.

Uso local no Mac:

    python3 gerar_embeddings_gemma300_benchmark.py --variant tr --limit 50
    python3 gerar_embeddings_gemma300_benchmark.py --variant tr
    python3 gerar_embeddings_gemma300_benchmark.py --variant ipc_hierarquia

Tambem gera embeddings das queries:

    python3 gerar_embeddings_gemma300_benchmark.py --kind queries
    python3 gerar_embeddings_gemma300_benchmark.py --kind queries \
        --query-instruction ipc --overwrite
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = REPO_DIR / "dados"
MODEL_NAME_DEFAULT = "google/embeddinggemma-300m"
DIMENSAO_ESPERADA = 768

DOC_INSTRUCTION = (
    "Represente este documento de patente para busca semântica. "
    "Foque no problema técnico, na solução proposta, no domínio de aplicação "
    "e na finalidade da invenção:"
)

QUERY_INSTRUCTION = (
    "Represente esta consulta de busca de patentes para recuperar documentos "
    "tecnicamente relevantes:"
)

# Instrucoes de query nomeadas. O nome entra no config.json e no sufixo da
# colecao, para que duas configuracoes nunca se confundam na avaliacao.
# Trocar a instrucao da query custa 2 linhas de embedding; trocar a do
# documento custa 974 x 4.
QUERY_INSTRUCTIONS = {
    "pt": QUERY_INSTRUCTION,
    "ipc": (
        "Represente esta consulta de busca de patentes para recuperar documentos "
        "cujo dominio tecnico e classificacao internacional de patentes (CIP/IPC) "
        "correspondam ao que a consulta descreve:"
    ),
}

# Corpus do benchmark piloto: 1.000 documentos com IPC, descricoes PT e
# hierarquia, avaliado contra os gabaritos em dados/gabaritos/.
CORPUS = BASE_DIR / "corpus_piloto_ipc.tsv"

DOC_VARIANTS = {
    "tr": {
        "input": CORPUS,
        "text_column": "texto_para_embedding",
        "output_name": "gemma300_tr_docs",
    },
    "ipc_grupo": {
        "input": CORPUS,
        "text_column": "texto_para_embedding_ipc_grupo_pt",
        "output_name": "gemma300_tr_ipc_grupo_pt_docs",
    },
    "ipc_direto": {
        "input": CORPUS,
        "text_column": "texto_para_embedding_ipc_pt",
        "output_name": "gemma300_tr_ipc_direto_pt_docs",
    },
    "ipc_hierarquia": {
        "input": CORPUS,
        "text_column": "texto_para_embedding_ipc_hierarquia_pt",
        "output_name": "gemma300_tr_ipc_hierarquia_pt_docs",
    },
}

# Fase 2: as queries elaboradas por humano + LLM, uma por gabarito em
# dados/gabaritos/. Cada query nova entra em queries_fase2.tsv e o script
# retoma so os blocos que faltam.
QUERY_CONFIG = {
    "input": BASE_DIR / "queries_fase2.tsv",
    "text_column": "query_text",
    "output_name": "gemma300_queries_fase2",
}


def limpar_memoria(torch_module: Any | None = None) -> None:
    gc.collect()
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def importar_dependencias_modelo() -> tuple[Any, Any]:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Dependencias ausentes. Instale com:\n"
            "  python3 -m pip install -U sentence-transformers torch pandas numpy\n"
        ) from exc
    return torch, SentenceTransformer


def escolher_device(device: str, torch_module: Any) -> str:
    if device != "auto":
        return device
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    if text == "\\N":
        return ""
    return " ".join(text.split()).strip()


def aplicar_instrucao(text: str, instruction: str, enabled: bool) -> str:
    text = clean_text(text)
    if not enabled:
        return text
    return f"{instruction}\n\n{text}".strip()


def batches_para_tentativa(batch_size: int) -> list[int]:
    batches = [max(1, batch_size)]
    while batches[-1] > 1:
        batches.append(max(1, batches[-1] // 2))
    return batches


def erro_de_memoria(exc: Exception) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in msg or "mps backend out of memory" in msg


def carregar_dataframe(path: Path, text_column: str, limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        low_memory=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )
    if text_column not in df.columns:
        raise ValueError(f"Coluna de texto ausente: {text_column}")
    if limit is not None:
        df = df.head(limit).copy()
    return df


def caminho_bloco(output_dir: Path, indice: int) -> Path:
    return output_dir / f"embeddings_bloco_{indice:05d}.npy"


def caminho_metadata(output_dir: Path, indice: int) -> Path:
    return output_dir / f"metadata_bloco_{indice:05d}.tsv"


def bloco_existente_valido(output_dir: Path, indice: int, linhas: int, overwrite: bool) -> bool:
    if overwrite:
        return False
    path = caminho_bloco(output_dir, indice)
    metadata = caminho_metadata(output_dir, indice)
    if not path.exists() or not metadata.exists():
        return False
    try:
        array = np.load(path, mmap_mode="r")
        ok = array.shape == (linhas, DIMENSAO_ESPERADA) and np.isfinite(array[: min(10, linhas)]).all()
        shape = array.shape
        del array
    except Exception as exc:
        print(f"Bloco {indice:05d} invalido: {exc}")
        return False
    if ok:
        print(f"Bloco {indice:05d} ja existe e parece valido: {shape}. Pulando.")
    return bool(ok)


def encode_texts(
    model: Any,
    texts: list[str],
    batch_size: int,
    torch_module: Any,
) -> tuple[np.ndarray, int]:
    for tentativa in batches_para_tentativa(batch_size):
        try:
            print(f"Gerando embeddings com batch {tentativa}...")
            embeddings = model.encode(
                texts,
                batch_size=tentativa,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(embeddings, dtype=np.float32), tentativa
        except Exception as exc:
            if not erro_de_memoria(exc) or tentativa == 1:
                raise
            print(f"Memoria insuficiente com batch {tentativa}; tentando batch menor.")
            limpar_memoria(torch_module)
    raise RuntimeError("Falha inesperada ao gerar embeddings.")


def metadata_docs(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "num_pedido_normalizado",
        "num_publicacao",
        "titulo",
        "ipc",
        "ipc_com_descricao_pt",
        "ipc_hierarquia_descricao_pt",
    ]
    existing = [col for col in columns if col in df.columns]
    return df[existing].fillna("")


def metadata_queries(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["query_id", "tema", "subtema", "tipo_query", "query_text"]
    existing = [col for col in columns if col in df.columns]
    return df[existing].fillna("")


def salvar_metadata(df: pd.DataFrame, path: Path, kind: str) -> None:
    meta = metadata_queries(df) if kind == "queries" else metadata_docs(df)
    meta.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")


def salvar_config(args: argparse.Namespace, output_dir: Path, input_path: Path, text_column: str, device: str) -> None:
    config = {
        "modelo": args.model_name,
        "kind": args.kind,
        "variant": args.variant if args.kind == "docs" else None,
        "input": str(input_path),
        "text_column": text_column,
        "output_dir": str(output_dir),
        "device": device,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "max_seq_length": args.max_seq_length,
        "limit": args.limit,
        "usar_instrucao": not args.no_instruction,
        "doc_instruction": DOC_INSTRUCTION if args.kind == "docs" and not args.no_instruction else "",
        "query_instruction_nome": args.query_instruction if args.kind == "queries" and not args.no_instruction else "",
        "query_instruction": QUERY_INSTRUCTIONS[args.query_instruction] if args.kind == "queries" and not args.no_instruction else "",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera embeddings Gemma 300M para o benchmark de patentes.")
    parser.add_argument("--kind", choices=["docs", "queries"], default="docs")
    parser.add_argument("--variant", choices=sorted(DOC_VARIANTS), default="tr")
    parser.add_argument("--input", type=Path, default=None, help="TSV de entrada. Padrao depende de --kind/--variant.")
    parser.add_argument("--text-column", default=None, help="Coluna de texto. Padrao depende de --kind/--variant.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=MODEL_NAME_DEFAULT)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="Processa so as primeiras N linhas para teste.")
    parser.add_argument("--no-instruction", action="store_true", help="Nao adiciona instrucao PT antes do texto.")
    parser.add_argument("--query-instruction", choices=sorted(QUERY_INSTRUCTIONS), default="pt",
                        help="Instrucao nomeada para --kind queries. O nome vira sufixo "
                             "da colecao, exceto 'pt' (o padrao).")
    parser.add_argument("--overwrite", action="store_true", help="Regenera blocos mesmo se ja existirem.")
    args = parser.parse_args()

    if args.kind == "queries":
        defaults = dict(QUERY_CONFIG)
        instruction = QUERY_INSTRUCTIONS[args.query_instruction]
        if args.query_instruction != "pt":
            defaults["output_name"] += f"_{args.query_instruction}"
    else:
        defaults = DOC_VARIANTS[args.variant]
        instruction = DOC_INSTRUCTION

    input_path = args.input or defaults["input"]
    text_column = args.text_column or defaults["text_column"]
    output_dir = args.output_dir or (REPO_DIR / "embeddings" / defaults["output_name"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_module, SentenceTransformer = importar_dependencias_modelo()
    device = escolher_device(args.device, torch_module)

    print(f"Modelo: {args.model_name}")
    print(f"Device: {device}")
    print(f"Entrada: {input_path}")
    print(f"Coluna texto: {text_column}")
    print(f"Saida: {output_dir}")

    df = carregar_dataframe(input_path, text_column, args.limit)
    if df.empty:
        raise ValueError("Entrada vazia.")

    textos_base = [clean_text(value) for value in df[text_column].tolist()]
    vazios = sum(not text for text in textos_base)
    if vazios:
        raise ValueError(f"Ha {vazios} textos vazios na coluna {text_column}.")

    textos = [aplicar_instrucao(text, instruction, not args.no_instruction) for text in textos_base]
    total_blocos = (len(textos) + args.block_size - 1) // args.block_size
    print(f"Registros: {len(textos)} | blocos: {total_blocos} | bloco: {args.block_size}")

    model = SentenceTransformer(args.model_name, device=device)
    model.max_seq_length = args.max_seq_length

    manifest_path = output_dir / "manifest.jsonl"
    salvar_config(args, output_dir, input_path, text_column, device)

    inicio_execucao = time.time()
    blocos_gerados = 0
    for indice in range(total_blocos):
        start = indice * args.block_size
        end = min(start + args.block_size, len(textos))
        bloco_textos = textos[start:end]
        bloco_df = df.iloc[start:end].copy()

        if bloco_existente_valido(output_dir, indice, len(bloco_textos), args.overwrite):
            continue

        print(f"\nBloco {indice + 1}/{total_blocos}: linhas {start}..{end - 1}")
        array, batch_usado = encode_texts(model, bloco_textos, args.batch_size, torch_module)
        if array.shape != (len(bloco_textos), DIMENSAO_ESPERADA):
            raise ValueError(f"Shape inesperado: {array.shape}; esperado {(len(bloco_textos), DIMENSAO_ESPERADA)}")

        emb_path = caminho_bloco(output_dir, indice)
        tmp_path = emb_path.with_suffix(".tmp.npy")
        np.save(tmp_path, array)
        os.replace(tmp_path, emb_path)

        meta_path = caminho_metadata(output_dir, indice)
        salvar_metadata(bloco_df, meta_path, args.kind)

        registro = {
            "indice_bloco": indice,
            "arquivo_embeddings": str(emb_path),
            "arquivo_metadata": str(meta_path),
            "inicio": start,
            "fim": end - 1,
            "num_registros": len(bloco_textos),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "batch_size_usado": batch_usado,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with manifest_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(registro, ensure_ascii=False) + "\n")

        print(f"Salvo: {emb_path} | {meta_path}")
        blocos_gerados += 1
        del array, bloco_df, bloco_textos
        limpar_memoria(torch_module)

    minutos = (time.time() - inicio_execucao) / 60
    print(f"\nConcluido: {blocos_gerados} blocos novos em {minutos:.1f} min.")
    print("Execute novamente para retomar: blocos validos serao pulados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
