#!/usr/bin/env python3
"""Gera embeddings densos do BGE-M3 no formato de colecao do projeto.

O BGE-M3 nao usa instrucao: a documentacao do modelo diz que, ao contrario
dos BGE anteriores, ele "nao requer mais adicionar instrucoes as queries".
Entao aqui nao ha prefixo nenhum, nos dois lados. Contexto de 8192 tokens
contra 2048 do Gemma, o que importa nas variantes com IPC.

    python3 src/gerar_embeddings_bgem3.py --kind queries
    python3 src/gerar_embeddings_bgem3.py --kind docs --variant tr
    python3 src/gerar_embeddings_bgem3.py --kind docs --variant ipc_grupo

Usa FlagEmbedding se estiver instalado, senao sentence-transformers.
Por padrao so o vetor denso. Com --sparse grava tambem os pesos lexicais
(a cabeca esparsa do M3, um BM25 aprendido) em sparse_bloco_00000.npz, no
formato CSR: indptr, ids de token e pesos. A avaliacao le esse npz com numpy
puro, sem precisar do modelo. O ColBERT fica para depois.
"""
from __future__ import annotations
import argparse, csv, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
EMB = RAIZ / "embeddings"
MODELO = "BAAI/bge-m3"
CORPUS = DADOS / "corpus_piloto_ipc.tsv"

VARIANTES = {
    "tr":        ("texto_para_embedding",              "bgem3_tr_docs"),
    "ipc_grupo": ("texto_para_embedding_ipc_grupo_pt", "bgem3_ipc_grupo_docs"),
}
QUERIES = (DADOS / "queries_fase2.tsv", "query_text", "bgem3_queries_fase2")


def carregar_modelo(device: str):
    """Devolve (encode_fn, backend). encode_fn(textos) -> np.ndarray."""
    try:
        from FlagEmbedding import BGEM3FlagModel
        try:
            m = BGEM3FlagModel(MODELO, use_fp16=(device == "cuda"), devices=device)
        except (TypeError, ValueError, RuntimeError) as exc:
            # Versoes antigas nao aceitam devices=, e nem toda build lida com mps.
            print(f"BGEM3FlagModel com devices={device!r} falhou ({exc}); "
                  f"tentando sem especificar device.")
            m = BGEM3FlagModel(MODELO, use_fp16=False)

        def encode(textos, batch_size, max_length, sparse=False):
            saida = m.encode(textos, batch_size=batch_size, max_length=max_length,
                             return_dense=True, return_sparse=sparse,
                             return_colbert_vecs=False)
            denso = np.asarray(saida["dense_vecs"], dtype=np.float32)
            return (denso, saida["lexical_weights"]) if sparse else (denso, None)
        return encode, "FlagEmbedding"
    except ImportError:
        pass
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(MODELO, device=None if device == "auto" else device)

        def encode(textos, batch_size, max_length, sparse=False):
            if sparse:
                raise SystemExit("A cabeca esparsa do M3 so existe no FlagEmbedding. "
                                 "Instale-o (pip install FlagEmbedding) para usar --sparse.")
            m.max_seq_length = max_length
            denso = np.asarray(m.encode(textos, batch_size=batch_size,
                                        show_progress_bar=True, convert_to_numpy=True,
                                        normalize_embeddings=True), dtype=np.float32)
            return denso, None
        return encode, "sentence-transformers"
    except ImportError:
        raise SystemExit("Instale FlagEmbedding ou sentence-transformers para usar o BGE-M3.")


def escolher_device(pedido: str) -> str:
    if pedido != "auto":
        return pedido
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["docs", "queries"], default="docs")
    p.add_argument("--variant", choices=sorted(VARIANTES), default="tr")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sparse", action="store_true",
                   help="Grava tambem os pesos lexicais (cabeca esparsa). "
                        "Requer FlagEmbedding.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.kind == "queries":
        entrada, coluna, nome = QUERIES
        chave = "query_id"
    else:
        coluna, nome = VARIANTES[args.variant]
        entrada, chave = CORPUS, "num_pedido_normalizado"
    saida = EMB / nome
    npy, tsv = saida / "embeddings_bloco_00000.npy", saida / "metadata_bloco_00000.tsv"

    df = pd.read_csv(entrada, sep="\t", dtype=str, low_memory=False,
                     quoting=csv.QUOTE_NONE, escapechar="\\").fillna("")
    if args.limit:
        df = df.head(args.limit).copy()
    if coluna not in df.columns:
        raise SystemExit(f"Coluna ausente em {entrada.name}: {coluna}")

    if npy.exists() and not args.overwrite:
        atual = np.load(npy, mmap_mode="r")
        if atual.shape[0] == len(df):
            print(f"{nome}: ja existe com {atual.shape}. Use --overwrite para regerar.")
            return 0
        print(f"{nome}: {atual.shape[0]} vetores para {len(df)} linhas. Regerando.")

    device = escolher_device(args.device)
    encode, backend = carregar_modelo(device)
    saida.mkdir(parents=True, exist_ok=True)
    print(f"Modelo: {MODELO} ({backend}) | device: {device}")
    print(f"Entrada: {entrada.name} | coluna: {coluna} | linhas: {len(df)}")
    print(f"Saida: {saida}")

    t0 = time.time()
    vetores, lexicais = encode(df[coluna].astype(str).tolist(), args.batch_size,
                               args.max_length, sparse=args.sparse)
    normas = np.linalg.norm(vetores, axis=1, keepdims=True)
    normas[normas == 0] = 1.0
    vetores = (vetores / normas).astype(np.float32)
    np.save(npy, vetores)

    if lexicais is not None:
        indptr, ids_tok, pesos = [0], [], []
        for d in lexicais:
            for tok, peso in d.items():
                ids_tok.append(int(tok))
                pesos.append(float(peso))
            indptr.append(len(ids_tok))
        np.savez(saida / "sparse_bloco_00000.npz",
                 indptr=np.asarray(indptr, dtype=np.int64),
                 ids=np.asarray(ids_tok, dtype=np.int32),
                 pesos=np.asarray(pesos, dtype=np.float32))
        print(f"pesos lexicais: {len(ids_tok)} pares em {len(lexicais)} textos "
              f"(media {len(ids_tok) / max(len(lexicais), 1):.0f} tokens/texto)")

    cols = [c for c in [chave, "num_publicacao", "titulo", "ipc", "tema",
                        "tipo_query", "query_text"] if c in df.columns]
    df[cols].to_csv(tsv, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")

    (saida / "config.json").write_text(json.dumps({
        "modelo": MODELO, "backend": backend, "kind": args.kind,
        "variant": args.variant if args.kind == "docs" else None,
        "input": str(entrada), "text_column": coluna, "output_dir": str(saida),
        "device": device, "batch_size": args.batch_size,
        "max_seq_length": args.max_length,
        "sparse": bool(lexicais is not None),
        "usar_instrucao": False, "doc_instruction": "", "query_instruction": "",
        "dimensao": int(vetores.shape[1]),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version, "platform": platform.platform(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (saida / "manifest.jsonl").write_text(json.dumps({
        "indice_bloco": 0, "arquivo_embeddings": str(npy), "arquivo_metadata": str(tsv),
        "inicio": 0, "fim": len(df) - 1, "num_registros": len(df),
        "shape": list(vetores.shape), "dtype": "float32",
        "batch_size_usado": args.batch_size,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Concluido: {vetores.shape} em {(time.time() - t0) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
