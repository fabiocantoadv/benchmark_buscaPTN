#!/usr/bin/env python3
"""Enriquece a amostra do benchmark com titulos IPC em portugues.

Usa apenas o JSON PT flat da IPC. Nao faz fallback para SQLite/ingles nem para
simbolos pais, para manter o experimento simples e auditavel.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "patentes_benchmark_amostra_1000.tsv"
DEFAULT_OUTPUT = BASE_DIR / "patentes_benchmark_amostra_1000_ipc_pt.tsv"
DEFAULT_MISSING = BASE_DIR / "ipc_simbolos_sem_descricao_json_pt.tsv"
DEFAULT_SUMMARY = BASE_DIR / "patentes_benchmark_amostra_1000_ipc_pt_resumo.tsv"
DEFAULT_IPC_JSON = Path("/Users/fabiocanto/Documents/ipc_net_beta/dist/ipc/ipc_titles_pt_flat_20260101.json")

csv.field_size_limit(sys.maxsize)


def clean(value: str | None) -> str:
    if value in (None, "\\N"):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def ipc_key(symbol: str) -> str:
    return clean(symbol).upper().replace(" ", "")


def ipc_display(symbol_key: str) -> str:
    if re.fullmatch(r"[A-H]", symbol_key):
        return symbol_key
    if re.fullmatch(r"[A-H]\d{2}", symbol_key):
        return symbol_key
    if re.fullmatch(r"[A-H]\d{2}[A-Z]", symbol_key):
        return symbol_key

    match = re.fullmatch(r"([A-H]\d{2}[A-Z])(\d+/\d+)", symbol_key)
    if match:
        return f"{match.group(1)} {match.group(2)}"

    return symbol_key


def split_ipc(value: str | None) -> list[str]:
    text = clean(value)
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def ipc_hierarchy_keys(symbol: str) -> list[str]:
    key = ipc_key(symbol)
    match = re.fullmatch(r"([A-H])(\d{2})([A-Z])(?:(\d+)/(\d+))?", key)
    if not match:
        return [key] if key else []

    section, klass, subclass, main_group, subgroup = match.groups()
    keys = [
        section,
        f"{section}{klass}",
        f"{section}{klass}{subclass}",
    ]
    if main_group and subgroup:
        keys.append(f"{section}{klass}{subclass}{main_group}/00")
        keys.append(key)

    deduped = []
    for item in keys:
        if item not in deduped:
            deduped.append(item)
    return deduped


def ipc_with_descriptions(symbols: list[str], titles: dict[str, str]) -> tuple[str, str, str]:
    found = []
    missing = []

    for symbol in symbols:
        title = clean(titles.get(ipc_key(symbol)))
        if title:
            found.append(f"{symbol} - {title}")
        else:
            missing.append(symbol)

    return "; ".join(found), "; ".join(missing), str(len(found))


def ipc_hierarchy_with_descriptions(
    symbols: list[str],
    titles: dict[str, str],
) -> tuple[str, str, str]:
    described_groups = []
    missing = []
    found_count = 0

    for symbol in symbols:
        parts = []
        for key in ipc_hierarchy_keys(symbol):
            title = clean(titles.get(key))
            display = ipc_display(key)
            if title:
                parts.append(f"{display} - {title}")
                found_count += 1
            else:
                missing.append(display)

        if parts:
            described_groups.append(f"{symbol}: " + " > ".join(parts))

    return "; ".join(described_groups), "; ".join(dict.fromkeys(missing)), str(found_count)


def main() -> int:
    if not DEFAULT_INPUT.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {DEFAULT_INPUT}")
    if not DEFAULT_IPC_JSON.exists():
        raise FileNotFoundError(f"JSON PT de IPC nao encontrado: {DEFAULT_IPC_JSON}")

    with DEFAULT_IPC_JSON.open(encoding="utf-8") as fp:
        titles = json.load(fp)

    total_docs = 0
    docs_with_any_description = 0
    docs_with_all_descriptions = 0
    total_symbols = 0
    total_symbols_found = 0
    total_hierarchy_items = 0
    total_hierarchy_items_found = 0
    missing_symbols: dict[str, int] = {}
    missing_hierarchy_symbols: dict[str, int] = {}

    with DEFAULT_INPUT.open(encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Arquivo de entrada sem cabecalho.")

        fieldnames = list(reader.fieldnames)
        for column in (
            "ipc_com_descricao_pt",
            "ipc_sem_descricao_pt",
            "ipc_descricoes_encontradas_qtd",
            "texto_para_embedding_ipc_pt",
            "ipc_hierarquia_descricao_pt",
            "ipc_hierarquia_sem_descricao_pt",
            "ipc_hierarquia_descricoes_encontradas_qtd",
            "texto_para_embedding_ipc_hierarquia_pt",
        ):
            if column not in fieldnames:
                fieldnames.append(column)

        with DEFAULT_OUTPUT.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(
                dst,
                fieldnames=fieldnames,
                delimiter="\t",
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
                quotechar=None,
                escapechar="\\",
                extrasaction="ignore",
            )
            writer.writeheader()

            for row in reader:
                total_docs += 1
                symbols = split_ipc(row.get("ipc"))
                described, missing, found_count_text = ipc_with_descriptions(symbols, titles)
                hierarchy_described, hierarchy_missing, hierarchy_found_count_text = ipc_hierarchy_with_descriptions(
                    symbols,
                    titles,
                )
                found_count = int(found_count_text)
                hierarchy_found_count = int(hierarchy_found_count_text)

                total_symbols += len(symbols)
                total_symbols_found += found_count
                hierarchy_items_count = sum(len(ipc_hierarchy_keys(symbol)) for symbol in symbols)
                total_hierarchy_items += hierarchy_items_count
                total_hierarchy_items_found += hierarchy_found_count
                if found_count:
                    docs_with_any_description += 1
                if symbols and found_count == len(symbols):
                    docs_with_all_descriptions += 1

                for symbol in split_ipc(missing):
                    missing_symbols[symbol] = missing_symbols.get(symbol, 0) + 1
                for symbol in split_ipc(hierarchy_missing):
                    missing_hierarchy_symbols[symbol] = missing_hierarchy_symbols.get(symbol, 0) + 1

                row["ipc_com_descricao_pt"] = described
                row["ipc_sem_descricao_pt"] = missing
                row["ipc_descricoes_encontradas_qtd"] = found_count_text
                row["ipc_hierarquia_descricao_pt"] = hierarchy_described
                row["ipc_hierarquia_sem_descricao_pt"] = hierarchy_missing
                row["ipc_hierarquia_descricoes_encontradas_qtd"] = hierarchy_found_count_text

                base_text = clean(row.get("texto_para_embedding"))
                ipc_text = f" Classificacao IPC: {described}" if described else ""
                ipc_hierarchy_text = (
                    f" Classificacao IPC hierarquica: {hierarchy_described}" if hierarchy_described else ""
                )
                row["texto_para_embedding_ipc_pt"] = f"{base_text}{ipc_text}".strip()
                row["texto_para_embedding_ipc_hierarquia_pt"] = f"{base_text}{ipc_hierarchy_text}".strip()

                for key, value in list(row.items()):
                    row[key] = clean(value)

                writer.writerow(row)

    with DEFAULT_MISSING.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t", lineterminator="\n")
        writer.writerow(["ipc", "ocorrencias_na_amostra"])
        for symbol, count in sorted(missing_symbols.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([symbol, count])

    summary_rows = [
        ("documentos", total_docs),
        ("documentos_com_alguma_descricao_ipc_pt", docs_with_any_description),
        ("documentos_com_todas_descricoes_ipc_pt", docs_with_all_descriptions),
        ("simbolos_ipc_total_ocorrencias", total_symbols),
        ("simbolos_ipc_com_descricao_pt_ocorrencias", total_symbols_found),
        ("simbolos_ipc_sem_descricao_pt_ocorrencias", total_symbols - total_symbols_found),
        ("simbolos_ipc_sem_descricao_pt_distintos", len(missing_symbols)),
        ("itens_hierarquia_ipc_total_ocorrencias", total_hierarchy_items),
        ("itens_hierarquia_ipc_com_descricao_pt_ocorrencias", total_hierarchy_items_found),
        ("itens_hierarquia_ipc_sem_descricao_pt_ocorrencias", total_hierarchy_items - total_hierarchy_items_found),
        ("itens_hierarquia_ipc_sem_descricao_pt_distintos", len(missing_hierarchy_symbols)),
    ]
    with DEFAULT_SUMMARY.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t", lineterminator="\n")
        writer.writerow(["metrica", "valor"])
        writer.writerows(summary_rows)

    print(f"saida: {DEFAULT_OUTPUT}")
    print(f"faltantes: {DEFAULT_MISSING}")
    print(f"resumo: {DEFAULT_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
