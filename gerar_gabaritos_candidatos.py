#!/usr/bin/env python3
"""Gera qrels candidatos para o benchmark de busca semantica de patentes.

Este gabarito e intencionalmente "fraco": combina regras de IPC com termos
tecnicos em titulo/resumo. Serve para triagem e revisao humana, nao como
verdade-terreno final.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


PATENTES_PATH = Path("patentes_benchmark_amostra_1000.tsv")
QUERIES_PATH = Path("queries_benchmark_patentes.tsv")
QRELS_PATH = Path("qrels_candidatos_queries_benchmark.tsv")
POOL_PATH = Path("gabaritos_candidatos_revisao.tsv")


@dataclass(frozen=True)
class Rule:
    high_ipc: tuple[str, ...]
    medium_ipc: tuple[str, ...]
    high_terms: tuple[str, ...]
    medium_terms: tuple[str, ...]


RULES: dict[str, Rule] = {
    "conjugados_anticorpo_farmaco_adc": Rule(
        high_ipc=("A61P 35", "A61K 39", "A61K 47", "C07K 16"),
        medium_ipc=("A61K", "A61P", "C07K"),
        high_terms=(
            "conjugado anticorpo farmaco",
            "anticorpo farmaco",
            "adc",
            "citotox",
            "ceacam5",
        ),
        medium_terms=("anticorpo", "antitumor", "tumor", "cancer", "neoplas"),
    ),
    "inibidores_pequenas_moleculas": Rule(
        high_ipc=("A61K 31", "A61P 35", "C07D"),
        medium_ipc=("A61K", "A61P", "C07", "C07D"),
        high_terms=("inibidor", "quinase", "jak", "btk", "mcl", "heterocic"),
        medium_terms=("composto", "molecula", "cancer", "tumor", "neoplas"),
    ),
    "imunoterapia_e_vacinas": Rule(
        high_ipc=("A61K 39", "A61P 35", "C12N", "C07K"),
        medium_ipc=("A61K", "A61P", "C12", "C07K"),
        high_terms=("vacina", "linfocito", "imune", "imunoterapia", "antigeno", "ctl"),
        medium_terms=("anticorpo", "tumor", "cancer", "neoplas", "resposta imune"),
    ),
    "formulacoes_e_entrega": Rule(
        high_ipc=("A61K 9", "A61K 47", "A61P 35"),
        medium_ipc=("A61K", "A61P"),
        high_terms=("formulacao", "nanopart", "lipossoma", "liberacao", "entrega", "administracao"),
        medium_terms=("farmaceutica", "farmaco", "quimioterap", "cancer", "tumor"),
    ),
    "combinacoes_terapeuticas": Rule(
        high_ipc=("A61K 45", "A61K 31", "A61K 39", "A61P 35"),
        medium_ipc=("A61K", "A61P"),
        high_terms=("combinacao", "combinacoes", "coadministracao", "sinerg", "associacao"),
        medium_terms=("antitumor", "cancer", "neoplas", "tratamento", "terapeutic"),
    ),
    "membranas_filtracao": Rule(
        high_ipc=("C02F", "B01D 61"),
        medium_ipc=("C02F", "B01D", "E03B"),
        high_terms=("membrana", "ultrafiltracao", "nanofiltracao", "osmose reversa", "filtro"),
        medium_terms=("agua", "purificacao", "filtracao", "efluente", "contamin"),
    ),
    "adsorcao_metais_pesados": Rule(
        high_ipc=("C02F 1/28", "C02F 1/62", "B01J"),
        medium_ipc=("C02F", "B01D", "B01J"),
        high_terms=("adsorvente", "adsorcao", "arsenio", "chumbo", "mercurio", "metal pesado"),
        medium_terms=("agua", "efluente", "contamin", "ion", "remocao"),
    ),
    "tratamento_biologico_efluentes": Rule(
        high_ipc=("C02F 3", "C02F 9", "C12N"),
        medium_ipc=("C02F", "C12"),
        high_terms=("biologico", "biorreator", "biofilme", "microorganismo", "lodo"),
        medium_terms=("efluente", "esgoto", "agua residual", "degradacao", "organica"),
    ),
    "oxidacao_desinfeccao": Rule(
        high_ipc=("C02F 1/32", "C02F 1/72", "C02F 1/78"),
        medium_ipc=("C02F", "B01J"),
        high_terms=("ozonio", "ultravioleta", "oxidacao", "fotocatalise", "desinfeccao", "peroxido"),
        medium_terms=("agua", "poluente", "microorganismo", "organico", "degradacao"),
    ),
    "dessalinizacao_e_reuso": Rule(
        high_ipc=("C02F 1/44", "C02F 1/469", "C02F 103", "B01D 61"),
        medium_ipc=("C02F", "B01D"),
        high_terms=("dessalinizacao", "salobra", "agua do mar", "reuso", "reutilizavel", "sais"),
        medium_terms=("agua", "efluente", "osmose", "purificacao", "tratamento"),
    ),
    "alocacao_recursos_radio": Rule(
        high_ipc=("H04W 72", "H04L 5"),
        medium_ipc=("H04W", "H04L", "H04B"),
        high_terms=("alocacao", "recurso", "uplink", "downlink", "escalonamento", "pdcch", "pusch"),
        medium_terms=("5g", "nr", "lte", "estacao base", "usuario", "sem fio"),
    ),
    "beamforming_mimo": Rule(
        high_ipc=("H04B 7", "H04W", "H01Q"),
        medium_ipc=("H04B", "H04W", "H01Q"),
        high_terms=("beamforming", "feixe", "mimo", "precod", "csi", "antena"),
        medium_terms=("5g", "nr", "sem fio", "canal", "cobertura", "capacidade"),
    ),
    "handover_mobilidade": Rule(
        high_ipc=("H04W 36", "H04W 48"),
        medium_ipc=("H04W",),
        high_terms=("handover", "mobilidade", "celula alvo", "celula servidora", "reselection"),
        medium_terms=("usuario", "ue", "estacao base", "rede movel", "sem fio", "conexao"),
    ),
    "latencia_urllc": Rule(
        high_ipc=("H04W 72", "H04L 1", "H04W 28"),
        medium_ipc=("H04W", "H04L"),
        high_terms=("urllc", "baixa latencia", "confiabilidade", "harq", "ack", "repeticao", "grant free"),
        medium_terms=("5g", "nr", "tempo real", "critica", "sem fio", "uplink"),
    ),
    "canais_sinalizacao": Rule(
        high_ipc=("H04L 5", "H04W 72", "H04W 76"),
        medium_ipc=("H04L", "H04W"),
        high_terms=("pdcch", "pucch", "dci", "harq", "ack", "sinalizacao", "canal de controle", "scell"),
        medium_terms=("5g", "nr", "lte", "transmissao", "recepcao", "sem fio"),
    ),
}


THEME_IPC = {
    "tratamentos_cancer_farmacos": ("A61K", "A61P", "C07D", "C07K", "C12N", "C12Q"),
    "purificacao_agua": ("C02F", "B01D", "E03B", "C01B", "C25B"),
    "comunicacao_5g": ("H04W", "H04L", "H04B", "H04N", "H01Q"),
}


def norm(value: str | None) -> str:
    value = "" if value in (None, "\\N") else value
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.lower()
    value = re.sub(r"[^a-z0-9/]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def ipc_tokens(ipc: str) -> list[str]:
    if not ipc or ipc == "\\N":
        return []
    return [part.strip() for part in ipc.split(";") if part.strip()]


def ipc_matches(ipcs: list[str], patterns: tuple[str, ...]) -> int:
    total = 0
    for pattern in patterns:
        pattern_n = norm(pattern).upper()
        for code in ipcs:
            code_n = norm(code).upper()
            if code_n.startswith(pattern_n):
                total += 1
                break
    return total


def term_hits(text: str, terms: tuple[str, ...]) -> list[str]:
    hits = []
    for term in terms:
        term_n = norm(term)
        if term_n and term_n in text:
            hits.append(term)
    return hits


def score_query_doc(query: dict[str, str], doc: dict[str, str]) -> tuple[int, float, str]:
    rule = RULES[query["subtema"]]
    text = norm(" ".join((doc.get(key) or "") for key in ("titulo", "resumo", "ipc")))
    ipcs = ipc_tokens(doc.get("ipc", ""))

    high_ipc = ipc_matches(ipcs, rule.high_ipc)
    med_ipc = ipc_matches(ipcs, rule.medium_ipc)
    theme_ipc = ipc_matches(ipcs, THEME_IPC[query["tema"]])
    high_terms = term_hits(text, rule.high_terms)
    med_terms = term_hits(text, rule.medium_terms)

    numeric_score = high_ipc * 4 + med_ipc * 1.5 + theme_ipc + len(high_terms) * 3 + len(med_terms)

    if high_ipc and high_terms:
        relevance = 2
    elif high_ipc >= 2 or len(high_terms) >= 2:
        relevance = 2
    elif theme_ipc and (med_ipc or high_terms or med_terms):
        relevance = 1
    elif med_ipc and med_terms:
        relevance = 1
    else:
        relevance = 0

    evidence = []
    if high_ipc:
        evidence.append(f"ipc_forte={high_ipc}")
    if med_ipc:
        evidence.append(f"ipc_media={med_ipc}")
    if theme_ipc:
        evidence.append(f"ipc_tema={theme_ipc}")
    if high_terms:
        evidence.append("termos_fortes=" + "|".join(high_terms[:8]))
    if med_terms:
        evidence.append("termos_medios=" + "|".join(med_terms[:8]))
    return relevance, numeric_score, "; ".join(evidence)


def main() -> int:
    with QUERIES_PATH.open(encoding="utf-8", newline="") as fp:
        queries = list(csv.DictReader(fp, delimiter="\t"))
    with PATENTES_PATH.open(encoding="utf-8", newline="") as fp:
        docs = list(csv.DictReader(fp, delimiter="\t"))

    qrels_rows = []
    review_rows = []

    for query in queries:
        scored = []
        for doc in docs:
            relevance, score, evidence = score_query_doc(query, doc)
            qrels_rows.append(
                {
                    "query_id": query["query_id"],
                    "num_pedido_normalizado": doc["num_pedido_normalizado"],
                    "relevance": str(relevance),
                }
            )
            scored.append((relevance, score, evidence, doc))

        for relevance in (2, 1):
            selected = sorted(
                (item for item in scored if item[0] == relevance),
                key=lambda item: (-item[1], item[3]["ordem_arquivo"]),
            )[:30]
            for rel, score, evidence, doc in selected:
                review_rows.append(review_row(query, doc, rel, score, evidence))

        hard_lows = sorted(
            (item for item in scored if item[0] == 0 and item[1] > 0),
            key=lambda item: (-item[1], item[3]["ordem_arquivo"]),
        )[:20]
        easy_lows = sorted(
            (item for item in scored if item[0] == 0 and item[1] == 0),
            key=lambda item: item[3]["ordem_arquivo"],
        )[:10]
        for rel, score, evidence, doc in hard_lows + easy_lows:
            review_rows.append(review_row(query, doc, rel, score, evidence))

    with QRELS_PATH.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=("query_id", "num_pedido_normalizado", "relevance"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(qrels_rows)

    review_fields = (
        "query_id",
        "tema",
        "subtema",
        "query_text",
        "num_pedido_normalizado",
        "num_publicacao",
        "relevance_candidato",
        "score_regra",
        "evidencia_regra",
        "ipc",
        "titulo",
    )
    with POOL_PATH.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=review_fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"queries: {len(queries)}")
    print(f"documentos: {len(docs)}")
    print(f"qrels: {len(qrels_rows)} -> {QRELS_PATH}")
    print(f"pool_revisao: {len(review_rows)} -> {POOL_PATH}")
    return 0


def review_row(
    query: dict[str, str],
    doc: dict[str, str],
    relevance: int,
    score: float,
    evidence: str,
) -> dict[str, str]:
    return {
        "query_id": query["query_id"],
        "tema": query["tema"],
        "subtema": query["subtema"],
        "query_text": query["query_text"],
        "num_pedido_normalizado": doc["num_pedido_normalizado"],
        "num_publicacao": doc.get("num_publicacao", ""),
        "relevance_candidato": str(relevance),
        "score_regra": f"{score:.1f}",
        "evidencia_regra": evidence,
        "ipc": doc.get("ipc", ""),
        "titulo": doc.get("titulo", ""),
    }


if __name__ == "__main__":
    raise SystemExit(main())
