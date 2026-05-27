"""
Busca esparsa BM25 e fusão de rankings (Reciprocal Rank Fusion).

BM25 é o algoritmo clássico de recuperação por palavra-chave. Complementa a
busca densa: o embedding captura similaridade semântica, mas pode perder
correspondências exatas de termos — e em medicina, nomes precisos de doenças
e fármacos importam muito.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence

from rich.console import Console

console = Console()

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def build_bm25(corpus_texts: Sequence[str]):
    """Constrói um índice BM25 (BM25Okapi) sobre o corpus."""
    from rank_bm25 import BM25Okapi

    console.print(f"[bold]Tokenizando {len(corpus_texts)} documentos para BM25[/]")
    tokenized = [tokenize(t) for t in corpus_texts]
    console.print("[bold]Construindo índice BM25[/]")
    return BM25Okapi(tokenized)


def bm25_search(
    bm25,
    query_texts: Sequence[str],
    corpus_ids: list[str],
    top_k: int = 50,
    query_ids: list[str] | None = None,
    exclude_self: bool = True,
) -> dict[str, list[str]]:
    """Busca BM25 top-k para cada query. Returns {query_id: [doc_id]}."""
    import numpy as np

    console.print(f"[bold]Busca BM25: top-{top_k} para {len(query_texts)} queries[/]")
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    fetch = top_k + (1 if exclude_self else 0)
    rankings: dict[str, list[str]] = {}

    for qi, qtext in enumerate(query_texts):
        scores = bm25.get_scores(tokenize(qtext))
        if fetch < len(scores):
            top_idx = np.argpartition(-scores, fetch)[:fetch]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        else:
            top_idx = np.argsort(-scores)

        qid = query_ids[qi] if query_ids is not None else str(qi)
        self_idx = id_to_idx.get(qid, -1) if exclude_self else -1

        ranked = []
        for idx in top_idx:
            if idx == self_idx:
                continue
            ranked.append(corpus_ids[idx])
            if len(ranked) >= top_k:
                break
        rankings[qid] = ranked

    return rankings


def reciprocal_rank_fusion(
    rankings_list: list[dict[str, list[str]]],
    k: int = 60,
    top_k: int = 50,
) -> dict[str, list[str]]:
    """Funde múltiplos rankings via RRF. Para cada doc, soma 1/(k+rank)
    sobre todos os rankings. k=60 é o padrão da literatura."""
    all_queries: set[str] = set()
    for r in rankings_list:
        all_queries.update(r.keys())

    fused: dict[str, list[str]] = {}
    for qid in all_queries:
        scores: dict[str, float] = defaultdict(float)
        for ranking in rankings_list:
            for rank, doc_id in enumerate(ranking.get(qid, []), start=1):
                scores[doc_id] += 1.0 / (k + rank)
        ordered = sorted(scores.items(), key=lambda x: -x[1])
        fused[qid] = [doc for doc, _ in ordered[:top_k]]

    return fused
