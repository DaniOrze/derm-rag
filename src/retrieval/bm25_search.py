"""
Busca esparsa BM25 (via bm25s, vetorizado) e fusão de rankings (RRF).

Usa a biblioteca bm25s em vez do rank_bm25: o bm25s é vetorizado (numpy/scipy
esparso) e roda a busca ~100x mais rápido. A busca de 1000 queries que levava
~20 min com rank_bm25 passa a levar segundos.

A interface das funções é a mesma da versão anterior, então o restante do
pipeline (run_hybrid.py) não muda.
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
    """Constrói um índice BM25 vetorizado (bm25s) sobre o corpus.

    Retorna um dict com o retriever e os tokens do corpus, para ser usado
    por bm25_search. Mantém compatibilidade com a assinatura anterior.
    """
    import bm25s

    console.print(f"[bold]Tokenizando {len(corpus_texts)} documentos para BM25[/]")
    corpus_tokens = [tokenize(t) for t in corpus_texts]

    console.print("[bold]Construindo índice BM25 (bm25s)[/]")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    return {"retriever": retriever}


def bm25_search(
    bm25,
    query_texts: Sequence[str],
    corpus_ids: list[str],
    top_k: int = 50,
    query_ids: list[str] | None = None,
    exclude_self: bool = True,
) -> dict[str, list[str]]:
    """Busca BM25 top-k para cada query. Returns {query_id: [doc_id]}.

    bm25 é o dict retornado por build_bm25.
    """
    import numpy as np

    retriever = bm25["retriever"]
    console.print(f"[bold]Busca BM25: top-{top_k} para {len(query_texts)} queries[/]")

    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    fetch = top_k + (1 if exclude_self else 0)
    fetch = min(fetch, len(corpus_ids))

    # Tokeniza todas as queries e busca de uma vez (vetorizado)
    query_tokens = [tokenize(q) for q in query_texts]
    # results: matriz (n_queries, fetch) de índices de documentos
    results, _scores = retriever.retrieve(query_tokens, k=fetch, show_progress=False)

    rankings: dict[str, list[str]] = {}
    for qi in range(len(query_texts)):
        qid = query_ids[qi] if query_ids is not None else str(qi)
        self_idx = id_to_idx.get(qid, -1) if exclude_self else -1

        ranked = []
        for doc_idx in results[qi]:
            doc_idx = int(doc_idx)
            if doc_idx == self_idx:
                continue
            ranked.append(corpus_ids[doc_idx])
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
