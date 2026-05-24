"""
Busca densa por similaridade de cosseno.

Para a escala deste projeto (~18k documentos), uma busca exata por produto
interno em numpy é rápida o suficiente e evita a complexidade de um vector
store externo. Se o corpus crescer muito, migra-se para FAISS/Qdrant.

Como os embeddings são normalizados, o produto interno = similaridade de
cosseno.
"""
from __future__ import annotations

import numpy as np
from rich.console import Console

console = Console()


def search(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    corpus_ids: list[str],
    top_k: int = 50,
    query_ids: list[str] | None = None,
    exclude_self: bool = True,
) -> dict[str, list[str]]:
    """Busca os top_k documentos mais similares para cada query.

    Args:
        query_emb: (n_queries, dim) embeddings das queries
        corpus_emb: (n_docs, dim) embeddings do corpus
        corpus_ids: ids dos documentos (mesma ordem de corpus_emb)
        top_k: quantos resultados retornar por query
        query_ids: ids das queries (necessário se exclude_self=True)
        exclude_self: remove o próprio documento do ranking (auto-match)

    Returns:
        {query_id: [doc_id ordenados por similaridade]}
        Se query_ids for None, usa o índice posicional como chave.
    """
    console.print(f"[bold]Buscando top-{top_k} para {len(query_emb)} queries "
                  f"em {len(corpus_emb)} documentos[/]")

    # Similaridade = produto interno (embeddings normalizados → cosseno)
    # (n_queries, n_docs)
    sims = query_emb @ corpus_emb.T

    # Mapa id->índice para exclusão de auto-match
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}

    rankings: dict[str, list[str]] = {}
    # Pega um pouco mais que top_k para compensar a remoção do self
    fetch = top_k + (1 if exclude_self else 0)

    for qi in range(len(query_emb)):
        scores = sims[qi]
        # índices dos maiores scores (sem ordenar tudo: argpartition)
        if fetch < len(scores):
            top_idx = np.argpartition(-scores, fetch)[:fetch]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        else:
            top_idx = np.argsort(-scores)

        qid = query_ids[qi] if query_ids is not None else str(qi)
        self_idx = id_to_idx.get(qid, -1) if exclude_self else -1

        ranked_ids = []
        for idx in top_idx:
            if idx == self_idx:
                continue
            ranked_ids.append(corpus_ids[idx])
            if len(ranked_ids) >= top_k:
                break
        rankings[qid] = ranked_ids

    return rankings
