"""
Métricas de avaliação de retrieval.

Implementa as métricas padrão para avaliar a qualidade da recuperação:
    - Recall@k: fração dos documentos relevantes que aparecem no top-k
    - MRR: Mean Reciprocal Rank (posição do primeiro relevante)
    - nDCG@k: Normalized Discounted Cumulative Gain (aproveita scores graduados)

Convenções:
    - Para cada query, temos um conjunto de documentos relevantes (qrels),
      no formato {doc_id: relevance_score}, onde score é 1 ou 2.
    - O retrieval devolve uma lista ordenada de doc_ids (ranking).
    - Para Recall e MRR, "relevante" = qualquer doc com score >= 1.
    - Para nDCG, o score (1 ou 2) entra como ganho graduado, aproveitando
      a distinção de níveis de similaridade do PMC-Patients.

Estas implementações são auto-contidas (sem dependências externas) para
facilitar testes e reprodutibilidade.
"""
from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(ranking: Sequence[str], qrels: dict[str, int], k: int) -> float:
    """Fração dos documentos relevantes recuperados no top-k.

    Args:
        ranking: lista de doc_ids ordenada por relevância (melhor primeiro)
        qrels: dict {doc_id: score} dos documentos relevantes para a query
        k: corte

    Returns:
        Recall@k em [0, 1]. Se não há relevantes, retorna 0.0.
    """
    relevant = set(qrels.keys())
    if not relevant:
        return 0.0
    top_k = set(ranking[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def reciprocal_rank(ranking: Sequence[str], qrels: dict[str, int]) -> float:
    """Recíproco da posição do primeiro documento relevante.

    Returns:
        1/rank do primeiro acerto (rank começa em 1), ou 0.0 se nenhum
        relevante aparece no ranking.
    """
    relevant = set(qrels.keys())
    for i, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranking: Sequence[str], qrels: dict[str, int], k: int) -> float:
    """Discounted Cumulative Gain no top-k, usando scores graduados."""
    dcg = 0.0
    for i, doc_id in enumerate(ranking[:k], start=1):
        gain = qrels.get(doc_id, 0)
        # desconto logarítmico padrão
        dcg += gain / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranking: Sequence[str], qrels: dict[str, int], k: int) -> float:
    """nDCG@k: DCG normalizado pelo DCG ideal (ranking perfeito).

    Aproveita os scores 1 e 2 como ganhos graduados.
    Returns:
        nDCG@k em [0, 1]. Se não há relevantes, retorna 0.0.
    """
    if not qrels:
        return 0.0
    actual = dcg_at_k(ranking, qrels, k)
    # ranking ideal: documentos ordenados por score decrescente
    ideal_ranking = [doc for doc, _ in sorted(qrels.items(), key=lambda x: -x[1])]
    ideal = dcg_at_k(ideal_ranking, qrels, k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def evaluate_rankings(
    rankings: dict[str, Sequence[str]],
    qrels: dict[str, dict[str, int]],
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Avalia um conjunto de queries e devolve métricas médias.

    Args:
        rankings: {query_id: [doc_id ordenados]}
        qrels: {query_id: {doc_id: score}}
        k_values: cortes para Recall e nDCG

    Returns:
        dict com métricas médias sobre todas as queries, ex.:
        {'recall@1': .., 'recall@5': .., 'mrr': .., 'ndcg@10': ..}
    """
    # Considera só queries que têm ground truth
    query_ids = [q for q in rankings if qrels.get(q)]
    n = len(query_ids)
    if n == 0:
        return {}

    results: dict[str, float] = {}

    for k in k_values:
        results[f"recall@{k}"] = sum(
            recall_at_k(rankings[q], qrels[q], k) for q in query_ids
        ) / n
        results[f"ndcg@{k}"] = sum(
            ndcg_at_k(rankings[q], qrels[q], k) for q in query_ids
        ) / n

    results["mrr"] = sum(
        reciprocal_rank(rankings[q], qrels[q]) for q in query_ids
    ) / n

    results["n_queries"] = n
    return results
