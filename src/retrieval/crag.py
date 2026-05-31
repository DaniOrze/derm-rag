"""
CRAG — Corrective Retrieval Augmented Generation (adaptado para recuperação).

Referência: Yan et al. (2024) "Corrective Retrieval Augmented Generation"
            https://arxiv.org/abs/2401.15884

Adaptação para recuperação de casos similares (sem geração de texto):

    1. Dense retrieval → candidatos iniciais
    2. Assess: cross-encoder pontua cada par (query, candidato)
               → por query: max_score > threshold → "confiante"
                            max_score ≤ threshold → "incerto"
    3. Correct: queries "incertas" recebem candidatos BM25 adicionais,
                que são fundidos com os densos via RRF e re-ranqueados
    4. Queries "confiantes" apenas têm seus candidatos re-ranqueados

A etapa corretiva é seletiva: só queries com retrieval inicial fraco
são corrigidas, evitando degradar queries onde o dense já funciona bem.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from rich.console import Console
from rich.progress import track

from src.retrieval.bm25_search import reciprocal_rank_fusion

console = Console()


def _load_reranker(model_name: str, device: str):
    """Carrega o reranker uma única vez e retorna (model, tokenizer)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    console.print(f"[bold]Carregando reranker {model_name} em {device}[/]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def _score_pairs(
    model,
    tokenizer,
    pairs: list[list[str]],
    device: str,
    batch_size: int,
    max_length: int,
) -> list[float]:
    """Pontua pares (query_text, doc_text) com o reranker. Retorna logits."""
    import torch

    scores = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            logits = model(**inputs, return_dict=True).logits.view(-1).float()
            scores.extend(logits.cpu().tolist())
    return scores


def crag_retrieval(
    query_texts: Sequence[str],
    query_ids: Sequence[str],
    dense_rankings: dict[str, list[str]],
    bm25_index,
    corpus_ids: list[str],
    corpus_texts: list[str],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cuda",
    threshold: float = 0.0,
    top_k: int = 50,
    correction_k: int = 100,
    batch_size: int = 32,
    max_length: int = 512,
    rrf_k: int = 60,
    score_cache_path: Optional[Path | str] = None,
) -> tuple[dict[str, list[str]], dict]:
    """Pipeline CRAG completo.

    Args:
        query_texts: textos das queries
        query_ids: IDs das queries
        dense_rankings: {query_id: [doc_id]} do retrieval denso inicial
        bm25_index: índice BM25 (dict retornado por build_bm25)
        corpus_ids: todos os IDs do corpus
        corpus_texts: todos os textos do corpus
        model_name: reranker cross-encoder
        device: dispositivo de inferência
        threshold: limiar para classificar query como "incerta"
                   (max logit ≤ threshold → aciona correção BM25)
        top_k: tamanho do ranking final por query
        correction_k: candidatos BM25 adicionados na correção
        batch_size: lote para o reranker
        max_length: truncamento de tokens
        rrf_k: constante RRF na correção

    Returns:
        (final_rankings, stats)
        stats: {
            "n_confident": int,
            "n_corrected": int,
            "threshold": float,
            "max_score_mean": float,
            "max_score_std": float,
        }
    """
    from src.retrieval.bm25_search import bm25_search  # importação local

    id_to_text = dict(zip(corpus_ids, corpus_texts))
    qid_to_text = dict(zip(query_ids, query_texts))

    # ------------------------------------------------------------------
    # Passo 1: pontua candidatos densos para determinar confiança
    # Scores são cacheados: ablação de threshold roda em segundos
    # ------------------------------------------------------------------
    query_max_scores: dict[str, float] = {}
    query_doc_scores: dict[str, dict[str, float]] = {}

    if score_cache_path is not None:
        score_cache_path = Path(score_cache_path)

    if score_cache_path is not None and score_cache_path.exists():
        console.print(f"[green]Carregando scores do cache: {score_cache_path}[/]")
        import json as _json
        with open(score_cache_path) as f:
            cached = _json.load(f)
        query_doc_scores = cached["query_doc_scores"]
        query_max_scores = cached["query_max_scores"]
    else:
        model, tokenizer = _load_reranker(model_name, device)
        console.print("[bold]CRAG — Passo 1: avaliando confiança do retrieval denso[/]")

        for qid in track(query_ids, description="Assess"):
            candidates = dense_rankings.get(qid, [])
            if not candidates:
                query_max_scores[qid] = -999.0
                query_doc_scores[qid] = {}
                continue

            qtext = qid_to_text[qid]
            pairs = [[qtext, id_to_text.get(d, "")] for d in candidates]
            scores = _score_pairs(model, tokenizer, pairs, device, batch_size, max_length)

            query_doc_scores[qid] = dict(zip(candidates, scores))
            query_max_scores[qid] = max(scores)

        if score_cache_path is not None:
            import json as _json
            score_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(score_cache_path, "w") as f:
                _json.dump({
                    "query_doc_scores": query_doc_scores,
                    "query_max_scores": query_max_scores,
                }, f)
            console.print(f"[green]Scores salvos em {score_cache_path}[/]")

    # Estatísticas de confiança
    all_max = list(query_max_scores.values())
    mean_max = float(np.mean(all_max))
    std_max = float(np.std(all_max))
    n_corrected_ids = [q for q, s in query_max_scores.items() if s <= threshold]
    n_confident_ids = [q for q, s in query_max_scores.items() if s > threshold]

    console.print(
        f"  max_score: média={mean_max:.3f} ± {std_max:.3f} | "
        f"threshold={threshold}"
    )
    console.print(
        f"  Confiantes: {len(n_confident_ids)} | "
        f"Incertas (correção): {len(n_corrected_ids)}"
    )

    # ------------------------------------------------------------------
    # Passo 2: BM25 para queries incertas
    # ------------------------------------------------------------------
    bm25_rankings: dict[str, list[str]] = {}
    if n_corrected_ids:
        console.print(
            f"[bold]CRAG — Passo 2: BM25 para {len(n_corrected_ids)} queries incertas[/]"
        )
        corrected_texts = [qid_to_text[q] for q in n_corrected_ids]
        bm25_rankings = bm25_search(
            bm25_index, corrected_texts, corpus_ids,
            top_k=correction_k, query_ids=n_corrected_ids, exclude_self=True,
        )

    # ------------------------------------------------------------------
    # Passo 3: monta ranking final por query
    #
    # Fiel ao CRAG original: o avaliador determina a ação, mas não
    # reordena o resultado final. O ranking é sempre baseado nos scores
    # do retrieval (dense ou BM25+RRF), nunca no cross-encoder.
    # ------------------------------------------------------------------
    console.print("[bold]CRAG — Passo 3: aplicando correção seletiva[/]")
    final_rankings: dict[str, list[str]] = {}

    # Pré-computa fusão RRF para as queries incertas (uma vez só)
    merged_rrf: dict[str, list[str]] = {}
    if bm25_rankings:
        merged_rrf = reciprocal_rank_fusion(
            [dense_rankings, bm25_rankings], k=rrf_k, top_k=top_k,
        )

    for qid in query_ids:
        if query_max_scores.get(qid, -999.0) <= threshold and qid in merged_rrf:
            # Query incerta: usa fusão Dense + BM25 via RRF
            final_rankings[qid] = merged_rrf[qid]
        else:
            # Query confiante: mantém ranking denso original
            final_rankings[qid] = dense_rankings.get(qid, [])[:top_k]

    stats = {
        "n_confident": len(n_confident_ids),
        "n_corrected": len(n_corrected_ids),
        "threshold": threshold,
        "max_score_mean": round(mean_max, 4),
        "max_score_std": round(std_max, 4),
    }
    return final_rankings, stats
