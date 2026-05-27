"""
Reranking com BGE-reranker-v2-m3 via transformers puro.

Carrega o modelo diretamente com AutoModelForSequenceClassification, que é o
método de baixo nível documentado pela BAAI. Evita tanto o FlagReranker
(que chama tokenizer.prepare_for_model, removido no transformers 5.x) quanto
o CrossEncoder do sentence-transformers (que estava produzindo scores
incorretos para este modelo no ambiente atual).

O modelo produz um logit por par (query, documento): quanto MAIOR o logit,
mais relevante. Ordenamos por logit decrescente.
"""
from __future__ import annotations

from typing import Sequence

from rich.console import Console
from rich.progress import track

console = Console()


def rerank(
    query_texts: Sequence[str],
    query_ids: Sequence[str],
    candidate_rankings: dict[str, list[str]],
    corpus_ids: list[str],
    corpus_texts: list[str],
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cuda",
    top_k: int = 10,
    batch_size: int = 32,
    max_length: int = 512,
) -> dict[str, list[str]]:
    """Reordena os candidatos de cada query. Returns {query_id: [doc_ids]}."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    console.print(f"[bold]Carregando reranker {model_name} em {device}[/]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    id_to_text = dict(zip(corpus_ids, corpus_texts))
    qid_to_text = dict(zip(query_ids, query_texts))

    console.print(f"[bold]Reranqueando {len(query_ids)} queries[/]")
    reranked: dict[str, list[str]] = {}

    for qid in track(query_ids, description="Reranking"):
        candidates = candidate_rankings.get(qid, [])
        if not candidates:
            reranked[qid] = []
            continue

        qtext = qid_to_text[qid]
        pairs = [[qtext, id_to_text.get(doc_id, "")] for doc_id in candidates]

        scores = []
        with torch.no_grad():
            # processa em lotes para não estourar memória
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start:start + batch_size]
                inputs = tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt",
                ).to(device)
                logits = model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())

        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        reranked[qid] = [doc for doc, _ in ranked[:top_k]]

    return reranked
