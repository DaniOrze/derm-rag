"""
Ablação de threshold do CRAG.

Reutiliza os scores do reranker já calculados (data/processed/crag_scores.json)
para testar múltiplos thresholds sem rodar o reranker novamente.
Cada threshold determina quantas queries são corrigidas com BM25.

Uso:
    python scripts/run_crag_ablation.py --config configs/crag.yaml

Pré-requisito: rodar run_crag.py pelo menos uma vez para gerar o cache
               de scores em data/processed/crag_scores.json.
"""
from __future__ import annotations

import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.prepare_eval import DataConfig, SplitConfig, prepare_dataset
from src.retrieval.embeddings import generate_embeddings
from src.retrieval.dense_search import search as dense_search
from src.retrieval.bm25_search import build_bm25, bm25_search, reciprocal_rank_fusion
from src.evaluation.retrieval_metrics import evaluate_rankings

console = Console()

# Thresholds a testar (logit do reranker)
THRESHOLDS = [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def apply_crag_threshold(
    query_ids: list[str],
    dense_rankings: dict[str, list[str]],
    bm25_rankings: dict[str, list[str]],
    query_max_scores: dict[str, float],
    threshold: float,
    top_k: int,
    rrf_k: int = 60,
) -> tuple[dict[str, list[str]], int]:
    """Aplica um threshold e retorna (rankings_corrigidos, n_corrigidos)."""
    uncertain_ids = [q for q in query_ids if query_max_scores.get(q, -999.0) <= threshold]

    merged_rrf: dict[str, list[str]] = {}
    if uncertain_ids:
        uncertain_dense = {q: dense_rankings[q] for q in uncertain_ids if q in dense_rankings}
        uncertain_bm25 = {q: bm25_rankings[q] for q in uncertain_ids if q in bm25_rankings}
        merged_rrf = reciprocal_rank_fusion(
            [uncertain_dense, uncertain_bm25], k=rrf_k, top_k=top_k,
        )

    rankings: dict[str, list[str]] = {}
    for qid in query_ids:
        if query_max_scores.get(qid, -999.0) <= threshold and qid in merged_rrf:
            rankings[qid] = merged_rrf[qid]
        else:
            rankings[qid] = dense_rankings.get(qid, [])[:top_k]

    return rankings, len(uncertain_ids)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/crag.yaml")
    args = p.parse_args()

    console.rule("[bold]Ablação de Threshold — CRAG[/]")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    score_cache_path = Path(cfg.get("score_cache_path", "data/processed/crag_scores.json"))
    if not score_cache_path.exists():
        console.print(
            f"[red]Cache de scores não encontrado: {score_cache_path}\n"
            "Execute run_crag.py primeiro para gerar o cache.[/]"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    console.rule("[bold]1. Preparando dados[/]")
    data = DataConfig(**cfg["data"])
    split = SplitConfig(**cfg["split"])
    ds = prepare_dataset(data, split)

    # ------------------------------------------------------------------
    console.rule("[bold]2. Rankings densos[/]")
    emb_cfg = cfg["embedding"]
    corpus_emb = generate_embeddings(
        ds["corpus_texts"],
        model_name=emb_cfg["model_name"], device=emb_cfg["device"],
        batch_size=emb_cfg["batch_size"], max_length=emb_cfg["max_length"],
        normalize=emb_cfg["normalize"], cache_path=emb_cfg["cache_path"],
    )
    id_to_idx = {cid: i for i, cid in enumerate(ds["corpus_ids"])}
    query_idx = [id_to_idx[q] for q in ds["query_ids"]]
    query_emb = corpus_emb[query_idx]

    ret_cfg = cfg["retrieval"]
    dense_rankings = dense_search(
        query_emb, corpus_emb, ds["corpus_ids"],
        top_k=ret_cfg["correction_k"], query_ids=ds["query_ids"], exclude_self=True,
    )

    # ------------------------------------------------------------------
    console.rule("[bold]3. BM25 para todas as queries[/]")
    bm25_index = build_bm25(ds["corpus_texts"])
    query_texts = [ds["corpus_texts"][i] for i in query_idx]
    bm25_rankings = bm25_search(
        bm25_index, query_texts, ds["corpus_ids"],
        top_k=ret_cfg["correction_k"], query_ids=ds["query_ids"], exclude_self=True,
    )

    # ------------------------------------------------------------------
    console.rule("[bold]4. Carregando scores cacheados[/]")
    with open(score_cache_path) as f:
        cached_scores = json.load(f)
    query_max_scores: dict[str, float] = cached_scores["query_max_scores"]
    console.print(f"[green]Scores carregados para {len(query_max_scores)} queries[/]")

    # ------------------------------------------------------------------
    console.rule("[bold]5. Ablação de threshold[/]")
    k_values = cfg["evaluation"]["k_values"]
    top_k = ret_cfg["top_k"]
    results_by_threshold = []

    for threshold in THRESHOLDS:
        rankings, n_corrected = apply_crag_threshold(
            ds["query_ids"], dense_rankings, bm25_rankings,
            query_max_scores, threshold, top_k,
        )
        metrics = evaluate_rankings(rankings, ds["qrels"], k_values=k_values)
        pct = 100 * n_corrected / len(ds["query_ids"])
        results_by_threshold.append({
            "threshold": threshold,
            "n_corrected": n_corrected,
            "pct_corrected": round(pct, 1),
            "recall@10": round(metrics.get("recall@10", 0), 4),
            "mrr": round(metrics.get("mrr", 0), 4),
        })

    # Tabela de resultados
    table = Table(title="Ablação de Threshold CRAG (score >= 1)")
    table.add_column("Threshold", style="cyan", justify="right")
    table.add_column("Queries corrigidas", justify="right")
    table.add_column("% corrigidas", justify="right")
    table.add_column("Recall@10", justify="right", style="bold green")
    table.add_column("MRR", justify="right", style="bold green")

    for r in results_by_threshold:
        table.add_row(
            str(r["threshold"]),
            str(r["n_corrected"]),
            f"{r['pct_corrected']}%",
            f"{r['recall@10']:.4f}",
            f"{r['mrr']:.4f}",
        )
    console.print(table)

    # ------------------------------------------------------------------
    console.rule("[bold]6. Salvando[/]")
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"crag_ablation_{ts}.json"
    out.write_text(json.dumps({
        "timestamp": ts,
        "config": cfg,
        "ablation": results_by_threshold,
    }, indent=2))
    console.print(f"[green]Ablação salva em {out}[/]")
    console.rule("[bold green]Concluído[/]")


if __name__ == "__main__":
    main()
