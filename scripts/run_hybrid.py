"""
Roda o experimento de retrieval HÍBRIDO (dense + BM25 + RRF + reranker).

Uso:
    python scripts/run_hybrid.py --config configs/hybrid.yaml
"""
from __future__ import annotations

import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.prepare_eval import DataConfig, SplitConfig, prepare_dataset
from src.retrieval.embeddings import generate_embeddings
from src.retrieval.dense_search import search as dense_search
from src.retrieval.bm25_search import build_bm25, bm25_search, reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.evaluation.retrieval_metrics import evaluate_rankings

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/hybrid.yaml")
    p.add_argument("--force-embeddings", action="store_true")
    return p.parse_args()


def print_metrics(title: str, metrics: dict) -> None:
    table = Table(title=title)
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", justify="right", style="bold green")
    for k, v in metrics.items():
        table.add_row(k, str(int(v)) if k == "n_queries" else f"{v:.4f}")
    console.print(table)


def main():
    args = parse_args()
    console.rule("[bold]Experimento de Retrieval Híbrido[/]")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    console.rule("[bold]1. Preparando conjunto de teste[/]")
    data = DataConfig(**cfg["data"])
    split = SplitConfig(**cfg["split"])
    ds = prepare_dataset(data, split)

    console.rule("[bold]2. Embeddings densos[/]")
    emb_cfg = cfg["embedding"]
    corpus_emb = generate_embeddings(
        ds["corpus_texts"],
        model_name=emb_cfg["model_name"], device=emb_cfg["device"],
        batch_size=emb_cfg["batch_size"], max_length=emb_cfg["max_length"],
        normalize=emb_cfg["normalize"], cache_path=emb_cfg["cache_path"],
        force_recompute=args.force_embeddings,
    )
    id_to_idx = {cid: i for i, cid in enumerate(ds["corpus_ids"])}
    query_idx = [id_to_idx[q] for q in ds["query_ids"]]
    query_emb = corpus_emb[query_idx]
    query_texts = [ds["corpus_texts"][i] for i in query_idx]

    console.rule("[bold]3. Busca densa + BM25[/]")
    candidates = cfg["retrieval"]["candidates"]
    dense_rank = dense_search(
        query_emb, corpus_emb, ds["corpus_ids"],
        top_k=candidates, query_ids=ds["query_ids"], exclude_self=True,
    )
    bm25 = build_bm25(ds["corpus_texts"])
    sparse_rank = bm25_search(
        bm25, query_texts, ds["corpus_ids"],
        top_k=candidates, query_ids=ds["query_ids"], exclude_self=True,
    )

    console.rule("[bold]4. Fusão RRF[/]")
    rrf_k = cfg["retrieval"].get("rrf_k", 60)
    fused = reciprocal_rank_fusion([dense_rank, sparse_rank], k=rrf_k,
                                   top_k=candidates)

    final_rank = fused
    if cfg["retrieval"].get("use_reranker", True):
        console.rule("[bold]5. Reranking[/]")
        rr = cfg["reranker"]
        final_rank = rerank(
            query_texts, ds["query_ids"], fused,
            ds["corpus_ids"], ds["corpus_texts"],
            model_name=rr["model_name"], device=rr["device"],
            top_k=cfg["retrieval"]["top_k"],
            batch_size=rr["batch_size"], max_length=rr["max_length"],
        )
    else:
        top_k = cfg["retrieval"]["top_k"]
        final_rank = {q: docs[:top_k] for q, docs in fused.items()}

    console.rule("[bold]6. Avaliação[/]")
    eval_cfg = cfg["evaluation"]
    k_values = eval_cfg["k_values"]
    metrics = evaluate_rankings(final_rank, ds["qrels"], k_values=k_values)
    print_metrics("Híbrido — Métricas (score >= 1)", metrics)

    metrics_strict = {}
    if eval_cfg.get("also_strict") and ds["qrels_strict"]:
        metrics_strict = evaluate_rankings(final_rank, ds["qrels_strict"],
                                           k_values=k_values)
        print_metrics("Híbrido — Métricas (score == 2)", metrics_strict)

    console.rule("[bold]7. Salvando[/]")
    results_dir = Path(eval_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = eval_cfg["run_name"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"{run_name}_{ts}.json"
    out.write_text(json.dumps({
        "run_name": run_name, "timestamp": ts, "config": cfg,
        "n_corpus": len(ds["corpus_ids"]), "n_queries": len(ds["query_ids"]),
        "metrics": metrics, "metrics_strict": metrics_strict,
    }, indent=2, default=str))
    console.print(f"[green]Resultados salvos em {out}[/]")

    rankings_file = results_dir / f"{run_name}_{ts}_rankings.json"
    rankings_file.write_text(json.dumps(
        {q: r[:10] for q, r in final_rank.items()}, indent=2))
    console.print(f"[green]Rankings salvos em {rankings_file}[/]")

    console.rule("[bold green]Concluído[/]")


if __name__ == "__main__":
    main()
