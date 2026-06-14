"""
BM25 puro como baseline esparso.

Uso:
    python scripts/run_bm25.py --config configs/bm25.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.prepare_eval import DataConfig, SplitConfig, prepare_dataset
from src.retrieval.bm25_search import build_bm25, bm25_search
from src.evaluation.retrieval_metrics import evaluate_rankings

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/bm25.yaml")
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
    console.rule("[bold]BM25 — Baseline Esparso[/]")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    console.rule("[bold]1. Preparando conjunto de teste[/]")
    data = DataConfig(**cfg["data"])
    split = SplitConfig(**cfg["split"])
    ds = prepare_dataset(data, split)

    console.rule("[bold]2. Construindo índice BM25[/]")
    bm25 = build_bm25(ds["corpus_texts"])

    console.rule("[bold]3. Busca BM25[/]")
    id_to_idx = {cid: i for i, cid in enumerate(ds["corpus_ids"])}
    query_idx = [id_to_idx[q] for q in ds["query_ids"]]
    query_texts = [ds["corpus_texts"][i] for i in query_idx]

    rankings = bm25_search(
        bm25, query_texts, ds["corpus_ids"],
        top_k=cfg["retrieval"]["top_k"],
        query_ids=ds["query_ids"],
        exclude_self=True,
    )

    console.rule("[bold]4. Avaliação[/]")
    k_values = cfg["evaluation"]["k_values"]
    metrics = evaluate_rankings(rankings, ds["qrels"], k_values=k_values)
    print_metrics("BM25 — Métricas (score >= 1)", metrics)

    metrics_strict = {}
    if cfg["evaluation"].get("also_strict") and ds["qrels_strict"]:
        metrics_strict = evaluate_rankings(rankings, ds["qrels_strict"], k_values=k_values)
        print_metrics("BM25 — Métricas (score == 2)", metrics_strict)

    console.rule("[bold]5. Salvando resultados[/]")
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg["evaluation"]["run_name"]
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
        {q: r[:10] for q, r in rankings.items()}, indent=2))
    console.print(f"[green]Rankings salvos em {rankings_file}[/]")

    console.rule("[bold green]Concluído[/]")


if __name__ == "__main__":
    main()
