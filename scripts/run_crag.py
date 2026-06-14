"""
Pipeline CRAG: dense retrieval → assess → correct → evaluate.

Uso:
    python scripts/run_crag.py --config configs/crag.yaml
    python scripts/run_crag.py --config configs/crag.yaml --force-embeddings

Referência: Yan et al. (2024) "Corrective Retrieval Augmented Generation"
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
from src.retrieval.bm25_search import build_bm25
from src.retrieval.crag import crag_retrieval
from src.evaluation.retrieval_metrics import evaluate_rankings

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/crag.yaml")
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
    console.rule("[bold]Experimento CRAG[/]")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    console.rule("[bold]1. Preparando conjunto de teste[/]")
    data = DataConfig(**cfg["data"])
    split = SplitConfig(**cfg["split"])
    ds = prepare_dataset(data, split)

    # ------------------------------------------------------------------
    console.rule("[bold]2. Embeddings densos (BGE-M3)[/]")
    emb_cfg = cfg["embedding"]
    corpus_emb = generate_embeddings(
        ds["corpus_texts"],
        model_name=emb_cfg["model_name"],
        device=emb_cfg["device"],
        batch_size=emb_cfg["batch_size"],
        max_length=emb_cfg["max_length"],
        normalize=emb_cfg["normalize"],
        cache_path=emb_cfg["cache_path"],
        force_recompute=args.force_embeddings,
    )
    id_to_idx = {cid: i for i, cid in enumerate(ds["corpus_ids"])}
    query_idx = [id_to_idx[q] for q in ds["query_ids"]]
    query_texts = [ds["corpus_texts"][i] for i in query_idx]

    query_instruction = emb_cfg.get("query_instruction")
    if query_instruction:
        console.print(f"  [bold]Instrução de query:[/] {query_instruction}")
        prefixed = [f"Instruct: {query_instruction}\nQuery: {t}" for t in query_texts]
        query_emb = generate_embeddings(
            prefixed,
            model_name=emb_cfg["model_name"],
            device=emb_cfg["device"],
            batch_size=emb_cfg["batch_size"],
            max_length=emb_cfg["max_length"],
            normalize=emb_cfg["normalize"],
            cache_path=emb_cfg.get("query_cache_path"),
            force_recompute=args.force_embeddings,
        )
    else:
        query_emb = corpus_emb[query_idx]

    # ------------------------------------------------------------------
    console.rule("[bold]3. Retrieval denso inicial[/]")
    ret_cfg = cfg["retrieval"]
    dense_rankings = dense_search(
        query_emb, corpus_emb, ds["corpus_ids"],
        top_k=ret_cfg["initial_k"],
        query_ids=ds["query_ids"],
        exclude_self=True,
    )

    # ------------------------------------------------------------------
    console.rule("[bold]4. Índice BM25 (para etapa corretiva)[/]")
    bm25_index = build_bm25(ds["corpus_texts"])

    # ------------------------------------------------------------------
    console.rule("[bold]5. CRAG: assess → correct → rerank[/]")
    rr_cfg = cfg["reranker"]
    final_rankings, stats = crag_retrieval(
        query_texts=query_texts,
        query_ids=ds["query_ids"],
        dense_rankings=dense_rankings,
        bm25_index=bm25_index,
        corpus_ids=ds["corpus_ids"],
        corpus_texts=ds["corpus_texts"],
        model_name=rr_cfg["model_name"],
        device=rr_cfg["device"],
        threshold=ret_cfg["threshold"],
        top_k=ret_cfg["top_k"],
        correction_k=ret_cfg["correction_k"],
        batch_size=rr_cfg["batch_size"],
        max_length=rr_cfg["max_length"],
        rrf_k=ret_cfg.get("rrf_k", 60),
        score_cache_path=cfg.get("score_cache_path", "data/processed/crag_scores.json"),
    )

    console.print(f"\n[bold]Estatísticas CRAG:[/]")
    console.print(f"  Queries confiantes (sem correção): {stats['n_confident']}")
    console.print(f"  Queries corrigidas (BM25+RRF):     {stats['n_corrected']}")
    console.print(f"  Threshold usado:                   {stats['threshold']}")
    console.print(f"  Max score médio:                   {stats['max_score_mean']:.4f} ± {stats['max_score_std']:.4f}")

    # ------------------------------------------------------------------
    console.rule("[bold]6. Avaliação[/]")
    k_values = cfg["evaluation"]["k_values"]
    metrics = evaluate_rankings(final_rankings, ds["qrels"], k_values=k_values)
    print_metrics("CRAG — Métricas (score >= 1)", metrics)

    metrics_strict = {}
    if cfg["evaluation"].get("also_strict") and ds["qrels_strict"]:
        metrics_strict = evaluate_rankings(
            final_rankings, ds["qrels_strict"], k_values=k_values
        )
        print_metrics("CRAG — Métricas (score == 2)", metrics_strict)

    # ------------------------------------------------------------------
    console.rule("[bold]7. Salvando resultados[/]")
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg["evaluation"]["run_name"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out = results_dir / f"{run_name}_{ts}.json"
    out.write_text(json.dumps({
        "run_name": run_name,
        "timestamp": ts,
        "config": cfg,
        "n_corpus": len(ds["corpus_ids"]),
        "n_queries": len(ds["query_ids"]),
        "crag_stats": stats,
        "metrics": metrics,
        "metrics_strict": metrics_strict,
    }, indent=2, default=str))
    console.print(f"[green]Resultados salvos em {out}[/]")

    rankings_file = results_dir / f"{run_name}_{ts}_rankings.json"
    rankings_file.write_text(
        json.dumps({q: r[:10] for q, r in final_rankings.items()}, indent=2)
    )
    console.print(f"[green]Rankings salvos em {rankings_file}[/]")

    console.rule("[bold green]Concluído[/]")


if __name__ == "__main__":
    main()
