"""
Roda o experimento de retrieval denso completo.

Pipeline:
    1. Prepara o conjunto de teste (queries, corpus, qrels)
    2. Gera embeddings BGE-M3 do corpus (cacheado)
    3. Busca top-k para cada query
    4. Avalia com Recall@k, MRR, nDCG (versões completa e estrita)
    5. Salva resultados em JSON

Uso:
    python scripts/run_retrieval.py --config configs/dense_baseline.yaml
"""
from __future__ import annotations

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
from src.retrieval.dense_search import search
from src.evaluation.retrieval_metrics import evaluate_rankings

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/dense_baseline.yaml")
    p.add_argument("--force-embeddings", action="store_true",
                   help="Recomputa embeddings ignorando o cache")
    return p.parse_args()


def print_metrics(title: str, metrics: dict) -> None:
    table = Table(title=title)
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", justify="right", style="bold green")
    for k, v in metrics.items():
        if k == "n_queries":
            table.add_row(k, str(int(v)))
        else:
            table.add_row(k, f"{v:.4f}")
    console.print(table)


def main():
    args = parse_args()
    console.rule("[bold]Experimento de Retrieval Denso — BGE-M3[/]")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1. Prepara dados
    console.rule("[bold]1. Preparando conjunto de teste[/]")
    data = DataConfig(**cfg["data"])
    split = SplitConfig(**cfg["split"])
    ds = prepare_dataset(data, split)

    # 2. Embeddings do corpus
    console.rule("[bold]2. Gerando embeddings do corpus[/]")
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

    # As queries são um subconjunto do corpus → reusa os embeddings
    id_to_idx = {cid: i for i, cid in enumerate(ds["corpus_ids"])}
    query_idx = [id_to_idx[q] for q in ds["query_ids"]]
    query_emb = corpus_emb[query_idx]

    # 3. Busca
    console.rule("[bold]3. Busca[/]")
    rankings = search(
        query_emb, corpus_emb, ds["corpus_ids"],
        top_k=cfg["retrieval"]["top_k"],
        query_ids=ds["query_ids"],
        exclude_self=True,
    )

    # 4. Avaliação
    console.rule("[bold]4. Avaliação[/]")
    eval_cfg = cfg["evaluation"]
    k_values = eval_cfg["k_values"]

    metrics = evaluate_rankings(rankings, ds["qrels"], k_values=k_values)
    print_metrics("Métricas (score >= 1)", metrics)

    metrics_strict = {}
    if eval_cfg.get("also_strict") and ds["qrels_strict"]:
        metrics_strict = evaluate_rankings(rankings, ds["qrels_strict"], k_values=k_values)
        print_metrics("Métricas (score == 2, estrito)", metrics_strict)

    # 5. Salva resultados
    console.rule("[bold]5. Salvando resultados[/]")
    results_dir = Path(eval_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    run_name = eval_cfg["run_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"{run_name}_{timestamp}.json"

    payload = {
        "run_name": run_name,
        "timestamp": timestamp,
        "config": cfg,
        "n_corpus": len(ds["corpus_ids"]),
        "n_queries": len(ds["query_ids"]),
        "metrics": metrics,
        "metrics_strict": metrics_strict,
    }
    out_file.write_text(json.dumps(payload, indent=2, default=str))
    console.print(f"[green]Resultados salvos em {out_file}[/]")

    # Salva também os rankings para inspeção qualitativa posterior
    rankings_file = results_dir / f"{run_name}_{timestamp}_rankings.json"
    rankings_file.write_text(json.dumps(
        {q: r[:10] for q, r in rankings.items()}, indent=2))
    console.print(f"[green]Rankings (top-10) salvos em {rankings_file}[/]")

    console.rule("[bold green]Concluído[/]")


if __name__ == "__main__":
    main()
