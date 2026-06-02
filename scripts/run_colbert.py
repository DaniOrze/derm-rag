"""
Pipeline ColBERT: Dense → ColBERT MaxSim reranking → avaliação.

Dois estágios:
    1. Dense retrieval (BGE-M3, top-100) — candidatos iniciais
    2. ColBERT MaxSim reranking — reordena com interação multi-vetor

Vetores ColBERT são computados apenas para os documentos necessários:
    queries (1.000) + candidatos únicos (~12-15k) — não o corpus inteiro.

Uso:
    python scripts/run_colbert.py --config configs/colbert.yaml
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
from src.retrieval.colbert_search import generate_colbert_vectors, colbert_rerank
from src.evaluation.retrieval_metrics import evaluate_rankings

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/colbert.yaml")
    p.add_argument("--force-colbert", action="store_true",
                   help="Recomputa vetores ColBERT mesmo que cache exista")
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
    console.rule("[bold]Pipeline ColBERT (Dense → MaxSim Reranking)[/]")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    console.rule("[bold]1. Preparando conjunto de teste[/]")
    data = DataConfig(**cfg["data"])
    split = SplitConfig(**cfg["split"])
    ds = prepare_dataset(data, split)

    id_to_text = dict(zip(ds["corpus_ids"], ds["corpus_texts"]))

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
    )
    id_to_idx  = {cid: i for i, cid in enumerate(ds["corpus_ids"])}
    query_idx  = [id_to_idx[q] for q in ds["query_ids"]]
    query_emb  = corpus_emb[query_idx]

    # ------------------------------------------------------------------
    console.rule("[bold]3. Retrieval denso inicial (top-100)[/]")
    candidates_k = cfg["retrieval"]["candidates"]
    dense_rankings = dense_search(
        query_emb, corpus_emb, ds["corpus_ids"],
        top_k=candidates_k,
        query_ids=ds["query_ids"],
        exclude_self=True,
    )

    # ------------------------------------------------------------------
    console.rule("[bold]4. Coletando IDs necessários para ColBERT[/]")
    needed_ids: set[str] = set(ds["query_ids"])
    for ranking in dense_rankings.values():
        needed_ids.update(ranking)

    console.print(f"  IDs únicos necessários: {len(needed_ids)} "
                  f"(queries: {len(ds['query_ids'])}, "
                  f"candidatos únicos: {len(needed_ids) - len(ds['query_ids'])})")

    needed_texts = {doc_id: id_to_text[doc_id]
                    for doc_id in needed_ids if doc_id in id_to_text}

    # ------------------------------------------------------------------
    console.rule("[bold]5. Gerando vetores ColBERT[/]")
    cb_cfg = cfg["colbert"]
    colbert_vecs = generate_colbert_vectors(
        id_to_text=needed_texts,
        model_name=cb_cfg["model_name"],
        device=cb_cfg["device"],
        batch_size=cb_cfg["batch_size"],
        max_length=cb_cfg["max_length"],
        cache_path=cb_cfg["cache_path"],
        force_recompute=args.force_colbert,
    )

    # Estatísticas dos vetores
    dims = [v.shape for v in list(colbert_vecs.values())[:5]]
    console.print(f"  Exemplo de shapes (primeiros 5): {dims}")

    # ------------------------------------------------------------------
    console.rule("[bold]6. ColBERT MaxSim Reranking[/]")
    final_rankings = colbert_rerank(
        query_ids=ds["query_ids"],
        colbert_vecs=colbert_vecs,
        initial_rankings=dense_rankings,
        top_k=cfg["retrieval"]["top_k"],
    )

    # ------------------------------------------------------------------
    console.rule("[bold]7. Avaliação[/]")
    k_values = cfg["evaluation"]["k_values"]
    metrics = evaluate_rankings(final_rankings, ds["qrels"], k_values=k_values)
    print_metrics("ColBERT Reranking — Métricas (score >= 1)", metrics)

    metrics_strict = {}
    if cfg["evaluation"].get("also_strict") and ds["qrels_strict"]:
        metrics_strict = evaluate_rankings(final_rankings, ds["qrels_strict"], k_values=k_values)
        print_metrics("ColBERT Reranking — Métricas (score == 2)", metrics_strict)

    # ------------------------------------------------------------------
    console.rule("[bold]8. Salvando[/]")
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
    rankings_file.write_text(json.dumps({q: r[:10] for q, r in final_rankings.items()}, indent=2))
    console.print(f"[green]Rankings salvos em {rankings_file}[/]")

    console.rule("[bold green]Concluído[/]")


if __name__ == "__main__":
    main()
