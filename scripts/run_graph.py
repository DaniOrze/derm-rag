"""
Pipeline GraphRAG: extração de entidades → grafo de conhecimento → recuperação → avaliação.

Uso:
    python scripts/run_graph.py --config configs/graph.yaml
    python scripts/run_graph.py --config configs/graph.yaml --force-entities
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

import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.prepare_eval import DataConfig, SplitConfig, prepare_dataset
from src.retrieval.entity_extractor import EntityExtractor
from src.retrieval.graph_search import build_graph, graph_search
from src.evaluation.retrieval_metrics import evaluate_rankings

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/graph.yaml")
    p.add_argument(
        "--force-entities", action="store_true",
        help="Reprocessa extração de entidades mesmo que o cache exista",
    )
    p.add_argument(
        "--force-graph", action="store_true",
        help="Reconstrói o grafo mesmo que o cache exista",
    )
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
    console.rule("[bold]Experimento GraphRAG[/]")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    console.rule("[bold]1. Preparando conjunto de teste[/]")
    data = DataConfig(**cfg["data"])
    split = SplitConfig(**cfg["split"])
    ds = prepare_dataset(data, split)

    # ------------------------------------------------------------------
    console.rule("[bold]2. Extração de entidades médicas[/]")
    ent_cfg = cfg["entity_extraction"]
    entity_cache = Path(ent_cfg["cache_path"])

    if args.force_entities and entity_cache.exists():
        console.print("[yellow]--force-entities: removendo cache de entidades[/]")
        entity_cache.unlink()

    extractor = EntityExtractor(
        use_scispacy=ent_cfg.get("use_scispacy", True),
        spacy_model=ent_cfg.get("spacy_model", "en_ner_bc5cdr_md"),
    )
    entities_by_case = extractor.extract_batch(
        ds["corpus_texts"],
        ds["corpus_ids"],
        cache_path=entity_cache,
    )

    # ------------------------------------------------------------------
    console.rule("[bold]3. Construindo grafo de conhecimento[/]")
    graph_cache = Path(cfg["graph"]["cache_path"])

    if args.force_graph and graph_cache.exists():
        console.print("[yellow]--force-graph: removendo cache do grafo[/]")
        graph_cache.unlink()

    graph = build_graph(
        ds["corpus_ids"],
        entities_by_case,
        cache_path=graph_cache,
    )

    # ------------------------------------------------------------------
    console.rule("[bold]4. Recuperação via grafo[/]")
    rankings = graph_search(
        query_ids=ds["query_ids"],
        corpus_ids=ds["corpus_ids"],
        graph=graph,
        entities_by_case=entities_by_case,
        top_k=cfg["retrieval"]["top_k"],
    )

    # ------------------------------------------------------------------
    console.rule("[bold]5. Avaliação[/]")
    k_values = cfg["evaluation"]["k_values"]
    metrics = evaluate_rankings(rankings, ds["qrels"], k_values=k_values)
    print_metrics("GraphRAG — Métricas (score >= 1)", metrics)

    metrics_strict = {}
    if cfg["evaluation"].get("also_strict") and ds["qrels_strict"]:
        metrics_strict = evaluate_rankings(
            rankings, ds["qrels_strict"], k_values=k_values
        )
        print_metrics("GraphRAG — Métricas (score == 2)", metrics_strict)

    # ------------------------------------------------------------------
    console.rule("[bold]6. Salvando resultados[/]")
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
        "metrics": metrics,
        "metrics_strict": metrics_strict,
    }, indent=2, default=str))
    console.print(f"[green]Resultados salvos em {out}[/]")

    rankings_file = results_dir / f"{run_name}_{ts}_rankings.json"
    rankings_file.write_text(
        json.dumps({q: r[:10] for q, r in rankings.items()}, indent=2)
    )
    console.print(f"[green]Rankings salvos em {rankings_file}[/]")

    console.rule("[bold green]Concluído[/]")


if __name__ == "__main__":
    main()
