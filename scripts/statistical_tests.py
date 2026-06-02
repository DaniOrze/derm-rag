"""
Testes de significância estatística entre sistemas de retrieval.

Método: paired bootstrap test (Efron & Tibshirani, 1993).
Para cada par (A, B) e cada métrica:
    H0: E[metric(A)] == E[metric(B)]
    p-value: proporção de amostras bootstrap com |diff_boot - diff_obs| >= |diff_obs|

Métricas testadas simultaneamente:
    Recall@10 — fração de relevantes recuperados no top-10
    MRR       — posição média do primeiro acerto
    nDCG@10   — DCG normalizado com ganhos graduados (scores 1 e 2)

Uso:
    python scripts/statistical_tests.py
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

COMPARISONS = [
    # (nome_A,             prefixo_A,                   nome_B,                prefixo_B)
    ("Dense BGE-M3",      "dense_bge_m3_baseline",      "Hybrid BGE-M3+BM25",  "hybrid_bge_m3_norr"),
    ("Dense BGE-M3",      "dense_bge_m3_baseline",      "Dense v1 ft",         "dense_bge_m3_finetuned_2"),
    ("Dense BGE-M3",      "dense_bge_m3_baseline",      "Dense v2 iterativo",  "dense_bge_m3_finetuned_v2"),
    ("Dense v1",          "dense_bge_m3_finetuned_2",   "Hybrid v1+BM25",      "hybrid_bge_m3_finetuned_2"),
    ("Dense v1",          "dense_bge_m3_finetuned_2",   "Dense v2 iterativo",  "dense_bge_m3_finetuned_v2"),
    ("Hybrid v1+BM25",    "hybrid_bge_m3_finetuned_2",  "CRAG (thr=3.0)",      "crag_thr3"),
    ("Hybrid v1+BM25",    "hybrid_bge_m3_finetuned_2",  "Dense v2 iterativo",  "dense_bge_m3_finetuned_v2"),
    ("Dense v2",          "dense_bge_m3_finetuned_v2",  "CRAG v2+reranker ft", "crag_v2_reranker_finetuned"),
    ("Dense v2",          "dense_bge_m3_finetuned_v2",  "Hybrid v2+BM25",      "hybrid_bge_m3_finetuned_v2"),
]

N_BOOTSTRAP = 10_000
SEED = 42
K = 10


# ---------------------------------------------------------------------------
# Métricas por query
# ---------------------------------------------------------------------------

def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranking: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranking, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: list[str], qrels_scored: dict[str, int], k: int) -> float:
    if not qrels_scored:
        return 0.0
    dcg = sum(
        qrels_scored.get(d, 0) / math.log2(i + 1)
        for i, d in enumerate(ranking[:k], 1)
    )
    ideal = sorted(qrels_scored.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Bootstrap test
# ---------------------------------------------------------------------------

def bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Retorna (observed_diff, p_value)."""
    n = len(scores_a)
    obs = scores_b.mean() - scores_a.mean()
    diffs = np.array([
        scores_b[idx].mean() - scores_a[idx].mean()
        for idx in (rng.integers(0, n, size=n) for _ in range(n_bootstrap))
    ])
    centered = diffs - diffs.mean()
    p = (np.abs(centered) >= abs(obs)).mean()
    return float(obs), float(p)


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_latest_rankings(results_dir: Path, prefix: str) -> dict | None:
    files = sorted(results_dir.glob(f"{prefix}*_rankings.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def parse_similar(val) -> dict[str, int]:
    if isinstance(val, str) and val.strip():
        try:
            d = ast.literal_eval(val)
            if isinstance(d, dict):
                return {str(k): int(v) for k, v in d.items()}
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--data",        default="data/processed/derm_cases.csv")
    p.add_argument("--out",         default="results/statistical_tests.json")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    console.rule("[bold]Testes de Significância Estatística[/]")
    console.print(f"  Bootstrap: {args.n_bootstrap} amostras | Métricas: Recall@{K}, MRR, nDCG@{K}")

    # ------------------------------------------------------------------
    # 1. Carrega dataset e qrels (binário e graduado)
    # ------------------------------------------------------------------
    console.rule("[bold]1. Carregando qrels[/]")
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["patient", "patient_uid"]).copy()
    df["patient_uid"] = df["patient_uid"].astype(str)
    corpus_set = set(df["patient_uid"])

    qrels_bin:   dict[str, set[str]]      = {}   # para Recall e MRR
    qrels_graded: dict[str, dict[str, int]] = {}  # para nDCG

    for _, row in df.iterrows():
        uid = row["patient_uid"]
        sims = parse_similar(row["similar_patients"])
        kept = {d: s for d, s in sims.items() if d in corpus_set and d != uid and s >= 1}
        if kept:
            qrels_bin[uid]    = set(kept.keys())
            qrels_graded[uid] = kept

    splits_path = Path("data/finetune/splits.json")
    if splits_path.exists():
        with open(splits_path) as f:
            test_ids = set(json.load(f)["test"])
        console.print(f"  Test IDs de {splits_path}")
    else:
        eligible = [q for q in qrels_bin]
        rng_split = random.Random(42)
        rng_split.shuffle(eligible)
        test_ids = set(eligible[:1000])
        console.print("  Test IDs via fallback (seed=42)")

    console.print(f"  Queries de teste: {len(test_ids)}")
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # 2. Calcula per-query scores para todos os sistemas
    # ------------------------------------------------------------------
    console.rule("[bold]2. Calculando scores por query[/]")
    prefixes_needed = set()
    for _, pa, _, pb in COMPARISONS:
        prefixes_needed.add(pa)
        prefixes_needed.add(pb)

    system_scores: dict[str, dict[str, np.ndarray]] = {}  # prefix → {metric: array}
    query_ids_ordered = [q for q in test_ids if q in qrels_bin]

    for prefix in prefixes_needed:
        rankings = load_latest_rankings(results_dir, prefix)
        if rankings is None:
            console.print(f"  [yellow]✗[/] {prefix}*: não encontrado")
            continue

        r10, mrr_vals, ndcg_vals = [], [], []
        for qid in query_ids_ordered:
            rank = rankings.get(qid, [])
            rel  = qrels_bin[qid]
            grad = qrels_graded[qid]
            r10.append(recall_at_k(rank, rel, K))
            mrr_vals.append(reciprocal_rank(rank, rel))
            ndcg_vals.append(ndcg_at_k(rank, grad, K))

        system_scores[prefix] = {
            "recall":  np.array(r10),
            "mrr":     np.array(mrr_vals),
            "ndcg":    np.array(ndcg_vals),
        }
        console.print(
            f"  [green]✓[/] {prefix}: "
            f"R@{K}={np.mean(r10):.4f}  "
            f"MRR={np.mean(mrr_vals):.4f}  "
            f"nDCG@{K}={np.mean(ndcg_vals):.4f}"
        )

    # ------------------------------------------------------------------
    # 3. Roda bootstrap para cada par × métrica
    # ------------------------------------------------------------------
    console.rule("[bold]3. Testes bootstrap[/]")
    results = []

    for name_a, prefix_a, name_b, prefix_b in COMPARISONS:
        if prefix_a not in system_scores or prefix_b not in system_scores:
            console.print(f"  [yellow]Pulando[/] {name_a} vs {name_b} (arquivo ausente)")
            continue

        sa = system_scores[prefix_a]
        sb = system_scores[prefix_b]

        row: dict = {"system_a": name_a, "system_b": name_b}
        for metric, key in [("Recall@10", "recall"), ("MRR", "mrr"), (f"nDCG@{K}", "ndcg")]:
            diff, pval = bootstrap_test(sa[key], sb[key], args.n_bootstrap, rng)
            row[f"mean_a_{key}"] = round(float(sa[key].mean()), 4)
            row[f"mean_b_{key}"] = round(float(sb[key].mean()), 4)
            row[f"diff_{key}"]   = round(diff, 4)
            row[f"p_{key}"]      = round(pval, 4)
            row[f"sig_{key}"]    = sig_stars(pval)

        results.append(row)
        console.print(
            f"  {name_a} → {name_b}: "
            f"R@10 Δ={row['diff_recall']:+.4f} {row['sig_recall']}  "
            f"MRR Δ={row['diff_mrr']:+.4f} {row['sig_mrr']}  "
            f"nDCG Δ={row['diff_ndcg']:+.4f} {row['sig_ndcg']}"
        )

    # ------------------------------------------------------------------
    # 4. Tabela Recall@10
    # ------------------------------------------------------------------
    console.rule("[bold]4. Tabela — Recall@10[/]")
    _print_metric_table(results, "recall", f"Bootstrap Recall@{K} (n={args.n_bootstrap})", K)

    # ------------------------------------------------------------------
    # 5. Tabela MRR
    # ------------------------------------------------------------------
    console.rule("[bold]5. Tabela — MRR[/]")
    _print_metric_table(results, "mrr", f"Bootstrap MRR (n={args.n_bootstrap})", K)

    # ------------------------------------------------------------------
    # 6. Tabela nDCG@10
    # ------------------------------------------------------------------
    console.rule("[bold]6. Tabela — nDCG@10[/]")
    _print_metric_table(results, "ndcg", f"Bootstrap nDCG@{K} (n={args.n_bootstrap})", K)

    # ------------------------------------------------------------------
    # 7. Tabela resumo — todas as métricas
    # ------------------------------------------------------------------
    console.rule("[bold]7. Resumo consolidado[/]")
    summary = Table(title="Resumo — Significância por métrica (p<0.05 = *)")
    summary.add_column("Comparação", style="cyan")
    summary.add_column("R@10 Δ", justify="right")
    summary.add_column("R@10 sig", justify="center")
    summary.add_column("MRR Δ", justify="right")
    summary.add_column("MRR sig", justify="center")
    summary.add_column("nDCG Δ", justify="right")
    summary.add_column("nDCG sig", justify="center")

    for r in results:
        def _colored(sig): return f"[green]{sig}[/green]" if sig != "n.s." else f"[yellow]{sig}[/yellow]"
        summary.add_row(
            f"{r['system_a']} → {r['system_b']}",
            f"{r['diff_recall']:+.4f}", _colored(r["sig_recall"]),
            f"{r['diff_mrr']:+.4f}",   _colored(r["sig_mrr"]),
            f"{r['diff_ndcg']:+.4f}",  _colored(r["sig_ndcg"]),
        )
    console.print(summary)
    console.print("\n[bold]* p<0.05   ** p<0.01   *** p<0.001   n.s. não significativo[/]")

    # ------------------------------------------------------------------
    # 8. Salva
    # ------------------------------------------------------------------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n_bootstrap": args.n_bootstrap,
        "k": K,
        "comparisons": results,
    }, indent=2))
    console.print(f"\n[green]Resultados salvos em {args.out}[/]")
    console.rule("[bold green]Concluído[/]")


def _print_metric_table(results: list[dict], key: str, title: str, k: int) -> None:
    label = {"recall": f"R@{k}", "mrr": "MRR", "ndcg": f"nDCG@{k}"}[key]
    table = Table(title=title)
    table.add_column("Sistema A", style="cyan")
    table.add_column("Sistema B", style="cyan")
    table.add_column(f"{label} A", justify="right")
    table.add_column(f"{label} B", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Sig.", justify="center", style="bold")

    for r in results:
        sig = r[f"sig_{key}"]
        color = "green" if r[f"p_{key}"] < 0.05 else "yellow"
        table.add_row(
            r["system_a"], r["system_b"],
            f"{r[f'mean_a_{key}']:.4f}",
            f"{r[f'mean_b_{key}']:.4f}",
            f"{r[f'diff_{key}']:+.4f}",
            f"{r[f'p_{key}']:.4f}",
            f"[{color}]{sig}[/{color}]",
        )
    console.print(table)


if __name__ == "__main__":
    main()
