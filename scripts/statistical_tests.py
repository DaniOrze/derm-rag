"""
Testes de significância estatística entre sistemas de retrieval.

Método: paired bootstrap test (Efron & Tibshirani, 1993).
Para cada par de sistemas (A, B):
    H0: E[metric(A)] == E[metric(B)]
    Estatística: diferença observada nas métricas médias
    p-value: proporção de amostras bootstrap com diferença >= observada

Interpretação:
    p < 0.05  → diferença significativa (95% confiança)
    p < 0.01  → altamente significativa (99% confiança)

Uso:
    python scripts/statistical_tests.py
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

# Pares de sistemas a comparar — (nome_A, prefixo_A, nome_B, prefixo_B)
# A é o sistema "pior" (baseline), B é o sistema "melhor" (novo)
# Prefixos correspondem ao run_name dos configs — únicos por design.
# hybrid_bge_m3_norr  → hybrid sem reranker (configs/hybrid_no_reranker.yaml)
# crag_thr3           → CRAG base BGE-M3 com threshold=3.0 (configs/crag_thr3.yaml)
COMPARISONS = [
    ("Dense BGE-M3",      "dense_bge_m3_baseline",        "Hybrid BGE-M3+BM25",    "hybrid_bge_m3_norr"),
    ("Dense BGE-M3",      "dense_bge_m3_baseline",        "Dense v1 fine-tuned",   "dense_bge_m3_finetuned_2"),
    ("Dense v1",          "dense_bge_m3_finetuned_2",     "Hybrid v1+BM25",        "hybrid_bge_m3_finetuned_2"),
    ("Hybrid v1+BM25",    "hybrid_bge_m3_finetuned_2",    "CRAG (thr=3.0)",        "crag_thr3"),
    ("Dense v1",          "dense_bge_m3_finetuned_2",     "Dense v2 iterativo",    "dense_bge_m3_finetuned_v2"),
    ("Hybrid v1+BM25",    "hybrid_bge_m3_finetuned_2",    "Dense v2 iterativo",    "dense_bge_m3_finetuned_v2"),
    ("Dense v2",          "dense_bge_m3_finetuned_v2",    "CRAG v2+reranker ft",   "crag_v2_reranker_finetuned"),
    ("Dense v2",          "dense_bge_m3_finetuned_v2",    "Hybrid v2+BM25",        "hybrid_bge_m3_finetuned_v2"),
]

N_BOOTSTRAP = 10_000
SEED        = 42


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


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def mrr(ranking: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranking, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    rng: np.random.Generator = None,
) -> tuple[float, float, float]:
    """
    Paired bootstrap test.
    Returns (observed_diff, p_value, ci_lower_95).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    a = np.array(scores_a)
    b = np.array(scores_b)
    n = len(a)
    observed_diff = b.mean() - a.mean()

    count_extreme = 0
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = b[idx].mean() - a[idx].mean()

    # p-value: proporção de bootstrap onde a diferença ≥ observada (sob H0 centrada)
    centered = diffs - diffs.mean()
    p_value = (np.abs(centered) >= np.abs(observed_diff)).mean()

    ci_lower = float(np.percentile(diffs, 2.5))
    return float(observed_diff), float(p_value), ci_lower


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--data",        default="data/processed/derm_cases.csv")
    p.add_argument("--k",           type=int, default=10)
    p.add_argument("--out",         default="results/statistical_tests.json")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    console.rule("[bold]Testes de Significância Estatística[/]")
    console.print(f"  Bootstrap amostras: {args.n_bootstrap} | k={args.k}")

    # ------------------------------------------------------------------
    # 1. Carrega dataset e reconstrói qrels
    # ------------------------------------------------------------------
    console.rule("[bold]1. Carregando qrels[/]")
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["patient", "patient_uid"]).copy()
    df["patient_uid"] = df["patient_uid"].astype(str)
    corpus_set = set(df["patient_uid"])

    qrels: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        uid = row["patient_uid"]
        sims = parse_similar(row["similar_patients"])
        kept = {d for d, s in sims.items() if d in corpus_set and d != uid and s >= 1}
        if kept:
            qrels[uid] = kept

    # Carrega os IDs de teste exatamente como gerados em prepare_finetune_data.py
    # (seed=42, ordem do DataFrame — idêntico ao prepare_eval.py)
    splits_path = Path("data/finetune/splits.json")
    if splits_path.exists():
        with open(splits_path) as f:
            test_ids = set(json.load(f)["test"])
        console.print(f"  Test IDs carregados de {splits_path}")
    else:
        # Fallback: reproduz a mesma lógica do prepare_eval.py
        eligible = [q for q in qrels if len(qrels[q]) >= 1]
        rng_split = random.Random(42)
        rng_split.shuffle(eligible)
        test_ids = set(eligible[:1000])
        console.print(f"  Test IDs gerados por fallback (splits.json não encontrado)")

    console.print(f"  Queries de teste: {len(test_ids)}")

    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # 2. Roda comparações
    # ------------------------------------------------------------------
    console.rule("[bold]2. Rodando comparações[/]")
    results = []

    for name_a, prefix_a, name_b, prefix_b in COMPARISONS:
        rankings_a = load_latest_rankings(results_dir, prefix_a)
        rankings_b = load_latest_rankings(results_dir, prefix_b)

        if rankings_a is None:
            console.print(f"  [yellow]✗[/] {name_a} ({prefix_a}*): não encontrado")
            continue
        if rankings_b is None:
            console.print(f"  [yellow]✗[/] {name_b} ({prefix_b}*): não encontrado")
            continue

        scores_a, scores_b = [], []
        for qid in test_ids:
            if qid not in qrels:
                continue
            rel = qrels[qid]
            r_a = recall_at_k(rankings_a.get(qid, []), rel, args.k)
            r_b = recall_at_k(rankings_b.get(qid, []), rel, args.k)
            scores_a.append(r_a)
            scores_b.append(r_b)

        diff, p_val, ci_low = bootstrap_test(scores_a, scores_b, args.n_bootstrap, rng)

        results.append({
            "system_a":  name_a,
            "system_b":  name_b,
            "mean_a":    round(np.mean(scores_a), 4),
            "mean_b":    round(np.mean(scores_b), 4),
            "diff":      round(diff, 4),
            "p_value":   round(p_val, 4),
            "ci_low_95": round(ci_low, 4),
            "sig":       sig_stars(p_val),
        })
        console.print(
            f"  {name_a} vs {name_b}: "
            f"Δ={diff:+.4f}  p={p_val:.4f}  {sig_stars(p_val)}"
        )

    # ------------------------------------------------------------------
    # 3. Tabela formatada
    # ------------------------------------------------------------------
    console.rule("[bold]3. Resultados[/]")
    table = Table(title=f"Testes Bootstrap (Recall@{args.k}, n={args.n_bootstrap})")
    table.add_column("Sistema A (baseline)", style="cyan")
    table.add_column("Sistema B (novo)", style="cyan")
    table.add_column("R@10 A", justify="right")
    table.add_column("R@10 B", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Sig.", justify="center", style="bold")

    for r in results:
        sig_color = "green" if r["p_value"] < 0.05 else "yellow"
        table.add_row(
            r["system_a"], r["system_b"],
            f"{r['mean_a']:.4f}", f"{r['mean_b']:.4f}",
            f"{r['diff']:+.4f}",
            f"{r['p_value']:.4f}",
            f"[{sig_color}]{r['sig']}[/{sig_color}]",
        )
    console.print(table)
    console.print("\n[bold]* p<0.05   ** p<0.01   *** p<0.001   n.s. não significativo[/]")

    # ------------------------------------------------------------------
    # 4. Salva
    # ------------------------------------------------------------------
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "metric": f"recall@{args.k}",
        "n_bootstrap": args.n_bootstrap,
        "comparisons": results,
    }, indent=2))
    console.print(f"\n[green]Resultados salvos em {args.out}[/]")
    console.rule("[bold green]Concluído[/]")


if __name__ == "__main__":
    main()
