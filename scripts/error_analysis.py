"""
Análise de erro dos sistemas de retrieval.

Responde as perguntas:
  1. Quantas queries cada sistema falha completamente (0 acertos no top-10)?
  2. Quais queries NENHUM sistema consegue acertar?
  3. As falhas têm padrão? (texto curto/longo, poucos positivos, doenças raras?)
  4. Exemplos concretos de falhas e acertos para a tese.

Uso:
    python scripts/error_analysis.py --config configs/dense_baseline.yaml

O script carrega automaticamente os rankings das runs mais recentes
para cada sistema definido em --systems.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

# Sistemas a comparar — nome legível → prefixo do arquivo de rankings em results/
SYSTEMS = {
    "Dense BGE-M3":           "dense_bge_m3_baseline",
    "Hybrid BGE-M3+BM25":     "hybrid_bge_m3",
    "CRAG BGE-M3 (thr=3.0)":  "crag",
    "Dense Fine-tuned":        "dense_bge_m3_finetuned",
    "Hybrid Fine-tuned+BM25":  "hybrid_bge_m3_finetuned",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_latest_rankings(results_dir: Path, run_name: str) -> dict[str, list[str]] | None:
    """Carrega o arquivo de rankings mais recente para um run_name."""
    pattern = f"{run_name}_*_rankings.json"
    files = sorted(results_dir.glob(pattern))
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


def hit_at_k(ranking: list[str], relevant: set[str], k: int) -> bool:
    return bool(set(ranking[:k]) & relevant)


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def text_length_bucket(n_tokens: int) -> str:
    if n_tokens < 200:
        return "curto (<200)"
    elif n_tokens < 500:
        return "médio (200-500)"
    elif n_tokens < 800:
        return "longo (500-800)"
    else:
        return "muito longo (>800)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--data",        default="data/processed/derm_cases.csv")
    p.add_argument("--entities",    default="data/processed/entities.json")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--n-queries",   type=int, default=1000)
    p.add_argument("--out",         default="results/error_analysis.json")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    console.rule("[bold]Análise de Erro — Sistemas de Retrieval[/]")

    # ------------------------------------------------------------------
    # 1. Carrega rankings de todos os sistemas
    # ------------------------------------------------------------------
    console.rule("[bold]1. Carregando rankings[/]")
    all_rankings: dict[str, dict[str, list[str]]] = {}
    for name, prefix in SYSTEMS.items():
        r = load_latest_rankings(results_dir, prefix)
        if r:
            all_rankings[name] = r
            console.print(f"  [green]✓[/] {name}: {len(r)} queries")
        else:
            console.print(f"  [yellow]✗[/] {name}: não encontrado (pulando)")

    if not all_rankings:
        console.print("[red]Nenhum ranking encontrado. Execute os experimentos primeiro.[/]")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Carrega dataset
    # ------------------------------------------------------------------
    console.rule("[bold]2. Carregando dataset[/]")
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["patient", "patient_uid"]).copy()
    df["patient_uid"] = df["patient_uid"].astype(str)
    corpus_set = set(df["patient_uid"])
    id_to_text = dict(zip(df["patient_uid"], df["patient"].astype(str)))
    id_to_row  = {row["patient_uid"]: row for _, row in df.iterrows()}

    # Reconstrói qrels (mesma lógica do prepare_eval.py)
    import random
    qrels: dict[str, dict[str, int]] = {}
    for _, row in df.iterrows():
        uid = row["patient_uid"]
        sims = parse_similar(row["similar_patients"])
        kept = {d: s for d, s in sims.items() if d in corpus_set and d != uid}
        if kept:
            qrels[uid] = kept

    eligible = [q for q, r in qrels.items() if len(r) >= 1]
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    test_ids = eligible[: args.n_queries]
    console.print(f"  Test queries: {len(test_ids)}")

    # Carrega entidades (se disponível)
    entities_by_case: dict[str, dict[str, list[str]]] = {}
    if Path(args.entities).exists():
        with open(args.entities) as f:
            entities_by_case = json.load(f)

    # ------------------------------------------------------------------
    # 3. Métricas por query por sistema
    # ------------------------------------------------------------------
    console.rule("[bold]3. Calculando métricas por query[/]")

    # {query_id: {system_name: recall@10}}
    per_query: dict[str, dict[str, float]] = defaultdict(dict)
    for name, rankings in all_rankings.items():
        for qid in test_ids:
            ranking = rankings.get(qid, [])
            relevant = set(qrels.get(qid, {}).keys())
            per_query[qid][name] = recall_at_k(ranking, relevant, k=10)

    # ------------------------------------------------------------------
    # 4. Visão geral — taxa de falha por sistema
    # ------------------------------------------------------------------
    console.rule("[bold]4. Taxa de falha por sistema (0 acertos em top-10)[/]")
    overview_table = Table(title="Falhas completas por sistema")
    overview_table.add_column("Sistema", style="cyan")
    overview_table.add_column("Falhas (0 acertos)", justify="right")
    overview_table.add_column("% falhas", justify="right")
    overview_table.add_column("Recall@10 médio", justify="right", style="green")

    system_stats = {}
    for name in all_rankings:
        recalls = [per_query[q].get(name, 0.0) for q in test_ids]
        n_fail  = sum(1 for r in recalls if r == 0.0)
        mean_r  = np.mean(recalls)
        system_stats[name] = {"n_fail": n_fail, "mean_recall": mean_r}
        overview_table.add_row(
            name,
            str(n_fail),
            f"{100*n_fail/len(test_ids):.1f}%",
            f"{mean_r:.4f}",
        )
    console.print(overview_table)

    # ------------------------------------------------------------------
    # 5. Queries que NENHUM sistema acerta
    # ------------------------------------------------------------------
    console.rule("[bold]5. Queries impossíveis (nenhum sistema acerta)[/]")
    impossible = [
        q for q in test_ids
        if all(per_query[q].get(n, 0.0) == 0.0 for n in all_rankings)
    ]
    console.print(f"  Queries sem acerto em nenhum sistema: [red]{len(impossible)}[/] "
                  f"({100*len(impossible)/len(test_ids):.1f}%)")

    # Caracteriza as queries impossíveis
    imp_n_pos   = [len(qrels.get(q, {})) for q in impossible]
    imp_lengths = [len(id_to_text.get(q, "").split()) for q in impossible]
    all_lengths = [len(id_to_text.get(q, "").split()) for q in test_ids]

    console.print(f"  Média de positivos (impossíveis): {np.mean(imp_n_pos):.2f} "
                  f"vs {np.mean([len(qrels.get(q,{})) for q in test_ids]):.2f} (todas)")
    console.print(f"  Média de tokens  (impossíveis): {np.mean(imp_lengths):.0f} "
                  f"vs {np.mean(all_lengths):.0f} (todas)")

    # ------------------------------------------------------------------
    # 6. Análise por tamanho de texto
    # ------------------------------------------------------------------
    console.rule("[bold]6. Performance por tamanho de texto[/]")
    best_system = max(system_stats, key=lambda n: system_stats[n]["mean_recall"])
    length_table = Table(title=f"Recall@10 ({best_system}) por tamanho do caso")
    length_table.add_column("Faixa de tokens")
    length_table.add_column("N queries", justify="right")
    length_table.add_column("Recall@10", justify="right", style="green")
    length_table.add_column("% falhas", justify="right", style="red")

    buckets: dict[str, list[float]] = defaultdict(list)
    for qid in test_ids:
        n_tok = len(id_to_text.get(qid, "").split())
        bucket = text_length_bucket(n_tok)
        buckets[bucket].append(per_query[qid].get(best_system, 0.0))

    for bucket in ["curto (<200)", "médio (200-500)", "longo (500-800)", "muito longo (>800)"]:
        vals = buckets.get(bucket, [])
        if vals:
            length_table.add_row(
                bucket,
                str(len(vals)),
                f"{np.mean(vals):.4f}",
                f"{100*sum(1 for v in vals if v==0)/len(vals):.1f}%",
            )
    console.print(length_table)

    # ------------------------------------------------------------------
    # 7. Análise por número de positivos
    # ------------------------------------------------------------------
    console.rule("[bold]7. Performance por número de casos similares (ground truth)[/]")
    pos_table = Table(title=f"Recall@10 ({best_system}) por tamanho do ground truth")
    pos_table.add_column("N° similares")
    pos_table.add_column("N queries", justify="right")
    pos_table.add_column("Recall@10", justify="right", style="green")

    pos_buckets: dict[str, list[float]] = defaultdict(list)
    for qid in test_ids:
        n = len(qrels.get(qid, {}))
        b = "1" if n == 1 else ("2-3" if n <= 3 else ("4-9" if n <= 9 else "10+"))
        pos_buckets[b].append(per_query[qid].get(best_system, 0.0))

    for b in ["1", "2-3", "4-9", "10+"]:
        vals = pos_buckets.get(b, [])
        if vals:
            pos_table.add_row(b, str(len(vals)), f"{np.mean(vals):.4f}")
    console.print(pos_table)

    # ------------------------------------------------------------------
    # 8. Análise por doenças (se entidades disponíveis)
    # ------------------------------------------------------------------
    if entities_by_case:
        console.rule("[bold]8. Performance por diagnóstico (top-15 doenças)[/]")
        disease_recalls: dict[str, list[float]] = defaultdict(list)
        for qid in test_ids:
            diseases = entities_by_case.get(qid, {}).get("disease", [])
            r = per_query[qid].get(best_system, 0.0)
            for d in diseases:
                disease_recalls[d].append(r)

        # Filtra doenças com pelo menos 10 queries
        disease_data = [
            (d, np.mean(vals), len(vals))
            for d, vals in disease_recalls.items()
            if len(vals) >= 10
        ]
        disease_data.sort(key=lambda x: x[1])  # ordena por recall crescente

        disease_table = Table(title=f"Recall@10 por diagnóstico ({best_system})")
        disease_table.add_column("Diagnóstico", style="cyan")
        disease_table.add_column("N queries", justify="right")
        disease_table.add_column("Recall@10", justify="right")

        # 5 piores + 5 melhores
        worst = disease_data[:5]
        best  = disease_data[-5:]
        disease_table.add_row("[bold red]— MAIS DIFÍCEIS —[/]", "", "")
        for d, r, n in worst:
            disease_table.add_row(d, str(n), f"[red]{r:.4f}[/]")
        disease_table.add_row("[bold green]— MAIS FÁCEIS —[/]", "", "")
        for d, r, n in reversed(best):
            disease_table.add_row(d, str(n), f"[green]{r:.4f}[/]")
        console.print(disease_table)

    # ------------------------------------------------------------------
    # 9. Exemplos concretos: falha vs acerto
    # ------------------------------------------------------------------
    console.rule("[bold]9. Exemplos concretos[/]")

    # 3 queries que o melhor sistema acerta mas o baseline falha
    baseline_name = "Dense BGE-M3" if "Dense BGE-M3" in all_rankings else list(all_rankings.keys())[0]
    improved = [
        q for q in test_ids
        if per_query[q].get(baseline_name, 0.0) == 0.0
        and per_query[q].get(best_system, 0.0) > 0.0
    ][:3]

    for qid in improved:
        text_preview = id_to_text.get(qid, "")[:300].replace("\n", " ")
        relevant = set(qrels.get(qid, {}).keys())
        baseline_rank = all_rankings[baseline_name].get(qid, [])
        best_rank     = all_rankings[best_system].get(qid, [])
        best_hit = next((d for d in best_rank if d in relevant), None)

        console.print(Panel(
            f"[bold]Query:[/] {text_preview}...\n\n"
            f"[bold]Ground truth:[/] {len(relevant)} caso(s) similar(es)\n"
            f"[bold]{baseline_name}:[/] FALHOU (0 acertos em top-10)\n"
            f"[bold]{best_system}:[/] ACERTOU — 1º hit: posição "
            f"{best_rank.index(best_hit)+1 if best_hit else '?'}\n"
            f"[bold]Doenças:[/] {entities_by_case.get(qid, {}).get('disease', [])[:5]}",
            title=f"Caso {qid} — melhora com fine-tuning",
            border_style="green",
        ))

    # ------------------------------------------------------------------
    # 10. Salva análise completa em JSON
    # ------------------------------------------------------------------
    output = {
        "n_test_queries": len(test_ids),
        "systems_evaluated": list(all_rankings.keys()),
        "system_stats": {
            name: {
                "n_fail": int(stats["n_fail"]),
                "pct_fail": round(100 * stats["n_fail"] / len(test_ids), 2),
                "mean_recall10": round(stats["mean_recall"], 4),
            }
            for name, stats in system_stats.items()
        },
        "impossible_queries": {
            "count": len(impossible),
            "pct": round(100 * len(impossible) / len(test_ids), 2),
            "mean_n_positives": round(float(np.mean(imp_n_pos)), 2) if imp_n_pos else 0,
            "mean_text_length": round(float(np.mean(imp_lengths)), 1) if imp_lengths else 0,
            "ids": impossible[:50],  # primeiros 50 para análise manual
        },
        "per_query_recall10": {
            qid: {name: round(per_query[qid].get(name, 0.0), 4) for name in all_rankings}
            for qid in test_ids
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))
    console.print(f"\n[green]Análise completa salva em {args.out}[/]")
    console.rule("[bold green]Concluído[/]")


if __name__ == "__main__":
    main()
