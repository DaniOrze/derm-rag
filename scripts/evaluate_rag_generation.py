"""
Avaliação da geração RAG.

Métricas computadas:
  1. Entity Hit Rate   — fração de queries onde ≥1 doença correta aparece na resposta
  2. Top-1 Hit Rate    — doença correta aparece na seção "Most Likely Diagnosis" da resposta
  3. BERTScore F1      — similaridade semântica resposta vs título (ground truth)
  4. Comprimento médio — palavras por resposta

Uso:
    python scripts/evaluate_rag_generation.py results/rag_gen_qwen3_8b_<ts>.json
    python scripts/evaluate_rag_generation.py results/rag_gen_qwen3_8b_<ts>.json --no-bertscore
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_top1_section(response: str) -> str:
    """Extrai só a seção 'Most Likely Diagnosis' da resposta estruturada."""
    patterns = [
        r"(?:Most Likely Diagnosis|Diagnóstico mais provável)[^\n]*\n(.*?)(?=\n#{1,3}|\n\d+\.|\Z)",
        r"\*\*Most Likely Diagnosis\*\*[^\n]*\n(.*?)(?=\n\*\*|\n#{1,3}|\Z)",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()[:500]
    # Fallback: primeiras 3 linhas
    return "\n".join(response.split("\n")[:3])


def entity_in_text(entities: list[str], text: str) -> bool:
    text_lower = text.lower()
    return any(e.lower() in text_lower for e in entities if len(e) > 2)


def compute_bert_score(responses: list[str], references: list[str]) -> dict | None:
    try:
        import torch
        from bert_score import score as bs_score
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        P, R, F1 = bs_score(responses, references, lang="en", verbose=True,
                             batch_size=32, device=dev)
        return {
            "precision": float(P.mean()),
            "recall":    float(R.mean()),
            "f1":        float(F1.mean()),
        }
    except ImportError:
        console.print("[yellow]bert_score não instalado (pip install bert-score). Pulando.[/]")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results_file", help="JSON com resultados da geração")
    p.add_argument("--entities",     default="data/processed/entities.json")
    p.add_argument("--no-bertscore", action="store_true")
    p.add_argument("--out",          default=None)
    args = p.parse_args()

    # ------------------------------------------------------------------
    # 1. Carrega geração
    # ------------------------------------------------------------------
    with open(args.results_file) as f:
        results: list[dict] = json.load(f)
    console.print(f"Resultados: {len(results)} queries de [cyan]{args.results_file}[/]")

    # ------------------------------------------------------------------
    # 2. Entidades
    # ------------------------------------------------------------------
    entities_by_case: dict[str, dict[str, list[str]]] = {}
    if Path(args.entities).exists():
        with open(args.entities) as f:
            entities_by_case = json.load(f)
        console.print(f"Entidades: {len(entities_by_case)} casos")
    else:
        console.print("[yellow]entities.json não encontrado — entity metrics desabilitadas[/]")

    # ------------------------------------------------------------------
    # 3. Métricas por query
    # ------------------------------------------------------------------
    per_query = []
    for item in results:
        qid      = item["query_id"]
        response = item.get("response", "")
        title    = item.get("ground_truth_title", "")
        top1_sec = extract_top1_section(response)

        diseases = entities_by_case.get(qid, {}).get("disease", [])

        entity_hit = entity_in_text(diseases, response)  if diseases else None
        top1_hit   = entity_in_text(diseases, top1_sec)  if diseases else None

        per_query.append({
            "query_id":    qid,
            "title":       title,
            "diseases":    diseases,
            "entity_hit":  entity_hit,
            "top1_hit":    top1_hit,
            "response_len": len(response.split()),
        })

    # ------------------------------------------------------------------
    # 4. Agrega
    # ------------------------------------------------------------------
    valid_e  = [x for x in per_query if x["entity_hit"]  is not None]
    valid_t  = [x for x in per_query if x["top1_hit"]    is not None]
    no_ents  = sum(1 for x in per_query if x["entity_hit"] is None)

    entity_hit_rate = float(np.mean([x["entity_hit"] for x in valid_e])) if valid_e else None
    top1_hit_rate   = float(np.mean([x["top1_hit"]   for x in valid_t])) if valid_t else None
    avg_len         = float(np.mean([x["response_len"] for x in per_query]))

    # ------------------------------------------------------------------
    # 5. BERTScore
    # ------------------------------------------------------------------
    bert = None
    if not args.no_bertscore:
        responses  = [r["response"]           for r in results]
        references = [r["ground_truth_title"]  for r in results]
        bert = compute_bert_score(responses, references)

    # ------------------------------------------------------------------
    # 6. Tabela de resultados
    # ------------------------------------------------------------------
    console.rule("[bold]Avaliação da Geração RAG[/]")
    tbl = Table(title=Path(args.results_file).name)
    tbl.add_column("Métrica",     style="cyan")
    tbl.add_column("Valor",       style="green", justify="right")
    tbl.add_column("N válido",    justify="right")

    if entity_hit_rate is not None:
        tbl.add_row("Entity Hit Rate (any@full response)",
                    f"{entity_hit_rate:.4f} ({entity_hit_rate*100:.1f}%)", str(len(valid_e)))
    if top1_hit_rate is not None:
        tbl.add_row("Top-1 Hit Rate (any@top1 section)",
                    f"{top1_hit_rate:.4f} ({top1_hit_rate*100:.1f}%)",   str(len(valid_t)))

    tbl.add_row("Queries sem entidades mapeadas", str(no_ents),    str(len(results)))
    tbl.add_row("Comprimento médio (palavras)",   f"{avg_len:.0f}", str(len(results)))

    if bert:
        tbl.add_row("BERTScore Precision", f"{bert['precision']:.4f}", str(len(results)))
        tbl.add_row("BERTScore Recall",    f"{bert['recall']:.4f}",    str(len(results)))
        tbl.add_row("BERTScore F1",        f"{bert['f1']:.4f}",        str(len(results)))

    console.print(tbl)

    # ------------------------------------------------------------------
    # 7. Exemplos acertos / erros
    # ------------------------------------------------------------------
    hits   = [x for x in per_query if x["entity_hit"] is True ][:3]
    misses = [x for x in per_query if x["entity_hit"] is False][:3]

    if hits:
        console.rule("[green]Acertos (doença correta encontrada na resposta)[/]")
        for ex in hits:
            console.print(f"  [green]✓[/] {ex['title']}")
            console.print(f"    Entidades: {ex['diseases']}")

    if misses:
        console.rule("[red]Falhas (doença correta ausente na resposta)[/]")
        for ex in misses:
            console.print(f"  [red]✗[/] {ex['title']}")
            console.print(f"    Entidades esperadas: {ex['diseases']}")

    # ------------------------------------------------------------------
    # 8. Salva
    # ------------------------------------------------------------------
    out_path = args.out or args.results_file.replace(".json", "_eval.json")
    out_data = {
        "source_file":       args.results_file,
        "n_queries":         len(results),
        "entity_hit_rate":   entity_hit_rate,
        "top1_hit_rate":     top1_hit_rate,
        "n_valid_entities":  len(valid_e),
        "n_no_entities":     no_ents,
        "avg_response_len":  avg_len,
        "bert_score":        bert,
        "per_query":         per_query,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    console.print(f"\nAvaliação salva em [cyan]{out_path}[/]")


if __name__ == "__main__":
    main()
