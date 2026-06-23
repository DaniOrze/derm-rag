#!/usr/bin/env python3
"""
RAGAS-style evaluation using Claude (Anthropic API) as LLM judge.

Follows the RAGAS framework (Shahul Es et al., 2023) for RAG evaluation:
  - Answer Relevancy    (all modes): how well the answer addresses the clinical presentation
  - Faithfulness        (RAG only):  fraction of answer claims supported by retrieved contexts
  - Context Recall      (RAG only):  coverage of ground-truth info in retrieved contexts
  - Context Precision   (RAG only):  precision@k of retrieved context relevance

Requires:
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python scripts/evaluate_ragas.py \\
        --results results/rag_gen_qwen3_8b_20260616_201755.json \\
        [--data data/processed/derm_cases.csv] \\
        [--n-samples 200] \\
        [--model claude-haiku-4-5] \\
        [--seed 42] \\
        [--output-dir results]

Cost estimate (claude-haiku-4-5, 200 RAG samples):
    ~4 calls × 200 × ~$0.002 ≈ $1.60
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import anthropic


# ─── LLM-judge prompts ───────────────────────────────────────────────────────

_FAITHFULNESS_SYS = (
    "You are a clinical expert evaluating whether a medical AI's answer faithfully "
    "reflects the retrieved case evidence. Respond with a single JSON object only — "
    "no preamble, no explanation outside the JSON."
)

_FAITHFULNESS_USR = """\
## Retrieved Clinical Cases (evidence)
{contexts}

## AI-Generated Diagnostic Response
{answer}

Extract the main factual claims from the AI response (diagnoses proposed, clinical \
features cited, management steps). For each claim, judge if it is directly supported \
by the retrieved cases above.

Respond with exactly:
{{"claims": ["claim 1", "claim 2"], "supported": [true, false], "score": <float 0-1>}}"""


_ANSWER_REL_SYS = (
    "You are a senior dermatologist rating the clinical relevance of an AI-generated "
    "diagnostic response. Respond with a single JSON object only — "
    "no preamble, no explanation outside the JSON."
)

_ANSWER_REL_USR = """\
## Clinical Presentation
{question}

## AI-Generated Diagnostic Response
{answer}

Rate how well the response addresses this specific clinical presentation (score 0–1):
  1.0 → Excellent: correct diagnostic direction, appropriate differential, key features cited
  0.7 → Good: mostly relevant with minor gaps
  0.4 → Partial: addresses some elements, misses key clinical aspects
  0.1 → Poor: mostly off-target or wrong direction
  0.0 → Irrelevant

Respond with exactly:
{{"strengths": ["strength 1"], "gaps": ["gap 1"], "score": <float 0-1>}}"""


_CTX_RECALL_SYS = (
    "You are a senior dermatologist evaluating whether retrieved clinical cases "
    "contain sufficient information to support the correct diagnosis. "
    "Respond with a single JSON object only — no preamble, no explanation outside the JSON."
)

_CTX_RECALL_USR = """\
## Clinical Presentation
{question}

## Correct Diagnosis (ground truth)
{ground_truth}

## Retrieved Clinical Cases
{contexts}

List the key diagnostic features needed to identify the ground-truth diagnosis. \
For each feature, indicate whether it is present in the retrieved cases.

Respond with exactly:
{{"key_features": ["feature 1", "feature 2"], "found": [true, false], "score": <float 0-1>}}"""


_CTX_PREC_SYS = (
    "You are a senior dermatologist judging whether each retrieved clinical case "
    "is helpful for reaching the correct diagnosis. "
    "Respond with a single JSON object only — no preamble, no explanation outside the JSON."
)

_CTX_PREC_USR = """\
## Clinical Presentation
{question}

## Correct Diagnosis (ground truth)
{ground_truth}

## Retrieved Cases (numbered)
{cases}

For each case (in order), judge whether it is relevant / helpful for identifying \
the ground-truth diagnosis.

Respond with exactly:
{{"relevant": [true, false, true], "score": <float 0-1>}}"""


# ─── Result container ────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    query_id: str
    is_rag: bool
    answer_relevancy: Optional[float] = None
    faithfulness: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None
    skipped: bool = False


# ─── Utility functions ───────────────────────────────────────────────────────

def truncate(text: str, max_chars: int = 1500) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def extract_json(raw: str) -> Optional[dict]:
    """Parse JSON from a Claude response, stripping markdown fences."""
    text = raw.strip()
    for fence in ("```json\n", "```json", "```\n", "```"):
        if fence in text:
            idx = text.find(fence) + len(fence)
            end = text.find("```", idx)
            if end != -1:
                text = text[idx:end].strip()
                break
    start = text.find("{")
    end   = text.rfind("}") + 1
    if 0 <= start < end:
        text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def call_claude(
    client: anthropic.Anthropic,
    system: str,
    user: str,
    model: str,
    max_retries: int = 3,
) -> Optional[dict]:
    """Call Claude with exponential backoff on rate-limit and server errors."""
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            parsed = extract_json(resp.content[0].text)
            if parsed is None:
                print(f"    [warn] JSON parse failed (attempt {attempt+1})")
            return parsed
        except anthropic.RateLimitError:
            delay = 5.0 * (2 ** attempt)
            print(f"    [rate-limit] sleeping {delay:.0f}s…", flush=True)
            time.sleep(delay)
        except anthropic.APIError as exc:
            delay = 2.0 * (2 ** attempt)
            print(f"    [API error attempt {attempt+1}] {exc}", flush=True)
            time.sleep(delay)
        except (IndexError, AttributeError) as exc:
            print(f"    [parse error attempt {attempt+1}] {exc}", flush=True)
    return None


# ─── Metric evaluators ───────────────────────────────────────────────────────

def _eval_faithfulness(
    client: anthropic.Anthropic,
    model: str,
    answer: str,
    contexts: list[str],
) -> Optional[float]:
    ctx = "\n\n".join(
        f"**Case {i+1}:**\n{truncate(c, 600)}" for i, c in enumerate(contexts)
    )
    result = call_claude(
        client, _FAITHFULNESS_SYS,
        _FAITHFULNESS_USR.format(contexts=ctx, answer=truncate(answer, 900)),
        model,
    )
    if result is None:
        return None
    if "score" in result:
        return max(0.0, min(1.0, float(result["score"])))
    claims    = result.get("claims", [])
    supported = result.get("supported", [])
    if claims and len(claims) == len(supported):
        return float(sum(supported)) / len(supported)
    return None


def _eval_answer_relevancy(
    client: anthropic.Anthropic,
    model: str,
    question: str,
    answer: str,
) -> Optional[float]:
    result = call_claude(
        client, _ANSWER_REL_SYS,
        _ANSWER_REL_USR.format(
            question=truncate(question, 700),
            answer=truncate(answer, 900),
        ),
        model,
    )
    if result is None:
        return None
    if "score" in result:
        return max(0.0, min(1.0, float(result["score"])))
    return None


def _eval_context_recall(
    client: anthropic.Anthropic,
    model: str,
    question: str,
    ground_truth: str,
    contexts: list[str],
) -> Optional[float]:
    ctx = "\n\n".join(
        f"**Case {i+1}:**\n{truncate(c, 600)}" for i, c in enumerate(contexts)
    )
    result = call_claude(
        client, _CTX_RECALL_SYS,
        _CTX_RECALL_USR.format(
            question=truncate(question, 500),
            ground_truth=ground_truth,
            contexts=ctx,
        ),
        model,
    )
    if result is None:
        return None
    if "score" in result:
        return max(0.0, min(1.0, float(result["score"])))
    features = result.get("key_features", [])
    found    = result.get("found", [])
    if features and len(features) == len(found):
        return float(sum(found)) / len(found)
    return None


def _eval_context_precision(
    client: anthropic.Anthropic,
    model: str,
    question: str,
    ground_truth: str,
    contexts: list[str],
) -> Optional[float]:
    cases = "\n\n".join(
        f"**Case {i+1}:**\n{truncate(c, 600)}" for i, c in enumerate(contexts)
    )
    result = call_claude(
        client, _CTX_PREC_SYS,
        _CTX_PREC_USR.format(
            question=truncate(question, 500),
            ground_truth=ground_truth,
            cases=cases,
        ),
        model,
    )
    if result is None:
        return None
    if "score" in result:
        return max(0.0, min(1.0, float(result["score"])))
    relevant = result.get("relevant", [])
    if relevant and sum(relevant) > 0:
        # Standard RAGAS precision@k formula: weighted average of precision at each rank
        prec_at_k = sum(
            (sum(relevant[:i+1]) / (i+1)) * int(relevant[i])
            for i in range(len(relevant))
        ) / sum(relevant)
        return max(0.0, min(1.0, prec_at_k))
    return 0.0 if relevant else None


# ─── Per-sample evaluation ───────────────────────────────────────────────────

def evaluate_sample(
    client: anthropic.Anthropic,
    model: str,
    item: dict,
    id_to_text: dict[str, str],
) -> SampleResult:
    qid           = item["query_id"]
    question      = item.get("query_presentation", "")
    answer        = item.get("response", "")
    ground_truth  = item.get("ground_truth_title", "")
    retrieved_ids = item.get("retrieved_ids", [])
    is_rag        = bool(retrieved_ids)

    contexts = [id_to_text[rid] for rid in retrieved_ids if rid in id_to_text]
    res = SampleResult(query_id=qid, is_rag=is_rag)

    # 1. Answer Relevancy — computed for all modes
    res.answer_relevancy = _eval_answer_relevancy(client, model, question, answer)

    # 2–4. Context-dependent metrics — RAG only
    if is_rag and contexts:
        time.sleep(0.2)
        res.faithfulness = _eval_faithfulness(client, model, answer, contexts)
        time.sleep(0.2)
        res.context_recall = _eval_context_recall(
            client, model, question, ground_truth, contexts
        )
        time.sleep(0.2)
        res.context_precision = _eval_context_precision(
            client, model, question, ground_truth, contexts
        )

    return res


# ─── Summary statistics ──────────────────────────────────────────────────────

def compute_stats(results: list[SampleResult]) -> dict:
    def avg(vals: list) -> Optional[float]:
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 4) if v else None

    valid     = [r for r in results if not r.skipped]
    rag       = [r for r in valid if r.is_rag]
    norag     = [r for r in valid if not r.is_rag]

    return {
        "n_total":            len(results),
        "n_rag":              len(rag),
        "n_norag":            len(norag),
        "n_skipped":          sum(1 for r in results if r.skipped),
        "answer_relevancy":   avg([r.answer_relevancy   for r in valid]),
        "faithfulness":       avg([r.faithfulness       for r in rag]),
        "context_recall":     avg([r.context_recall     for r in rag]),
        "context_precision":  avg([r.context_precision  for r in rag]),
    }


# ─── Console output ──────────────────────────────────────────────────────────

def print_table(stats: dict, run_name: str, model: str) -> None:
    W = 58
    print()
    print("=" * W)
    print(f"  RAGAS Evaluation — {run_name}")
    print(f"  Judge model  : {model}")
    print(
        f"  Samples      : {stats['n_total']} "
        f"(RAG={stats['n_rag']}, NoRAG={stats['n_norag']}, "
        f"skipped={stats['n_skipped']})"
    )
    print("=" * W)

    rows = [
        ("Answer Relevancy",  stats["answer_relevancy"],  "all"),
        ("Faithfulness",      stats["faithfulness"],      "RAG only"),
        ("Context Recall",    stats["context_recall"],    "RAG only"),
        ("Context Precision", stats["context_precision"], "RAG only"),
    ]
    for name, val, scope in rows:
        if val is None:
            print(f"  {name:<22}  N/A ({scope})")
        else:
            bar_len = int(val * 24)
            bar = "█" * bar_len + "░" * (24 - bar_len)
            print(f"  {name:<22}  {val:.4f}  [{bar}]  ({scope})")

    print("=" * W)
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="RAGAS-style RAG evaluation using Claude as LLM judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--results",    required=True,
                   help="Path to generation results JSON")
    p.add_argument("--data",       default="data/processed/derm_cases.csv",
                   help="Path to derm_cases.csv (context lookup)")
    p.add_argument("--n-samples",  type=int, default=200,
                   help="Queries to evaluate (default: 200)")
    p.add_argument("--model",      default="claude-haiku-4-5",
                   help="Anthropic judge model (default: claude-haiku-4-5)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output-dir", default="results")
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY not set")

    # ── Load generation results
    results_path = Path(args.results)
    if not results_path.exists():
        sys.exit(f"File not found: {results_path}")
    with open(results_path) as f:
        generations: list[dict] = json.load(f)
    print(f"Loaded {len(generations)} generations  ←  {results_path.name}")

    # ── Load CSV for context lookup
    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"CSV not found: {data_path}")
    df = pd.read_csv(data_path)
    id_to_text: dict[str, str] = dict(
        zip(df["patient_uid"].astype(str), df["patient"].astype(str))
    )
    print(f"Loaded {len(id_to_text)} cases from CSV")

    # ── Sample
    rng    = random.Random(args.seed)
    sample = list(generations)
    if args.n_samples < len(sample):
        sample = rng.sample(sample, args.n_samples)

    n_rag   = sum(1 for s in sample if s.get("retrieved_ids"))
    n_norag = len(sample) - n_rag
    print(f"Sample: {len(sample)} queries  (RAG={n_rag}, NoRAG={n_norag})")

    # ── Cost estimate (claude-haiku-4-5 rates: $1/$5 per 1M in/out)
    total_calls = n_rag * 4 + n_norag * 1
    cost_est    = total_calls * (800e-6 * 1.0 + 200e-6 * 5.0)
    print(f"Judge: {args.model}  |  Est. calls: {total_calls}  |  Est. cost: ~${cost_est:.2f}\n")

    # ── Evaluate
    client = anthropic.Anthropic(api_key=api_key)
    eval_results: list[SampleResult] = []

    for i, item in enumerate(sample, 1):
        qid  = item.get("query_id", "?")
        mode = "RAG" if item.get("retrieved_ids") else "NoRAG"
        print(f"[{i:4d}/{len(sample)}] {qid[:28]:<28} ({mode})", end=" ", flush=True)

        try:
            res = evaluate_sample(client, args.model, item, id_to_text)
            eval_results.append(res)
            parts = []
            if res.answer_relevancy  is not None: parts.append(f"AR={res.answer_relevancy:.2f}")
            if res.faithfulness      is not None: parts.append(f" F={res.faithfulness:.2f}")
            if res.context_recall    is not None: parts.append(f"CR={res.context_recall:.2f}")
            if res.context_precision is not None: parts.append(f"CP={res.context_precision:.2f}")
            print("  " + "  ".join(parts) if parts else "  (all None)")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            eval_results.append(
                SampleResult(
                    query_id=qid,
                    is_rag=bool(item.get("retrieved_ids")),
                    skipped=True,
                )
            )

        # Small inter-sample delay to stay within rate limits
        time.sleep(0.3)

    # ── Summary
    stats    = compute_stats(eval_results)
    run_name = results_path.stem
    print_table(stats, run_name, args.model)

    # ── Save
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{run_name}_ragas_{ts}.json"

    payload = {
        "run_name":    run_name,
        "judge_model": args.model,
        "n_samples":   len(eval_results),
        "seed":        args.seed,
        "timestamp":   ts,
        "summary":     stats,
        "samples":     [asdict(r) for r in eval_results],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
