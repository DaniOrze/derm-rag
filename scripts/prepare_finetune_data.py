"""
Prepara os dados de fine-tuning do encoder denso.

Divisão:
    Test  : 1.000 queries (seed=42 — imutável, mesmo das avaliações anteriores)
    Val   :   700 queries (seed=43 — sem overlap com test)
    Train : ~5.141 queries restantes

Para cada query de treino:
    Positivos : score==2 prioritariamente; score==1 como fallback
    Hard negatives: top-100 BGE-M3 que NÃO estão no ground truth da query

Formato de saída (JSONL):
    {"query": "...", "pos": ["...", ...], "neg": ["...", ...]}

Compatível com sentence-transformers e FlagEmbedding finetune.

Uso:
    python scripts/prepare_finetune_data.py --config configs/finetune.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.prepare_eval import DataConfig, SplitConfig, prepare_dataset
from src.retrieval.dense_search import search as dense_search

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/finetune.yaml")
    p.add_argument(
        "--positive-min-score", type=int, default=None,
        help="Score mínimo para um caso ser positivo. "
             "Default: usa o valor do config (2 para encoder, 1 para reranker).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    console.rule("[bold]Preparando dados de fine-tuning[/]")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = DataConfig(**cfg["data"])
    out_dir = Path(cfg["finetune"]["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Score mínimo para positivo: argumento > config > padrão 2
    positive_min_score = (
        args.positive_min_score
        if args.positive_min_score is not None
        else cfg["finetune"].get("positive_min_score", 2)
    )
    console.print(f"  Positive min score: {positive_min_score}")

    # ------------------------------------------------------------------
    # 1. Carrega dataset completo para descobrir TODOS os elegíveis
    # ------------------------------------------------------------------
    console.rule("[bold]1. Carregando dataset[/]")
    split_all = SplitConfig(min_similars=1, n_queries=None, seed=42)
    ds_all = prepare_dataset(data_cfg, split_all)

    corpus_ids   = ds_all["corpus_ids"]
    corpus_texts = ds_all["corpus_texts"]
    id_to_text   = dict(zip(corpus_ids, corpus_texts))
    qrels_all    = ds_all["qrels"]           # todos os elegíveis com score>=1
    qrels_strict = ds_all["qrels_strict"]    # todos os elegíveis com score==2

    all_eligible = ds_all["query_ids"]       # já embaralhados com seed=42

    # ------------------------------------------------------------------
    # 2. Divisão train / val / test
    # Reproduz exatamente a seleção do test set (seed=42, primeiros 1000)
    # ------------------------------------------------------------------
    console.rule("[bold]2. Divisão train / val / test[/]")
    n_test = cfg["finetune"]["n_test"]
    n_val  = cfg["finetune"]["n_val"]

    test_ids = set(all_eligible[:n_test])
    remaining = [q for q in all_eligible if q not in test_ids]

    rng = random.Random(cfg["finetune"]["val_seed"])
    rng.shuffle(remaining)
    val_ids   = remaining[:n_val]
    train_ids = remaining[n_val:]

    console.print(f"  Test  : {len(test_ids):>5} queries (seed=42, imutável)")
    console.print(f"  Val   : {len(val_ids):>5} queries (seed={cfg['finetune']['val_seed']})")
    console.print(f"  Train : {len(train_ids):>5} queries")

    # Salva splits para referência
    splits = {
        "test":  sorted(test_ids),
        "val":   sorted(val_ids),
        "train": sorted(train_ids),
    }
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2))
    console.print(f"  Splits salvos em {out_dir}/splits.json")

    # ------------------------------------------------------------------
    # 3. Hard negative mining com BGE-M3
    # ------------------------------------------------------------------
    console.rule("[bold]3. Hard negative mining (BGE-M3)[/]")
    emb_cfg = cfg["embedding"]
    emb_path = Path(emb_cfg["cache_path"])

    if not emb_path.exists():
        console.print(f"[red]Embeddings não encontrados em {emb_path}. "
                      "Execute scripts/run_retrieval.py primeiro.[/]")
        sys.exit(1)

    corpus_emb = np.load(emb_path)
    console.print(f"  Embeddings carregados: {corpus_emb.shape}")

    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    train_idx = [id_to_idx[q] for q in train_ids]
    train_emb = corpus_emb[train_idx]

    n_hard = cfg["finetune"]["n_hard_negatives"]
    top_k_mining = max(100, n_hard + 20)

    console.print(f"  Minerando top-{top_k_mining} para {len(train_ids)} queries de treino...")
    hard_neg_rankings = dense_search(
        train_emb, corpus_emb, corpus_ids,
        top_k=top_k_mining, query_ids=train_ids, exclude_self=True,
    )

    # ------------------------------------------------------------------
    # 4. Gera JSONL de treino
    # ------------------------------------------------------------------
    console.rule("[bold]4. Gerando JSONL de treino[/]")
    train_examples = []
    n_strict_only = 0
    n_fallback = 0
    n_skipped = 0

    for qid in train_ids:
        qtext = id_to_text[qid]

        # Positivos: depende de positive_min_score
        if positive_min_score >= 2:
            # score==2 prioritário, score==1 como fallback
            if qid in qrels_strict and qrels_strict[qid]:
                pos_ids = list(qrels_strict[qid].keys())
                n_strict_only += 1
            else:
                pos_ids = list(qrels_all[qid].keys())
                n_fallback += 1
        else:
            # score>=1: todos os similares são positivos
            pos_ids = list(qrels_all[qid].keys())
            n_fallback += 1

        pos_texts = [id_to_text[d] for d in pos_ids if d in id_to_text]
        if not pos_texts:
            n_skipped += 1
            continue

        # Hard negatives: top-k retrieval excluindo positivos
        pos_set = set(pos_ids)
        hard_negs = [
            d for d in hard_neg_rankings.get(qid, [])
            if d not in pos_set
        ][:n_hard]
        neg_texts = [id_to_text[d] for d in hard_negs if d in id_to_text]

        train_examples.append({
            "query": qtext,
            "pos":   pos_texts,
            "neg":   neg_texts,
        })

    train_path = out_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    console.print(f"  Exemplos gerados: {len(train_examples)}")
    console.print(f"    Com score==2:     {n_strict_only}")
    console.print(f"    Fallback score=1: {n_fallback}")
    console.print(f"    Pulados (sem texto): {n_skipped}")

    # ------------------------------------------------------------------
    # 5. Gera JSONL de validação (sem hard negatives — só pairs)
    # ------------------------------------------------------------------
    console.rule("[bold]5. Gerando JSONL de validação[/]")
    val_examples = []
    for qid in val_ids:
        qtext = id_to_text[qid]
        if qid in qrels_strict and qrels_strict[qid]:
            pos_ids = list(qrels_strict[qid].keys())
        else:
            pos_ids = list(qrels_all[qid].keys())
        pos_texts = [id_to_text[d] for d in pos_ids if d in id_to_text]
        if pos_texts:
            val_examples.append({"query": qtext, "pos": pos_texts, "neg": []})

    val_path = out_dir / "val.jsonl"
    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")
    console.print(f"  Exemplos de val: {len(val_examples)}")

    # ------------------------------------------------------------------
    # Estatísticas finais
    # ------------------------------------------------------------------
    table = Table(title="Resumo dos dados de fine-tuning")
    table.add_column("Split")
    table.add_column("Queries", justify="right")
    table.add_column("Exemplos", justify="right")
    table.add_column("Arquivo")
    table.add_row("Train", str(len(train_ids)), str(len(train_examples)), str(train_path))
    table.add_row("Val",   str(len(val_ids)),   str(len(val_examples)),   str(val_path))
    table.add_row("Test",  str(len(test_ids)),  "—",  "configs/dense_baseline.yaml (seed=42)")
    console.print(table)
    console.rule("[bold green]Concluído[/]")


if __name__ == "__main__":
    main()
