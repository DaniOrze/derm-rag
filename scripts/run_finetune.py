"""
Fine-tuning do encoder denso BGE-M3 para similaridade de casos dermatológicos.
Usa a API sentence-transformers v3+ (SentenceTransformerTrainer).

Loss: MultipleNegativesRankingLoss com triplets (query, pos, hard_neg)

Uso:
    python scripts/run_finetune.py --config configs/finetune.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

# Seleção de GPU: lê do config ANTES de qualquer import torch/CUDA.
# Se CUDA_VISIBLE_DEVICES já estiver no ambiente (ex: via linha de comando
# ou SLURM), respeita o valor existente sem sobrescrever.
def _set_gpu_from_config() -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    try:
        _args, _ = argparse.ArgumentParser(add_help=False).parse_known_args()
        cfg_path = None
        for i, a in enumerate(sys.argv):
            if a == "--config" and i + 1 < len(sys.argv):
                cfg_path = sys.argv[i + 1]
        if cfg_path:
            with open(cfg_path) as f:
                _cfg = yaml.safe_load(f)
            gpu = str(_cfg.get("training", {}).get("gpu", "7"))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    except Exception:
        pass

_set_gpu_from_config()

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/finetune.yaml")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_triplet_dataset(examples: list[dict]):
    """Converte JSONL em Dataset HuggingFace com triplets (anchor, positive, negative)."""
    from datasets import Dataset

    anchors, positives, negatives = [], [], []
    for ex in examples:
        q = ex["query"]
        negs = ex.get("neg", [])
        for pos in ex["pos"]:
            anchors.append(q)
            positives.append(pos)
            # Se não há hard negative, repete um positivo diferente ou string vazia
            neg = negs[0] if negs else (ex["pos"][1] if len(ex["pos"]) > 1 else "")
            negatives.append(neg)

    return Dataset.from_dict({
        "anchor":   anchors,
        "positive": positives,
        "negative": negatives,
    })


def build_val_evaluator(val_data: list[dict], batch_size: int):
    """Cria InformationRetrievalEvaluator para monitorar Recall@10 no val."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    queries, corpus, relevant = {}, {}, {}
    for i, ex in enumerate(val_data):
        qid = f"q{i}"
        queries[qid] = ex["query"]
        relevant[qid] = set()
        for j, pos in enumerate(ex["pos"]):
            did = f"d{i}_{j}"
            corpus[did] = pos
            relevant[qid].add(did)

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        batch_size=batch_size,
        show_progress_bar=False,
        name="val",
    )


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    console.print(f"[bold]GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'não definido')}[/]")
    console.rule("[bold]Fine-tuning BGE-M3 (sentence-transformers v3)[/]")

    ft_cfg = cfg["finetune"]
    tr_cfg = cfg["training"]
    data_dir = Path(ft_cfg["data_dir"])
    out_dir  = Path(ft_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    console.rule("[bold]1. Carregando dados[/]")
    train_data = load_jsonl(data_dir / "train.jsonl")
    val_data   = load_jsonl(data_dir / "val.jsonl")
    console.print(f"  Train: {len(train_data)} exemplos")
    console.print(f"  Val:   {len(val_data)} exemplos")

    # ------------------------------------------------------------------
    console.rule("[bold]2. Carregando modelo base[/]")
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    from sentence_transformers.trainer import SentenceTransformerTrainer

    model = SentenceTransformer(
        cfg["embedding"]["model_name"],
        device=tr_cfg["device"],
    )
    model.max_seq_length = cfg["embedding"]["max_length"]
    console.print(f"  Modelo: {cfg['embedding']['model_name']}")
    console.print(f"  max_seq_length: {model.max_seq_length}")

    # ------------------------------------------------------------------
    console.rule("[bold]3. Preparando datasets[/]")
    train_dataset = build_triplet_dataset(train_data)
    console.print(f"  Triplets de treino: {len(train_dataset)}")

    # ------------------------------------------------------------------
    console.rule("[bold]4. Configurando avaliador de validação[/]")
    evaluator = build_val_evaluator(val_data, tr_cfg["batch_size"])

    # ------------------------------------------------------------------
    console.rule("[bold]5. Configurando treino[/]")
    loss = MultipleNegativesRankingLoss(model)

    epochs       = tr_cfg["epochs"]
    batch_size   = tr_cfg["batch_size"]
    eval_steps   = tr_cfg["eval_steps"]
    n_steps      = (len(train_dataset) // batch_size) * epochs
    warmup_steps = int(n_steps * tr_cfg["warmup_ratio"])

    console.print(f"  Epochs:       {epochs}")
    console.print(f"  Batch size:   {batch_size}")
    console.print(f"  Steps total:  {n_steps}")
    console.print(f"  Warmup steps: {warmup_steps}")
    console.print(f"  Output:       {out_dir}")

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=tr_cfg.get("grad_accum", 1),
        learning_rate=float(tr_cfg["learning_rate"]),
        warmup_steps=warmup_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="val_cosine_ndcg@10",
        greater_is_better=True,
        fp16=tr_cfg.get("use_amp", True),
        gradient_checkpointing=tr_cfg.get("gradient_checkpointing", False),
        logging_steps=50,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    # ------------------------------------------------------------------
    console.rule("[bold]6. Treinando[/]")
    trainer.train()

    # ------------------------------------------------------------------
    console.rule("[bold]7. Salvando modelo final[/]")
    model.save_pretrained(str(out_dir))
    console.print(f"[green]Modelo salvo em {out_dir}[/]")
    console.print(
        "\n[bold]Próximo passo:[/]\n"
        "  python scripts/run_retrieval.py --config configs/dense_finetuned.yaml\n"
        "  python scripts/run_hybrid.py    --config configs/hybrid_finetuned.yaml"
    )
    console.rule("[bold green]Concluído[/]")


if __name__ == "__main__":
    main()
