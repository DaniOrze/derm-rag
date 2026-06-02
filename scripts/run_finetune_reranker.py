"""
Fine-tuning do reranker (cross-encoder) para similaridade de casos dermatológicos.

O BGE-reranker-v2-m3 original nunca viu pares de casos clínicos — por isso
prejudica o ranking quando usado como ranqueador final. Após fine-tuning:
  - Pode ser usado no ranking final do CRAG (desbloqueando o pipeline completo)
  - Scores mais calibrados para o domínio dermatológico

Dados:
  Positivos: pares (query, caso_similar) com score==2
  Negativos: top-k do fine-tuned encoder que NÃO são relevantes (hard negatives)
  Proporção: 1 positivo : N negativos por query

Loss: BinaryCrossEntropy (positivo=1, negativo=0)

Uso:
    python scripts/run_finetune_reranker.py --config configs/finetune_reranker.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/finetune_reranker.yaml")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_reranker_dataset(examples: list[dict], max_neg_per_query: int = 5):
    """Constrói dataset de pares para cross-encoder.

    Cada exemplo gera:
      - 1 par positivo  (query, pos, label=1.0)
      - N pares negativos (query, neg, label=0.0)
    """
    from datasets import Dataset

    queries, docs, labels = [], [], []
    for ex in examples:
        q    = ex["query"]
        negs = ex.get("neg", [])[:max_neg_per_query]
        for pos in ex["pos"]:
            queries.append(q)
            docs.append(pos)
            labels.append(1.0)
        for neg in negs:
            queries.append(q)
            docs.append(neg)
            labels.append(0.0)

    return Dataset.from_dict({"query": queries, "doc": docs, "label": labels})


def main():
    args = parse_args()
    console.rule("[bold]Fine-tuning Reranker (cross-encoder)[/]")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    import os
    gpu = str(cfg["training"].get("gpu", "7"))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    console.print(f"[bold]GPU: {gpu}[/]")

    tr_cfg  = cfg["training"]
    ft_cfg  = cfg["finetune"]
    data_dir = Path(ft_cfg["data_dir"])
    out_dir  = Path(ft_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    console.rule("[bold]1. Carregando dados[/]")
    # Reutiliza os dados de treino já preparados (com hard negatives)
    train_data = load_jsonl(data_dir / "train.jsonl")
    val_data   = load_jsonl(data_dir / "val.jsonl")
    console.print(f"  Train: {len(train_data)} exemplos")
    console.print(f"  Val:   {len(val_data)} exemplos")

    # ------------------------------------------------------------------
    console.rule("[bold]2. Carregando cross-encoder base[/]")
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset

    model_name = cfg["reranker"]["model_name"]
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    )
    console.print(f"  Modelo: {model_name}")

    # ------------------------------------------------------------------
    console.rule("[bold]3. Preparando dataset[/]")
    max_neg = cfg["reranker"].get("max_neg_per_query", 5)
    train_ds = build_reranker_dataset(train_data, max_neg)
    val_ds   = build_reranker_dataset(val_data[:200], max_neg)  # val menor para velocidade
    console.print(f"  Pares treino: {len(train_ds)} (pos + neg)")
    console.print(f"  Pares val:    {len(val_ds)}")

    max_len = cfg["reranker"]["max_length"]

    def tokenize(batch):
        return tokenizer(
            batch["query"], batch["doc"],
            padding="max_length", truncation=True,
            max_length=max_len,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["query", "doc"])
    val_ds   = val_ds.map(tokenize,   batched=True, remove_columns=["query", "doc"])
    train_ds = train_ds.rename_column("label", "labels")
    val_ds   = val_ds.rename_column("label",   "labels")
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # ------------------------------------------------------------------
    console.rule("[bold]4. Configurando treino[/]")

    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        preds  = (logits.squeeze() > 0).astype(int)
        labels = labels.astype(int)
        acc    = (preds == labels).mean()
        return {"accuracy": acc}

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=tr_cfg["epochs"],
        per_device_train_batch_size=tr_cfg["batch_size"],
        per_device_eval_batch_size=tr_cfg["batch_size"] * 2,
        learning_rate=float(tr_cfg["learning_rate"]),
        warmup_ratio=tr_cfg["warmup_ratio"],
        eval_strategy="steps",
        eval_steps=tr_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=tr_cfg["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=tr_cfg.get("use_amp", True),
        logging_steps=50,
        report_to="none",
    )

    def loss_fn(logits, labels):
        return torch.nn.BCEWithLogitsLoss()(logits.view(-1), labels.float())

    class RerankerTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels  = inputs.pop("labels")
            outputs = model(**inputs)
            loss    = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # ------------------------------------------------------------------
    console.rule("[bold]5. Treinando[/]")
    trainer.train()

    # ------------------------------------------------------------------
    console.rule("[bold]6. Salvando[/]")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    console.print(f"[green]Reranker salvo em {out_dir}[/]")
    console.print(
        "\n[bold]Próximo passo:[/]\n"
        "  Atualize reranker.model_name nos configs de hybrid/crag para usar o modelo fine-tunado."
    )
    console.rule("[bold green]Concluído[/]")


if __name__ == "__main__":
    main()
