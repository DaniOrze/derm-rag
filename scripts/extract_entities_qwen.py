"""
Extração de entidades médicas via Qwen3-8B local — sem custo de API.

Usa o mesmo modelo já em cache no servidor (Qwen/Qwen3-8B) com greedy
decoding e batch processing para extrair entidades estruturadas em JSON.

Estimativa: ~1-2h para 18K casos (batch_size=16, GPU única).

Saída: data/processed/entities_qwen.json
  {case_id: {disease: [...], morphology: [...], location: [...], symptom: [...], chemical: []}}

Uso:
    CUDA_VISIBLE_DEVICES=7 python scripts/extract_entities_qwen.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

SYSTEM = """\
You are a medical entity extractor for dermatology. Extract entities from the clinical text and return ONLY a JSON object with these keys:
- "disease": diagnoses and conditions (singular lowercase, e.g. "psoriasis", "melanoma")
- "morphology": skin lesion types (singular, e.g. "plaque", "papule", "erythema", "vesicle")
- "location": body locations (e.g. "scalp", "face", "lower extremity")
- "symptom": symptoms (e.g. "pruritus", "pain", "alopecia")
- "chemical": drugs and treatments (e.g. "methotrexate", "dupilumab")
Use singular lowercase. Empty list [] if none. Return ONLY the JSON, no explanation."""

USER_TMPL = "Extract entities from this dermatology case:\n\n{text}"

MAX_INPUT_WORDS = 400   # trunca input para controlar memória
MAX_NEW_TOKENS  = 180   # suficiente para JSON com ~10 entidades por tipo
CHECKPOINT_EVERY = 500  # guarda progresso a cada N casos


def truncate(text: str, max_words: int = MAX_INPUT_WORDS) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def apply_chat_template(tokenizer, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": USER_TMPL.format(text=truncate(text))},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def parse_json(text: str) -> dict[str, list[str]]:
    empty = {"disease": [], "morphology": [], "location": [], "symptom": [], "chemical": []}
    try:
        # Remove markdown fences se existirem
        clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        # Extrai o primeiro { ... }
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            return empty
        data = json.loads(m.group())
        result = {}
        for key in ("disease", "morphology", "location", "symptom", "chemical"):
            val = data.get(key, [])
            result[key] = sorted(set(str(v).lower().strip() for v in val if str(v).strip())) \
                          if isinstance(val, list) else []
        return result
    except Exception:
        return empty


def load_checkpoint(path: Path) -> dict[str, dict]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_checkpoint(results: dict, path: Path) -> None:
    path.write_text(json.dumps(results, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",      default="configs/graph_qwen.yaml")
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--force",       action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_path = Path(cfg["entity_extraction"]["cache_path"])
    ckpt_path   = output_path.with_suffix(".ckpt.json")

    if output_path.exists() and not args.force:
        console.print(f"[green]{output_path} já existe. Usa --force para re-extrair.[/]")
        return

    # Carrega corpus
    import pandas as pd
    df = pd.read_csv(cfg["data"]["derm_csv"])
    id_col, text_col = cfg["data"]["id_col"], cfg["data"]["text_col"]
    df = df.dropna(subset=[id_col, text_col]).copy()
    df[id_col] = df[id_col].astype(str)
    ids   = df[id_col].tolist()
    texts = df[text_col].astype(str).tolist()
    console.print(f"Corpus: {len(ids)} casos")

    # Retoma do checkpoint se existir
    results = load_checkpoint(ckpt_path)
    done_ids = set(results.keys())
    todo = [(i, t) for i, t in zip(ids, texts) if i not in done_ids]
    console.print(f"Já processados: {len(done_ids)} | Restantes: {len(todo)}")

    if not todo:
        console.print("[green]Todos os casos já processados.[/]")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # Carrega modelo
    model_name = cfg["generation"]["model_name"]
    console.print(f"\nCarregando modelo: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    console.print(f"Modelo pronto em: {next(model.parameters()).device}\n")

    batch_size = args.batch_size
    n_saved_since_ckpt = 0

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extraindo entidades...", total=len(todo))

        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            batch_ids   = [x[0] for x in batch]
            batch_texts = [x[1] for x in batch]

            # Monta prompts
            prompts = [apply_chat_template(tokenizer, t) for t in batch_texts]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,          # greedy — mais rápido e determinístico
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for case_id, out_ids in zip(batch_ids, output_ids):
                new_tokens = out_ids[input_len:]
                text_out   = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                results[case_id] = parse_json(text_out)
                n_saved_since_ckpt += 1

            progress.advance(task, len(batch))

            # Checkpoint periódico
            if n_saved_since_ckpt >= CHECKPOINT_EVERY:
                save_checkpoint(results, ckpt_path)
                n_saved_since_ckpt = 0

    # Garante que todos os IDs têm entrada
    empty = {"disease": [], "morphology": [], "location": [], "symptom": [], "chemical": []}
    for case_id in ids:
        if case_id not in results:
            results[case_id] = empty

    # Estatísticas
    n = len(results)
    console.rule("Cobertura")
    for key in ("disease", "morphology", "location", "symptom", "chemical"):
        has = sum(1 for e in results.values() if e.get(key))
        avg = sum(len(e.get(key, [])) for e in results.values()) / n
        console.print(f"  {key:<12}: {has:>5}/{n} ({has/n*100:.0f}%)  média {avg:.1f}/caso")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    console.print(f"\n[green bold]Entidades salvas em {output_path}[/]")

    if ckpt_path.exists():
        ckpt_path.unlink()


if __name__ == "__main__":
    main()
