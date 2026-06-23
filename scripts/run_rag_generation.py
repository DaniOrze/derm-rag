"""
Pipeline RAG de geração: retrieval → LLM → diagnóstico diferencial.

Usa HuggingFace Transformers com batching (batch_size configurável).
Recomenda-se rodar em GPU dedicada: CUDA_VISIBLE_DEVICES=5 python ...

Fluxo:
  1. Carrega rankings do melhor retriever (já computados)
  2. Para cada query do test set:
     a. Trunca o texto à apresentação clínica (sem diagnóstico embutido)
     b. Recupera os top-K casos similares
     c. Monta prompt: apresentação atual + K casos da literatura
  3. Gera em mini-batches para maior throughput
  4. Salva resultados em JSON para avaliação

Uso:
    CUDA_VISIBLE_DEVICES=5 python scripts/run_rag_generation.py --config configs/rag_generation.yaml
    CUDA_VISIBLE_DEVICES=5 python scripts/run_rag_generation.py --config configs/rag_generation.yaml --n-queries 1000
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


SYSTEM_PROMPT = (
    "You are an expert dermatologist providing clinical decision support in an emergency setting. "
    "Given a patient's clinical presentation and similar cases from the dermatology literature, "
    "provide a concise differential diagnosis and initial management plan."
)

TASK_PROMPT_RAG = """\
## Current Patient Presentation
{query}

## Similar Cases from the Dermatology Literature
{cases}

## Task
Based on the current patient presentation and the similar cases above, provide:
1. **Most Likely Diagnosis** and **Differential Diagnoses** (in order of probability)
2. **Key Supporting Features** from this case and the similar cases
3. **Recommended Initial Workup and Management**

Be concise and clinically focused."""

TASK_PROMPT_NORAG = """\
## Current Patient Presentation
{query}

## Task
Based on the current patient presentation alone, provide:
1. **Most Likely Diagnosis** and **Differential Diagnoses** (in order of probability)
2. **Key Supporting Features** from this case
3. **Recommended Initial Workup and Management**

Be concise and clinically focused."""


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


def tokenize_truncate(text: str, tokenizer, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def build_user_message(query_presentation: str, retrieved: list[tuple[str, str]]) -> str:
    if not retrieved:
        return TASK_PROMPT_NORAG.format(query=query_presentation)
    case_blocks = [f"### Case {i}\n{text}" for i, (_, text) in enumerate(retrieved, 1)]
    return TASK_PROMPT_RAG.format(
        query=query_presentation,
        cases="\n\n".join(case_blocks),
    )


def apply_chat_template(tokenizer, messages: list[dict]) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def main():
    # Resolve paths relative to the project root (parent of scripts/)
    # so the script works regardless of the calling directory.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    import os
    os.chdir(PROJECT_ROOT)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rag_generation.yaml")
    p.add_argument("--n-queries", type=int, default=None,
                   help="Override split.n_queries do config")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.n_queries is not None:
        cfg["split"]["n_queries"] = args.n_queries

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------
    df = pd.read_csv(cfg["data"]["derm_csv"])
    df = df.dropna(subset=[cfg["data"]["text_col"], cfg["data"]["id_col"]]).copy()
    df[cfg["data"]["id_col"]] = df[cfg["data"]["id_col"]].astype(str)
    corpus_set = set(df[cfg["data"]["id_col"]])

    id_to_text  = dict(zip(df[cfg["data"]["id_col"]], df[cfg["data"]["text_col"]].astype(str)))
    id_to_title = dict(zip(df[cfg["data"]["id_col"]], df[cfg["data"]["title_col"]].astype(str)))

    qrels: dict[str, dict[str, int]] = {}
    for _, row in df.iterrows():
        uid = row[cfg["data"]["id_col"]]
        sims = parse_similar(row["similar_patients"])
        kept = {d: s for d, s in sims.items() if d in corpus_set and d != uid and s >= 1}
        if kept:
            qrels[uid] = kept

    eligible = list(qrels.keys())
    rng = random.Random(cfg["split"]["seed"])
    rng.shuffle(eligible)
    test_ids = eligible[: cfg["split"]["n_queries"]]
    print(f"Test set: {len(test_ids)} queries")

    # ------------------------------------------------------------------
    # 2. Rankings
    # ------------------------------------------------------------------
    results_dir = Path(cfg["retrieval"]["results_dir"])
    top_k       = cfg["retrieval"]["top_k"]
    rankings    = {}

    if top_k > 0:
        rankings = load_latest_rankings(results_dir, cfg["retrieval"]["rankings_prefix"])
        if rankings is None:
            sys.exit(f"Rankings não encontrado: {cfg['retrieval']['rankings_prefix']}")
        print(f"Rankings: {len(rankings)} queries carregadas")
    else:
        print("Modo No-RAG: sem recuperação de casos")
    pres_max = cfg["presentation"]["max_tokens"]
    ctx_max  = cfg["generation"].get("context_case_max_tokens", 600)

    # ------------------------------------------------------------------
    # 3. Tokenizer (só para truncar textos — separado do vLLM)
    # ------------------------------------------------------------------
    model_name = cfg["generation"]["model_name"]
    print(f"Carregando tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ------------------------------------------------------------------
    # 4. Monta todos os prompts antes de carregar o modelo
    # ------------------------------------------------------------------
    print("Montando prompts...")
    prompts: list[str] = []
    meta:    list[dict] = []
    skipped = 0

    for qid in test_ids:
        if top_k > 0 and qid not in rankings:
            skipped += 1
            continue

        query_pres = tokenize_truncate(id_to_text[qid], tokenizer, pres_max)
        if top_k > 0:
            top_ids   = rankings[qid][:top_k]
            retrieved = [
                (cid, tokenize_truncate(id_to_text.get(cid, ""), tokenizer, ctx_max))
                for cid in top_ids
            ]
        else:
            top_ids   = []
            retrieved = []

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(query_pres, retrieved)},
        ]

        prompts.append(apply_chat_template(tokenizer, messages))
        meta.append({
            "query_id":           qid,
            "query_presentation": query_pres,
            "retrieved_ids":      top_ids,
            "retrieved_titles":   [id_to_title.get(cid, "") for cid in top_ids],
            "ground_truth_title": id_to_title.get(qid, ""),
        })

    print(f"Prompts prontos: {len(prompts)} | Skipped: {skipped}")

    # ------------------------------------------------------------------
    # 5. Carrega modelo (transformers)
    # ------------------------------------------------------------------
    gen_cfg = cfg["generation"]
    dtype = getattr(torch, gen_cfg.get("torch_dtype", "bfloat16"))
    batch_size = gen_cfg.get("batch_size", 4)

    print(f"\nCarregando modelo: {model_name}")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map_cfg = gen_cfg.get("device_map", "single_gpu")
    device_map = "auto" if device_map_cfg == "auto" else {"": "cuda:0"}

    quantization = gen_cfg.get("quantization", None)
    quant_config = None
    if quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("Quantização: 4-bit NF4 (bnb_4bit_use_double_quant=True)")
    elif quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Quantização: 8-bit")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        dtype=dtype if quant_config is None else None,
        quantization_config=quant_config,
    )
    model.eval()
    print(f"Modelo em: {next(model.parameters()).device}")
    print(f"Modelo pronto. Batch size: {batch_size}\n")

    temperature = gen_cfg.get("temperature", 0.1)
    max_new     = gen_cfg["max_new_tokens"]

    # ------------------------------------------------------------------
    # 6. Gera em mini-batches
    # ------------------------------------------------------------------
    responses: list[str] = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Gerando"):
        batch_prompts = prompts[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=gen_cfg.get("max_model_len", 8192) - max_new,
        ).to(model.device)

        gen_kwargs: dict = dict(
            max_new_tokens=max_new,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["do_sample"]   = True

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Com left-padding, todos os inputs têm o mesmo comprimento padded
        input_len = inputs["input_ids"].shape[1]
        for out_ids in output_ids:
            new_tokens = out_ids[input_len:]
            responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    # ------------------------------------------------------------------
    # 7. Monta resultados
    # ------------------------------------------------------------------
    outputs = []
    for m, response in zip(meta, responses):
        outputs.append({**m, "response": response})

    # ------------------------------------------------------------------
    # 7. Salva
    # ------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg["evaluation"]["run_name"]
    out_path = results_dir / f"{run_name}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print(f"\nResultados salvos em {out_path}")
    print(f"Total geradas: {len(outputs)}")

    if outputs:
        ex = outputs[0]
        print("\n" + "=" * 60)
        print("EXEMPLO DE OUTPUT")
        print("=" * 60)
        print(f"Query (primeiros 300 chars):\n{ex['query_presentation'][:300]}...\n")
        print(f"Casos recuperados: {ex['retrieved_ids']}")
        print(f"Ground truth (título): {ex['ground_truth_title']}\n")
        print(f"Resposta:\n{ex['response'][:600]}...")


if __name__ == "__main__":
    main()
