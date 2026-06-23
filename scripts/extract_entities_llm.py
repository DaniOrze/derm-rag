"""
Extração de entidades médicas via Claude Haiku — Anthropic Batch API.

Processa todos os casos do corpus e salva entities_llm.json no mesmo
formato que EntityExtractor.extract_batch(), permitindo substituição
directa no pipeline run_graph.py.

Custo estimado (~18K casos, claude-haiku-4-5, Batch API com 50% desc.):
    Input:  ~10M tokens → ~$5
    Output: ~2M tokens  → ~$5
    Total:  ~$10

Uso:
    python scripts/extract_entities_llm.py --config configs/graph_llm.yaml
    python scripts/extract_entities_llm.py --config configs/graph_llm.yaml --poll  # retomar
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.prepare_eval import DataConfig, SplitConfig, prepare_dataset

console = Console()

BATCH_ID_FILE = "data/processed/entities_llm_batch_id.txt"
OUTPUT_FILE   = "data/processed/entities_llm.json"

MAX_TEXT_WORDS = 500   # trunca textos longos para controlar custo

SYSTEM = """\
You are a medical entity extractor specializing in dermatology.
Given a clinical case text, extract medical entities and return ONLY valid JSON with these exact keys:
- "disease": diagnoses, conditions, syndromes (e.g., "psoriasis", "melanoma", "atopic dermatitis")
- "morphology": skin lesion characteristics (e.g., "plaque", "papule", "erythema", "vesicle", "ulcer", "scale")
- "location": anatomical locations (e.g., "scalp", "lower extremity", "face", "trunk", "nail")
- "symptom": clinical symptoms (e.g., "pruritus", "pain", "swelling", "burning", "alopecia")
- "chemical": drugs and treatments (e.g., "methotrexate", "dupilumab", "clobetasol", "prednisone")

Rules:
- Normalize to singular lowercase (e.g., "plaque" not "plaques", "erythema" not "erythematous")
- Expand abbreviations (e.g., "atopic dermatitis" not "AD")
- Only include entities explicitly present or clearly implied in the text
- Use empty list [] if no entities of that type
- No explanation, no markdown — ONLY the JSON object"""


def truncate(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def submit_batch(client: anthropic.Anthropic, cases: list[tuple[str, str]]) -> str:
    """Submete batch e devolve batch_id."""
    console.print(f"[bold]Submetendo batch com {len(cases)} casos...[/]")
    requests = [
        {
            "custom_id": case_id,
            "params": {
                "model": "claude-haiku-4-5",
                "max_tokens": 256,
                "system": SYSTEM,
                "messages": [{"role": "user", "content": truncate(text, MAX_TEXT_WORDS)}],
            },
        }
        for case_id, text in cases
    ]
    batch = client.messages.batches.create(requests=requests)
    console.print(f"[green]Batch criado: {batch.id}[/]")
    Path(BATCH_ID_FILE).write_text(batch.id)
    console.print(f"[dim]batch_id salvo em {BATCH_ID_FILE}[/]")
    return batch.id


def poll_batch(client: anthropic.Anthropic, batch_id: str, poll_interval: int = 60) -> None:
    """Aguarda o batch terminar, mostrando progresso."""
    console.print(f"[bold]Aguardando batch {batch_id}...[/]")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        done   = counts.succeeded + counts.errored + counts.canceled + counts.expired
        total  = counts.processing + done
        pct    = done / total * 100 if total else 0
        console.print(
            f"  status={batch.processing_status} | "
            f"done={done}/{total} ({pct:.1f}%) | "
            f"succeeded={counts.succeeded} errored={counts.errored}"
        )
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval)


def parse_entity_json(text: str) -> dict[str, list[str]]:
    """Extrai e valida o JSON de entidades da resposta do modelo."""
    empty = {"disease": [], "morphology": [], "location": [], "symptom": [], "chemical": []}
    try:
        # Remove possíveis blocos markdown
        clean = text.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
            clean = clean.rstrip("`").strip()
        data = json.loads(clean)
        if not isinstance(data, dict):
            return empty
        result = {}
        for key in ("disease", "morphology", "location", "symptom", "chemical"):
            val = data.get(key, [])
            result[key] = sorted(set(str(v).lower().strip() for v in val if v)) if isinstance(val, list) else []
        return result
    except (json.JSONDecodeError, Exception):
        return empty


def collect_results(client: anthropic.Anthropic, batch_id: str) -> dict[str, dict]:
    """Itera pelos resultados do batch e devolve {case_id: entities}."""
    console.print("[bold]Recolhendo resultados...[/]")
    entities: dict[str, dict] = {}
    n_ok, n_err = 0, 0

    for result in client.messages.batches.results(batch_id):
        case_id = result.custom_id
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            entities[case_id] = parse_entity_json(text)
            n_ok += 1
        else:
            entities[case_id] = {"disease": [], "morphology": [], "location": [], "symptom": [], "chemical": []}
            n_err += 1

    console.print(f"  Resultados: {n_ok} OK | {n_err} erros")
    return entities


def print_stats(entities: dict) -> None:
    n = len(entities)
    for key in ("disease", "morphology", "location", "symptom", "chemical"):
        has = sum(1 for e in entities.values() if e.get(key))
        avg = sum(len(e.get(key, [])) for e in entities.values()) / n
        console.print(f"  {key:<12}: {has}/{n} casos com ≥1 entidade  (média {avg:.1f}/caso)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/graph_llm.yaml")
    p.add_argument("--poll", action="store_true",
                   help="Apenas aguarda batch já submetido (lê batch_id do ficheiro)")
    p.add_argument("--poll-interval", type=int, default=60,
                   help="Segundos entre verificações do batch (default 60)")
    args = p.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY não definida[/]")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        console.print(f"[green]{output_path} já existe. Nada a fazer.[/]")
        console.print("  (apaga o ficheiro para re-extrair)")
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Carrega todos os casos do corpus (não só o split de teste)
    import pandas as pd  # só importa se necessário
    df = pd.read_csv(cfg["data"]["derm_csv"])
    id_col   = cfg["data"]["id_col"]
    text_col = cfg["data"]["text_col"]
    df = df.dropna(subset=[id_col, text_col]).copy()
    df[id_col] = df[id_col].astype(str)
    cases = list(zip(df[id_col].tolist(), df[text_col].astype(str).tolist()))
    console.print(f"Corpus: {len(cases)} casos carregados de {cfg['data']['derm_csv']}")

    # Submete ou retoma
    batch_id_path = Path(BATCH_ID_FILE)
    if args.poll:
        if not batch_id_path.exists():
            console.print("[red]--poll: ficheiro batch_id não encontrado[/]")
            sys.exit(1)
        batch_id = batch_id_path.read_text().strip()
        console.print(f"Retomando batch: {batch_id}")
    else:
        batch_id = submit_batch(client, cases)

    poll_batch(client, batch_id, poll_interval=args.poll_interval)

    entities = collect_results(client, batch_id)

    # Garante que todos os casos têm entrada (mesmo os sem texto)
    empty = {"disease": [], "morphology": [], "location": [], "symptom": [], "chemical": []}
    for case_id, _ in cases:
        if case_id not in entities:
            entities[case_id] = empty

    console.rule("[bold]Estatísticas de cobertura[/]")
    print_stats(entities)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entities, indent=2, ensure_ascii=False))
    console.print(f"\n[green bold]Entidades salvas em {output_path}[/]")

    # Limpa o ficheiro de batch_id após sucesso
    if batch_id_path.exists():
        batch_id_path.unlink()


if __name__ == "__main__":
    main()
