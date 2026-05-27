"""
Preparação do conjunto de teste para avaliação de retrieval.

A partir do subset dermatológico do PMC-Patients, este módulo:
    1. Lê os casos e a coluna similar_patients (ground truth)
    2. Constrói os qrels {query_uid: {doc_uid: score}}
    3. Separa um conjunto de queries de teste (casos com similares conhecidos)
    4. Define o corpus de busca (todos os casos)

Decisão metodológica importante:
    - O CORPUS de busca contém TODOS os casos (inclusive as queries), mas na
      hora de avaliar removemos o próprio caso do ranking (auto-match).
    - RELEVANTE = qualquer similar com score >= 1 (decisão: usar ambos os
      níveis). A versão "só score 2" é calculada à parte como robustez.
    - Importante: só contam como ground truth os similares que ESTÃO no corpus
      (alguns similares podem ter sido filtrados fora do subset dermatológico).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rich.console import Console

console = Console()


@dataclass
class DataConfig:
    derm_csv: str
    text_col: str
    id_col: str
    similar_col: str


@dataclass
class SplitConfig:
    min_similars: int
    n_queries: int | None
    seed: int


def load_configs(path: str | Path) -> tuple[DataConfig, SplitConfig, dict]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    data = DataConfig(**cfg["data"])
    split = SplitConfig(**cfg["split"])
    return data, split, cfg


def _parse_similar(val: Any) -> dict[str, int]:
    """Converte a string do dict de similares em dict real {uid: score}."""
    if isinstance(val, dict):
        return {str(k): int(v) for k, v in val.items()}
    if isinstance(val, str) and val.strip():
        try:
            d = ast.literal_eval(val)
            if isinstance(d, dict):
                return {str(k): int(v) for k, v in d.items()}
        except (ValueError, SyntaxError):
            return {}
    return {}


def prepare_dataset(data: DataConfig, split: SplitConfig) -> dict[str, Any]:
    """Carrega os dados e monta corpus, queries e qrels.

    Returns dict com:
        - corpus_ids: list[str]      todos os uids no corpus
        - corpus_texts: list[str]    textos correspondentes (mesma ordem)
        - query_ids: list[str]       uids escolhidos como queries de teste
        - qrels: {query_uid: {doc_uid: score}}     ground truth (score >=1)
        - qrels_strict: idem, mas só score == 2
    """
    csv_path = Path(data.derm_csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {csv_path}. "
            "Copie o derm_cases.csv da etapa anterior para data/processed/."
        )

    console.print(f"[bold]Carregando {csv_path}[/]")
    df = pd.read_csv(csv_path)
    console.print(f"  {len(df)} casos")

    # Remove linhas sem texto ou sem id
    df = df.dropna(subset=[data.text_col, data.id_col]).reset_index(drop=True)
    df[data.id_col] = df[data.id_col].astype(str)

    # Corpus = todos os casos
    corpus_ids = df[data.id_col].tolist()
    corpus_texts = df[data.text_col].astype(str).tolist()
    corpus_set = set(corpus_ids)
    console.print(f"  Corpus: {len(corpus_ids)} documentos")

    # Constrói qrels, mantendo só similares que estão no corpus
    qrels_full: dict[str, dict[str, int]] = {}
    qrels_strict: dict[str, dict[str, int]] = {}
    for _, row in df.iterrows():
        uid = row[data.id_col]
        sims = _parse_similar(row[data.similar_col])
        # mantém só os que existem no corpus e não são o próprio caso
        kept = {d: s for d, s in sims.items() if d in corpus_set and d != uid}
        if kept:
            qrels_full[uid] = kept
            strict = {d: s for d, s in kept.items() if s >= 2}
            if strict:
                qrels_strict[uid] = strict

    console.print(f"  Casos com ground truth (score>=1): {len(qrels_full)}")
    console.print(f"  Casos com ground truth estrito (score==2): {len(qrels_strict)}")

    # Seleciona queries: casos com pelo menos min_similars relevantes
    eligible = [q for q, rels in qrels_full.items() if len(rels) >= split.min_similars]
    console.print(f"  Queries elegíveis (>= {split.min_similars} similar): {len(eligible)}")

    # Amostra n_queries com seed fixa
    import random
    rng = random.Random(split.seed)
    rng.shuffle(eligible)
    if split.n_queries is not None:
        query_ids = eligible[: split.n_queries]
    else:
        query_ids = eligible
    console.print(f"  Queries de teste selecionadas: {len(query_ids)}")

    return {
        "corpus_ids": corpus_ids,
        "corpus_texts": corpus_texts,
        "query_ids": query_ids,
        "qrels": {q: qrels_full[q] for q in query_ids},
        "qrels_strict": {q: qrels_strict[q] for q in query_ids if q in qrels_strict},
    }
