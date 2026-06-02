"""
ColBERT retrieval via MaxSim (late interaction multi-vector).

Diferença fundamental em relação ao dense:
    Dense:   1 vetor por documento (CLS token, 1024-d)
             score = dot(q_vec, d_vec)

    ColBERT: N vetores por documento (1 por token, 1024-d)
             score = Σ_{i ∈ tokens(q)}  max_{j ∈ tokens(d)} cos_sim(q_i, d_j)
             ("MaxSim" — cada token da query encontra o token mais similar do doc)

Pipeline de dois estágios (padrão na literatura):
    1. Dense retrieval → top-K candidatos por query (rápido)
    2. ColBERT MaxSim reranking nos candidatos (preciso)

Isso evita computar MaxSim contra o corpus inteiro (18k × tokens × tokens),
o que seria inviável em tempo/memória sem índice especializado (PLAID/FAISS).

Os vetores ColBERT são computados só para os documentos necessários
(queries + candidatos únicos), reduzindo a ~12-15k docs.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, track

console = Console()


# ---------------------------------------------------------------------------
# Geração de vetores ColBERT
# ---------------------------------------------------------------------------

def generate_colbert_vectors(
    id_to_text: dict[str, str],
    model_name: str = "BAAI/bge-m3",
    device: str = "cuda",
    batch_size: int = 16,   # menor que dense: mais memória por doc (N vetores)
    max_length: int = 128,  # truncamento conservador para eficiência
    cache_path: Optional[Path | str] = None,
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """Gera vetores ColBERT para um dict {doc_id: text}.

    Retorna {doc_id: np.ndarray de shape (n_tokens, colbert_dim) em float16}.
    Cache em pickle — inclui só os IDs solicitados.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_recompute:
            console.print(f"[green]Carregando vetores ColBERT do cache: {cache_path}[/]")
            with open(cache_path, "rb") as f:
                cached: dict[str, np.ndarray] = pickle.load(f)
            missing = [k for k in id_to_text if k not in cached]
            if not missing:
                return {k: cached[k] for k in id_to_text}
            console.print(f"  Cache incompleto ({len(missing)} IDs ausentes). Computando faltantes.")
            to_compute = {k: id_to_text[k] for k in missing}
        else:
            cached = {}
            to_compute = id_to_text
    else:
        cached = {}
        to_compute = id_to_text

    if not to_compute:
        return {k: cached[k] for k in id_to_text}

    from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415

    console.print(f"[bold]Carregando BGE-M3 (ColBERT) em {device}[/]")
    use_fp16 = device == "cuda"
    model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)

    ids   = list(to_compute.keys())
    texts = list(to_compute.values())

    console.print(
        f"[bold]Gerando vetores ColBERT para {len(texts)} documentos "
        f"(batch={batch_size}, max_len={max_length})[/]"
    )

    new_vecs: dict[str, np.ndarray] = {}
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("ColBERT encode...", total=len(texts))
        for start in range(0, len(texts), batch_size):
            batch_ids   = ids[start : start + batch_size]
            batch_texts = texts[start : start + batch_size]
            output = model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                max_length=max_length,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )
            for doc_id, vecs in zip(batch_ids, output["colbert_vecs"]):
                arr = np.asarray(vecs, dtype=np.float16)
                # L2-normalize cada token vector (para cosine via dot product)
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                new_vecs[doc_id] = (arr / norms).astype(np.float16)
            progress.advance(task, len(batch_ids))

    cached.update(new_vecs)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cached, f)
        console.print(f"[green]Vetores ColBERT salvos em {cache_path}[/]")

    return {k: cached[k] for k in id_to_text}


# ---------------------------------------------------------------------------
# MaxSim scoring
# ---------------------------------------------------------------------------

def maxsim_score(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """Score ColBERT: Σ_{qi} max_{dj} cos_sim(q_i, d_j).

    query_vecs: (n_q_tokens, dim) float16 ou float32, L2-normalizado
    doc_vecs:   (n_d_tokens, dim) float16 ou float32, L2-normalizado
    """
    # (n_q_tokens, n_d_tokens) — produto interno = cosine (vetores normalizados)
    sim = query_vecs.astype(np.float32) @ doc_vecs.astype(np.float32).T
    return float(sim.max(axis=1).sum())


# ---------------------------------------------------------------------------
# Reranking ColBERT
# ---------------------------------------------------------------------------

def colbert_rerank(
    query_ids: list[str],
    colbert_vecs: dict[str, np.ndarray],
    initial_rankings: dict[str, list[str]],
    top_k: int = 50,
) -> dict[str, list[str]]:
    """Reranqueia candidatos densos usando ColBERT MaxSim.

    Args:
        query_ids: IDs das queries
        colbert_vecs: {doc_id: (n_tokens, dim)} — inclui queries e candidatos
        initial_rankings: {query_id: [candidate_doc_ids]} do retrieval denso
        top_k: tamanho do ranking final

    Returns:
        {query_id: [doc_id ordenados por MaxSim decrescente]}
    """
    console.print(
        f"[bold]ColBERT reranking: {len(query_ids)} queries "
        f"(top-{top_k})[/]"
    )

    reranked: dict[str, list[str]] = {}
    n_missing_query = 0
    n_missing_candidate = 0

    for qid in track(query_ids, description="ColBERT MaxSim"):
        if qid not in colbert_vecs:
            reranked[qid] = initial_rankings.get(qid, [])[:top_k]
            n_missing_query += 1
            continue

        q_vecs = colbert_vecs[qid]
        candidates = initial_rankings.get(qid, [])

        scored = []
        for doc_id in candidates:
            if doc_id not in colbert_vecs:
                n_missing_candidate += 1
                continue
            score = maxsim_score(q_vecs, colbert_vecs[doc_id])
            scored.append((doc_id, score))

        # Ordena por score decrescente
        scored.sort(key=lambda x: -x[1])
        reranked[qid] = [d for d, _ in scored[:top_k]]

    if n_missing_query:
        console.print(f"[yellow]  {n_missing_query} queries sem vetores ColBERT (fallback para dense)[/]")
    if n_missing_candidate:
        console.print(f"[yellow]  {n_missing_candidate} candidatos sem vetores (ignorados)[/]")

    return reranked
