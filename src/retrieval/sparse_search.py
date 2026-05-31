"""
Busca esparsa com vetores léxicos BGE-M3 (SPLADE-style).

Diferença em relação ao BM25:
    BM25       → pesos calculados por estatísticas de frequência (TF-IDF)
    BGE-M3 sparse → pesos aprendidos pelo modelo (expansão semântica de tokens)

O BGE-M3 no modo sparse produz um dict {token_id: weight} por documento,
onde token_id vem do tokenizador XLM-RoBERTa. Tokens semanticamente
relacionados à consulta recebem pesos mesmo que não apareçam literalmente,
tornando a busca mais robusta que BM25.

A busca é feita por produto interno entre vetores esparsos, implementado
eficientemente via matrizes scipy.sparse.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

console = Console()

BGEM3_VOCAB_SIZE = 250002  # tamanho do vocabulário XLM-RoBERTa


# ---------------------------------------------------------------------------
# Geração de vetores esparsos
# ---------------------------------------------------------------------------

def generate_sparse_vectors(
    texts: list[str],
    model_name: str = "BAAI/bge-m3",
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 1024,
    cache_path: Optional[Path | str] = None,
    force_recompute: bool = False,
) -> list[dict[int, float]]:
    """Gera vetores esparsos BGE-M3 para todos os textos.

    Retorna lista de dicts {token_id: weight}, um por documento.
    Cacheado em pickle para evitar reprocessamento.
    """
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_recompute:
            console.print(f"[green]Carregando vetores esparsos do cache: {cache_path}[/]")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if len(cached) == len(texts):
                return cached
            console.print(
                f"[yellow]Cache tem {len(cached)} vetores mas há {len(texts)} textos. "
                "Recomputando.[/]"
            )

    from FlagEmbedding import BGEM3FlagModel

    console.print(f"[bold]Carregando BGE-M3 (sparse) em {device}[/]")
    use_fp16 = device == "cuda"
    model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)

    console.print(f"[bold]Gerando vetores esparsos para {len(texts)} textos[/]")
    all_sparse: list[dict[int, float]] = []

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Sparse encode...", total=len(texts))

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            output = model.encode(
                batch,
                batch_size=len(batch),
                max_length=max_length,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            # lexical_weights: list of {token_id: weight}
            # keys podem ser str ou int dependendo da versão do FlagEmbedding
            for vec in output["lexical_weights"]:
                all_sparse.append({int(k): float(v) for k, v in vec.items()})

            progress.advance(task, len(batch))

    if cache_path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(all_sparse, f)
        console.print(f"[green]Vetores esparsos salvos em {cache_path}[/]")

    return all_sparse


# ---------------------------------------------------------------------------
# Construção do índice e busca
# ---------------------------------------------------------------------------

def build_sparse_matrix(
    sparse_vecs: list[dict[int, float]],
    vocab_size: int = BGEM3_VOCAB_SIZE,
) -> csr_matrix:
    """Constrói matriz esparsa scipy (n_docs × vocab_size)."""
    rows, cols, data = [], [], []
    for i, vec in enumerate(sparse_vecs):
        for tok_id, weight in vec.items():
            rows.append(i)
            cols.append(tok_id)
            data.append(weight)
    return csr_matrix((data, (rows, cols)), shape=(len(sparse_vecs), vocab_size))


def sparse_search(
    query_sparse_vecs: list[dict[int, float]],
    corpus_sparse_matrix: csr_matrix,
    corpus_ids: list[str],
    top_k: int = 50,
    query_ids: Optional[list[str]] = None,
    exclude_self: bool = True,
    vocab_size: int = BGEM3_VOCAB_SIZE,
) -> dict[str, list[str]]:
    """Busca esparsa top-k por produto interno de vetores BGE-M3.

    Args:
        query_sparse_vecs: lista de {token_id: weight} para as queries
        corpus_sparse_matrix: índice esparso do corpus (n_docs × vocab_size)
        corpus_ids: IDs do corpus (mesma ordem da matriz)
        top_k: quantos resultados por query
        query_ids: IDs das queries (necessário se exclude_self=True)
        exclude_self: remove auto-match do resultado
        vocab_size: tamanho do vocabulário (XLM-RoBERTa = 250002)

    Returns:
        {query_id: [doc_id ordenados por score decrescente]}
    """
    console.print(
        f"[bold]Busca esparsa BGE-M3: top-{top_k} para "
        f"{len(query_sparse_vecs)} queries em {len(corpus_ids)} documentos[/]"
    )

    # Constrói matriz de queries (n_queries × vocab_size)
    query_matrix = build_sparse_matrix(query_sparse_vecs, vocab_size)

    # Scores: (n_queries × n_docs) = query_matrix @ corpus_matrix.T
    score_matrix = (query_matrix @ corpus_sparse_matrix.T).toarray()

    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    fetch = top_k + (1 if exclude_self else 0)
    rankings: dict[str, list[str]] = {}

    for qi in range(len(query_sparse_vecs)):
        qid = query_ids[qi] if query_ids is not None else str(qi)
        scores = score_matrix[qi]
        self_idx = id_to_idx.get(qid, -1) if exclude_self else -1

        if fetch < len(scores):
            top_idx = np.argpartition(-scores, fetch)[:fetch]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        else:
            top_idx = np.argsort(-scores)

        ranked = []
        for idx in top_idx:
            if idx == self_idx:
                continue
            ranked.append(corpus_ids[idx])
            if len(ranked) >= top_k:
                break
        rankings[qid] = ranked

    return rankings
