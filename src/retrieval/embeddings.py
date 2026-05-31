"""
Geração de embeddings densos.

Suporta dois backends:
    BGE-M3 (FlagEmbedding): modelo geral multilíngue, 1024-d
    SentenceTransformer:    modelos como BioLORD-2023-C (768-d, domínio biomédico)

A detecção do backend é automática pelo nome do modelo:
    "BAAI/bge-m3"          → FlagEmbedding (BGEM3FlagModel)
    qualquer outro modelo  → SentenceTransformer

Os embeddings são cacheados em disco (.npy) — gerar embeddings dos 18k casos
é a etapa cara, mas roda só uma vez por modelo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()

_BGE_M3_MODELS = {"BAAI/bge-m3", "BAAI/bge-m3-unsupervised"}


def generate_embeddings(
    texts: list[str],
    model_name: str = "BAAI/bge-m3",
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 1024,
    normalize: bool = True,
    cache_path: str | Path | None = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """Gera embeddings densos para uma lista de textos.

    Args:
        texts: lista de textos a embedar
        model_name: modelo HuggingFace. BGE-M3 usa FlagEmbedding;
                    outros modelos usam SentenceTransformer.
        device: 'cuda', 'cpu' ou 'mps'
        batch_size: tamanho do lote
        max_length: truncamento em tokens
        normalize: normaliza para norma 1 (cosine via dot product)
        cache_path: se fornecido, salva/carrega os embeddings deste arquivo
        force_recompute: ignora o cache e recomputa

    Returns:
        np.ndarray de shape (len(texts), dim)
    """
    cache_path = Path(cache_path) if cache_path else None

    if cache_path and cache_path.exists() and not force_recompute:
        console.print(f"[green]Carregando embeddings do cache: {cache_path}[/]")
        emb = np.load(cache_path)
        if len(emb) == len(texts):
            return emb
        console.print(
            f"[yellow]Cache tem {len(emb)} embeddings mas há {len(texts)} textos. "
            "Recomputando.[/]"
        )

    console.print(f"[bold]Carregando modelo {model_name} em {device}[/]")

    if model_name in _BGE_M3_MODELS:
        emb = _encode_bge_m3(texts, model_name, device, batch_size, max_length)
    else:
        emb = _encode_sentence_transformer(texts, model_name, device, batch_size, max_length)

    emb = np.asarray(emb, dtype=np.float32)

    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
        console.print(f"[green]Embeddings salvos em {cache_path}[/]")

    return emb


def _encode_bge_m3(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    from FlagEmbedding import BGEM3FlagModel

    use_fp16 = device == "cuda"
    model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
    console.print(
        f"[bold]Gerando embeddings BGE-M3 de {len(texts)} textos "
        f"(batch={batch_size}, max_len={max_length})[/]"
    )
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return output["dense_vecs"]


def _encode_sentence_transformer(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_length
    console.print(
        f"[bold]Gerando embeddings SentenceTransformer de {len(texts)} textos "
        f"(batch={batch_size}, max_len={max_length})[/]"
    )
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,  # normalização feita em generate_embeddings
        convert_to_numpy=True,
    )
    return emb
