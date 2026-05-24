"""
Geração de embeddings densos com BGE-M3.

BGE-M3 é multilíngue (bom para PT-BR depois) e produz embeddings densos de
1024 dimensões. Aqui usamos só o modo denso; os modos sparse e multi-vector
ficam para experimentos futuros (hybrid retrieval).

Os embeddings são cacheados em disco (.npy) para não recomputar a cada run —
gerar embeddings dos 18k casos é a parte cara, mas roda só uma vez.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()


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
        model_name: modelo do HuggingFace (default BGE-M3)
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

    # Tenta carregar do cache
    if cache_path and cache_path.exists() and not force_recompute:
        console.print(f"[green]Carregando embeddings do cache: {cache_path}[/]")
        emb = np.load(cache_path)
        if len(emb) == len(texts):
            return emb
        console.print(
            f"[yellow]Cache tem {len(emb)} embeddings mas há {len(texts)} textos. "
            "Recomputando.[/]"
        )

    # Importa aqui para não exigir torch/FlagEmbedding em quem só usa métricas
    from FlagEmbedding import BGEM3FlagModel

    console.print(f"[bold]Carregando modelo {model_name} em {device}[/]")
    use_fp16 = device == "cuda"
    model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)

    console.print(f"[bold]Gerando embeddings de {len(texts)} textos "
                  f"(batch={batch_size}, max_len={max_length})[/]")
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    emb = np.asarray(output["dense_vecs"], dtype=np.float32)

    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
        console.print(f"[green]Embeddings salvos em {cache_path}[/]")

    return emb
