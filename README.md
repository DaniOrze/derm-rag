# Derm-RAG — Experimento de Retrieval (Dense Baseline)

Primeiro experimento real do RAG: retrieval denso com BGE-M3, avaliado
contra o ground truth de similaridade do PMC-Patients.

## O que este projeto faz

1. Prepara o conjunto de teste a partir do subset dermatológico
   (queries, corpus e qrels derivados de `similar_patients`)
2. Gera embeddings BGE-M3 do corpus (cacheados)
3. Faz busca densa (similaridade de cosseno) top-k
4. Avalia com Recall@k, MRR e nDCG
   - versão completa (score >= 1) e estrita (score == 2)

Este é o **baseline denso**. Hybrid (BM25 + reranker), GraphRAG e CRAG
vêm como incrementos depois, reusando este mesmo pipeline de avaliação.

## Pré-requisito

Copie o `derm_cases.csv` gerado na etapa anterior (projeto derm-rag-pmc)
para `data/processed/derm_cases.csv`.

## Setup (HPC)

```bash
mamba env create -f environment.yml
mamba activate derm-rag-retrieval
```

## Rodar

```bash
# Direto (em nó com GPU):
python scripts/run_retrieval.py --config configs/dense_baseline.yaml

# Ou via Slurm:
sbatch slurm/run_retrieval.sbatch
```

A primeira execução gera os embeddings dos ~18k casos (alguns minutos em
GPU) e os cacheia em `data/processed/embeddings_bge_m3.npy`. Execuções
seguintes reusam o cache.

## Configuração

Tudo em `configs/dense_baseline.yaml`:
- `split.n_queries`: quantas queries de teste (default 1000)
- `embedding.device`: 'cuda' no HPC, 'cpu'/'mps' no Mac
- `retrieval.top_k`: profundidade da busca
- `evaluation.k_values`: cortes das métricas

## Estrutura

```
derm-rag-retrieval/
├── configs/dense_baseline.yaml
├── src/
│   ├── data/prepare_eval.py        ← split + qrels
│   ├── retrieval/
│   │   ├── embeddings.py           ← BGE-M3
│   │   └── dense_search.py         ← busca por cosseno
│   └── evaluation/
│       └── retrieval_metrics.py    ← Recall@k, MRR, nDCG
├── scripts/run_retrieval.py        ← orquestra tudo
├── slurm/run_retrieval.sbatch
└── results/                        ← JSONs de resultado
```

## Notas metodológicas

- **Relevância:** score >= 1 conta como relevante (decisão de incluir ambos
  os níveis). nDCG aproveita os scores 1 e 2 como ganhos graduados. A versão
  estrita (só score 2) é reportada como análise de robustez.
- **Auto-match:** o corpus inclui as próprias queries, mas o caso é removido
  do seu próprio ranking na avaliação.
- **Escala:** busca exata em numpy basta para ~18k docs. Para corpus maiores,
  migrar para FAISS/Qdrant.
```
