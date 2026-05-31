"""
GraphRAG: recuperação via grafo bipartito de entidades médicas.

Estrutura do grafo:
    Nós tipo "document": cada caso de paciente
    Nós tipo "entity":   cada entidade médica única (prefixada por tipo,
                         ex: "disease::psoriasis", "location::scalp")
    Arestas:             (document, entity) com peso = type_weight

Pontuação de relevância (IDF-ponderada):
    Para cada par (query, candidato), o score é a soma dos pesos
    IDF × type_weight sobre todas as entidades compartilhadas.
    O IDF penaliza entidades muito comuns (ex: "erythema" em milhares
    de casos) e recompensa entidades raras e discriminativas.

    score(q, d) = Σ_{e ∈ ent(q) ∩ ent(d)} type_weight(e) × idf(e)
    idf(e)      = log( N / df(e) )    (N = total docs, df = doc freq)

Referência metodológica: a ontologia e os pesos foram definidos com base
na relevância clínica de cada tipo de entidade para similaridade de casos
dermatológicos (disease > morphology > location ≈ symptom ≈ chemical).
"""
from __future__ import annotations

import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx
from rich.console import Console

console = Console()

# Peso por tipo de entidade (defensável clinicamente na tese)
ENTITY_TYPE_WEIGHTS: dict[str, float] = {
    "disease":    3.0,   # diagnóstico é o eixo mais discriminativo
    "morphology": 2.0,   # lesão morfológica é o segundo mais informativo
    "location":   1.5,   # localização anatômica complementa
    "symptom":    1.0,
    "chemical":   1.0,
}


# ---------------------------------------------------------------------------
# Construção do grafo
# ---------------------------------------------------------------------------

def build_graph(
    case_ids: list[str],
    entities_by_case: dict[str, dict[str, list[str]]],
    cache_path: Optional[Path | str] = None,
) -> nx.Graph:
    """Constrói o grafo bipartito casos ↔ entidades.

    Cada aresta leva o entity_type como atributo, permitindo
    pesquisas filtradas por tipo se necessário.

    Args:
        case_ids: todos os IDs do corpus (mesma ordem do corpus)
        entities_by_case: {case_id: {entity_type: [entity_str, ...]}}
        cache_path: se informado, salva/carrega o grafo em pickle

    Returns:
        nx.Graph com nós anotados por node_type
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            console.print(f"[green]Carregando grafo do cache: {cache_path}[/]")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    console.print(f"[bold]Construindo grafo para {len(case_ids)} documentos[/]")
    G = nx.Graph()

    # Adiciona nós de documento
    for case_id in case_ids:
        G.add_node(case_id, node_type="document")

    # Adiciona nós de entidade e arestas (document, entity)
    for case_id, entities in entities_by_case.items():
        for entity_type, entity_list in entities.items():
            weight = ENTITY_TYPE_WEIGHTS.get(entity_type, 1.0)
            for entity_str in entity_list:
                # Nó de entidade é único por (tipo, valor)
                entity_node = f"{entity_type}::{entity_str}"
                if not G.has_node(entity_node):
                    G.add_node(
                        entity_node,
                        node_type="entity",
                        entity_type=entity_type,
                        entity_value=entity_str,
                    )
                G.add_edge(case_id, entity_node, weight=weight)

    n_docs = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "document")
    n_ent = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "entity")
    console.print(
        f"  Grafo: {n_docs} nós-documento | "
        f"{n_ent} nós-entidade | "
        f"{G.number_of_edges()} arestas"
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(G, f)
        console.print(f"[green]Grafo salvo em {cache_path}[/]")

    return G


# ---------------------------------------------------------------------------
# IDF por nó de entidade
# ---------------------------------------------------------------------------

def compute_entity_idf(graph: nx.Graph, n_docs: int) -> dict[str, float]:
    """Calcula IDF para cada nó de entidade no grafo.

    df(e) = número de documentos vizinhos do nó e
    idf(e) = log( N / df(e) )

    Entidades muito comuns (df alto) recebem IDF baixo,
    reduzindo seu peso na pontuação final.
    """
    idf: dict[str, float] = {}
    for node, data in graph.nodes(data=True):
        if data.get("node_type") != "entity":
            continue
        df = graph.degree(node)  # número de documentos que contêm essa entidade
        idf[node] = math.log(n_docs / (df + 1))
    return idf


# ---------------------------------------------------------------------------
# Recuperação via grafo
# ---------------------------------------------------------------------------

def graph_search(
    query_ids: list[str],
    corpus_ids: list[str],
    graph: nx.Graph,
    entities_by_case: dict[str, dict[str, list[str]]],
    top_k: int = 50,
) -> dict[str, list[str]]:
    """Recuperação por traversal de 2 saltos no grafo de entidades.

    Algoritmo:
        1. Para cada query, percorre seus nós de entidade (1º salto)
        2. De cada entidade, coleta documentos vizinhos (2º salto)
        3. Pontua candidatos por: Σ type_weight × idf(entidade compartilhada)
        4. Remove auto-match e retorna top-k

    Args:
        query_ids: IDs das queries de teste
        corpus_ids: IDs do corpus completo (para filtro)
        graph: grafo bipartito construído por build_graph
        entities_by_case: cache de entidades (para acesso direto)
        top_k: tamanho do ranking final por query

    Returns:
        {query_id: [doc_id ordenados por score decrescente]}
    """
    console.print(
        f"[bold]GraphRAG: recuperando top-{top_k} "
        f"para {len(query_ids)} queries[/]"
    )

    corpus_set = set(corpus_ids)
    n_docs = len(corpus_ids)
    idf = compute_entity_idf(graph, n_docs)

    rankings: dict[str, list[str]] = {}

    for query_id in query_ids:
        scores: dict[str, float] = defaultdict(float)

        if query_id not in graph:
            rankings[query_id] = []
            continue

        # 1º salto: entidades do documento-query
        for entity_node in graph.neighbors(query_id):
            node_data = graph.nodes[entity_node]
            if node_data.get("node_type") != "entity":
                continue

            edge_weight = graph.edges[query_id, entity_node]["weight"]
            entity_idf = idf.get(entity_node, 0.0)
            contribution = edge_weight * entity_idf

            # 2º salto: documentos que compartilham essa entidade
            for neighbor in graph.neighbors(entity_node):
                if neighbor == query_id:
                    continue
                if neighbor not in corpus_set:
                    continue
                scores[neighbor] += contribution

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        rankings[query_id] = [doc_id for doc_id, _ in ranked]

    # Queries sem nenhum candidato
    n_empty = sum(1 for r in rankings.values() if not r)
    if n_empty:
        console.print(
            f"[yellow]  Atenção: {n_empty} queries sem candidatos no grafo[/]"
        )

    return rankings
