"""Benchmarks for shortest path algorithms"""

import random

import pytest

import networkx as nx


@pytest.fixture(
    scope="module", params=["unweighted", "uniform", "increasing", "random"]
)
def graph_type(request):
    return request.param


@pytest.fixture(scope="module")
def connected_seven_node_graphs(graph_type):
    """Generate connected graphs with 7 nodes from graph atlas"""
    seed = 0xDEADC0DE
    connected_sevens = [
        G for G in nx.graph_atlas_g() if (len(G) == 7) and nx.is_connected(G)
    ]

    if graph_type == "uniform":
        for G in connected_sevens:
            nx.set_edge_attributes(G, values=5, name="weight")
    elif graph_type == "random":
        random.seed(seed)
        for G in connected_sevens:
            nx.set_edge_attributes(
                G,
                values={e: random.randint(1, len(G)) for e in G.edges},
                name="weight",
            )
    elif graph_type == "increasing":
        for G in connected_sevens:
            nx.set_edge_attributes(G, {e: max(e) for e in G.edges}, name="weight")

    return connected_sevens


@pytest.mark.benchmark
def test_multi_source_dijkstra_over_atlas(benchmark, connected_seven_node_graphs):
    """How long it takes to compute dijkstra multisource paths over many
    small graphs."""

    def run_dijkstra():
        for G in connected_seven_node_graphs:
            _ = nx.multi_source_dijkstra(G, sources=[0, 1])

    benchmark(run_dijkstra)


@pytest.mark.benchmark
def test_multi_source_dijkstra_over_atlas_with_target(
    benchmark, connected_seven_node_graphs
):
    """Dijkstra with target specification"""

    def run_dijkstra():
        for G in connected_seven_node_graphs:
            _ = nx.multi_source_dijkstra(G, sources=[0, 1], target=6)

    benchmark(run_dijkstra)
