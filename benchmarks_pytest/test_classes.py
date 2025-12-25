"""Benchmarks for graph classes"""

import pytest

import networkx as nx


@pytest.fixture(
    scope="module", params=["Graph", "DiGraph", "MultiGraph", "MultiDiGraph"]
)
def graph_type(request):
    return request.param


@pytest.fixture
def empty_graph(graph_type):
    """Create an empty graph of the specified type"""
    return getattr(nx, graph_type)()


@pytest.fixture
def nodes():
    return list(range(1, 1000))


@pytest.fixture
def edges():
    return []


@pytest.fixture
def subgraph_nodes():
    return list(range(1, 100))


@pytest.fixture
def subgraph_nodes_large():
    return list(range(1, 900))


@pytest.mark.benchmark
def test_graph_create(benchmark, graph_type):
    """Benchmark graph creation"""
    benchmark(lambda: getattr(nx, graph_type)())


@pytest.mark.benchmark
def test_add_nodes_from(benchmark, empty_graph, nodes):
    """Benchmark adding nodes from list"""

    def add_nodes():
        G = empty_graph.copy()
        G.add_nodes_from(nodes)

    benchmark(add_nodes)


@pytest.mark.benchmark
def test_add_edges_from(benchmark, empty_graph, edges):
    """Benchmark adding edges from list"""

    def add_edges():
        G = empty_graph.copy()
        G.add_edges_from(edges)

    benchmark(add_edges)


@pytest.mark.benchmark
def test_copy(benchmark, empty_graph):
    """Benchmark graph copy"""
    benchmark(lambda: empty_graph.copy())


@pytest.mark.benchmark
def test_to_directed(benchmark, empty_graph):
    """Benchmark converting to directed"""
    benchmark(lambda: empty_graph.to_directed())


@pytest.mark.benchmark
def test_to_undirected(benchmark, empty_graph):
    """Benchmark converting to undirected"""
    benchmark(lambda: empty_graph.to_undirected())


@pytest.mark.benchmark
def test_subgraph(benchmark, empty_graph, subgraph_nodes):
    """Benchmark small subgraph creation"""

    def create_subgraph():
        _ = empty_graph.subgraph(subgraph_nodes).copy()

    benchmark(create_subgraph)


@pytest.mark.benchmark
def test_subgraph_large(benchmark, empty_graph, subgraph_nodes_large):
    """Benchmark large subgraph creation"""

    def create_subgraph():
        _ = empty_graph.subgraph(subgraph_nodes_large).copy()

    benchmark(create_subgraph)
