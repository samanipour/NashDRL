import torch

from data import CSRMap
from domain import DirectedGraph, Edge


def test_csr_construction_matches_document_example():
    graph = DirectedGraph(4, (
        Edge(0, 0, 1, 10, 100), Edge(1, 1, 3, 20, 80), Edge(2, 0, 2, 15, 120),
        Edge(3, 2, 3, 25, 60), Edge(4, 1, 2, 5, 50),
    ))
    csr = CSRMap.from_graph(graph)
    assert csr.row_ptr.tolist() == [0, 2, 4, 5, 5]
    assert csr.col_idx.tolist() == [1, 2, 2, 3, 3]
    assert csr.edge_ids.tolist() == [0, 2, 4, 1, 3]
    assert torch.allclose(csr.edge_flow, torch.zeros(5))
