import torch

from nash_drl.data import Action, CSRMap, GlobalState
from nash_drl.domain import DirectedGraph, Edge
from nash_drl.routing import DijkstraMapper


def test_dijkstra_mapper_uses_negative_and_positive_weights_without_saturation():
    graph = DirectedGraph(
        3,
        (
            Edge(0, 0, 1, 1.0, 10.0),
            Edge(1, 1, 2, 1.0, 10.0),
            Edge(2, 0, 2, 5.0, 10.0),
        ),
    )
    csr = CSRMap.from_graph(graph)
    state = GlobalState(
        csr, torch.zeros(1, 6), csr.edge_flow,
        torch.tensor([0]), torch.tensor([2]), torch.tensor([2])
    )
    mapper = DijkstraMapper()
    # Higher weight on the direct edge still means lower traversal cost under
    # the smooth transform, while negative values remain finite.
    paths = mapper.map(Action(torch.tensor([[1.0, 1.0, -5.0]])), state)
    assert paths.edge_ids[0] == [0, 1]
