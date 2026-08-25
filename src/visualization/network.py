from __future__ import annotations

from data import CSRMap


def to_networkx(csr_map: CSRMap):
    import networkx as nx
    graph = nx.DiGraph()
    graph.add_nodes_from(range(csr_map.num_nodes))
    for edge_id in range(csr_map.num_edges):
        graph.add_edge(int(csr_map.edge_from[edge_id]), int(csr_map.col_idx[(csr_map.edge_ids == edge_id).nonzero(as_tuple=False)[0]].item()), id=edge_id)
    return graph
