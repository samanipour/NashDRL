from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from nash_drl.domain import DirectedGraph


@dataclass(slots=True)
class CSRMap:
    """CSR adjacency plus edge-property arrays.

    Shapes:
        row_ptr: [V+1]
        col_idx: [E]
        edge_ids: [E]
        edge_len: [E]
        edge_cap: [E]
        edge_flow: [E]
        edge_from: [E]
    """

    row_ptr: Tensor
    col_idx: Tensor
    edge_ids: Tensor
    edge_len: Tensor
    edge_cap: Tensor
    edge_flow: Tensor
    edge_from: Tensor

    @classmethod
    def from_graph(cls, graph: DirectedGraph, *, device: torch.device | None = None) -> "CSRMap":
        edges = sorted(graph.edges, key=lambda e: (e.source, e.destination, e.id))
        row_ptr = [0]
        col_idx: list[int] = []
        edge_ids: list[int] = []
        edge_len = [0.0] * graph.num_edges
        edge_cap = [0.0] * graph.num_edges
        edge_from = [0] * graph.num_edges

        by_source: list[list[int]] = [[] for _ in range(graph.num_nodes)]
        for e in edges:
            by_source[e.source].append(e.id)
            edge_len[e.id] = e.length_km
            edge_cap[e.id] = e.capacity_vph
            edge_from[e.id] = e.source

        edge_by_id = {e.id: e for e in edges}
        for source in range(graph.num_nodes):
            ids = sorted(by_source[source], key=lambda i: edge_by_id[i].destination)
            col_idx.extend(edge_by_id[i].destination for i in ids)
            edge_ids.extend(ids)
            row_ptr.append(len(col_idx))

        kwargs = {"device": device} if device is not None else {}
        return cls(
            row_ptr=torch.tensor(row_ptr, dtype=torch.long, **kwargs),
            col_idx=torch.tensor(col_idx, dtype=torch.long, **kwargs),
            edge_ids=torch.tensor(edge_ids, dtype=torch.long, **kwargs),
            edge_len=torch.tensor(edge_len, dtype=torch.float32, **kwargs),
            edge_cap=torch.tensor(edge_cap, dtype=torch.float32, **kwargs),
            edge_flow=torch.zeros(graph.num_edges, dtype=torch.float32, **kwargs),
            edge_from=torch.tensor(edge_from, dtype=torch.long, **kwargs),
        )

    @property
    def num_nodes(self) -> int:
        return int(self.row_ptr.numel() - 1)

    @property
    def num_edges(self) -> int:
        return int(self.edge_len.numel())

    def outgoing_edge_ids(self, node: int) -> Tensor:
        start = int(self.row_ptr[node].item())
        end = int(self.row_ptr[node + 1].item())
        return self.edge_ids[start:end]

    def edge_by_id(self, edge_id: int):
        """Return the domain edge metadata by stable edge id."""
        idx = (self.edge_ids == int(edge_id)).nonzero(as_tuple=False)
        if idx.numel() == 0:
            raise KeyError(f"Unknown edge id: {edge_id}")
        pos = int(idx[0].item())
        # edge_from + col_idx identify endpoints; length/capacity are indexed by edge id.
        from nash_drl.domain import Edge
        source = int(self.edge_from[edge_id].item())
        destination = int(self.col_idx[pos].item())
        return Edge(edge_id, source, destination, float(self.edge_len[edge_id]), float(self.edge_cap[edge_id]))
