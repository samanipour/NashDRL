from __future__ import annotations

import heapq

import torch

from nash_drl.data import Action, GlobalState, Paths

from .mapper import ActionToPathMapper


class DijkstraMapper(ActionToPathMapper):
    """Deterministic mapper using reciprocal positive edge weights as costs."""

    def map(self, action: Action, state: GlobalState) -> Paths:
        action.validate()
        paths: list[list[int]] = []
        for agent_idx in range(action.num_agents):
            source = int(state.current_nodes[agent_idx].item())
            destination = int(state.next_destinations[agent_idx].item())
            paths.append(self._shortest_path(state, action.edge_weights[agent_idx], source, destination))
        return Paths(paths)

    def _shortest_path(self, state: GlobalState, weights: torch.Tensor, source: int, destination: int) -> list[int]:
        if source == destination:
            return []
        dist = {source: 0.0}
        prev: dict[int, tuple[int, int]] = {}
        heap = [(0.0, source)]
        while heap:
            d, node = heapq.heappop(heap)
            if node == destination:
                break
            if d > dist.get(node, float("inf")):
                continue
            for edge_tensor in state.csr_map.outgoing_edge_ids(node):
                edge_id = int(edge_tensor.item())
                edge_positions = (state.csr_map.edge_ids == edge_id).nonzero(as_tuple=False)
                if edge_positions.numel() == 0:
                    continue
                pos = int(edge_positions[0].item())
                next_node = int(state.csr_map.col_idx[pos].item())
                weight = float(weights[edge_id].item())
                cost = 1.0 / max(weight, 1e-6)
                candidate = d + cost
                if candidate < dist.get(next_node, float("inf")):
                    dist[next_node] = candidate
                    prev[next_node] = (node, edge_id)
                    heapq.heappush(heap, (candidate, next_node))
        if destination not in prev:
            return []
        path: list[int] = []
        node = destination
        while node != source:
            parent, edge_id = prev[node]
            path.append(edge_id)
            node = parent
        path.reverse()
        return path
