from __future__ import annotations

from nash_drl.data import Action, GlobalState, Paths

from .mapper import ActionToPathMapper


class GreedyMapper(ActionToPathMapper):
    def map(self, action: Action, state: GlobalState) -> Paths:
        action.validate()
        paths: list[list[int]] = []
        for i in range(action.num_agents):
            current = 0
            visited: set[int] = set()
            path: list[int] = []
            for _ in range(state.csr_map.num_nodes):
                if current in visited:
                    break
                visited.add(current)
                outgoing = state.csr_map.outgoing_edge_ids(current)
                if outgoing.numel() == 0:
                    break
                best = max((int(e), float(action.edge_weights[i, int(e)].item())) for e in outgoing)
                edge_id = best[0]
                path.append(edge_id)
                loc = (state.csr_map.edge_ids == edge_id).nonzero(as_tuple=False)
                if loc.numel() == 0:
                    break
                current = int(state.csr_map.col_idx[int(loc[0])].item())
            paths.append(path)
        return Paths(paths)
