from __future__ import annotations

import heapq
import math
from collections import Counter

import torch

from nash_drl.data import Action, GlobalState, Paths

from .dijkstra import DijkstraMapper


class BudgetAwareDijkstraMapper(DijkstraMapper):
    """Deterministic budget-feasible path mapper for NashDRL.

    The Actor supplies edge preferences. The mapper first computes preference
    routes, then repairs routes that exceed remaining budget using live SUMO
    background flow plus the other vehicles' currently proposed routes.
    """

    def __init__(self, weight_clip: float = 10.0, energy_rate_kwh_per_km: float = 1.0,
                 charging_overhead: float = 6.0, charging_fixed_cost: float = 1.0,
                 charging_floor_price: float = 0.0, max_repair_passes: int = 2) -> None:
        super().__init__(weight_clip=weight_clip)
        if max_repair_passes < 1:
            raise ValueError("max_repair_passes must be >= 1")
        self.energy_rate = float(energy_rate_kwh_per_km)
        self.overhead = float(charging_overhead)
        self.fixed_cost = float(charging_fixed_cost)
        self.floor_price = float(charging_floor_price)
        self.max_repair_passes = int(max_repair_passes)
        self.last_diagnostics: dict[str, object] = {}

    def map(self, action: Action, state: GlobalState) -> Paths:
        action.validate()
        n = action.num_agents
        active = self._active(state, n)
        budgets = self._budgets(state, n)
        paths = [[] for _ in range(n)]

        for i in range(n):
            if not active[i]:
                continue
            source = int(state.current_nodes[i].item())
            destination = int(state.next_destinations[i].item())
            paths[i] = self._shortest_path(state, action.edge_weights[i], source, destination)

        repaired: list[int] = []
        initial_excess: dict[int, float] = {}

        for _ in range(self.max_repair_passes):
            changed = False
            planned_flow = self._planned_flow(paths, state.edge_flow.numel())
            for i in range(n):
                if not active[i]:
                    continue
                current_counter = Counter(paths[i])
                estimated = self._route_cost(paths[i], state, planned_flow, current_counter)
                excess = estimated - budgets[i]
                if excess <= 1e-6:
                    continue
                initial_excess.setdefault(i, float(excess))
                source = int(state.current_nodes[i].item())
                destination = int(state.next_destinations[i].item())
                candidate = self._budget_feasible_path(
                    state, action.edge_weights[i], source, destination, budgets[i],
                    planned_flow, current_counter,
                )
                if candidate is not None and candidate != paths[i]:
                    paths[i] = candidate
                    repaired.append(i)
                    planned_flow = self._planned_flow(paths, state.edge_flow.numel())
                    changed = True
            if not changed:
                break

        final_flow = self._planned_flow(paths, state.edge_flow.numel())
        final_costs = []
        residual = []
        for i in range(n):
            if not active[i]:
                final_costs.append(0.0)
                residual.append(0.0)
                continue
            c = self._route_cost(paths[i], state, final_flow, Counter(paths[i]))
            final_costs.append(c)
            residual.append(max(0.0, c - budgets[i]))

        self.last_diagnostics = {
            "repaired_agents": sorted(set(repaired)),
            "initial_excess": initial_excess,
            "final_estimated_costs": final_costs,
            "residual_budget_excess": residual,
            "repair_count": len(set(repaired)),
        }
        return Paths(paths)

    @staticmethod
    def _active(state: GlobalState, n: int) -> list[bool]:
        if state.active_mask is not None:
            return [bool(x) for x in state.active_mask.detach().cpu().tolist()]
        return [True] * n

    @staticmethod
    def _budgets(state: GlobalState, n: int) -> list[float]:
        if state.remaining_budgets is None:
            raise ValueError("Budget-aware Nash routing requires state.remaining_budgets")
        values = [float(x) for x in state.remaining_budgets.detach().cpu().tolist()]
        if len(values) != n:
            raise ValueError("remaining_budgets must contain one value per agent")
        return values

    @staticmethod
    def _planned_flow(paths: list[list[int]], num_edges: int) -> torch.Tensor:
        flow = torch.zeros(num_edges, dtype=torch.float32)
        for path in paths:
            for eid in path:
                flow[eid] += 1.0
        return flow

    def _edge_charge(self, edge_id: int, state: GlobalState, other_flow: float) -> float:
        edge = state.csr_map.edge_by_id(edge_id)
        total_flow = float(state.edge_flow[edge_id]) + other_flow + 1.0
        price = max(self.floor_price, self.overhead / max(total_flow, 1e-6) + self.fixed_cost)
        return self.energy_rate * float(edge.length_km) * price

    def _route_cost(self, path: list[int], state: GlobalState, planned_flow: torch.Tensor, own_counter: Counter[int]) -> float:
        cost = 0.0
        consumed = Counter()
        for eid in path:
            consumed[eid] += 1
            other_flow = max(0.0, float(planned_flow[eid]) - float(own_counter[eid]))
            # The candidate agent contributes one vehicle; _edge_charge adds it.
            cost += self._edge_charge(eid, state, other_flow)
        return cost

    def _budget_feasible_path(self, state: GlobalState, weights: torch.Tensor, source: int, destination: int,
                              budget: float, planned_flow: torch.Tensor, own_counter: Counter[int]) -> list[int] | None:
        if source == destination:
            return []
        # Positive edge costs + visited-node labels prevent cycling. The label
        # stores (preference objective, economic cost, parent, edge).
        labels: dict[int, tuple[float, float, int | None, int | None]] = {source: (0.0, 0.0, None, None)}
        heap = [(0.0, 0.0, source)]
        while heap:
            objective, spent, node = heapq.heappop(heap)
            current = labels.get(node)
            if current is None or objective > current[0] + 1e-12 or spent > current[1] + 1e-12:
                continue
            if node == destination:
                result: list[int] = []
                cur = node
                while cur != source:
                    _, _, parent, eid = labels[cur]
                    assert parent is not None and eid is not None
                    result.append(eid)
                    cur = parent
                result.reverse()
                return result

            for edge_tensor in state.csr_map.outgoing_edge_ids(node):
                eid = int(edge_tensor.item())
                edge = state.csr_map.edge_by_id(eid)
                # Current candidate agent is not represented in planned_flow yet;
                # subtract its old route contribution and add this candidate edge.
                other_flow = max(0.0, float(planned_flow[eid]) - float(own_counter[eid]))
                charge = self._edge_charge(eid, state, other_flow)
                new_spent = spent + charge
                if new_spent > budget + 1e-6:
                    continue
                pref = self._edge_cost(float(weights[eid].item()))
                new_objective = objective + pref + 1e-4 * charge
                old = labels.get(edge.destination)
                if old is None or new_objective < old[0] - 1e-12 or (abs(new_objective-old[0]) < 1e-12 and new_spent < old[1]):
                    labels[edge.destination] = (new_objective, new_spent, node, eid)
                    heapq.heappush(heap, (new_objective, new_spent, edge.destination))
        return None
