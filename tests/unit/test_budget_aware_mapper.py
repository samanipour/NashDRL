import torch

from nash_drl.data import Action, CSRMap, GlobalState
from nash_drl.domain import DirectedGraph, Edge
from nash_drl.routing import BudgetAwareDijkstraMapper


def _state():
    graph=DirectedGraph(4, (
        Edge(0,0,1,1.0,100), Edge(1,1,3,10.0,100),
        Edge(2,0,2,1.0,100), Edge(3,2,3,1.0,100),
    ))
    csr=CSRMap.from_graph(graph)
    return GlobalState(
        csr_map=csr, agent_features=torch.zeros(1,6), edge_flow=torch.zeros(4),
        current_nodes=torch.tensor([0]), next_destinations=torch.tensor([3]),
        final_destinations=torch.tensor([3]), remaining_budgets=torch.tensor([20.0]),
        active_mask=torch.tensor([1.0]),
    )


def test_budget_mapper_repairs_expensive_route():
    state=_state()
    # Prefer 0->1->3, but it is too expensive under the charging model.
    action=Action(torch.tensor([[10.0,10.0,0.0,0.0]]))
    mapper=BudgetAwareDijkstraMapper(charging_overhead=6.0, charging_fixed_cost=1.0)
    paths=mapper.map(action,state)
    assert paths.edge_ids[0] == [2,3]
    assert mapper.last_diagnostics["repair_count"] == 1
