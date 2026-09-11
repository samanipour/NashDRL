from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from nash_drl.models.actor import ActorOutput


@dataclass(frozen=True, slots=True)
class NashDecision:
    """Unconstrained local LQ equilibrium proposal plus constraint diagnostics."""

    action: Tensor  # [N,E] or [B,N,E]
    equilibrium_type: str = "local_lq_unconstrained"


class AnalyticalNashPolicy:
    """Compute the local LQ equilibrium implied by Actor parameters.

    For the quadratic form in NashDRL-Model-V12, the first-order conditions
    have the zero-deviation solution z_i = u_i - mu_i = 0 whenever the
    interaction system is nonsingular. Thus the unconstrained local equilibrium
    is mu. Hard budgets are enforced separately by the constrained route layer.
    """

    def select(self, params: ActorOutput) -> Tensor:
        return params.mu

    def decide(self, params: ActorOutput) -> NashDecision:
        return NashDecision(action=self.select(params))


class ConstrainedNashPolicy(AnalyticalNashPolicy):
    """Named policy interface used by the constrained Nash training stack.

    The policy produces the LQ equilibrium proposal; the budget-aware route
    mapper defines the feasible action set before execution. Keeping these
    responsibilities separate prevents the Actor API from drifting from the
    five-channel Section-4 design.
    """

    def decide(self, params: ActorOutput) -> NashDecision:
        return NashDecision(action=self.select(params), equilibrium_type="budget_constrained_lq")
