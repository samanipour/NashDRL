from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChargingStationModel:
    overhead: float = 6.0
    fixed_unit_cost: float = 1.0
    floor_price: float = 0.0

    def unit_price(self, flow: float) -> float:
        if flow <= 0:
            return self.floor_price
        return max(self.floor_price, self.overhead / flow + self.fixed_unit_cost)
