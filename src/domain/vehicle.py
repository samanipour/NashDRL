from __future__ import annotations

from dataclasses import dataclass, field

from .trip import Trip


@dataclass(slots=True)
class Vehicle:
    id: int
    trips: list[Trip]
    free_flow_speed_kmh: float
    budget: float
    current_trip_index: int = 0
    current_node: int | None = None
    remaining_budget: float | None = None

    def __post_init__(self) -> None:
        if self.remaining_budget is None:
            self.remaining_budget = self.budget
        if self.current_node is None and self.trips:
            self.current_node = self.trips[0].origin

    @property
    def next_destination(self) -> int:
        if not self.trips:
            raise ValueError("Vehicle has no trips.")
        return self.trips[self.current_trip_index].destination

    @property
    def final_destination(self) -> int:
        if not self.trips:
            raise ValueError("Vehicle has no trips.")
        return self.trips[-1].destination

    @property
    def remaining_trip_count(self) -> int:
        return max(0, len(self.trips) - self.current_trip_index)
