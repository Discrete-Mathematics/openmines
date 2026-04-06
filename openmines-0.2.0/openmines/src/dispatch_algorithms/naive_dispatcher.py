from __future__ import annotations

import numpy as np
from openmines.src.dispatcher import BaseDispatcher


class NaiveDispatcher(BaseDispatcher):
    """A lightweight baseline dispatcher.

    strategy:
    - first: always pick index 0 (legacy behavior)
    - round_robin: cycle through candidate indices
    - queue_aware: choose the site with the smallest simple score
      score = estimated_queue_wait + travel_time + 0.5 * road_truck_count
    """

    def __init__(self, strategy: str = "queue_aware"):
        super().__init__()
        self.name = "NaiveDispatcher"
        valid_strategies = {"first", "round_robin", "queue_aware"}
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. valid={sorted(valid_strategies)}")
        self.strategy = strategy
        self._load_cursor = 0
        self._dump_cursor = 0

    @staticmethod
    def _safe_speed(speed: float) -> float:
        return max(float(speed), 1e-6)

    @staticmethod
    def _safe_argmin(values) -> int:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return 0
        return int(np.argmin(arr))

    def _round_robin(self, count: int, cursor_attr: str) -> int:
        if count <= 0:
            return 0
        cursor = getattr(self, cursor_attr)
        target = cursor % count
        setattr(self, cursor_attr, cursor + 1)
        return int(target)

    def _pick_load_site(self, truck: "Truck", mine: "Mine", from_charging: bool) -> int:
        load_count = len(mine.load_sites)
        if load_count <= 0:
            return 0

        if self.strategy == "first":
            return 0
        if self.strategy == "round_robin":
            return self._round_robin(load_count, "_load_cursor")

        queue_wait = np.array([site.estimated_queue_wait_time for site in mine.load_sites], dtype=float)

        if from_charging:
            distances = np.array(mine.road.charging_to_load, dtype=float)
            road_counts = np.array(
                [
                    len(mine.road.truck_on_road(start=mine.charging_site, end=mine.load_sites[i]))
                    for i in range(load_count)
                ],
                dtype=float,
            )
        else:
            cur_index = mine.dump_sites.index(truck.current_location)
            distances = np.array(mine.road.d2l_road_matrix[:, cur_index], dtype=float)
            road_counts = np.array(
                [
                    len(mine.road.truck_on_road(start=truck.current_location, end=mine.load_sites[i]))
                    for i in range(load_count)
                ],
                dtype=float,
            )

        travel_time = 60.0 * distances / self._safe_speed(truck.truck_speed)
        score = queue_wait + travel_time + 0.5 * road_counts
        return self._safe_argmin(score)

    def _pick_dump_site(self, truck: "Truck", mine: "Mine") -> int:
        dump_count = len(mine.dump_sites)
        if dump_count <= 0:
            return 0

        if self.strategy == "first":
            return 0
        if self.strategy == "round_robin":
            return self._round_robin(dump_count, "_dump_cursor")

        cur_index = mine.load_sites.index(truck.current_location)
        distances = np.array(mine.road.l2d_road_matrix[cur_index, :], dtype=float)
        queue_wait = np.array([site.estimated_queue_wait_time for site in mine.dump_sites], dtype=float)
        road_counts = np.array(
            [
                len(mine.road.truck_on_road(start=truck.current_location, end=mine.dump_sites[i]))
                for i in range(dump_count)
            ],
            dtype=float,
        )

        travel_time = 60.0 * distances / self._safe_speed(truck.truck_speed)
        score = queue_wait + travel_time + 0.5 * road_counts
        return self._safe_argmin(score)

    def give_init_order(self, truck: "Truck", mine: "Mine") -> int:
        return self._pick_load_site(truck=truck, mine=mine, from_charging=True)

    def give_haul_order(self, truck: "Truck", mine: "Mine") -> int:
        return self._pick_dump_site(truck=truck, mine=mine)

    def give_back_order(self, truck: "Truck", mine: "Mine") -> int:
        return self._pick_load_site(truck=truck, mine=mine, from_charging=False)