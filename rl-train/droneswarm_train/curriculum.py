"""Win-rate-based curriculum with advancement and demotion."""

import math
from typing import Optional

from .constants import BASE_SPEED, DT, MIN_MAX_TICKS, SPEED_MULTIPLIER


def compute_max_ticks(world_size: float, speed_multiplier: float = SPEED_MULTIPLIER) -> int:
    """Compute dynamic max_ticks from world size and speed."""
    distance_per_tick = BASE_SPEED * DT * speed_multiplier
    diagonal = world_size * math.sqrt(2.0)
    return max(MIN_MAX_TICKS, math.ceil(diagonal / distance_per_tick))


class Curriculum:
    """Win-rate-based curriculum with advancement and demotion."""

    def __init__(self, final_drones: int = 24, final_targets: int = 6):
        self.stages = [
            {"drones": 4, "targets": 2, "world_size": 2500.0},
            {"drones": 8, "targets": 4, "world_size": 5000.0},
            {"drones": 16, "targets": 6, "world_size": 7500.0},
            {
                "drones": final_drones,
                "targets": final_targets,
                "world_size": 10000.0,
            },
        ]
        self.current_stage = 0
        self.advance_threshold = 0.70
        self.demotion_threshold = 0.30
        self.required_consecutive = 5
        self.demotion_consecutive = 10
        self.consecutive_above = 0
        self.consecutive_below = 0

    def report_eval(self, win_rate: float) -> Optional[str]:
        """Report an evaluation win rate.

        Returns 'advanced', 'demoted', or None.
        """
        if self.current_stage < len(self.stages) - 1:
            if win_rate >= self.advance_threshold:
                self.consecutive_above += 1
            else:
                self.consecutive_above = 0

            if self.consecutive_above >= self.required_consecutive:
                self.current_stage += 1
                self.consecutive_above = 0
                self.consecutive_below = 0
                return "advanced"

        if win_rate < self.demotion_threshold:
            self.consecutive_below += 1
        else:
            self.consecutive_below = 0

        if (
            self.consecutive_below >= self.demotion_consecutive
            and self.current_stage > 0
        ):
            self.current_stage -= 1
            self.consecutive_above = 0
            self.consecutive_below = 0
            return "demoted"

        return None

    def current(self) -> dict:
        return self.stages[self.current_stage]

    def is_final(self) -> bool:
        return self.current_stage >= len(self.stages) - 1

    def stage_label(self) -> str:
        s = self.stages[self.current_stage]
        return f"{s['drones']}v{s['drones']}/{s['targets']}t"
