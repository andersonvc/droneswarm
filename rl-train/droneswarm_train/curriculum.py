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
        self.advance_threshold = 0.55
        self.demotion_threshold = 0.20
        self.required_consecutive = 3
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
            # Don't reset — count total evals above threshold, not consecutive.

            if self.consecutive_above >= self.required_consecutive:
                self.current_stage += 1
                self.consecutive_above = 0
                self.consecutive_below = 0
                return "advanced"

        return None

    def current(self) -> dict:
        return self.stages[self.current_stage]

    def is_final(self) -> bool:
        return self.current_stage >= len(self.stages) - 1

    def stage_label(self) -> str:
        s = self.stages[self.current_stage]
        return f"{s['drones']}v{s['drones']}/{s['targets']}t"

    def compute_n_steps(self) -> int:
        """Compute rollout n_steps to cover ~1.5 full episodes at this stage.

        Uses the world diagonal to estimate max episode length in decision steps,
        then adds 50% headroom so most episodes complete within a single rollout.
        """
        stage = self.current()
        decision_interval = 10  # MULTI_DECISION_INTERVAL
        distance_per_tick = BASE_SPEED * DT * SPEED_MULTIPLIER
        diagonal = stage["world_size"] * math.sqrt(2.0)
        max_ticks = math.ceil(diagonal / distance_per_tick)
        episode_steps = max_ticks / decision_interval
        # 1.5x episode length, rounded up to nearest 64.
        n_steps = int(math.ceil(episode_steps * 1.5 / 64.0)) * 64
        return max(n_steps, 64)

    def compute_n_envs(self, memory_budget_gb: float = 12.0) -> int:
        """Compute n_envs that fits within memory budget.

        Estimates per-transition memory: ego + entities + overhead.
        Accounts for drone count scaling with stage.
        """
        stage = self.current()
        n_steps = self.compute_n_steps()
        drones_per_env = stage["drones"] * 2  # both groups
        # Rough per-transition bytes: (25 + 64*10 + 10) * 4 bytes = ~2.7KB
        bytes_per_transition = 2800
        transitions_per_env = n_steps * stage["drones"]  # Group A drones only
        bytes_per_env = transitions_per_env * bytes_per_transition
        # Leave 4GB headroom for model, optimizer, GPU overhead.
        usable_bytes = (memory_budget_gb - 4.0) * 1e9
        n_envs = int(usable_bytes / bytes_per_env)
        # Clamp to reasonable range and round down to power of 2.
        n_envs = max(32, min(n_envs, 2048))
        # Round down to nearest power of 2.
        n_envs = 2 ** int(math.log2(n_envs))
        return n_envs
