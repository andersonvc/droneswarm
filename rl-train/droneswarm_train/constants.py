"""Constants re-exported from the Rust droneswarm_env module.

All observation layout values come from drone-lib/src/game/obs_layout.rs
via the PyO3 bindings. This module re-exports them so Python code has
a single import point and never hardcodes values.
"""

import droneswarm_env

EGO_DIM: int = droneswarm_env.EGO_DIM
ENTITY_DIM: int = droneswarm_env.ENTITY_DIM
ACT_DIM: int = droneswarm_env.ACT_DIM
MAX_ENTITIES: int = droneswarm_env.MAX_ENTITIES

# Entity token field indices
TYPE_FLAG_IDX = 6

# Entity type flag values (encoded in obs_layout.rs)
ENEMY_DRONE = 0.0
FRIENDLY_DRONE = 0.33
ENEMY_TARGET = 0.67
FRIENDLY_TARGET = 1.0
TYPE_THRESHOLD = 0.17

# Logit masking fill value — must be large negative but finite for MPS stability.
MASK_FILL_VALUE = -1e4

# Simulation speed constants
SPEED_MULTIPLIER = 4.0
BASE_SPEED = 20.0
DT = 0.05
MIN_MAX_TICKS = 10_000
