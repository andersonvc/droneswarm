/// Observation layout for entity-attention RL architecture (V2).
///
/// Observations are split into:
/// 1. Ego features: fixed 25-dim vector for the drone's own state
/// 2. Entity tokens: variable-length 8-dim tokens for all visible entities
///
/// V1 legacy constants retained for wasm-lib backward compatibility.

// ============================================================================
// V2 Constants
// ============================================================================

/// Ego feature dimensionality (25 dimensions).
pub const EGO_DIM: usize = 25;

/// Per-entity token dimensionality (8 dimensions).
pub const ENTITY_DIM: usize = 8;

/// Discrete action space size (19 actions, no Hold).
pub const ACT_DIM: usize = 19;

/// Maximum entity tokens for batched padding.
pub const MAX_ENTITIES: usize = 64;

/// Max velocity for observation normalization.
pub const MAX_VELOCITY: f32 = 20.0;

/// Flattened V2 observation size for MLP compatibility (Phase 1).
pub const OBS_DIM_V2: usize = EGO_DIM + MAX_ENTITIES * ENTITY_DIM;

/// Ego feature indices [0..25].
pub mod ego_idx {
    // Position & velocity
    pub const X: usize = 0;
    pub const Y: usize = 1;
    pub const VX: usize = 2;
    pub const VY: usize = 3;
    pub const HEADING: usize = 4;
    pub const LAST_ACTION: usize = 5;
    // Task type one-hot [6..15] (9 task types)
    pub const TASK_TYPE_START: usize = 6;
    pub const TASK_TYPE_COUNT: usize = 9;
    // Task phase ordinal [15]
    pub const TASK_PHASE: usize = 15;
    // Relative alive index [16]
    pub const RELATIVE_ALIVE_INDEX: usize = 16;
    // Global features [17..25]
    pub const GLOBAL_OWN_DRONES: usize = 17;
    pub const GLOBAL_ENEMY_DRONES: usize = 18;
    pub const GLOBAL_OWN_TARGETS: usize = 19;
    pub const GLOBAL_ENEMY_TARGETS: usize = 20;
    pub const GLOBAL_TIME_FRAC: usize = 21;
    pub const GLOBAL_THREATS: usize = 22;
    pub const GLOBAL_NEAREST_FRIENDLY_DIST: usize = 23;
    pub const GLOBAL_FRIENDLIES_IN_BLAST: usize = 24;
}

/// Entity token field indices [0..8].
pub mod entity_idx {
    pub const DX: usize = 0;
    pub const DY: usize = 1;
    pub const DIST: usize = 2;
    pub const VX: usize = 3;
    pub const VY: usize = 4;
    /// Relative heading: atan2(dy, dx) - my_heading, normalized to [-1, 1].
    pub const HEADING_REL: usize = 5;
    /// Type: 0=enemy_drone, 1=friendly_drone, 2=enemy_target, 3=friendly_target.
    pub const TYPE_FLAG: usize = 6;
    /// 1.0 for real entities, 0.0 for padding.
    pub const ALIVE_FLAG: usize = 7;
}

/// Entity type flag values for entity_idx::TYPE_FLAG.
pub mod entity_type {
    pub const ENEMY_DRONE: f32 = 0.0;
    pub const FRIENDLY_DRONE: f32 = 1.0;
    pub const ENEMY_TARGET: f32 = 2.0;
    pub const FRIENDLY_TARGET: f32 = 3.0;
}

// ============================================================================
// V1 Legacy Constants (wasm-lib backward compatibility)
// ============================================================================

/// V1 per-drone observation dimensionality (flat 64-dim vector).
pub const OBS_DIM: usize = 64;

/// Named indices for the V1 64-dim per-drone observation vector.
pub mod obs_idx {
    // [0..5] Ego state
    pub const EGO_X: usize = 0;
    pub const EGO_Y: usize = 1;
    pub const EGO_VX: usize = 2;
    pub const EGO_VY: usize = 3;
    pub const EGO_LAST_ACTION: usize = 4;

    // [5..25] 4 nearest enemy drones: (dx/w, dy/w, dist/diag, vx/max_v, vy/max_v) x 4
    pub const ENEMY_DRONES_START: usize = 5;
    pub const ENEMY_DRONES_COUNT: usize = 4;
    pub const ENEMY_DRONE_STRIDE: usize = 5;

    // [25..40] 3 nearest friendly drones: (dx/w, dy/w, dist/diag, vx/max_v, vy/max_v) x 3
    pub const FRIENDLY_DRONES_START: usize = 25;
    pub const FRIENDLY_DRONES_COUNT: usize = 3;
    pub const FRIENDLY_DRONE_STRIDE: usize = 5;

    // [40..46] 3 nearest enemy targets: (dx/w, dy/w) x 3
    pub const ENEMY_TARGETS_START: usize = 40;
    pub const ENEMY_TARGETS_COUNT: usize = 3;
    pub const ENEMY_TARGET_STRIDE: usize = 2;

    // [46..52] 3 nearest friendly targets: (dx/w, dy/w) x 3
    pub const FRIENDLY_TARGETS_START: usize = 46;
    pub const FRIENDLY_TARGETS_COUNT: usize = 3;
    pub const FRIENDLY_TARGET_STRIDE: usize = 2;

    // [52..60] Global features
    pub const GLOBAL_OWN_DRONES: usize = 52;
    pub const GLOBAL_ENEMY_DRONES: usize = 53;
    pub const GLOBAL_OWN_TARGETS: usize = 54;
    pub const GLOBAL_ENEMY_TARGETS: usize = 55;
    pub const GLOBAL_TIME_FRAC: usize = 56;
    pub const GLOBAL_THREATS: usize = 57;
    pub const GLOBAL_NEAREST_FRIENDLY_DIST: usize = 58;
    pub const GLOBAL_FRIENDLIES_IN_BLAST: usize = 59;

    // [60] Agent ID
    pub const AGENT_ID: usize = 60;

    // [61..64] Last actions of 3 nearest friendly drones
    pub const FRIENDLY_ACTIONS_START: usize = 61;
    pub const FRIENDLY_ACTIONS_COUNT: usize = 3;

    // Per-drone relative offsets within a drone tuple
    pub const REL_DX: usize = 0;
    pub const REL_DY: usize = 1;
    pub const REL_DIST: usize = 2;
    pub const REL_VX: usize = 3;
    pub const REL_VY: usize = 4;
}
