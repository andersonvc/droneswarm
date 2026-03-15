/// Named indices for the 64-dim per-drone observation vector.
///
/// Using named constants prevents off-by-one errors when both sim_runner and
/// wasm-lib construct or read observations.
pub mod obs_idx {
    // [0..5] Ego state
    pub const EGO_X: usize = 0;
    pub const EGO_Y: usize = 1;
    pub const EGO_VX: usize = 2;
    pub const EGO_VY: usize = 3;
    pub const EGO_LAST_ACTION: usize = 4;

    // [5..25] 4 nearest enemy drones: (dx/w, dy/w, dist/diag, vx/max_v, vy/max_v) × 4
    pub const ENEMY_DRONES_START: usize = 5;
    pub const ENEMY_DRONES_COUNT: usize = 4;
    pub const ENEMY_DRONE_STRIDE: usize = 5;

    // [25..40] 3 nearest friendly drones: (dx/w, dy/w, dist/diag, vx/max_v, vy/max_v) × 3
    pub const FRIENDLY_DRONES_START: usize = 25;
    pub const FRIENDLY_DRONES_COUNT: usize = 3;
    pub const FRIENDLY_DRONE_STRIDE: usize = 5;

    // [40..46] 3 nearest enemy targets: (dx/w, dy/w) × 3
    pub const ENEMY_TARGETS_START: usize = 40;
    pub const ENEMY_TARGETS_COUNT: usize = 3;
    pub const ENEMY_TARGET_STRIDE: usize = 2;

    // [46..52] 3 nearest friendly targets: (dx/w, dy/w) × 3
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

/// Per-drone observation dimensionality.
pub const OBS_DIM: usize = 64;
/// Per-drone action space size (14 discrete actions).
pub const ACT_DIM: usize = 14;
/// Max velocity for observation normalization.
pub const MAX_VELOCITY: f32 = 20.0;
