/// Linear Congruential Generator — deterministic, fast, no-dependency RNG.
/// Used by both sim_runner and wasm-lib for reproducible simulations.
pub fn lcg_next(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1103515245).wrapping_add(12345);
    *state
}

/// Return a float in [0, 1) from the LCG.
pub fn lcg_f32(state: &mut u32) -> f32 {
    lcg_next(state);
    *state as f32 / u32::MAX as f32
}
