//! Shared reward computation for multi-agent RL training.

use crate::types::Bounds;

use super::result::GameResult;
use super::state::GameDrone;

/// Inputs for computing the multi-agent reward.
pub struct RewardInput {
    pub game_result: GameResult,
    pub snap_targets_a_alive: usize,
    pub snap_targets_b_alive: usize,
    pub snap_drones_a_alive: usize,
    pub snap_drones_b_alive: usize,
    pub current_targets_a_alive: usize,
    pub current_targets_b_alive: usize,
    pub current_drones_a_alive: usize,
    pub current_drones_b_alive: usize,
    pub detonation_radius: f32,
    pub tick_count: u32,
    pub max_ticks: u32,
}

/// Compute multi-agent reward (shared team reward, Group A perspective).
///
/// Uses potential-based reward shaping with advantage ratios, a ramping
/// time-pressure penalty, and a cluster density penalty.
pub fn compute_multi_reward(
    input: &RewardInput,
    group_a_drones: &[&GameDrone],
    bounds: &Bounds,
) -> f32 {
    let mut reward = 0.0;

    // Terminal reward.
    match input.game_result {
        GameResult::AWins => reward += 10.0,
        GameResult::BWins => reward -= 10.0,
        GameResult::Draw => reward -= 5.0,
        GameResult::InProgress => {}
    }

    // Potential-based reward shaping using advantage ratios.
    const W_TARGETS: f32 = 5.0;
    const W_DRONES: f32 = 1.0;

    let phi_before = W_TARGETS
        * (input.snap_targets_a_alive as f32 / (1.0 + input.snap_targets_b_alive as f32))
        + W_DRONES
            * (input.snap_drones_a_alive as f32 / (1.0 + input.snap_drones_b_alive as f32));

    let phi_after = W_TARGETS
        * (input.current_targets_a_alive as f32 / (1.0 + input.current_targets_b_alive as f32))
        + W_DRONES
            * (input.current_drones_a_alive as f32 / (1.0 + input.current_drones_b_alive as f32));

    reward += phi_after - phi_before;

    // Time pressure: step penalty ramps up as the game progresses.
    // Early game (~0%): -0.001/step. Late game (~100%): -0.01/step.
    // Quadratic ramp so agents feel increasing urgency to finish.
    let time_frac = input.tick_count as f32 / input.max_ticks.max(1) as f32;
    let step_penalty = 0.001 + 0.009 * time_frac * time_frac;
    reward -= step_penalty;

    // Cluster density penalty.
    let mut cluster_penalty = 0.0f32;
    for drone in group_a_drones {
        let pos = drone.agent.state().pos.as_vec2();
        let nearby = group_a_drones
            .iter()
            .filter(|d| {
                d.id != drone.id
                    && bounds.distance(pos, d.agent.state().pos.as_vec2())
                        <= input.detonation_radius
            })
            .count() as f32;
        cluster_penalty += (nearby - 2.0).max(0.0);
    }
    let n_drones = group_a_drones.len().max(1) as f32;
    reward -= 0.01 * cluster_penalty / n_drones;

    reward
}
