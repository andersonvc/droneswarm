//! Shared reward computation for multi-agent RL training.

use std::collections::HashMap;

use crate::types::{Bounds, Vec2};

use super::result::GameResult;
use super::state::{GameDrone, TargetState};

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
    _group_a_drones: &[&GameDrone],
    _bounds: &Bounds,
) -> f32 {
    let mut reward = 0.0;

    // Terminal reward.
    match input.game_result {
        GameResult::AWins => reward += 10.0,
        GameResult::BWins => reward -= 10.0,
        GameResult::Draw => reward -= 8.0,
        GameResult::InProgress => {}
    }

    // Target destruction: the primary objective.
    let enemy_targets_destroyed = input.snap_targets_b_alive.saturating_sub(input.current_targets_b_alive);
    let friendly_targets_lost = input.snap_targets_a_alive.saturating_sub(input.current_targets_a_alive);
    reward += 3.0 * enemy_targets_destroyed as f32;
    reward -= 3.0 * friendly_targets_lost as f32;

    // Enemy drone kills (reduces their effective target penalty on us).
    let enemy_drones_killed = input.snap_drones_b_alive.saturating_sub(input.current_drones_b_alive);
    reward += 0.3 * enemy_drones_killed as f32;

    // Own drone losses are NEUTRAL — drones are expendable.

    // Time pressure.
    let time_frac = input.tick_count as f32 / input.max_ticks.max(1) as f32;
    let step_penalty = 0.001 + 0.009 * time_frac * time_frac;
    reward -= step_penalty;

    reward
}

/// Per-drone detonation event for individual reward computation.
pub struct DetonationEvent {
    /// ID of the drone that detonated.
    pub drone_id: usize,
    /// Group of the detonating drone.
    pub group: u32,
    /// Position where detonation occurred.
    pub blast_pos: Vec2,
    /// Whether the blast hit any enemy target.
    pub hit_target: bool,
    /// Whether the blast hit any enemy drone(s).
    pub hit_enemy_drone: bool,
}

/// Compute individual rewards for Group A drones.
///
/// Aligned with win condition:
///   - Destroying enemy targets is the primary objective
///   - Intercepting enemies threatening friendly targets is critical defense
///   - Drones are expendable weapons, no death penalty
///
/// Returns a HashMap of drone_id → individual_reward.
pub fn compute_individual_rewards(
    detonation_events: &[DetonationEvent],
    _dead_drone_ids: &[usize],
    group_a_drones: &[&GameDrone],
    all_drones: &[GameDrone],
    friendly_targets: &[TargetState],
    threat_radius: f32,
    bounds: &Bounds,
) -> HashMap<usize, f32> {
    let mut rewards: HashMap<usize, f32> = HashMap::new();

    // +2.0 for blast hitting enemy target (primary objective).
    // +0.5 for blast hitting enemy drone (reduces enemy force).
    // +1.0 bonus if that enemy drone was near a friendly target (critical intercept).
    for event in detonation_events {
        if event.group != 0 {
            continue;
        }
        let mut r = 0.0f32;
        if event.hit_target {
            r += 2.0;
        }
        if event.hit_enemy_drone {
            r += 0.5;
            // Check if the blast was near any friendly target (defensive intercept).
            let alive_friendly: Vec<Vec2> = friendly_targets
                .iter()
                .filter(|t| !t.destroyed)
                .map(|t| t.pos.as_vec2())
                .collect();
            let near_friendly = alive_friendly.iter().any(|&tpos| {
                let dx = event.blast_pos.x - tpos.x;
                let dy = event.blast_pos.y - tpos.y;
                (dx * dx + dy * dy).sqrt() <= threat_radius
            });
            if near_friendly {
                r += 1.0; // Critical intercept: stopped an attack on our target
            }
        }
        *rewards.entry(event.drone_id).or_insert(0.0) += r;
    }

    // Small reward for drones actively engaging enemies near friendly targets.
    // This teaches drones to move toward threats, not just wait.
    let alive_friendly: Vec<Vec2> = friendly_targets
        .iter()
        .filter(|t| !t.destroyed)
        .map(|t| t.pos.as_vec2())
        .collect();

    if !alive_friendly.is_empty() {
        // Find enemy drones that are within threat_radius of any friendly target.
        let threatening_enemies: Vec<Vec2> = all_drones
            .iter()
            .filter(|d| d.group != 0)
            .filter(|d| {
                let epos = d.agent.state().pos.as_vec2();
                alive_friendly.iter().any(|&tpos| bounds.distance(tpos, epos) <= threat_radius)
            })
            .map(|d| d.agent.state().pos.as_vec2())
            .collect();

        if !threatening_enemies.is_empty() {
            for drone in group_a_drones {
                let dpos = drone.agent.state().pos.as_vec2();
                // Reward for being close to a threatening enemy (intercepting).
                let min_dist = threatening_enemies
                    .iter()
                    .map(|&epos| bounds.distance(dpos, epos))
                    .fold(f32::INFINITY, f32::min);
                if min_dist <= threat_radius {
                    // Scaled: closer = more reward. Max 0.05 per step.
                    let proximity = 1.0 - (min_dist / threat_radius).min(1.0);
                    *rewards.entry(drone.id).or_insert(0.0) += 0.05 * proximity;
                }
            }
        }
    }

    rewards
}
