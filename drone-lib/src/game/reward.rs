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
/// Returns a HashMap of drone_id → individual_reward.
/// Rewards:
///   +0.5 for detonating drone whose blast hit a target
///   +0.2 for detonating drone whose blast hit enemy drone(s)
///   -0.3 for drone that died with no active task
///   +defensive positioning bonus for drones between enemies and friendly targets
pub fn compute_individual_rewards(
    detonation_events: &[DetonationEvent],
    dead_drone_ids: &[usize],
    group_a_drones: &[&GameDrone],
    all_drones: &[GameDrone],
    friendly_targets: &[TargetState],
    threat_radius: f32,
    bounds: &Bounds,
) -> HashMap<usize, f32> {
    let mut rewards: HashMap<usize, f32> = HashMap::new();

    // Detonation rewards for group A drones.
    for event in detonation_events {
        if event.group != 0 {
            continue;
        }
        let mut r = 0.0f32;
        if event.hit_target {
            r += 0.5;
        }
        if event.hit_enemy_drone {
            r += 0.2;
        }
        *rewards.entry(event.drone_id).or_insert(0.0) += r;
    }

    // Penalty for dying with no active task.
    for &dead_id in dead_drone_ids {
        // Check if this was a group A drone (ID < group_split).
        // We check by looking in group_a_drones list (might already be removed).
        // Since dead drones are removed, check detonation events or just track group.
        let was_group_a = dead_id < all_drones.len(); // approximate; caller should filter
        if was_group_a {
            // Check if drone had no task (already destroyed, so we can't query).
            // The caller tracks this via DroneAgent::task_info() before destruction.
            // For now, only apply if not in detonation_events (didn't detonate intentionally).
            if !detonation_events.iter().any(|e| e.drone_id == dead_id) {
                *rewards.entry(dead_id).or_insert(0.0) -= 0.3;
            }
        }
    }

    // Defensive positioning reward: reward drones between enemies and friendly targets.
    let alive_friendly: Vec<Vec2> = friendly_targets
        .iter()
        .filter(|t| !t.destroyed)
        .map(|t| t.pos.as_vec2())
        .collect();

    if !alive_friendly.is_empty() {
        for drone in group_a_drones {
            let drone_pos = drone.agent.state().pos.as_vec2();

            // Find enemies within threat_radius of any friendly target.
            for &tpos in &alive_friendly {
                for enemy in all_drones.iter().filter(|d| d.group != 0) {
                    let epos = enemy.agent.state().pos.as_vec2();
                    if bounds.distance(tpos, epos) > threat_radius {
                        continue;
                    }

                    // Check if drone is in the interception cone.
                    let target_to_enemy = Vec2::new(epos.x - tpos.x, epos.y - tpos.y);
                    let target_to_friendly =
                        Vec2::new(drone_pos.x - tpos.x, drone_pos.y - tpos.y);

                    let dot = target_to_enemy.x * target_to_friendly.x
                        + target_to_enemy.y * target_to_friendly.y;

                    // Positive dot = drone is roughly between target and enemy.
                    if dot > 0.0 {
                        let te_len = (target_to_enemy.x * target_to_enemy.x
                            + target_to_enemy.y * target_to_enemy.y)
                            .sqrt();
                        let tf_len = (target_to_friendly.x * target_to_friendly.x
                            + target_to_friendly.y * target_to_friendly.y)
                            .sqrt();
                        if te_len > 0.0 && tf_len > 0.0 {
                            let cos_angle = dot / (te_len * tf_len);
                            // Small reward proportional to how well positioned.
                            let interception_bonus = 0.01 * cos_angle.max(0.0);
                            *rewards.entry(drone.id).or_insert(0.0) += interception_bonus;
                        }
                    }
                }
            }
        }
    }

    rewards
}
