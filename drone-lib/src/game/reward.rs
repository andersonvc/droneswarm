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

    // Terminal reward (Refactor 3: scaled down from ±10 to ±5).
    match input.game_result {
        GameResult::AWins => reward += 5.0,
        GameResult::BWins => reward -= 5.0,
        GameResult::Draw => reward -= 3.0,
        GameResult::InProgress => {}
    }

    // Target destruction: the primary objective.
    let enemy_targets_destroyed = input.snap_targets_b_alive.saturating_sub(input.current_targets_b_alive);
    let friendly_targets_lost = input.snap_targets_a_alive.saturating_sub(input.current_targets_a_alive);
    reward += 3.0 * enemy_targets_destroyed as f32;
    reward -= 3.0 * friendly_targets_lost as f32;

    // Enemy drone kills.
    let enemy_drones_killed = input.snap_drones_b_alive.saturating_sub(input.current_drones_b_alive);
    reward += 0.3 * enemy_drones_killed as f32;

    // Time pressure (reduced coefficient for cleaner signal).
    let time_frac = input.tick_count as f32 / input.max_ticks.max(1) as f32;
    let step_penalty = 0.0005 + 0.0005 * time_frac;
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
    /// Number of friendly drones killed by this blast (excluding self).
    pub friendly_kills: usize,
}

/// Compute individual rewards for Group A drones.
///
/// Aligned with win condition:
///   - Destroying enemy targets is the primary objective
///   - Intercepting enemies threatening friendly targets is critical defense
///   - Drones are expendable weapons, no death penalty
///
/// Returns a HashMap of drone_id → individual_reward.
/// Per-drone approach distances from previous step (for approach reward delta).
/// Passed in from sim_runner, which tracks these across steps.
pub struct ApproachContext {
    /// Per-drone distance to assigned attack target last step.
    pub prev_distances: HashMap<usize, f32>,
}

pub fn compute_individual_rewards(
    detonation_events: &[DetonationEvent],
    _dead_drone_ids: &[usize],
    group_a_drones: &[&GameDrone],
    all_drones: &[GameDrone],
    friendly_targets: &[TargetState],
    enemy_targets: &[TargetState],
    attack_targets: &HashMap<usize, crate::types::Position>,
    threat_radius: f32,
    bounds: &Bounds,
    approach_ctx: &mut ApproachContext,
) -> HashMap<usize, f32> {
    let mut rewards: HashMap<usize, f32> = HashMap::new();

    // === Refactor 2: Boosted detonation rewards ===
    // +5.0 for blast hitting enemy target (was +2.0).
    // +1.5 for blast hitting enemy drone (was +0.5).
    // +3.0 bonus for critical intercept near friendly target (was +1.0).
    for event in detonation_events {
        if event.group != 0 {
            continue;
        }
        let mut r = 0.0f32;
        if event.hit_target {
            r += 5.0;
        }
        if event.hit_enemy_drone {
            r += 1.5;
            let alive_friendly: Vec<Vec2> = friendly_targets
                .iter()
                .filter(|t| !t.destroyed)
                .map(|t| t.pos.as_vec2())
                .collect();
            let near_friendly = alive_friendly.iter().any(|&tpos| {
                bounds.distance(event.blast_pos, tpos) <= threat_radius
            });
            if near_friendly {
                r += 3.0;
            }
        }
        // Friendly fire penalty: -2.0 per friendly drone killed by this blast.
        if event.friendly_kills > 0 {
            r -= 2.0 * event.friendly_kills as f32;
        }
        *rewards.entry(event.drone_id).or_insert(0.0) += r;
    }

    // === Clustering penalty: discourage drones from bunching within blast radius ===
    // -0.05 per step for each nearby friendly drone within detonation radius.
    // Teaches drones to spread out so a single enemy blast can't kill multiple.
    let detonation_radius = threat_radius / 5.0; // threat_radius = detonation_radius * 5
    for drone in group_a_drones {
        let dpos = drone.agent.state().pos.as_vec2();
        let nearby_friendlies = group_a_drones
            .iter()
            .filter(|d| d.id != drone.id)
            .filter(|d| bounds.distance(dpos, d.agent.state().pos.as_vec2()) <= detonation_radius)
            .count();
        if nearby_friendlies > 0 {
            *rewards.entry(drone.id).or_insert(0.0) -= 0.05 * nearby_friendlies as f32;
        }
    }

    // === Refactor 1: Dense per-drone approach reward ===
    // For each drone with an active attack target, reward closing distance.
    // +0.1 * (distance_reduced / 100m), capped at +0.1 per step.
    let mut new_distances: HashMap<usize, f32> = HashMap::new();
    for drone in group_a_drones {
        if let Some(target_pos) = attack_targets.get(&drone.id) {
            let dpos = drone.agent.state().pos.as_vec2();
            let current_dist = bounds.distance(dpos, target_pos.as_vec2());
            new_distances.insert(drone.id, current_dist);

            if let Some(&prev_dist) = approach_ctx.prev_distances.get(&drone.id) {
                let dist_reduced = prev_dist - current_dist;
                if dist_reduced > 0.0 {
                    let approach_reward = (0.1 * dist_reduced / 100.0).min(0.1);
                    *rewards.entry(drone.id).or_insert(0.0) += approach_reward;
                }
            }
        }
    }
    approach_ctx.prev_distances = new_distances;

    // === Refactor 4: Strengthened proximity shaping ===
    // 4x stronger: max 0.2 per step (was 0.05).
    let alive_friendly: Vec<Vec2> = friendly_targets
        .iter()
        .filter(|t| !t.destroyed)
        .map(|t| t.pos.as_vec2())
        .collect();

    if !alive_friendly.is_empty() {
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
                let min_dist = threatening_enemies
                    .iter()
                    .map(|&epos| bounds.distance(dpos, epos))
                    .fold(f32::INFINITY, f32::min);
                if min_dist <= threat_radius {
                    let proximity = 1.0 - (min_dist / threat_radius).min(1.0);
                    *rewards.entry(drone.id).or_insert(0.0) += 0.2 * proximity;
                }
            }
        }

        // === Refactor 5: Defense positioning reward ===
        // +0.05/step for drones holding position near a friendly target.
        for drone in group_a_drones {
            let dpos = drone.agent.state().pos.as_vec2();
            let near_friendly_target = alive_friendly
                .iter()
                .any(|&tpos| bounds.distance(dpos, tpos) <= 300.0);
            if near_friendly_target {
                *rewards.entry(drone.id).or_insert(0.0) += 0.05;
            }
        }
    }

    rewards
}
