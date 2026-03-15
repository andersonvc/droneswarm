//! Shared RL action mapping for per-drone discrete actions.
//!
//! Translates action indices (0-13) into task assignments that both
//! sim_runner and wasm-lib can apply to drones.

use std::collections::HashMap;

use crate::tasks::attack::AttackTask;
use crate::tasks::defend::DefendTask;
use crate::tasks::evade::EvadeTask;
use crate::tasks::intercept::InterceptTask;
use crate::tasks::intercept_group::InterceptGroupTask;
use crate::tasks::patrol::PatrolTask;
use crate::types::{Bounds, Position, Vec2};

use super::patrol::build_patrol_route;
use super::state::{GameDrone, TargetState};

/// Apply an RL action (0-13) to a drone in the game engine.
///
/// Action space:
///   0-2   = Attack nth nearest enemy target (direct)
///   3-5   = Attack nth nearest enemy target (evasive)
///   6-7   = Intercept nth nearest enemy drone
///   8     = Intercept enemy cluster
///   9     = Defend nearest friendly target (tight)
///   10    = Defend nearest friendly target (wide)
///   11    = Patrol perimeter
///   12    = Evade nearest threat
///   13    = Hold (no-op)
pub fn apply_rl_action(
    drone_id: usize,
    action: u32,
    drones: &mut [GameDrone],
    targets_a: &[TargetState],
    targets_b: &[TargetState],
    attack_targets: &mut HashMap<usize, Position>,
    intercept_targets: &mut HashMap<usize, usize>,
    detonation_radius: f32,
    patrol_standoff: f32,
    bounds: &Bounds,
) {
    // 13 = Hold: keep current task
    if action == 13 {
        return;
    }

    // Find the drone's group and position.
    let (group, drone_pos) = match drones.iter().find(|d| d.id == drone_id) {
        Some(d) => (d.group, d.agent.state().pos.as_vec2()),
        None => return,
    };

    match action {
        // --- Attack (direct): target by nth-nearest ---
        0 | 1 | 2 => {
            let nth = action as usize;
            let target_pos = nth_nearest_enemy_target(drone_pos, group, nth, targets_a, targets_b, bounds);
            if let Some(target) = target_pos {
                attack_targets.remove(&drone_id);
                intercept_targets.remove(&drone_id);
                if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
                    drone.agent.set_task(Box::new(AttackTask::new(target, detonation_radius)));
                }
                attack_targets.insert(drone_id, target);
            }
        }

        // --- Attack (evasive): target by nth-nearest ---
        3 | 4 | 5 => {
            let nth = (action - 3) as usize;
            let target_pos = nth_nearest_enemy_target(drone_pos, group, nth, targets_a, targets_b, bounds);
            if let Some(target) = target_pos {
                attack_targets.remove(&drone_id);
                intercept_targets.remove(&drone_id);
                if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
                    drone.agent.set_task(Box::new(AttackTask::new_evasive(target, detonation_radius)));
                }
                attack_targets.insert(drone_id, target);
            }
        }

        // --- Intercept nearest / 2nd nearest enemy drone ---
        6 | 7 => {
            let nth = (action - 6) as usize;
            let enemy_id = nth_nearest_enemy_drone(drone_pos, group, nth, drones, bounds);
            if let Some(eid) = enemy_id {
                attack_targets.remove(&drone_id);
                intercept_targets.remove(&drone_id);
                intercept_drone_task(drone_id, eid, group, drones, intercept_targets, detonation_radius);
            }
        }

        // --- Intercept enemy cluster ---
        8 => {
            attack_targets.remove(&drone_id);
            intercept_targets.remove(&drone_id);
            if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
                drone.agent.set_task(Box::new(InterceptGroupTask::new(group, detonation_radius)));
            }
        }

        // --- Defend nearest friendly target (tight: 100m orbit, 300m engage) ---
        9 => {
            let target_pos = nth_nearest_friendly_target(drone_pos, group, 0, targets_a, targets_b, bounds);
            if let Some(center) = target_pos {
                attack_targets.remove(&drone_id);
                intercept_targets.remove(&drone_id);
                if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
                    drone.agent.set_task(Box::new(DefendTask::new(
                        drone_id, group, center, 100.0, 300.0, detonation_radius,
                    )));
                }
            }
        }

        // --- Defend nearest friendly target (wide: 250m orbit, 600m engage) ---
        10 => {
            let target_pos = nth_nearest_friendly_target(drone_pos, group, 0, targets_a, targets_b, bounds);
            if let Some(center) = target_pos {
                attack_targets.remove(&drone_id);
                intercept_targets.remove(&drone_id);
                if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
                    drone.agent.set_task(Box::new(DefendTask::new(
                        drone_id, group, center, 250.0, 600.0, detonation_radius,
                    )));
                }
            }
        }

        // --- Patrol perimeter ---
        11 => {
            let friendly_targets = if group == 0 { targets_a } else { targets_b };
            let friendly_positions: Vec<Position> = friendly_targets
                .iter()
                .filter(|t| !t.destroyed)
                .map(|t| t.pos)
                .collect();
            if !friendly_positions.is_empty() {
                let waypoints = build_patrol_route(&friendly_positions, patrol_standoff);
                attack_targets.remove(&drone_id);
                intercept_targets.remove(&drone_id);
                if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
                    drone.agent.set_task(Box::new(PatrolTask::new(waypoints, 50.0, 2.0)));
                }
            }
        }

        // --- Evade nearest threat ---
        12 => {
            attack_targets.remove(&drone_id);
            intercept_targets.remove(&drone_id);
            if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
                drone.agent.set_task(Box::new(EvadeTask::new(group)));
            }
        }

        _ => {} // Unknown action — treat as hold
    }
}

/// Get the Nth nearest alive enemy target position (0-indexed, group-aware).
pub fn nth_nearest_enemy_target(
    drone_pos: Vec2,
    group: u32,
    nth: usize,
    targets_a: &[TargetState],
    targets_b: &[TargetState],
    bounds: &Bounds,
) -> Option<Position> {
    let enemy_targets = if group == 0 { targets_b } else { targets_a };
    let mut targets: Vec<(f32, Position)> = enemy_targets
        .iter()
        .filter(|t| !t.destroyed)
        .map(|t| (bounds.distance(drone_pos, t.pos.as_vec2()), t.pos))
        .collect();
    targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    targets.get(nth).map(|(_, pos)| *pos)
}

/// Get the Nth nearest alive friendly target position (0-indexed, group-aware).
pub fn nth_nearest_friendly_target(
    drone_pos: Vec2,
    group: u32,
    nth: usize,
    targets_a: &[TargetState],
    targets_b: &[TargetState],
    bounds: &Bounds,
) -> Option<Position> {
    let friendly_targets = if group == 0 { targets_a } else { targets_b };
    let mut targets: Vec<(f32, Position)> = friendly_targets
        .iter()
        .filter(|t| !t.destroyed)
        .map(|t| (bounds.distance(drone_pos, t.pos.as_vec2()), t.pos))
        .collect();
    targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    targets.get(nth).map(|(_, pos)| *pos)
}

/// Get the ID of the Nth nearest enemy drone (0-indexed).
pub fn nth_nearest_enemy_drone(
    drone_pos: Vec2,
    group: u32,
    nth: usize,
    drones: &[GameDrone],
    bounds: &Bounds,
) -> Option<usize> {
    let mut enemies: Vec<(f32, usize)> = drones
        .iter()
        .filter(|d| d.group != group)
        .map(|d| (bounds.distance(drone_pos, d.agent.state().pos.as_vec2()), d.id))
        .collect();
    enemies.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    enemies.get(nth).map(|(_, id)| *id)
}

/// Set up intercept: one drone chases another.
fn intercept_drone_task(
    attacker_id: usize,
    target_id: usize,
    group: u32,
    drones: &mut [GameDrone],
    intercept_targets: &mut HashMap<usize, usize>,
    detonation_radius: f32,
) {
    if attacker_id == target_id {
        return;
    }
    if !drones.iter().any(|d| d.id == attacker_id) {
        return;
    }
    if !drones.iter().any(|d| d.id == target_id) {
        return;
    }

    if let Some(drone) = drones.iter_mut().find(|d| d.id == attacker_id) {
        let task = InterceptTask::new(attacker_id, target_id, group, detonation_radius);
        drone.agent.set_task(Box::new(task));
    }
    intercept_targets.insert(attacker_id, target_id);
}
