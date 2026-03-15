//! RL action mapping for per-drone discrete actions (V2, 19 actions).
//!
//! Translates action indices (0-18) into task assignments.
//!
//! Action space:
//!   0-2:   Attack {nearest, farthest, least-defended} enemy target (direct)
//!   3-5:   Attack {nearest, farthest, least-defended} enemy target (evasive)
//!   6-11:  Attack target by index 0-5 (direct, wraps if fewer targets)
//!   12-13: Intercept {nearest, 2nd nearest} enemy drone
//!   14:    Intercept enemy cluster
//!   15-16: Defend nearest friendly target {tight 100m/300m, wide 250m/600m}
//!   17:    Patrol perimeter
//!   18:    Evade nearest threat

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

/// Apply an RL action (0-18) to a drone in the game engine.
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
    // Find the drone's group and position.
    let (group, drone_pos) = match drones.iter().find(|d| d.id == drone_id) {
        Some(d) => (d.group, d.agent.state().pos.as_vec2()),
        None => return,
    };

    match action {
        // --- Attack nearest enemy target (direct) ---
        0 => {
            if let Some(target) =
                nth_nearest_enemy_target(drone_pos, group, 0, targets_a, targets_b, bounds)
            {
                clear_tasks(drone_id, attack_targets, intercept_targets);
                set_attack_direct(drone_id, target, drones, attack_targets, detonation_radius);
            }
        }
        // --- Attack farthest enemy target (direct) ---
        1 => {
            if let Some(target) =
                farthest_enemy_target(drone_pos, group, targets_a, targets_b, bounds)
            {
                clear_tasks(drone_id, attack_targets, intercept_targets);
                set_attack_direct(drone_id, target, drones, attack_targets, detonation_radius);
            }
        }
        // --- Attack least-defended enemy target (direct) ---
        2 => {
            if let Some(target) = least_defended_enemy_target(
                drone_pos,
                group,
                targets_a,
                targets_b,
                attack_targets,
                bounds,
            ) {
                clear_tasks(drone_id, attack_targets, intercept_targets);
                set_attack_direct(drone_id, target, drones, attack_targets, detonation_radius);
            }
        }
        // --- Attack nearest enemy target (evasive) ---
        3 => {
            if let Some(target) =
                nth_nearest_enemy_target(drone_pos, group, 0, targets_a, targets_b, bounds)
            {
                clear_tasks(drone_id, attack_targets, intercept_targets);
                set_attack_evasive(drone_id, target, drones, attack_targets, detonation_radius);
            }
        }
        // --- Attack farthest enemy target (evasive) ---
        4 => {
            if let Some(target) =
                farthest_enemy_target(drone_pos, group, targets_a, targets_b, bounds)
            {
                clear_tasks(drone_id, attack_targets, intercept_targets);
                set_attack_evasive(drone_id, target, drones, attack_targets, detonation_radius);
            }
        }
        // --- Attack least-defended enemy target (evasive) ---
        5 => {
            if let Some(target) = least_defended_enemy_target(
                drone_pos,
                group,
                targets_a,
                targets_b,
                attack_targets,
                bounds,
            ) {
                clear_tasks(drone_id, attack_targets, intercept_targets);
                set_attack_evasive(drone_id, target, drones, attack_targets, detonation_radius);
            }
        }
        // --- Attack target by index 0-5 (direct, wraps if fewer targets) ---
        6..=11 => {
            let idx = (action - 6) as usize;
            if let Some(target) =
                enemy_target_by_index(drone_pos, group, idx, targets_a, targets_b, bounds)
            {
                clear_tasks(drone_id, attack_targets, intercept_targets);
                set_attack_direct(drone_id, target, drones, attack_targets, detonation_radius);
            }
        }
        // --- Intercept nearest enemy drone ---
        12 => {
            if let Some(eid) = nth_nearest_enemy_drone(drone_pos, group, 0, drones, bounds) {
                clear_tasks(drone_id, attack_targets, intercept_targets);
                intercept_drone_task(
                    drone_id,
                    eid,
                    group,
                    drones,
                    intercept_targets,
                    detonation_radius,
                );
            }
        }
        // --- Intercept 2nd nearest enemy drone ---
        13 => {
            if let Some(eid) = nth_nearest_enemy_drone(drone_pos, group, 1, drones, bounds) {
                clear_tasks(drone_id, attack_targets, intercept_targets);
                intercept_drone_task(
                    drone_id,
                    eid,
                    group,
                    drones,
                    intercept_targets,
                    detonation_radius,
                );
            }
        }
        // --- Intercept enemy cluster ---
        14 => {
            clear_tasks(drone_id, attack_targets, intercept_targets);
            if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
                drone
                    .agent
                    .set_task(Box::new(InterceptGroupTask::new(group, detonation_radius)));
            }
        }
        // --- Defend nearest friendly target (tight: 100m orbit, 300m engage) ---
        15 => {
            let target_pos =
                nth_nearest_friendly_target(drone_pos, group, 0, targets_a, targets_b, bounds);
            if let Some(center) = target_pos {
                clear_tasks(drone_id, attack_targets, intercept_targets);
                if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
                    drone.agent.set_task(Box::new(DefendTask::new(
                        drone_id,
                        group,
                        center,
                        100.0,
                        300.0,
                        detonation_radius,
                    )));
                }
            }
        }
        // --- Defend nearest friendly target (wide: 250m orbit, 600m engage) ---
        16 => {
            let target_pos =
                nth_nearest_friendly_target(drone_pos, group, 0, targets_a, targets_b, bounds);
            if let Some(center) = target_pos {
                clear_tasks(drone_id, attack_targets, intercept_targets);
                if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
                    drone.agent.set_task(Box::new(DefendTask::new(
                        drone_id,
                        group,
                        center,
                        250.0,
                        600.0,
                        detonation_radius,
                    )));
                }
            }
        }
        // --- Patrol perimeter ---
        17 => {
            let friendly_targets = if group == 0 { targets_a } else { targets_b };
            let friendly_positions: Vec<Position> = friendly_targets
                .iter()
                .filter(|t| !t.destroyed)
                .map(|t| t.pos)
                .collect();
            if !friendly_positions.is_empty() {
                let waypoints = build_patrol_route(&friendly_positions, patrol_standoff);
                clear_tasks(drone_id, attack_targets, intercept_targets);
                if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
                    drone
                        .agent
                        .set_task(Box::new(PatrolTask::new(waypoints, 50.0, 2.0)));
                }
            }
        }
        // --- Evade nearest threat ---
        18 => {
            clear_tasks(drone_id, attack_targets, intercept_targets);
            if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
                drone.agent.set_task(Box::new(EvadeTask::new(group)));
            }
        }

        _ => {} // Unknown action — no-op
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn clear_tasks(
    drone_id: usize,
    attack_targets: &mut HashMap<usize, Position>,
    intercept_targets: &mut HashMap<usize, usize>,
) {
    attack_targets.remove(&drone_id);
    intercept_targets.remove(&drone_id);
}

fn set_attack_direct(
    drone_id: usize,
    target: Position,
    drones: &mut [GameDrone],
    attack_targets: &mut HashMap<usize, Position>,
    detonation_radius: f32,
) {
    if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
        drone
            .agent
            .set_task(Box::new(AttackTask::new(target, detonation_radius)));
    }
    attack_targets.insert(drone_id, target);
}

fn set_attack_evasive(
    drone_id: usize,
    target: Position,
    drones: &mut [GameDrone],
    attack_targets: &mut HashMap<usize, Position>,
    detonation_radius: f32,
) {
    if let Some(drone) = drones.iter_mut().find(|d| d.id == drone_id) {
        drone
            .agent
            .set_task(Box::new(AttackTask::new_evasive(target, detonation_radius)));
    }
    attack_targets.insert(drone_id, target);
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

/// Get the farthest alive enemy target position.
fn farthest_enemy_target(
    drone_pos: Vec2,
    group: u32,
    targets_a: &[TargetState],
    targets_b: &[TargetState],
    bounds: &Bounds,
) -> Option<Position> {
    let enemy_targets = if group == 0 { targets_b } else { targets_a };
    enemy_targets
        .iter()
        .filter(|t| !t.destroyed)
        .max_by(|a, b| {
            let da = bounds.distance(drone_pos, a.pos.as_vec2());
            let db = bounds.distance(drone_pos, b.pos.as_vec2());
            da.partial_cmp(&db).unwrap()
        })
        .map(|t| t.pos)
}

/// Get the least-defended enemy target: fewest drones assigned to attack it.
fn least_defended_enemy_target(
    drone_pos: Vec2,
    group: u32,
    targets_a: &[TargetState],
    targets_b: &[TargetState],
    attack_targets: &HashMap<usize, Position>,
    bounds: &Bounds,
) -> Option<Position> {
    let enemy_targets = if group == 0 { targets_b } else { targets_a };
    let alive: Vec<Position> = enemy_targets
        .iter()
        .filter(|t| !t.destroyed)
        .map(|t| t.pos)
        .collect();
    if alive.is_empty() {
        return None;
    }

    // Count attackers per target (within 50m of target position).
    let mut best_target = alive[0];
    let mut min_attackers = usize::MAX;
    let mut best_dist = f32::MAX;

    for &tpos in &alive {
        let attackers = attack_targets
            .values()
            .filter(|&&at_pos| {
                let dx = at_pos.x() - tpos.x();
                let dy = at_pos.y() - tpos.y();
                (dx * dx + dy * dy).sqrt() < 50.0
            })
            .count();
        let dist = bounds.distance(drone_pos, tpos.as_vec2());
        // Prefer fewer attackers, break ties by distance.
        if attackers < min_attackers || (attackers == min_attackers && dist < best_dist) {
            min_attackers = attackers;
            best_target = tpos;
            best_dist = dist;
        }
    }

    Some(best_target)
}

/// Get enemy target by index (sorted by distance), wrapping if fewer targets.
fn enemy_target_by_index(
    drone_pos: Vec2,
    group: u32,
    idx: usize,
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
    if targets.is_empty() {
        return None;
    }
    targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let wrapped_idx = idx % targets.len();
    Some(targets[wrapped_idx].1)
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
        .map(|d| {
            (
                bounds.distance(drone_pos, d.agent.state().pos.as_vec2()),
                d.id,
            )
        })
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
