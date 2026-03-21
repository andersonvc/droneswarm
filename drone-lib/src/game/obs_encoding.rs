//! Entity-based observation encoding for per-drone RL (V2).
//!
//! Produces structured observations: ego features (fixed 25-dim) + entity tokens
//! (variable-length 10-dim each). Used by sim_runner for training and wasm-lib for inference.

use std::collections::HashMap;

use crate::types::{Bounds, Position, Vec2};

use super::obs_layout::{
    ego_idx, entity_idx, entity_type, ACT_DIM, EGO_DIM, ENTITY_DIM, MAX_VELOCITY,
};
use super::state::{GameDrone, TargetState};

/// Configuration for observation encoding.
pub struct ObservationEncoderConfig {
    pub world_size: f32,
    pub max_ticks: u32,
    pub initial_drones_per_side: u32,
    pub initial_targets_per_side: u32,
    pub detonation_radius: f32,
    pub threat_radius_multiplier: f32,
    pub max_nearby_threats: f32,
}

/// Encode a V2 observation for a single drone.
///
/// Returns (ego, entities_flat, n_entities):
/// - ego: EGO_DIM floats
/// - entities_flat: n_entities * ENTITY_DIM floats (all entities concatenated)
/// - n_entities: number of real entities
pub fn encode_drone_observation(
    drone: &GameDrone,
    all_drones: &[GameDrone],
    targets_a: &[TargetState],
    targets_b: &[TargetState],
    last_actions: &HashMap<usize, u32>,
    attack_targets: &HashMap<usize, Position>,
    intercept_targets: &HashMap<usize, usize>,
    tick_count: u32,
    bounds: &Bounds,
    config: &ObservationEncoderConfig,
) -> (Vec<f32>, Vec<f32>, usize) {
    let w = config.world_size;
    let diag = (w * w + w * w).sqrt();
    let state = drone.agent.state();
    let my_pos = state.pos.as_vec2();
    let my_heading = state.hdg.radians();

    // ========================================================================
    // Ego features [0..25]
    // ========================================================================
    let mut ego = vec![0.0f32; EGO_DIM];

    ego[ego_idx::X] = state.pos.x() / w;
    ego[ego_idx::Y] = state.pos.y() / w;
    ego[ego_idx::VX] = state.vel.as_vec2().x / MAX_VELOCITY;
    ego[ego_idx::VY] = state.vel.as_vec2().y / MAX_VELOCITY;
    ego[ego_idx::HEADING] = my_heading / (2.0 * std::f32::consts::PI);
    let last_action = last_actions.get(&drone.id).copied().unwrap_or(0) as f32;
    ego[ego_idx::LAST_ACTION] = last_action / ACT_DIM as f32;

    // Task type one-hot [6..15]
    if let Some((task_name, phase_name)) = drone.agent.task_info() {
        let task_idx = task_name_to_index(task_name);
        if task_idx > 0 && task_idx < ego_idx::TASK_TYPE_COUNT {
            ego[ego_idx::TASK_TYPE_START + task_idx] = 1.0;
        }
        ego[ego_idx::TASK_PHASE] = phase_to_ordinal(phase_name);
    }
    // If no task, all task one-hot bits remain 0 (index 0 = None is implicit)

    // Relative alive index: position of this drone in sorted alive same-group IDs
    let mut alive_same_group: Vec<usize> = all_drones
        .iter()
        .filter(|d| d.group == drone.group)
        .map(|d| d.id)
        .collect();
    alive_same_group.sort_unstable();
    let alive_count = alive_same_group.len().max(1) as f32;
    let my_rank = alive_same_group
        .iter()
        .position(|&id| id == drone.id)
        .unwrap_or(0) as f32;
    ego[ego_idx::RELATIVE_ALIVE_INDEX] = my_rank / alive_count;

    // Global features [17..25]
    let initial_d = config.initial_drones_per_side as f32;
    let initial_t = config.initial_targets_per_side as f32;
    let own_group = drone.group;
    let enemy_group = 1 - drone.group;
    let count_group = |g: u32| all_drones.iter().filter(|d| d.group == g).count();
    let alive_a = targets_a.iter().filter(|t| !t.destroyed).count();
    let alive_b = targets_b.iter().filter(|t| !t.destroyed).count();

    ego[ego_idx::GLOBAL_OWN_DRONES] = count_group(own_group) as f32 / initial_d;
    ego[ego_idx::GLOBAL_ENEMY_DRONES] = count_group(enemy_group) as f32 / initial_d;
    let (own_targets_alive, enemy_targets_alive) = if own_group == 0 {
        (alive_a, alive_b)
    } else {
        (alive_b, alive_a)
    };
    ego[ego_idx::GLOBAL_OWN_TARGETS] = own_targets_alive as f32 / initial_t;
    ego[ego_idx::GLOBAL_ENEMY_TARGETS] = enemy_targets_alive as f32 / initial_t;
    ego[ego_idx::GLOBAL_TIME_FRAC] = tick_count as f32 / config.max_ticks as f32;

    let threats = count_nearby_threats(
        all_drones,
        targets_a,
        targets_b,
        own_group,
        config.detonation_radius,
        config.threat_radius_multiplier,
        bounds,
    );
    ego[ego_idx::GLOBAL_THREATS] = (threats as f32 / config.max_nearby_threats).clamp(0.0, 1.0);

    let nearest_friendly_dist = all_drones
        .iter()
        .filter(|d| d.group == drone.group && d.id != drone.id)
        .map(|d| bounds.distance(my_pos, d.agent.state().pos.as_vec2()))
        .fold(f32::INFINITY, f32::min);
    ego[ego_idx::GLOBAL_NEAREST_FRIENDLY_DIST] = if nearest_friendly_dist.is_finite() {
        nearest_friendly_dist / diag
    } else {
        1.0
    };

    let friendlies_in_blast = all_drones
        .iter()
        .filter(|d| d.group == drone.group && d.id != drone.id)
        .filter(|d| {
            bounds.distance(my_pos, d.agent.state().pos.as_vec2()) <= config.detonation_radius
        })
        .count() as f32;
    ego[ego_idx::GLOBAL_FRIENDLIES_IN_BLAST] = (friendlies_in_blast / 8.0).clamp(0.0, 1.0);

    // ========================================================================
    // Entity tokens (variable length, sorted by distance within each type group)
    // ========================================================================
    let mut entities: Vec<[f32; ENTITY_DIM]> = Vec::new();

    // Determine the drone's current target for IS_CURRENT_TARGET.
    // attack_targets maps drone_id -> target Position (for attack tasks).
    // intercept_targets maps drone_id -> enemy_drone_id (for intercept tasks).
    let my_attack_target: Option<Position> = attack_targets.get(&drone.id).copied();
    let my_intercept_target_id: Option<usize> = intercept_targets.get(&drone.id).copied();

    // Enemy drones
    let mut enemy_drones: Vec<(f32, &GameDrone)> = all_drones
        .iter()
        .filter(|d| d.group != drone.group)
        .map(|d| {
            let dist = bounds.distance(my_pos, d.agent.state().pos.as_vec2());
            (dist, d)
        })
        .collect();
    enemy_drones.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (_, d) in &enemy_drones {
        let mut token = make_drone_entity(
            d,
            my_pos,
            my_heading,
            w,
            diag,
            bounds,
            entity_type::ENEMY_DRONE,
        );
        // Mark as current target if this drone is being intercepted.
        if my_intercept_target_id == Some(d.id) {
            token[entity_idx::IS_CURRENT_TARGET] = 1.0;
        }
        // Count how many friendly drones are assigned to intercept this enemy drone.
        let intercept_count = intercept_targets
            .values()
            .filter(|&&target_id| target_id == d.id)
            .count() as f32;
        token[entity_idx::ASSIGNMENT_COUNT] =
            intercept_count / config.initial_drones_per_side as f32;
        entities.push(token);
    }

    // Friendly drones (excluding self)
    let mut friendly_drones: Vec<(f32, &GameDrone)> = all_drones
        .iter()
        .filter(|d| d.group == drone.group && d.id != drone.id)
        .map(|d| {
            let dist = bounds.distance(my_pos, d.agent.state().pos.as_vec2());
            (dist, d)
        })
        .collect();
    friendly_drones.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (_, d) in &friendly_drones {
        entities.push(make_drone_entity(
            d,
            my_pos,
            my_heading,
            w,
            diag,
            bounds,
            entity_type::FRIENDLY_DRONE,
        ));
    }

    // Enemy targets
    let enemy_target_list = if drone.group == 0 {
        targets_b
    } else {
        targets_a
    };
    let mut enemy_targets: Vec<(f32, &TargetState)> = enemy_target_list
        .iter()
        .filter(|t| !t.destroyed)
        .map(|t| {
            let dist = bounds.distance(my_pos, t.pos.as_vec2());
            (dist, t)
        })
        .collect();
    enemy_targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (_, t) in &enemy_targets {
        let mut token = make_target_entity(
            t,
            my_pos,
            my_heading,
            w,
            diag,
            bounds,
            entity_type::ENEMY_TARGET,
        );
        // Mark as current target if this is the attack target position.
        if let Some(atk_pos) = my_attack_target {
            let dist_to_atk = bounds.distance(t.pos.as_vec2(), atk_pos.as_vec2());
            if dist_to_atk < 1.0 {
                token[entity_idx::IS_CURRENT_TARGET] = 1.0;
            }
        }
        // Count how many friendly drones are assigned to attack this target.
        let attack_count = attack_targets
            .values()
            .filter(|atk_pos| {
                bounds.distance(t.pos.as_vec2(), atk_pos.as_vec2()) < 50.0
            })
            .count() as f32;
        token[entity_idx::ASSIGNMENT_COUNT] =
            attack_count / config.initial_drones_per_side as f32;
        entities.push(token);
    }

    // Friendly targets
    let friendly_target_list = if drone.group == 0 {
        targets_a
    } else {
        targets_b
    };
    let mut friendly_targets: Vec<(f32, &TargetState)> = friendly_target_list
        .iter()
        .filter(|t| !t.destroyed)
        .map(|t| {
            let dist = bounds.distance(my_pos, t.pos.as_vec2());
            (dist, t)
        })
        .collect();
    friendly_targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (_, t) in &friendly_targets {
        entities.push(make_target_entity(
            t,
            my_pos,
            my_heading,
            w,
            diag,
            bounds,
            entity_type::FRIENDLY_TARGET,
        ));
    }

    // Flatten entity tokens
    let n_entities = entities.len();
    let mut entities_flat = Vec::with_capacity(n_entities * ENTITY_DIM);
    for token in &entities {
        entities_flat.extend_from_slice(token);
    }

    (ego, entities_flat, n_entities)
}

/// Create a 10-dim entity token for a drone.
fn make_drone_entity(
    d: &GameDrone,
    my_pos: Vec2,
    my_heading: f32,
    w: f32,
    diag: f32,
    bounds: &Bounds,
    type_flag: f32,
) -> [f32; ENTITY_DIM] {
    let es = d.agent.state();
    let epos = es.pos.as_vec2();
    let dx = epos.x - my_pos.x;
    let dy = epos.y - my_pos.y;
    let dist = bounds.distance(my_pos, epos);
    let heading_to = dy.atan2(dx);
    let heading_rel = normalize_angle(heading_to - my_heading) / std::f32::consts::PI;

    let mut token = [0.0f32; ENTITY_DIM];
    token[entity_idx::DX] = dx / w;
    token[entity_idx::DY] = dy / w;
    token[entity_idx::DIST] = dist / diag;
    token[entity_idx::VX] = es.vel.as_vec2().x / MAX_VELOCITY;
    token[entity_idx::VY] = es.vel.as_vec2().y / MAX_VELOCITY;
    token[entity_idx::HEADING_REL] = heading_rel;
    token[entity_idx::TYPE_FLAG] = type_flag;
    token[entity_idx::ALIVE_FLAG] = 1.0;
    token
}

/// Create a 10-dim entity token for a target (no velocity/heading).
fn make_target_entity(
    t: &TargetState,
    my_pos: Vec2,
    my_heading: f32,
    w: f32,
    diag: f32,
    bounds: &Bounds,
    type_flag: f32,
) -> [f32; ENTITY_DIM] {
    let tpos = t.pos.as_vec2();
    let dx = tpos.x - my_pos.x;
    let dy = tpos.y - my_pos.y;
    let dist = bounds.distance(my_pos, tpos);
    let heading_to = dy.atan2(dx);
    let heading_rel = normalize_angle(heading_to - my_heading) / std::f32::consts::PI;

    let mut token = [0.0f32; ENTITY_DIM];
    token[entity_idx::DX] = dx / w;
    token[entity_idx::DY] = dy / w;
    token[entity_idx::DIST] = dist / diag;
    // vx, vy = 0 for targets
    token[entity_idx::HEADING_REL] = heading_rel;
    token[entity_idx::TYPE_FLAG] = type_flag;
    token[entity_idx::ALIVE_FLAG] = 1.0;
    token
}

/// Map task name string to one-hot index (0-8).
/// Index 0 = None (no task), represented by all-zeros in the one-hot.
fn task_name_to_index(name: &str) -> usize {
    match name {
        "Attack" => 1,
        "AttackEvasive" => 2,
        "Defend" => 3,
        "Intercept" => 4,
        "InterceptGroup" => 5,
        "Evade" => 6,
        "Loiter" => 7,
        "Patrol" => 8,
        _ => 0,
    }
}

/// Map task phase string to ordinal in [0, 1].
fn phase_to_ordinal(phase: &str) -> f32 {
    match phase {
        // Early/positioning phases
        "navigate" | "approach" | "transit" | "orbit" => 0.0,
        // Active engagement phases
        "engage" | "pursue" | "pursue_cluster" | "flee" | "hold" | "loiter" => 0.33,
        // Terminal phases
        "terminal" => 0.67,
        // Complete
        "complete" | "done" => 1.0,
        _ => 0.0,
    }
}

/// Normalize angle to [-PI, PI].
fn normalize_angle(angle: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    let mut a = angle % two_pi;
    if a > std::f32::consts::PI { a -= two_pi; }
    if a < -std::f32::consts::PI { a += two_pi; }
    a
}

/// Count enemy drones near friendly target centroid.
fn count_nearby_threats(
    drones: &[GameDrone],
    targets_a: &[TargetState],
    targets_b: &[TargetState],
    group: u32,
    detonation_radius: f32,
    threat_radius_multiplier: f32,
    bounds: &Bounds,
) -> usize {
    let targets = if group == 0 { targets_a } else { targets_b };
    let alive: Vec<Position> = targets
        .iter()
        .filter(|t| !t.destroyed)
        .map(|t| t.pos)
        .collect();
    if alive.is_empty() {
        return 0;
    }

    let cx = alive.iter().map(|p| p.x()).sum::<f32>() / alive.len() as f32;
    let cy = alive.iter().map(|p| p.y()).sum::<f32>() / alive.len() as f32;
    let centroid = Vec2::new(cx, cy);
    let threat_radius = detonation_radius * threat_radius_multiplier;

    drones
        .iter()
        .filter(|d| {
            d.group != group
                && bounds.distance(centroid, d.agent.state().pos.as_vec2()) <= threat_radius
        })
        .count()
}
