//! Shared observation encoding for per-drone RL observations.
//!
//! Used by both `sim_runner` and `wasm-lib` to produce identical 64-dim
//! observation vectors from game state.

use std::collections::HashMap;

use crate::types::{Bounds, Position, Vec2};

use super::obs_layout::obs_idx;
use super::obs_layout::{ACT_DIM, MAX_VELOCITY, OBS_DIM};
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

/// Encode a 64-dim observation vector for a single drone.
///
/// All distances are normalized by world size or diagonal; all counts by initial counts.
/// The observation is group-symmetric: "friendly" and "enemy" are relative to the drone's group.
pub fn encode_drone_observation(
    drone: &GameDrone,
    all_drones: &[GameDrone],
    targets_a: &[TargetState],
    targets_b: &[TargetState],
    last_actions: &HashMap<usize, u32>,
    tick_count: u32,
    bounds: &Bounds,
    config: &ObservationEncoderConfig,
) -> [f32; OBS_DIM] {
    let w = config.world_size;
    let diag = (w * w + w * w).sqrt();
    let state = drone.agent.state();
    let my_pos = state.pos.as_vec2();

    let mut obs = [0.0f32; OBS_DIM];

    // [0..5] Ego state
    obs[obs_idx::EGO_X] = state.pos.x() / w;
    obs[obs_idx::EGO_Y] = state.pos.y() / w;
    obs[obs_idx::EGO_VX] = state.vel.as_vec2().x / MAX_VELOCITY;
    obs[obs_idx::EGO_VY] = state.vel.as_vec2().y / MAX_VELOCITY;
    let last_action = last_actions.get(&drone.id).copied().unwrap_or(13) as f32;
    obs[obs_idx::EGO_LAST_ACTION] = last_action / ACT_DIM as f32;

    // [5..25] 4 nearest enemy drones
    let mut enemy_drones: Vec<(f32, f32, f32, f32, f32)> = all_drones
        .iter()
        .filter(|d| d.group != drone.group)
        .map(|d| {
            let es = d.agent.state();
            let epos = es.pos.as_vec2();
            let dist = bounds.distance(my_pos, epos);
            (
                (epos.x - my_pos.x) / w,
                (epos.y - my_pos.y) / w,
                dist / diag,
                es.vel.as_vec2().x / MAX_VELOCITY,
                es.vel.as_vec2().y / MAX_VELOCITY,
            )
        })
        .collect();
    enemy_drones.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    for i in 0..obs_idx::ENEMY_DRONES_COUNT {
        let base = obs_idx::ENEMY_DRONES_START + i * obs_idx::ENEMY_DRONE_STRIDE;
        if i < enemy_drones.len() {
            let e = &enemy_drones[i];
            obs[base + obs_idx::REL_DX] = e.0;
            obs[base + obs_idx::REL_DY] = e.1;
            obs[base + obs_idx::REL_DIST] = e.2;
            obs[base + obs_idx::REL_VX] = e.3;
            obs[base + obs_idx::REL_VY] = e.4;
        } else {
            obs[base + obs_idx::REL_DX] = 1.0;
            obs[base + obs_idx::REL_DY] = 1.0;
            obs[base + obs_idx::REL_DIST] = 1.0;
        }
    }

    // [25..40] 3 nearest friendly drones
    let mut friendly_drones: Vec<(f32, f32, f32, f32, f32)> = all_drones
        .iter()
        .filter(|d| d.group == drone.group && d.id != drone.id)
        .map(|d| {
            let fs = d.agent.state();
            let fpos = fs.pos.as_vec2();
            let dist = bounds.distance(my_pos, fpos);
            (
                (fpos.x - my_pos.x) / w,
                (fpos.y - my_pos.y) / w,
                dist / diag,
                fs.vel.as_vec2().x / MAX_VELOCITY,
                fs.vel.as_vec2().y / MAX_VELOCITY,
            )
        })
        .collect();
    friendly_drones.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    for i in 0..obs_idx::FRIENDLY_DRONES_COUNT {
        let base = obs_idx::FRIENDLY_DRONES_START + i * obs_idx::FRIENDLY_DRONE_STRIDE;
        if i < friendly_drones.len() {
            let f = &friendly_drones[i];
            obs[base + obs_idx::REL_DX] = f.0;
            obs[base + obs_idx::REL_DY] = f.1;
            obs[base + obs_idx::REL_DIST] = f.2;
            obs[base + obs_idx::REL_VX] = f.3;
            obs[base + obs_idx::REL_VY] = f.4;
        } else {
            obs[base + obs_idx::REL_DX] = 1.0;
            obs[base + obs_idx::REL_DY] = 1.0;
            obs[base + obs_idx::REL_DIST] = 1.0;
        }
    }

    // [40..46] 3 nearest enemy targets
    let enemy_target_list = if drone.group == 0 { targets_b } else { targets_a };
    let mut enemy_targets: Vec<(f32, f32, f32)> = enemy_target_list
        .iter()
        .filter(|t| !t.destroyed)
        .map(|t| {
            let dist = bounds.distance(my_pos, t.pos.as_vec2());
            (
                (t.pos.x() - state.pos.x()) / w,
                (t.pos.y() - state.pos.y()) / w,
                dist,
            )
        })
        .collect();
    enemy_targets.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    for i in 0..obs_idx::ENEMY_TARGETS_COUNT {
        let base = obs_idx::ENEMY_TARGETS_START + i * obs_idx::ENEMY_TARGET_STRIDE;
        if i < enemy_targets.len() {
            obs[base] = enemy_targets[i].0;
            obs[base + 1] = enemy_targets[i].1;
        }
    }

    // [46..52] 3 nearest friendly targets
    let friendly_target_list = if drone.group == 0 { targets_a } else { targets_b };
    let mut friendly_targets: Vec<(f32, f32, f32)> = friendly_target_list
        .iter()
        .filter(|t| !t.destroyed)
        .map(|t| {
            let dist = bounds.distance(my_pos, t.pos.as_vec2());
            (
                (t.pos.x() - state.pos.x()) / w,
                (t.pos.y() - state.pos.y()) / w,
                dist,
            )
        })
        .collect();
    friendly_targets.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    for i in 0..obs_idx::FRIENDLY_TARGETS_COUNT {
        let base = obs_idx::FRIENDLY_TARGETS_START + i * obs_idx::FRIENDLY_TARGET_STRIDE;
        if i < friendly_targets.len() {
            obs[base] = friendly_targets[i].0;
            obs[base + 1] = friendly_targets[i].1;
        }
    }

    // [52..60] Global features
    let initial_d = config.initial_drones_per_side as f32;
    let initial_t = config.initial_targets_per_side as f32;
    let own_group = drone.group;
    let enemy_group = 1 - drone.group;
    let count_group = |g: u32| all_drones.iter().filter(|d| d.group == g).count();
    let alive_a = targets_a.iter().filter(|t| !t.destroyed).count();
    let alive_b = targets_b.iter().filter(|t| !t.destroyed).count();

    obs[obs_idx::GLOBAL_OWN_DRONES] = count_group(own_group) as f32 / initial_d;
    obs[obs_idx::GLOBAL_ENEMY_DRONES] = count_group(enemy_group) as f32 / initial_d;
    let (own_targets_alive, enemy_targets_alive) = if own_group == 0 {
        (alive_a, alive_b)
    } else {
        (alive_b, alive_a)
    };
    obs[obs_idx::GLOBAL_OWN_TARGETS] = own_targets_alive as f32 / initial_t;
    obs[obs_idx::GLOBAL_ENEMY_TARGETS] = enemy_targets_alive as f32 / initial_t;
    obs[obs_idx::GLOBAL_TIME_FRAC] = tick_count as f32 / config.max_ticks as f32;

    let threats = count_nearby_threats(all_drones, targets_a, targets_b, own_group, config.detonation_radius, config.threat_radius_multiplier, bounds);
    obs[obs_idx::GLOBAL_THREATS] = (threats as f32 / config.max_nearby_threats).clamp(0.0, 1.0);

    // Nearest friendly distance
    let nearest_friendly_dist = all_drones
        .iter()
        .filter(|d| d.group == drone.group && d.id != drone.id)
        .map(|d| bounds.distance(my_pos, d.agent.state().pos.as_vec2()))
        .fold(f32::INFINITY, f32::min);
    obs[obs_idx::GLOBAL_NEAREST_FRIENDLY_DIST] =
        if nearest_friendly_dist.is_finite() { nearest_friendly_dist / diag } else { 1.0 };

    // Friendly density within detonation radius
    let friendlies_in_blast = all_drones
        .iter()
        .filter(|d| d.group == drone.group && d.id != drone.id)
        .filter(|d| bounds.distance(my_pos, d.agent.state().pos.as_vec2()) <= config.detonation_radius)
        .count() as f32;
    obs[obs_idx::GLOBAL_FRIENDLIES_IN_BLAST] = (friendlies_in_blast / 8.0).clamp(0.0, 1.0);

    // [60] Agent ID
    obs[obs_idx::AGENT_ID] = drone.id as f32 / config.initial_drones_per_side as f32;

    // [61..64] Last actions of 3 nearest friendly drones
    let mut friendly_ids_sorted: Vec<(f32, usize)> = all_drones
        .iter()
        .filter(|d| d.group == drone.group && d.id != drone.id)
        .map(|d| (bounds.distance(my_pos, d.agent.state().pos.as_vec2()), d.id))
        .collect();
    friendly_ids_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for i in 0..obs_idx::FRIENDLY_ACTIONS_COUNT {
        if i < friendly_ids_sorted.len() {
            let fid = friendly_ids_sorted[i].1;
            let action = last_actions.get(&fid).copied().unwrap_or(13) as f32;
            obs[obs_idx::FRIENDLY_ACTIONS_START + i] = action / ACT_DIM as f32;
        }
    }

    obs
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
    let alive: Vec<Position> = targets.iter().filter(|t| !t.destroyed).map(|t| t.pos).collect();
    if alive.is_empty() {
        return 0;
    }

    let cx = alive.iter().map(|p| p.x()).sum::<f32>() / alive.len() as f32;
    let cy = alive.iter().map(|p| p.y()).sum::<f32>() / alive.len() as f32;
    let centroid = Vec2::new(cx, cy);
    let threat_radius = detonation_radius * threat_radius_multiplier;

    drones
        .iter()
        .filter(|d| d.group != group && bounds.distance(centroid, d.agent.state().pos.as_vec2()) <= threat_radius)
        .count()
}
