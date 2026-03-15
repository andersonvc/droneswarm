use std::collections::HashSet;

use crate::types::{Bounds, Position};

use super::state::{GameDrone, TargetState};

/// Result of processing pending detonations.
pub struct DetonationResult {
    pub destroyed_ids: HashSet<usize>,
    pub blast_positions: Vec<Position>,
}

/// Process pending detonations: find all drones within blast radius of each detonation.
pub fn process_detonations(
    drones: &[GameDrone],
    pending: &mut HashSet<usize>,
    blast_radius: f32,
    bounds: &Bounds,
) -> DetonationResult {
    let mut destroyed_ids = HashSet::new();
    let mut blast_positions = Vec::new();

    if pending.is_empty() {
        return DetonationResult { destroyed_ids, blast_positions };
    }

    let detonations: Vec<usize> = pending.drain().collect();
    for det_id in &detonations {
        let det_pos = drones
            .iter()
            .find(|d| d.id == *det_id)
            .map(|d| d.agent.state().pos);

        let Some(det_pos) = det_pos else { continue };

        destroyed_ids.insert(*det_id);
        blast_positions.push(det_pos);

        for drone in drones {
            if drone.id == *det_id {
                continue;
            }
            let dist = bounds.distance(det_pos.as_vec2(), drone.agent.state().pos.as_vec2());
            if dist <= blast_radius {
                destroyed_ids.insert(drone.id);
            }
        }
    }

    DetonationResult { destroyed_ids, blast_positions }
}

/// Detect collisions between drones (pairwise distance check).
pub fn detect_collisions(
    drones: &[GameDrone],
    collision_distance: f32,
    bounds: &Bounds,
) -> HashSet<usize> {
    let mut collided = HashSet::new();
    for i in 0..drones.len() {
        for j in (i + 1)..drones.len() {
            let pos_i = drones[i].agent.state().pos;
            let pos_j = drones[j].agent.state().pos;
            let dist = bounds.distance(pos_i.as_vec2(), pos_j.as_vec2());
            if dist < collision_distance {
                collided.insert(drones[i].id);
                collided.insert(drones[j].id);
            }
        }
    }
    collided
}

/// Check if any blast positions hit targets. Returns true if any target was destroyed.
pub fn check_target_hits(
    targets: &mut [TargetState],
    blast_positions: &[Position],
    hit_radius: f32,
    bounds: &Bounds,
) -> bool {
    let mut any_hit = false;
    for det_pos in blast_positions {
        for target in targets.iter_mut() {
            if target.destroyed {
                continue;
            }
            let dist = bounds.distance(det_pos.as_vec2(), target.pos.as_vec2());
            if dist <= hit_radius {
                target.destroyed = true;
                any_hit = true;
            }
        }
    }
    any_hit
}
