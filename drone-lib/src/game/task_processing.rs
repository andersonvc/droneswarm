use std::collections::{HashMap, HashSet};

use crate::tasks::TaskStatus;
use crate::types::{Bounds, Objective, Position};

use super::state::GameDrone;

/// Result of processing task statuses across all drones.
pub struct TaskProcessingResult {
    pub detonation_requests: Vec<usize>,
    pub completed: Vec<usize>,
    pub failed: Vec<usize>,
}

/// Scan all drones for task status changes: detonation requests, completions, failures.
/// Interceptors near protected zones have detonations blocked (turned into failures).
pub fn process_task_status(
    drones: &[GameDrone],
    intercept_targets: &HashMap<usize, usize>,
    protected_zones: &HashMap<u32, Vec<Position>>,
    detonation_radius: f32,
    bounds: &Bounds,
) -> TaskProcessingResult {
    let mut detonation_requests = Vec::new();
    let mut completed = Vec::new();
    let mut failed = Vec::new();

    for drone in drones {
        let id = drone.id;

        if drone.agent.should_detonate() {
            // Protected zone safety check for interceptors.
            if intercept_targets.contains_key(&id) {
                let drone_pos = drone.agent.state().pos;
                let would_hit_friendly = protected_zones
                    .get(&drone.group)
                    .map(|zones| {
                        zones.iter().any(|z| {
                            bounds.distance(drone_pos.as_vec2(), z.as_vec2()) <= detonation_radius
                        })
                    })
                    .unwrap_or(false);

                if would_hit_friendly {
                    failed.push(id);
                    continue;
                }
            }
            detonation_requests.push(id);
            continue;
        }

        if let Some(status) = drone.agent.task_status() {
            match status {
                TaskStatus::Complete => completed.push(id),
                TaskStatus::Failed => failed.push(id),
                TaskStatus::Active => {}
            }
        }
    }

    TaskProcessingResult { detonation_requests, completed, failed }
}

/// Apply task processing results: update tracking maps and clear tasks on completed/failed drones.
pub fn apply_task_results(
    result: &TaskProcessingResult,
    drones: &mut [GameDrone],
    pending_detonations: &mut HashSet<usize>,
    attack_targets: &mut HashMap<usize, Position>,
    intercept_targets: &mut HashMap<usize, usize>,
) {
    for &id in &result.detonation_requests {
        intercept_targets.remove(&id);
        attack_targets.remove(&id);
        pending_detonations.insert(id);
    }
    for &id in &result.completed {
        intercept_targets.remove(&id);
        attack_targets.remove(&id);
        if let Some(drone) = drones.iter_mut().find(|d| d.id == id) {
            drone.agent.clear_task();
            drone.agent.set_objective(Objective::Sleep);
        }
    }
    for &id in &result.failed {
        intercept_targets.remove(&id);
        attack_targets.remove(&id);
        if let Some(drone) = drones.iter_mut().find(|d| d.id == id) {
            drone.agent.clear_task();
            drone.agent.set_objective(Objective::Sleep);
        }
    }

    // Clean up stale tracking.
    let alive_ids: HashSet<usize> = drones.iter().map(|d| d.id).collect();
    attack_targets.retain(|id, _| alive_ids.contains(id));
    intercept_targets.retain(|id, _| alive_ids.contains(id));
}
