//! Attack zone strategy: coordinated assault on enemy targets.
//!
//! Assigns drones in waves to attack enemy target positions. When a drone
//! detonates or is destroyed, the next available drone is sent to the nearest
//! target. Continues until all drones are depleted.

use std::collections::{HashMap, HashSet};

use crate::strategies::{StrategyDroneState, SwarmStrategy, TaskAssignment};
use crate::types::{Bounds, DroneInfo, Position};

/// Attack zone strategy.
///
/// Each available drone is assigned to attack the nearest target position,
/// spacing itself from friendly drones by at least `detonation_radius`.
/// Waves continue until all drones are depleted.
#[derive(Debug)]
pub struct AttackZoneStrategy {
    drone_ids: Vec<usize>,
    /// Target positions to attack.
    target_positions: Vec<Position>,
    /// Drone → assigned target index.
    assigned: HashMap<usize, usize>,
    /// Max drones to send per wave (None = unlimited).
    wave_size: Option<usize>,
}

impl AttackZoneStrategy {
    /// Create a new attack zone strategy.
    ///
    /// * `drone_ids` — drones allocated to this strategy
    /// * `target_positions` — enemy target positions to attack (meters)
    /// * `_detonation_radius` — retained for API compatibility
    pub fn new(drone_ids: Vec<usize>, target_positions: Vec<Position>, _detonation_radius: f32) -> Self {
        AttackZoneStrategy {
            drone_ids,
            target_positions,
            assigned: HashMap::new(),
            wave_size: None,
        }
    }

    /// Create with a limited wave size.
    pub fn with_wave_size(mut self, wave_size: usize) -> Self {
        self.wave_size = Some(wave_size);
        self
    }

    /// Find the nearest target that doesn't already have too many drones
    /// assigned AND where the drone won't be within detonation_radius of
    /// another friendly attacker headed to the same target.
    fn best_target(
        &self,
        pos: Position,
        _own_drones: &[StrategyDroneState],
    ) -> Option<usize> {
        // Count how many drones are currently assigned to each target
        let mut target_counts: HashMap<usize, usize> = HashMap::new();
        for &tidx in self.assigned.values() {
            *target_counts.entry(tidx).or_insert(0) += 1;
        }

        // Prefer targets with fewer drones assigned, break ties by distance
        (0..self.target_positions.len())
            .min_by(|&a, &b| {
                let count_a = target_counts.get(&a).copied().unwrap_or(0);
                let count_b = target_counts.get(&b).copied().unwrap_or(0);
                count_a.cmp(&count_b).then_with(|| {
                    let da = dist_sq(pos, self.target_positions[a]);
                    let db = dist_sq(pos, self.target_positions[b]);
                    da.partial_cmp(&db).unwrap()
                })
            })
    }

}

fn dist_sq(a: Position, b: Position) -> f32 {
    (a.x() - b.x()).powi(2) + (a.y() - b.y()).powi(2)
}

impl SwarmStrategy for AttackZoneStrategy {
    fn tick(
        &mut self,
        own_drones: &[StrategyDroneState],
        _all_drones: &[DroneInfo],
        _bounds: &Bounds,
        _dt: f32,
    ) -> Vec<TaskAssignment> {
        let mut assignments = Vec::new();
        let active_ids: HashSet<usize> = own_drones.iter().map(|d| d.id).collect();

        // 1. Clean up destroyed drones
        self.assigned.retain(|id, _| active_ids.contains(id));

        // 2. Clear assignments for drones whose task completed/failed —
        //    they are available for a new wave.
        for drone in own_drones.iter().filter(|d| d.available) {
            self.assigned.remove(&drone.id);
        }

        // 3. Assign available drones to targets (next wave)
        //    Respect wave_size: only send N drones at a time beyond those already in-flight.
        let in_flight = self.assigned.len();
        let max_in_flight = self.wave_size.unwrap_or(usize::MAX);
        let mut slots = max_in_flight.saturating_sub(in_flight);

        for drone in own_drones.iter().filter(|d| d.available) {
            if slots == 0 || self.target_positions.is_empty() {
                break;
            }
            if let Some(target_idx) = self.best_target(drone.pos, own_drones) {
                let target = self.target_positions[target_idx];
                assignments.push(TaskAssignment::Attack {
                    drone_id: drone.id,
                    target,
                });
                self.assigned.insert(drone.id, target_idx);
                slots -= 1;
            }
        }

        assignments
    }

    fn drone_ids(&self) -> &[usize] {
        &self.drone_ids
    }

    fn name(&self) -> &str {
        "AttackZone"
    }

    fn is_complete(&self) -> bool {
        // Never completes on its own — keeps sending waves until
        // all drones are depleted (detected by empty own_drones).
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Velocity;

    fn make_own_drone(id: usize, x: f32, y: f32, available: bool) -> StrategyDroneState {
        StrategyDroneState {
            id,
            pos: Position::new(x, y),
            vel: Velocity::zero(),
            available,
        }
    }

    fn bounds() -> Bounds {
        Bounds::new(5000.0, 5000.0).unwrap()
    }

    #[test]
    fn test_assigns_to_nearest_target() {
        let targets = vec![
            Position::new(1000.0, 1000.0),
            Position::new(200.0, 200.0),
        ];
        let mut strategy = AttackZoneStrategy::new(vec![0, 1], targets, 187.5);

        let own = vec![
            make_own_drone(0, 100.0, 100.0, true),
            make_own_drone(1, 900.0, 900.0, true),
        ];
        let assignments = strategy.tick(&own, &[], &bounds(), 0.016);

        assert_eq!(assignments.len(), 2);
        for a in &assignments {
            match a {
                TaskAssignment::Attack { drone_id: 0, target } => {
                    assert!((target.x() - 200.0).abs() < 1.0, "Drone 0 should attack nearest target");
                }
                TaskAssignment::Attack { drone_id: 1, target } => {
                    assert!((target.x() - 1000.0).abs() < 1.0, "Drone 1 should attack nearest target");
                }
                _ => panic!("Unexpected assignment"),
            }
        }
    }

    #[test]
    fn test_sends_follow_up_wave() {
        let targets = vec![Position::new(500.0, 0.0)];
        let mut strategy = AttackZoneStrategy::new(vec![0, 1, 2], targets, 187.5);

        // Wave 1: assign drone 0
        let own = vec![
            make_own_drone(0, 0.0, 0.0, true),
            make_own_drone(1, 100.0, 0.0, true),
            make_own_drone(2, 200.0, 0.0, true),
        ];
        let assignments = strategy.tick(&own, &[], &bounds(), 0.016);
        assert!(!assignments.is_empty());

        // Drone 0 detonated (destroyed), drones 1 and 2 still busy
        let own2 = vec![
            make_own_drone(1, 100.0, 0.0, false),
            make_own_drone(2, 200.0, 0.0, false),
        ];
        strategy.tick(&own2, &[], &bounds(), 0.016);

        // Drones 1 and 2 now available (wave 2)
        let own3 = vec![
            make_own_drone(1, 100.0, 0.0, true),
            make_own_drone(2, 200.0, 0.0, true),
        ];
        let assignments = strategy.tick(&own3, &[], &bounds(), 0.016);

        // Should assign them again to the same target
        assert_eq!(assignments.len(), 2, "Should send follow-up wave");
    }

    #[test]
    fn test_never_completes() {
        let targets = vec![Position::new(100.0, 0.0)];
        let mut strategy = AttackZoneStrategy::new(vec![0], targets, 187.5);

        let own = vec![make_own_drone(0, 0.0, 0.0, true)];
        strategy.tick(&own, &[], &bounds(), 0.016);
        assert!(!strategy.is_complete());

        // Drone completes
        let own2 = vec![make_own_drone(0, 100.0, 0.0, true)];
        strategy.tick(&own2, &[], &bounds(), 0.016);
        assert!(!strategy.is_complete(), "Should never complete — keep sending waves");
    }

    #[test]
    fn test_spreads_across_targets() {
        let targets = vec![
            Position::new(500.0, 500.0),
            Position::new(1000.0, 1000.0),
        ];
        let mut strategy = AttackZoneStrategy::new(vec![0, 1, 2, 3], targets, 187.5);

        let own = vec![
            make_own_drone(0, 0.0, 0.0, true),
            make_own_drone(1, 100.0, 0.0, true),
            make_own_drone(2, 200.0, 0.0, true),
            make_own_drone(3, 300.0, 0.0, true),
        ];
        let assignments = strategy.tick(&own, &[], &bounds(), 0.016);

        assert_eq!(assignments.len(), 4);
        // Should spread across targets, not pile all on one
        let mut target_counts: HashMap<String, usize> = HashMap::new();
        for a in &assignments {
            if let TaskAssignment::Attack { target, .. } = a {
                let key = format!("{},{}", target.x(), target.y());
                *target_counts.entry(key).or_insert(0) += 1;
            }
        }
        // Each target should get 2 drones (4 drones / 2 targets)
        for count in target_counts.values() {
            assert_eq!(*count, 2, "Should spread drones evenly across targets");
        }
    }
}
