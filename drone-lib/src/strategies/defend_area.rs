//! Defend area strategy: protect a zone with orbiting defenders and interceptors.
//!
//! Layered defense:
//! - **Inner ring**: Drones orbit the center with DefendTask, engaging local threats
//! - **Outer ring**: When enemies approach from afar, the strategy redirects a
//!   defender as an interceptor to meet them early
//! - **Reassignment**: After interception (complete/fail), drones return to defense
//! - **Safe spacing**: Orbit radius is enforced to keep drones at least
//!   detonation_radius apart so a single explosion can't chain-kill friendlies

use std::collections::{HashMap, HashSet};

use crate::strategies::{StrategyDroneState, SwarmStrategy, TaskAssignment};
use crate::types::{Bounds, DroneInfo, Position};

/// Role assigned to a drone within the defend area strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
enum DefendRole {
    /// Orbiting the defense center (DefendTask).
    Defender,
    /// Pursuing a specific approaching enemy (InterceptTask).
    Interceptor(usize),
}

/// Defend area strategy.
///
/// Allocates drones as defenders (orbit + local engage) and interceptors
/// (proactive pursuit of approaching threats). Automatically reallocates
/// drones when tasks complete, fail, or the threat situation changes.
#[derive(Debug)]
pub struct DefendAreaStrategy {
    center: Position,
    drone_ids: Vec<usize>,
    self_group: u32,
    /// Role assignments for each drone.
    roles: HashMap<usize, DefendRole>,
    /// Enemies currently being intercepted: enemy_id → interceptor_drone_id.
    engaged_enemies: HashMap<usize, usize>,
    /// Orbit radius for defenders (inner ring).
    orbit_radius: f32,
    /// DefendTask engage radius (local threat engagement).
    defend_engage_radius: f32,
    /// Distance at which strategy sends interceptors (outer ring).
    intercept_trigger_radius: f32,
    /// Detonation/blast radius — minimum safe spacing between drones.
    detonation_radius: f32,
    /// When true, alternate between Intercept and InterceptGroup assignments.
    mix_intercept_group: bool,
    /// Counter for alternating intercept types.
    intercept_counter: usize,
}

impl DefendAreaStrategy {
    /// Create a new defend area strategy.
    ///
    /// * `drone_ids` — drones allocated to this strategy
    /// * `self_group` — group ID of defending drones (enemies have different group)
    /// * `center` — center of the defense zone (meters)
    /// * `radius` — overall defense zone radius (meters)
    /// * `detonation_radius` — blast radius used for safe spacing
    pub fn new(
        drone_ids: Vec<usize>,
        self_group: u32,
        center: Position,
        radius: f32,
        detonation_radius: f32,
    ) -> Self {
        // Enforce minimum orbit radius: drones evenly spaced on a circle
        // must be > detonation_radius apart. Arc spacing = 2*pi*r / n.
        // Solve for r: r >= n * detonation_radius / (2*pi).
        let n = drone_ids.len().max(1) as f32;
        let min_orbit_radius = n * detonation_radius / std::f32::consts::TAU;
        let desired_orbit = radius * 0.4;
        let orbit_radius = desired_orbit.max(min_orbit_radius);

        DefendAreaStrategy {
            center,
            drone_ids,
            self_group,
            roles: HashMap::new(),
            engaged_enemies: HashMap::new(),
            orbit_radius,
            defend_engage_radius: radius * 0.8,
            intercept_trigger_radius: radius * 2.0,
            detonation_radius,
            mix_intercept_group: false,
            intercept_counter: 0,
        }
    }

    /// Enable mixed intercept mode: alternate between Intercept and InterceptGroup.
    pub fn with_mixed_intercept(mut self, enabled: bool) -> Self {
        self.mix_intercept_group = enabled;
        self
    }

    /// Find enemy drones approaching the defense zone that aren't already engaged.
    fn find_approaching_enemies(
        &self,
        all_drones: &[DroneInfo],
        bounds: &Bounds,
    ) -> Vec<usize> {
        all_drones
            .iter()
            .filter(|d| {
                d.group != self.self_group
                    && !self.engaged_enemies.contains_key(&d.uid)
                    && bounds.distance(self.center.as_vec2(), d.pos.as_vec2())
                        <= self.intercept_trigger_radius
            })
            .map(|d| d.uid)
            .collect()
    }

    /// Find the best drone to send as interceptor.
    /// Prefers available (idle) drones, falls back to pulling a defender off orbit.
    /// Skips drones that are too close to the center (within detonation_radius)
    /// to avoid detonating near friendly targets.
    fn find_drone_for_intercept(
        &self,
        enemy_pos: Position,
        own_drones: &[StrategyDroneState],
        bounds: &Bounds,
        already_assigned: &HashSet<usize>,
    ) -> Option<usize> {
        let safe_to_dispatch = |d: &&StrategyDroneState| -> bool {
            // Only dispatch drones far enough from center that detonation won't hit targets
            bounds.distance(d.pos.as_vec2(), self.center.as_vec2()) > self.detonation_radius
        };

        let closest_by = |iter: Box<dyn Iterator<Item = &StrategyDroneState> + '_>| {
            iter.min_by(|a, b| {
                let da = bounds.distance(a.pos.as_vec2(), enemy_pos.as_vec2());
                let db = bounds.distance(b.pos.as_vec2(), enemy_pos.as_vec2());
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|d| d.id)
        };

        // First: try available (idle) drones that are safely away from center
        let available = Box::new(
            own_drones
                .iter()
                .filter(|d| d.available && !already_assigned.contains(&d.id))
                .filter(safe_to_dispatch),
        );
        if let Some(id) = closest_by(available) {
            return Some(id);
        }

        // Second: pull the closest defender off orbit (also must be safe)
        let defenders = Box::new(
            own_drones
                .iter()
                .filter(|d| {
                    !already_assigned.contains(&d.id)
                        && matches!(self.roles.get(&d.id), Some(DefendRole::Defender))
                })
                .filter(safe_to_dispatch),
        );
        closest_by(defenders)
    }
}

impl SwarmStrategy for DefendAreaStrategy {
    fn tick(
        &mut self,
        own_drones: &[StrategyDroneState],
        all_drones: &[DroneInfo],
        bounds: &Bounds,
        _dt: f32,
    ) -> Vec<TaskAssignment> {
        let mut assignments = Vec::new();
        let active_ids: HashSet<usize> = own_drones.iter().map(|d| d.id).collect();

        // 1. Remove destroyed drones from tracking
        self.roles.retain(|id, _| active_ids.contains(id));
        self.engaged_enemies
            .retain(|_, interceptor_id| active_ids.contains(interceptor_id));

        // 2. Handle available drones (task complete/failed) — clear their roles
        for drone in own_drones.iter().filter(|d| d.available) {
            if let Some(DefendRole::Interceptor(enemy_id)) = self.roles.remove(&drone.id) {
                self.engaged_enemies.remove(&enemy_id);
            } else {
                self.roles.remove(&drone.id);
            }
        }

        // 3. Find approaching enemies
        let threats = self.find_approaching_enemies(all_drones, bounds);

        // 4. Assign interceptors to new threats
        let mut just_assigned: HashSet<usize> = HashSet::new();
        for enemy_id in &threats {
            let enemy_pos = all_drones
                .iter()
                .find(|d| d.uid == *enemy_id)
                .map(|d| d.pos);
            let Some(enemy_pos) = enemy_pos else {
                continue;
            };

            if let Some(drone_id) =
                self.find_drone_for_intercept(enemy_pos, own_drones, bounds, &just_assigned)
            {
                if self.mix_intercept_group && self.intercept_counter % 2 == 1 {
                    assignments.push(TaskAssignment::InterceptGroup { drone_id });
                } else {
                    assignments.push(TaskAssignment::Intercept {
                        drone_id,
                        target_id: *enemy_id,
                    });
                }
                self.intercept_counter += 1;
                self.roles
                    .insert(drone_id, DefendRole::Interceptor(*enemy_id));
                self.engaged_enemies.insert(*enemy_id, drone_id);
                just_assigned.insert(drone_id);
            }
        }

        // 5. Assign remaining unassigned drones as defenders
        for drone in own_drones {
            if just_assigned.contains(&drone.id) {
                continue;
            }
            if self.roles.contains_key(&drone.id) {
                continue;
            }
            assignments.push(TaskAssignment::Defend {
                drone_id: drone.id,
                center: self.center,
                orbit_radius: self.orbit_radius,
                engage_radius: self.defend_engage_radius,
            });
            self.roles.insert(drone.id, DefendRole::Defender);
        }

        assignments
    }

    fn drone_ids(&self) -> &[usize] {
        &self.drone_ids
    }

    fn name(&self) -> &str {
        "DefendArea"
    }

    fn is_complete(&self) -> bool {
        false // Continuous defense
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Heading, Velocity};

    fn make_own_drone(id: usize, x: f32, y: f32, available: bool) -> StrategyDroneState {
        StrategyDroneState {
            id,
            pos: Position::new(x, y),
            vel: Velocity::zero(),
            available,
        }
    }

    fn make_drone_info(uid: usize, x: f32, y: f32, group: u32) -> DroneInfo {
        DroneInfo {
            uid,
            pos: Position::new(x, y),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
            group,
        }
    }

    fn bounds() -> crate::types::Bounds {
        crate::types::Bounds::new(5000.0, 5000.0).unwrap()
    }

    #[test]
    fn test_initial_assignment_as_defenders() {
        let mut strategy =
            DefendAreaStrategy::new(vec![0, 1, 2], 0, Position::new(500.0, 500.0), 300.0, 187.5);

        let own = vec![
            make_own_drone(0, 400.0, 500.0, true),
            make_own_drone(1, 500.0, 400.0, true),
            make_own_drone(2, 600.0, 500.0, true),
        ];
        let all = vec![
            make_drone_info(0, 400.0, 500.0, 0),
            make_drone_info(1, 500.0, 400.0, 0),
            make_drone_info(2, 600.0, 500.0, 0),
        ];

        let assignments = strategy.tick(&own, &all, &bounds(), 0.016);

        assert_eq!(assignments.len(), 3, "All drones should get initial DefendTask");
        for a in &assignments {
            assert!(
                matches!(a, TaskAssignment::Defend { .. }),
                "Initial assignment should be Defend"
            );
        }
    }

    #[test]
    fn test_orbit_radius_enforces_safe_spacing() {
        // With 10 drones and detonation_radius=187.5, minimum orbit radius
        // = 10 * 187.5 / (2*pi) ≈ 298.4m which is > 300*0.4=120m
        let strategy = DefendAreaStrategy::new(
            (0..10).collect(),
            0,
            Position::new(500.0, 500.0),
            300.0,
            187.5,
        );

        let min_required = 10.0 * 187.5 / std::f32::consts::TAU;
        assert!(
            strategy.orbit_radius >= min_required,
            "Orbit radius {} should be >= {} for safe spacing",
            strategy.orbit_radius,
            min_required
        );
    }

    #[test]
    fn test_intercept_approaching_enemy() {
        let mut strategy =
            DefendAreaStrategy::new(vec![0, 1, 2], 0, Position::new(500.0, 500.0), 300.0, 187.5);

        // First tick: assign defenders (all available, far from center)
        let own = vec![
            make_own_drone(0, 200.0, 500.0, true),
            make_own_drone(1, 500.0, 200.0, true),
            make_own_drone(2, 800.0, 500.0, true),
        ];
        let all_friendly = vec![
            make_drone_info(0, 200.0, 500.0, 0),
            make_drone_info(1, 500.0, 200.0, 0),
            make_drone_info(2, 800.0, 500.0, 0),
        ];
        strategy.tick(&own, &all_friendly, &bounds(), 0.016);

        // Second tick: enemy approaches, drones are busy as defenders
        let own_busy = vec![
            make_own_drone(0, 200.0, 500.0, false),
            make_own_drone(1, 500.0, 200.0, false),
            make_own_drone(2, 800.0, 500.0, false),
        ];
        let all_with_enemy = vec![
            make_drone_info(0, 200.0, 500.0, 0),
            make_drone_info(1, 500.0, 200.0, 0),
            make_drone_info(2, 800.0, 500.0, 0),
            make_drone_info(10, 900.0, 500.0, 1), // Enemy approaching from east
        ];

        let assignments = strategy.tick(&own_busy, &all_with_enemy, &bounds(), 0.016);

        assert_eq!(assignments.len(), 1, "Should send one interceptor");
        match &assignments[0] {
            TaskAssignment::Intercept {
                drone_id,
                target_id,
            } => {
                assert_eq!(*target_id, 10, "Should target the enemy");
                // Drone 2 at (800, 500) is closest to enemy at (900, 500)
                assert_eq!(*drone_id, 2, "Should pick closest defender");
            }
            other => panic!("Expected Intercept, got {:?}", other),
        }
    }

    #[test]
    fn test_no_double_intercept() {
        let mut strategy =
            DefendAreaStrategy::new(vec![0, 1], 0, Position::new(500.0, 500.0), 300.0, 187.5);

        // Initial assignment — drones far from center
        let own = vec![
            make_own_drone(0, 200.0, 500.0, true),
            make_own_drone(1, 800.0, 500.0, true),
        ];
        let all = vec![
            make_drone_info(0, 200.0, 500.0, 0),
            make_drone_info(1, 800.0, 500.0, 0),
        ];
        strategy.tick(&own, &all, &bounds(), 0.016);

        // Enemy appears
        let own_busy = vec![
            make_own_drone(0, 200.0, 500.0, false),
            make_own_drone(1, 800.0, 500.0, false),
        ];
        let all_enemy = vec![
            make_drone_info(0, 200.0, 500.0, 0),
            make_drone_info(1, 800.0, 500.0, 0),
            make_drone_info(10, 900.0, 500.0, 1),
        ];
        strategy.tick(&own_busy, &all_enemy, &bounds(), 0.016);

        // Same enemy still there — should NOT send another interceptor
        let assignments = strategy.tick(&own_busy, &all_enemy, &bounds(), 0.016);

        let intercept_count = assignments
            .iter()
            .filter(|a| matches!(a, TaskAssignment::Intercept { .. }))
            .count();
        assert_eq!(intercept_count, 0, "Same enemy should not get second interceptor");
    }

    #[test]
    fn test_reassign_after_intercept_complete() {
        let mut strategy =
            DefendAreaStrategy::new(vec![0, 1], 0, Position::new(500.0, 500.0), 300.0, 187.5);

        // Initial assignment — drones far from center
        let own = vec![
            make_own_drone(0, 200.0, 500.0, true),
            make_own_drone(1, 800.0, 500.0, true),
        ];
        let all = vec![
            make_drone_info(0, 200.0, 500.0, 0),
            make_drone_info(1, 800.0, 500.0, 0),
        ];
        strategy.tick(&own, &all, &bounds(), 0.016);

        // Drone 1 sent to intercept
        let own_busy = vec![
            make_own_drone(0, 200.0, 500.0, false),
            make_own_drone(1, 800.0, 500.0, false),
        ];
        let all_enemy = vec![
            make_drone_info(0, 200.0, 500.0, 0),
            make_drone_info(1, 800.0, 500.0, 0),
            make_drone_info(10, 900.0, 500.0, 1),
        ];
        strategy.tick(&own_busy, &all_enemy, &bounds(), 0.016);

        // Intercept complete — drone 1 becomes available, enemy gone
        let own_after = vec![
            make_own_drone(0, 200.0, 500.0, false),
            make_own_drone(1, 800.0, 500.0, true), // available again
        ];
        let all_no_enemy = vec![
            make_drone_info(0, 200.0, 500.0, 0),
            make_drone_info(1, 800.0, 500.0, 0),
        ];
        let assignments = strategy.tick(&own_after, &all_no_enemy, &bounds(), 0.016);

        // Drone 1 should be reassigned as defender
        assert!(
            assignments
                .iter()
                .any(|a| matches!(a, TaskAssignment::Defend { drone_id: 1, .. })),
            "Available drone should be reassigned as defender"
        );
    }

    #[test]
    fn test_ignore_friendly_drones() {
        let mut strategy =
            DefendAreaStrategy::new(vec![0], 0, Position::new(500.0, 500.0), 300.0, 187.5);

        let own = vec![make_own_drone(0, 200.0, 500.0, true)];
        let all = vec![
            make_drone_info(0, 200.0, 500.0, 0),
            make_drone_info(5, 500.0, 500.0, 0), // Friendly inside zone
        ];

        let assignments = strategy.tick(&own, &all, &bounds(), 0.016);

        let intercepts = assignments
            .iter()
            .filter(|a| matches!(a, TaskAssignment::Intercept { .. }))
            .count();
        assert_eq!(intercepts, 0, "Should not intercept friendly drones");
    }

    #[test]
    fn test_handle_destroyed_drone() {
        let mut strategy =
            DefendAreaStrategy::new(vec![0, 1], 0, Position::new(500.0, 500.0), 300.0, 187.5);

        // Initial assignment
        let own = vec![
            make_own_drone(0, 200.0, 500.0, true),
            make_own_drone(1, 800.0, 500.0, true),
        ];
        let all = vec![
            make_drone_info(0, 200.0, 500.0, 0),
            make_drone_info(1, 800.0, 500.0, 0),
        ];
        strategy.tick(&own, &all, &bounds(), 0.016);

        // Drone 1 destroyed (no longer in own_drones)
        let own_after = vec![make_own_drone(0, 200.0, 500.0, false)];
        let all_after = vec![make_drone_info(0, 200.0, 500.0, 0)];

        // Should not panic, should clean up drone 1
        let assignments = strategy.tick(&own_after, &all_after, &bounds(), 0.016);
        assert!(assignments.is_empty(), "No reassignment needed");
    }
}
