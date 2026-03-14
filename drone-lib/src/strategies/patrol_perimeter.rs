//! Patrol perimeter strategy: formation-based circuit patrol with intercept.
//!
//! Drones are organized into squads of up to 6, each in a chevron formation
//! with an auto-assigned leader. Squads are staggered around the circuit
//! for even coverage. When an enemy approaches, the nearest patroller is
//! dispatched to intercept.

use std::collections::{HashMap, HashSet};

use crate::strategies::{StrategyDroneState, SwarmStrategy, TaskAssignment};
use crate::types::{Bounds, DroneInfo, Position};

/// Maximum drones per formation squad.
const SQUAD_SIZE: usize = 6;

/// Role for a drone in the patrol perimeter strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PatrolRole {
    /// Part of a patrol squad (leader or follower).
    Patroller,
    /// Intercepting a specific enemy drone.
    Interceptor(usize),
}

/// A squad of drones patrolling together in formation.
#[derive(Debug, Clone)]
struct Squad {
    leader_id: usize,
    follower_ids: Vec<usize>,
}

/// Patrol perimeter strategy.
///
/// Divides drones into squads of up to 6, each patrolling the full circuit
/// in chevron formation with staggered starting positions. Dispatches
/// interceptors when enemies approach.
#[derive(Debug)]
pub struct PatrolPerimeterStrategy {
    waypoints: Vec<Position>,
    drone_ids: Vec<usize>,
    loiter_duration: f32,
    /// Group ID for friend/foe distinction.
    self_group: u32,
    /// Detonation radius — minimum safe spacing.
    detonation_radius: f32,
    /// Role assignments.
    roles: HashMap<usize, PatrolRole>,
    /// Current squads.
    squads: Vec<Squad>,
    /// Enemies currently being intercepted: enemy_id → interceptor_drone_id.
    engaged_enemies: HashMap<usize, usize>,
    /// Number of active drones last tick (to detect changes requiring reassignment).
    last_active_count: usize,
    /// Center of the patrol zone (average of waypoints).
    center: Position,
    /// Detection radius for incoming threats.
    detection_radius: f32,
    /// When true, alternate between Intercept and InterceptGroup assignments.
    mix_intercept_group: bool,
    /// Counter for alternating intercept types.
    intercept_counter: usize,
}

impl PatrolPerimeterStrategy {
    /// Create a new patrol perimeter strategy.
    pub fn new(
        drone_ids: Vec<usize>,
        waypoints: Vec<Position>,
        loiter_duration: f32,
        self_group: u32,
        detonation_radius: f32,
    ) -> Self {
        let center = if waypoints.is_empty() {
            Position::new(0.0, 0.0)
        } else {
            let cx = waypoints.iter().map(|p| p.x()).sum::<f32>() / waypoints.len() as f32;
            let cy = waypoints.iter().map(|p| p.y()).sum::<f32>() / waypoints.len() as f32;
            Position::new(cx, cy)
        };

        let max_wp_dist = waypoints
            .iter()
            .map(|p| {
                let dx = p.x() - center.x();
                let dy = p.y() - center.y();
                (dx * dx + dy * dy).sqrt()
            })
            .fold(0.0_f32, f32::max);
        let detection_radius = max_wp_dist + detonation_radius * 3.0;

        PatrolPerimeterStrategy {
            waypoints,
            drone_ids,
            loiter_duration,
            self_group,
            detonation_radius,
            roles: HashMap::new(),
            squads: Vec::new(),
            engaged_enemies: HashMap::new(),
            last_active_count: 0,
            center,
            detection_radius,
            mix_intercept_group: false,
            intercept_counter: 0,
        }
    }

    /// Enable mixed intercept mode: alternate between Intercept and InterceptGroup.
    pub fn with_mixed_intercept(mut self, enabled: bool) -> Self {
        self.mix_intercept_group = enabled;
        self
    }

    /// Build the full circuit of waypoints starting from a given offset.
    fn circuit_from_offset(&self, offset: usize) -> Vec<Position> {
        let n = self.waypoints.len();
        if n == 0 {
            return Vec::new();
        }
        (0..n)
            .map(|i| self.waypoints[(offset + i) % n])
            .collect()
    }

    /// Divide drones into squads of up to SQUAD_SIZE.
    /// The first drone in each squad is the leader.
    fn build_squads(drone_ids: &[usize]) -> Vec<Squad> {
        drone_ids
            .chunks(SQUAD_SIZE)
            .map(|chunk| {
                let leader_id = chunk[0];
                let follower_ids = chunk[1..].to_vec();
                Squad {
                    leader_id,
                    follower_ids,
                }
            })
            .collect()
    }

    /// Find enemy drones approaching the patrol zone.
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
                        <= self.detection_radius
            })
            .map(|d| d.uid)
            .collect()
    }

    /// Find the best patrolling drone to intercept an enemy.
    fn find_interceptor(
        &self,
        enemy_pos: Position,
        own_drones: &[StrategyDroneState],
        bounds: &Bounds,
        already_assigned: &HashSet<usize>,
    ) -> Option<usize> {
        own_drones
            .iter()
            .filter(|d| {
                !already_assigned.contains(&d.id)
                    && matches!(
                        self.roles.get(&d.id),
                        Some(PatrolRole::Patroller) | None
                    )
                    && bounds.distance(d.pos.as_vec2(), self.center.as_vec2())
                        > self.detonation_radius
            })
            .min_by(|a, b| {
                let da = bounds.distance(a.pos.as_vec2(), enemy_pos.as_vec2());
                let db = bounds.distance(b.pos.as_vec2(), enemy_pos.as_vec2());
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|d| d.id)
    }

    /// Generate patrol formation assignments for all squads.
    fn assign_squads(&self) -> Vec<TaskAssignment> {
        let n_wps = self.waypoints.len();
        let n_squads = self.squads.len();
        if n_squads == 0 || n_wps == 0 {
            return Vec::new();
        }

        self.squads
            .iter()
            .enumerate()
            .map(|(i, squad)| {
                let offset = (i * n_wps) / n_squads;
                let waypoints = self.circuit_from_offset(offset);
                TaskAssignment::PatrolFormation {
                    leader_id: squad.leader_id,
                    follower_ids: squad.follower_ids.clone(),
                    waypoints,
                    loiter_duration: self.loiter_duration,
                }
            })
            .collect()
    }
}

impl SwarmStrategy for PatrolPerimeterStrategy {
    fn tick(
        &mut self,
        own_drones: &[StrategyDroneState],
        all_drones: &[DroneInfo],
        bounds: &Bounds,
        _dt: f32,
    ) -> Vec<TaskAssignment> {
        let mut assignments = Vec::new();

        let active_ids: HashSet<usize> = own_drones.iter().map(|d| d.id).collect();
        let active_count = active_ids.len();

        if active_count == 0 || self.waypoints.is_empty() {
            return assignments;
        }

        // Clean up destroyed drones
        self.roles.retain(|id, _| active_ids.contains(id));
        self.engaged_enemies
            .retain(|_, interceptor_id| active_ids.contains(interceptor_id));

        // Handle completed interceptors — return them to the patrol pool.
        // Only clear Interceptor roles; Patroller roles stay (followers are
        // always "available" per the task system but are actively in formation).
        let mut interceptors_returned = false;
        for drone in own_drones.iter().filter(|d| d.available) {
            if matches!(self.roles.get(&drone.id), Some(PatrolRole::Interceptor(_))) {
                if let Some(PatrolRole::Interceptor(enemy_id)) = self.roles.remove(&drone.id) {
                    self.engaged_enemies.remove(&enemy_id);
                    interceptors_returned = true;
                }
            }
        }

        // --- Intercept approaching enemies ---
        let threats = self.find_approaching_enemies(all_drones, bounds);
        let mut just_assigned_intercept: HashSet<usize> = HashSet::new();

        let min_patrollers = (active_count / 2).max(1);
        let current_patrollers = own_drones
            .iter()
            .filter(|d| !matches!(self.roles.get(&d.id), Some(PatrolRole::Interceptor(_))))
            .count();

        for enemy_id in &threats {
            if current_patrollers - just_assigned_intercept.len() <= min_patrollers {
                break;
            }
            let enemy_pos = all_drones
                .iter()
                .find(|d| d.uid == *enemy_id)
                .map(|d| d.pos);
            let Some(enemy_pos) = enemy_pos else { continue };

            if let Some(drone_id) =
                self.find_interceptor(enemy_pos, own_drones, bounds, &just_assigned_intercept)
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
                    .insert(drone_id, PatrolRole::Interceptor(*enemy_id));
                self.engaged_enemies.insert(*enemy_id, drone_id);
                just_assigned_intercept.insert(drone_id);
            }
        }

        // --- Build squads from non-interceptor drones ---
        // Only rebuild when something structurally changed:
        //   - drone count changed (drone destroyed)
        //   - an intercept was just dispatched (squad lost a member)
        //   - an interceptor returned (squad gains a member)
        let needs_rebuild = active_count != self.last_active_count
            || !just_assigned_intercept.is_empty()
            || interceptors_returned;
        self.last_active_count = active_count;

        if needs_rebuild {
            // Collect patroller IDs (exclude interceptors)
            let patroller_ids: Vec<usize> = own_drones
                .iter()
                .filter(|d| {
                    !matches!(self.roles.get(&d.id), Some(PatrolRole::Interceptor(_)))
                        && !just_assigned_intercept.contains(&d.id)
                })
                .map(|d| d.id)
                .collect();

            // Build squads and assign
            self.squads = Self::build_squads(&patroller_ids);
            for id in &patroller_ids {
                self.roles.insert(*id, PatrolRole::Patroller);
            }
            assignments.extend(self.assign_squads());
        }

        assignments
    }

    fn drone_ids(&self) -> &[usize] {
        &self.drone_ids
    }

    fn name(&self) -> &str {
        "PatrolPerimeter"
    }

    fn is_complete(&self) -> bool {
        false
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

    fn route_waypoints() -> Vec<Position> {
        vec![
            Position::new(0.0, 0.0),
            Position::new(100.0, 0.0),
            Position::new(200.0, 0.0),
            Position::new(300.0, 0.0),
            Position::new(400.0, 0.0),
            Position::new(500.0, 0.0),
        ]
    }

    fn bounds() -> Bounds {
        Bounds::new(5000.0, 5000.0).unwrap()
    }

    #[test]
    fn test_squads_of_six() {
        // 13 drones should produce 3 squads: 6, 6, 1
        let ids: Vec<usize> = (0..13).collect();
        let squads = PatrolPerimeterStrategy::build_squads(&ids);

        assert_eq!(squads.len(), 3);
        assert_eq!(squads[0].follower_ids.len(), 5); // leader + 5 followers
        assert_eq!(squads[1].follower_ids.len(), 5);
        assert_eq!(squads[2].follower_ids.len(), 0); // solo leader
    }

    #[test]
    fn test_emits_patrol_formation_assignments() {
        let drone_ids: Vec<usize> = (0..12).collect();
        let mut strategy =
            PatrolPerimeterStrategy::new(drone_ids, route_waypoints(), 2.0, 0, 187.5);

        let own: Vec<StrategyDroneState> = (0..12)
            .map(|i| make_own_drone(i, i as f32 * 100.0, 500.0, true))
            .collect();

        let assignments = strategy.tick(&own, &[], &bounds(), 0.016);

        let formation_count = assignments
            .iter()
            .filter(|a| matches!(a, TaskAssignment::PatrolFormation { .. }))
            .count();
        assert_eq!(formation_count, 2, "12 drones should produce 2 squads");

        for a in &assignments {
            if let TaskAssignment::PatrolFormation {
                follower_ids,
                waypoints,
                ..
            } = a
            {
                assert_eq!(follower_ids.len(), 5, "Each squad has 5 followers");
                assert_eq!(waypoints.len(), 6, "Full circuit");
            }
        }
    }

    #[test]
    fn test_staggered_squad_starts() {
        let drone_ids: Vec<usize> = (0..12).collect();
        let mut strategy =
            PatrolPerimeterStrategy::new(drone_ids, route_waypoints(), 2.0, 0, 187.5);

        let own: Vec<StrategyDroneState> = (0..12)
            .map(|i| make_own_drone(i, i as f32 * 50.0, 500.0, true))
            .collect();

        let assignments = strategy.tick(&own, &[], &bounds(), 0.016);
        let formations: Vec<_> = assignments
            .iter()
            .filter_map(|a| {
                if let TaskAssignment::PatrolFormation { waypoints, .. } = a {
                    Some(waypoints)
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(formations.len(), 2);
        // First waypoints should differ between squads
        assert!(
            (formations[0][0].x() - formations[1][0].x()).abs() > f32::EPSILON,
            "Squads should start at different offsets"
        );
    }

    #[test]
    fn test_intercept_dispatches_from_patrol() {
        let drone_ids: Vec<usize> = (0..6).collect();
        let mut strategy =
            PatrolPerimeterStrategy::new(drone_ids, route_waypoints(), 2.0, 0, 187.5);

        // Initial patrol
        let own: Vec<StrategyDroneState> = (0..6)
            .map(|i| make_own_drone(i, -500.0 + i as f32 * 300.0, 0.0, true))
            .collect();
        strategy.tick(&own, &[], &bounds(), 0.016);

        // Enemy approaches
        let own_busy: Vec<StrategyDroneState> = (0..6)
            .map(|i| make_own_drone(i, -500.0 + i as f32 * 300.0, 0.0, false))
            .collect();
        let all_with_enemy = vec![make_drone_info(10, 700.0, 0.0, 1)];

        let assignments = strategy.tick(&own_busy, &all_with_enemy, &bounds(), 0.016);

        let intercepts = assignments
            .iter()
            .filter(|a| matches!(a, TaskAssignment::Intercept { .. }))
            .count();
        assert_eq!(intercepts, 1, "Should dispatch one interceptor");
    }

    #[test]
    fn test_no_reassignment_when_stable() {
        let drone_ids: Vec<usize> = (0..6).collect();
        let mut strategy =
            PatrolPerimeterStrategy::new(drone_ids, route_waypoints(), 2.0, 0, 187.5);

        let own: Vec<StrategyDroneState> = (0..6)
            .map(|i| make_own_drone(i, 0.0, i as f32 * 100.0, true))
            .collect();
        strategy.tick(&own, &[], &bounds(), 0.016);

        // Same drones, still busy
        let own_busy: Vec<StrategyDroneState> = (0..6)
            .map(|i| make_own_drone(i, 0.0, i as f32 * 100.0, false))
            .collect();
        let assignments = strategy.tick(&own_busy, &[], &bounds(), 0.016);

        assert!(assignments.is_empty(), "No reassignment when stable");
    }
}
