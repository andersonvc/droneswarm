//! Doctrine layer: autonomous force allocation to achieve a win condition.
//!
//! Doctrine sits above strategies in the operational hierarchy:
//! **Doctrine → Strategy → Task → Safety → Platform**
//!
//! It observes game state (drone counts, target counts, threat proximity)
//! and dynamically allocates drones between defense (patrol_perimeter) and
//! offense (attack_zone) to maximize the chance of winning.
//!
//! Win condition: destroy all enemy targets while preserving friendly ones.

use std::collections::HashSet;

use crate::strategies::attack_zone::AttackZoneStrategy;
use crate::strategies::patrol_perimeter::PatrolPerimeterStrategy;
use crate::strategies::{StrategyDroneState, SwarmStrategy, TaskAssignment};
use crate::types::{Bounds, DroneInfo, Position};

/// How often to re-evaluate force allocation (in ticks).
const REALLOC_INTERVAL: u32 = 60;

/// Threat detection radius: enemy drones within this distance of
/// the friendly target centroid trigger increased defense allocation.
const THREAT_RADIUS_MULTIPLIER: f32 = 5.0;

/// Doctrine posture: controls the balance between defense and offense.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoctrineMode {
    /// Prioritize defense. Min 50% defenders, attack wave = 2 drones per target.
    Defensive,
    /// Prioritize offense. Min 15% defenders, unlimited attack waves.
    Aggressive,
}

impl DoctrineMode {
    /// Minimum fraction of drones kept on defense.
    fn min_defend_fraction(self) -> f32 {
        match self {
            DoctrineMode::Defensive => 0.5,
            DoctrineMode::Aggressive => 0.15,
        }
    }

    /// Max simultaneous attackers (None = unlimited).
    fn wave_size(self, enemy_target_count: usize) -> Option<usize> {
        match self {
            DoctrineMode::Defensive => Some(enemy_target_count * 2),
            DoctrineMode::Aggressive => None,
        }
    }
}

/// The Doctrine layer for one group.
///
/// Manages force allocation between defense and offense, dynamically
/// creating and ticking child strategies. Implements `SwarmStrategy`
/// so it plugs into the existing pipeline.
#[derive(Debug)]
pub struct SwarmDoctrine {
    /// Group ID (0 or 1).
    group_id: u32,
    /// All drone IDs managed by this objective.
    drone_ids: Vec<usize>,
    /// Detonation radius for child strategy construction.
    detonation_radius: f32,
    /// Posture: aggressive vs defensive.
    mode: DoctrineMode,

    /// Surviving friendly target positions.
    friendly_targets: Vec<Position>,
    /// Surviving enemy target positions.
    enemy_targets: Vec<Position>,
    /// Patrol waypoints (convex hull around friendly targets).
    patrol_waypoints: Vec<Position>,

    // --- Child strategies ---
    patrol_strategy: Option<PatrolPerimeterStrategy>,
    attack_strategy: Option<AttackZoneStrategy>,

    // --- Allocation state ---
    defend_ids: Vec<usize>,
    attack_ids: Vec<usize>,

    // --- Decision cooldown ---
    ticks_since_realloc: u32,
    /// Force reallocation on next tick (e.g., after target update).
    force_realloc: bool,

    // --- Change detection ---
    last_own_count: usize,
    last_enemy_target_count: usize,
    last_friendly_target_count: usize,
}

impl SwarmDoctrine {
    /// Create a new doctrine for a group.
    ///
    /// * `drone_ids` — all drones in this group
    /// * `group_id` — 0 or 1
    /// * `friendly_targets` — positions of friendly targets to defend (meters)
    /// * `enemy_targets` — positions of enemy targets to destroy (meters)
    /// * `patrol_waypoints` — convex hull route around friendly targets (meters)
    /// * `detonation_radius` — blast radius for safe spacing
    /// * `mode` — aggressive or defensive posture
    pub fn new(
        drone_ids: Vec<usize>,
        group_id: u32,
        friendly_targets: Vec<Position>,
        enemy_targets: Vec<Position>,
        patrol_waypoints: Vec<Position>,
        detonation_radius: f32,
        mode: DoctrineMode,
    ) -> Self {
        SwarmDoctrine {
            group_id,
            drone_ids,
            detonation_radius,
            mode,
            friendly_targets,
            enemy_targets,
            patrol_waypoints,
            patrol_strategy: None,
            attack_strategy: None,
            defend_ids: Vec::new(),
            attack_ids: Vec::new(),
            ticks_since_realloc: REALLOC_INTERVAL, // trigger immediate allocation
            force_realloc: true,
            last_own_count: 0,
            last_enemy_target_count: 0,
            last_friendly_target_count: 0,
        }
    }

    /// Set the doctrine mode, triggering reallocation on next tick.
    pub fn set_mode(&mut self, mode: DoctrineMode) {
        if self.mode != mode {
            self.mode = mode;
            self.force_realloc = true;
        }
    }

    /// Get the current doctrine mode.
    pub fn mode(&self) -> DoctrineMode {
        self.mode
    }

    /// Number of drones currently allocated to defense.
    pub fn defend_count(&self) -> usize {
        self.defend_ids.len()
    }

    /// Number of drones currently allocated to attack.
    pub fn attack_count(&self) -> usize {
        self.attack_ids.len()
    }

    /// Count enemy drones within threat radius of the friendly target centroid.
    fn count_nearby_threats(&self, all_drones: &[DroneInfo], bounds: &Bounds) -> usize {
        if self.friendly_targets.is_empty() {
            return 0;
        }
        let cx = self.friendly_targets.iter().map(|p| p.x()).sum::<f32>()
            / self.friendly_targets.len() as f32;
        let cy = self.friendly_targets.iter().map(|p| p.y()).sum::<f32>()
            / self.friendly_targets.len() as f32;
        let centroid = Position::new(cx, cy);
        let threat_radius = self.detonation_radius * THREAT_RADIUS_MULTIPLIER;

        all_drones
            .iter()
            .filter(|d| {
                d.group != self.group_id
                    && bounds.distance(centroid.as_vec2(), d.pos.as_vec2()) <= threat_radius
            })
            .count()
    }

    /// Compute force allocation: how many drones for defense vs. attack.
    fn compute_allocation(
        own_count: usize,
        friendly_target_count: usize,
        enemy_target_count: usize,
        nearby_threats: usize,
        mode: DoctrineMode,
    ) -> (usize, usize) {
        if own_count == 0 {
            return (0, 0);
        }

        // No enemy targets → all defense
        if enemy_target_count == 0 {
            return (own_count, 0);
        }

        // No friendly targets → all attack
        if friendly_target_count == 0 {
            return (0, own_count);
        }

        // Base: 2 drones per enemy target for attack, rest for defense
        let desired_attack = (enemy_target_count * 2).min(own_count);

        // Minimum defense based on mode, minimum 2
        let min_defend_frac = mode.min_defend_fraction();
        let min_defend = ((own_count as f32 * min_defend_frac).ceil() as usize)
            .max(2)
            .min(own_count);

        // Threat adjustment: +1 defender per nearby threat
        let threat_boost = nearby_threats.min(own_count / 3);
        let adjusted_defend = (min_defend + threat_boost).min(own_count);

        // Attack gets the remainder, but at least 1 if enemy targets exist
        let attack_count = own_count
            .saturating_sub(adjusted_defend)
            .min(desired_attack)
            .max(1);

        let defend_count = own_count - attack_count;

        (defend_count, attack_count)
    }

    /// Partition drone IDs: defenders are closest to friendly centroid,
    /// attackers are furthest (closer to enemy territory).
    fn partition_drones(
        &self,
        own_drones: &[StrategyDroneState],
        defend_count: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        if self.friendly_targets.is_empty() || defend_count == 0 {
            let all: Vec<usize> = own_drones.iter().map(|d| d.id).collect();
            return (Vec::new(), all);
        }

        let cx = self.friendly_targets.iter().map(|p| p.x()).sum::<f32>()
            / self.friendly_targets.len() as f32;
        let cy = self.friendly_targets.iter().map(|p| p.y()).sum::<f32>()
            / self.friendly_targets.len() as f32;

        // Sort by distance to friendly centroid (closest first → defenders)
        let mut sorted: Vec<(usize, f32)> = own_drones
            .iter()
            .map(|d| {
                let dx = d.pos.x() - cx;
                let dy = d.pos.y() - cy;
                (d.id, dx * dx + dy * dy)
            })
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let defend_ids: Vec<usize> = sorted.iter().take(defend_count).map(|(id, _)| *id).collect();
        let attack_ids: Vec<usize> = sorted.iter().skip(defend_count).map(|(id, _)| *id).collect();

        (defend_ids, attack_ids)
    }

    /// Rebuild child strategies with new drone partitions.
    fn rebuild_strategies(&mut self) {
        // Defense: patrol perimeter around friendly targets
        if !self.defend_ids.is_empty() && !self.patrol_waypoints.is_empty() {
            let mix = self.mode == DoctrineMode::Defensive;
            self.patrol_strategy = Some(
                PatrolPerimeterStrategy::new(
                    self.defend_ids.clone(),
                    self.patrol_waypoints.clone(),
                    0.0, // no loiter — continuous circuit
                    self.group_id,
                    self.detonation_radius,
                )
                .with_mixed_intercept(mix),
            );
        } else {
            self.patrol_strategy = None;
        }

        // Offense: attack enemy targets
        if !self.attack_ids.is_empty() && !self.enemy_targets.is_empty() {
            let mut attack = AttackZoneStrategy::new(
                self.attack_ids.clone(),
                self.enemy_targets.clone(),
                self.detonation_radius,
            );
            if let Some(ws) = self.mode.wave_size(self.enemy_targets.len()) {
                attack = attack.with_wave_size(ws);
            }
            self.attack_strategy = Some(attack);
        } else {
            self.attack_strategy = None;
        }
    }

    /// Check if reallocation is needed and perform it.
    fn maybe_reallocate(&mut self, own_drones: &[StrategyDroneState], all_drones: &[DroneInfo], bounds: &Bounds) -> bool {
        self.ticks_since_realloc += 1;

        let own_count = own_drones.len();
        let enemy_target_count = self.enemy_targets.len();
        let friendly_target_count = self.friendly_targets.len();

        // Check if anything changed that warrants reallocation
        let state_changed = own_count != self.last_own_count
            || enemy_target_count != self.last_enemy_target_count
            || friendly_target_count != self.last_friendly_target_count;

        let should_realloc = self.force_realloc
            || (state_changed && self.ticks_since_realloc >= REALLOC_INTERVAL);

        if !should_realloc {
            return false;
        }

        self.force_realloc = false;
        self.ticks_since_realloc = 0;
        self.last_own_count = own_count;
        self.last_enemy_target_count = enemy_target_count;
        self.last_friendly_target_count = friendly_target_count;

        let nearby_threats = self.count_nearby_threats(all_drones, bounds);
        let (defend_count, _attack_count) =
            Self::compute_allocation(own_count, friendly_target_count, enemy_target_count, nearby_threats, self.mode);

        let (defend_ids, attack_ids) = self.partition_drones(own_drones, defend_count);

        // Only rebuild if the partition actually changed
        let defend_set: HashSet<usize> = defend_ids.iter().copied().collect();
        let old_defend_set: HashSet<usize> = self.defend_ids.iter().copied().collect();
        if defend_set != old_defend_set {
            self.defend_ids = defend_ids;
            self.attack_ids = attack_ids;
            self.rebuild_strategies();
            return true;
        }

        false
    }
}

impl SwarmStrategy for SwarmDoctrine {
    fn tick(
        &mut self,
        own_drones: &[StrategyDroneState],
        all_drones: &[DroneInfo],
        bounds: &Bounds,
        dt: f32,
    ) -> Vec<TaskAssignment> {
        let mut assignments = Vec::new();

        // Re-evaluate force allocation
        let rebuilt = self.maybe_reallocate(own_drones, all_drones, bounds);

        // Build drone lookup for filtering
        let defend_set: HashSet<usize> = self.defend_ids.iter().copied().collect();
        let attack_set: HashSet<usize> = self.attack_ids.iter().copied().collect();

        // Tick patrol strategy with its drones
        if let Some(ref mut patrol) = self.patrol_strategy {
            let patrol_drones: Vec<StrategyDroneState> = own_drones
                .iter()
                .filter(|d| defend_set.contains(&d.id))
                .cloned()
                .collect();

            // If we just rebuilt, mark all patrol drones as available
            // so the strategy emits fresh assignments
            let patrol_drones = if rebuilt {
                patrol_drones
                    .into_iter()
                    .map(|mut d| {
                        d.available = true;
                        d
                    })
                    .collect()
            } else {
                patrol_drones
            };

            assignments.extend(patrol.tick(&patrol_drones, all_drones, bounds, dt));
        }

        // Tick attack strategy with its drones
        if let Some(ref mut attack) = self.attack_strategy {
            let attack_drones: Vec<StrategyDroneState> = own_drones
                .iter()
                .filter(|d| attack_set.contains(&d.id))
                .cloned()
                .collect();

            let attack_drones = if rebuilt {
                attack_drones
                    .into_iter()
                    .map(|mut d| {
                        d.available = true;
                        d
                    })
                    .collect()
            } else {
                attack_drones
            };

            assignments.extend(attack.tick(&attack_drones, all_drones, bounds, dt));
        }

        assignments
    }

    fn drone_ids(&self) -> &[usize] {
        &self.drone_ids
    }

    fn name(&self) -> &str {
        "Doctrine"
    }

    fn is_complete(&self) -> bool {
        false
    }

    fn update_targets(
        &mut self,
        friendly_targets: &[Position],
        enemy_targets: &[Position],
    ) {
        self.friendly_targets = friendly_targets.to_vec();
        self.enemy_targets = enemy_targets.to_vec();
        self.force_realloc = true;
    }

    fn group(&self) -> Option<u32> {
        Some(self.group_id)
    }

    fn set_doctrine_mode(&mut self, mode: DoctrineMode) {
        self.set_mode(mode);
    }

    fn defend_attack_counts(&self) -> Option<(usize, usize)> {
        Some((self.defend_count(), self.attack_count()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Velocity;

    fn make_own_drone(id: usize, x: f32, y: f32) -> StrategyDroneState {
        StrategyDroneState {
            id,
            pos: Position::new(x, y),
            vel: Velocity::zero(),
            available: true,
        }
    }

    fn make_drone_info(uid: usize, x: f32, y: f32, group: u32) -> DroneInfo {
        DroneInfo {
            uid,
            pos: Position::new(x, y),
            hdg: crate::types::Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
            group,
        }
    }

    fn bounds() -> Bounds {
        Bounds::new(5000.0, 5000.0).unwrap()
    }

    #[test]
    fn test_allocation_splits_defend_attack() {
        let (defend, attack) =
            SwarmDoctrine::compute_allocation(10, 6, 6, 0, DoctrineMode::Aggressive);
        // Aggressive: at least 15% defend = 2, attack gets the rest
        assert!(defend >= 2, "Defend should be >= 15%, got {}", defend);
        assert!(attack >= 1, "Attack should be >= 1, got {}", attack);
        assert_eq!(defend + attack, 10);
    }

    #[test]
    fn test_no_enemy_targets_all_defend() {
        let (defend, attack) =
            SwarmDoctrine::compute_allocation(10, 6, 0, 0, DoctrineMode::Aggressive);
        assert_eq!(defend, 10);
        assert_eq!(attack, 0);
    }

    #[test]
    fn test_no_friendly_targets_all_attack() {
        let (defend, attack) =
            SwarmDoctrine::compute_allocation(10, 0, 6, 0, DoctrineMode::Aggressive);
        assert_eq!(defend, 0);
        assert_eq!(attack, 10);
    }

    #[test]
    fn test_threat_increases_defense() {
        let (defend_no_threat, _) =
            SwarmDoctrine::compute_allocation(10, 6, 6, 0, DoctrineMode::Aggressive);
        let (defend_with_threat, _) =
            SwarmDoctrine::compute_allocation(10, 6, 6, 3, DoctrineMode::Aggressive);
        assert!(
            defend_with_threat > defend_no_threat,
            "Threats should increase defense: {} vs {}",
            defend_with_threat,
            defend_no_threat
        );
    }

    #[test]
    fn test_partition_defenders_closest_to_friendly() {
        let friendly_targets = vec![Position::new(100.0, 100.0)];
        let enemy_targets = vec![Position::new(900.0, 900.0)];
        let waypoints = vec![
            Position::new(50.0, 50.0),
            Position::new(150.0, 50.0),
            Position::new(150.0, 150.0),
            Position::new(50.0, 150.0),
        ];

        let obj = SwarmDoctrine::new(
            vec![0, 1, 2, 3],
            0,
            friendly_targets,
            enemy_targets,
            waypoints,
            187.5,
            DoctrineMode::Aggressive,
        );

        let own = vec![
            make_own_drone(0, 80.0, 80.0),   // close to friendly
            make_own_drone(1, 120.0, 120.0),  // close to friendly
            make_own_drone(2, 700.0, 700.0),  // far from friendly
            make_own_drone(3, 800.0, 800.0),  // far from friendly
        ];

        let (defend, attack) = obj.partition_drones(&own, 2);
        assert_eq!(defend.len(), 2);
        assert_eq!(attack.len(), 2);
        // Closest drones (0, 1) should be defenders
        assert!(defend.contains(&0), "Drone 0 should defend");
        assert!(defend.contains(&1), "Drone 1 should defend");
        // Farthest drones (2, 3) should attack
        assert!(attack.contains(&2), "Drone 2 should attack");
        assert!(attack.contains(&3), "Drone 3 should attack");
    }

    #[test]
    fn test_objective_emits_assignments() {
        let friendly = vec![Position::new(100.0, 100.0)];
        let enemy = vec![Position::new(900.0, 900.0)];
        let waypoints = vec![
            Position::new(50.0, 50.0),
            Position::new(150.0, 50.0),
            Position::new(150.0, 150.0),
            Position::new(50.0, 150.0),
        ];

        let mut obj = SwarmDoctrine::new(
            vec![0, 1, 2, 3, 4, 5],
            0,
            friendly,
            enemy,
            waypoints,
            187.5,
            DoctrineMode::Aggressive,
        );

        let own: Vec<StrategyDroneState> = (0..6)
            .map(|i| make_own_drone(i, 100.0 + i as f32 * 150.0, 100.0))
            .collect();

        let all: Vec<DroneInfo> = own
            .iter()
            .map(|d| make_drone_info(d.id, d.pos.x(), d.pos.y(), 0))
            .collect();

        let assignments = obj.tick(&own, &all, &bounds(), 0.016);
        assert!(!assignments.is_empty(), "Should emit assignments on first tick");
    }

    #[test]
    fn test_update_targets_forces_realloc() {
        let friendly = vec![Position::new(100.0, 100.0)];
        let enemy = vec![Position::new(900.0, 900.0)];
        let waypoints = vec![
            Position::new(50.0, 50.0),
            Position::new(150.0, 50.0),
            Position::new(150.0, 150.0),
            Position::new(50.0, 150.0),
        ];

        let mut obj = SwarmDoctrine::new(
            vec![0, 1, 2, 3],
            0,
            friendly.clone(),
            enemy,
            waypoints,
            187.5,
            DoctrineMode::Aggressive,
        );

        let own: Vec<StrategyDroneState> = (0..4)
            .map(|i| make_own_drone(i, 100.0 + i as f32 * 200.0, 100.0))
            .collect();
        let all: Vec<DroneInfo> = own
            .iter()
            .map(|d| make_drone_info(d.id, d.pos.x(), d.pos.y(), 0))
            .collect();

        // First tick
        obj.tick(&own, &all, &bounds(), 0.016);

        // Remove all enemy targets
        obj.update_targets(&friendly, &[]);

        // Make own drones available again
        let own_avail: Vec<StrategyDroneState> = own.iter().map(|d| {
            let mut d2 = d.clone();
            d2.available = true;
            d2
        }).collect();

        let assignments = obj.tick(&own_avail, &all, &bounds(), 0.016);

        // With no enemy targets, all should be defense — patrol formations
        let has_attack = assignments
            .iter()
            .any(|a| matches!(a, TaskAssignment::Attack { .. }));
        assert!(!has_attack, "No attack assignments when enemy targets are gone");
    }
}
