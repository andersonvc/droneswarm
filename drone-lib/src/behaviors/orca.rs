//! Optimal Reciprocal Collision Avoidance (ORCA).
//!
//! ORCA provides guaranteed collision-free navigation by computing half-planes
//! of safe velocities for each neighbor pair, then finding the optimal velocity
//! within the intersection of all half-planes.
//!
//! # Algorithm Overview
//!
//! For each neighbor:
//! 1. Compute the velocity obstacle (VO) - set of velocities leading to collision
//! 2. Take responsibility for half the avoidance (ORCA half-plane)
//! 3. The half-plane boundary passes through `v_opt + 0.5 * u` with normal `n`
//!
//! Where:
//! - `u` = smallest change to current velocity to escape the VO
//! - `n` = outward normal of the VO at the escape point
//!
//! The optimal velocity is found by projecting the preferred velocity onto
//! the intersection of all ORCA half-planes.

use crate::types::{units, Bounds, DroneInfo, Position, Vec2, Velocity};

/// ORCA algorithm default constants.
pub(crate) mod defaults {
    /// Time horizon for collision prediction (seconds).
    pub const TIME_HORIZON: f32 = 2.0;

    /// Agent radius for ORCA (meters) - slightly larger than physical for safety margin.
    pub const AGENT_RADIUS: f32 = 0.5;

    /// Neighbor detection distance (meters).
    pub const NEIGHBOR_DIST: f32 = 5.0;

    /// Maximum number of neighbors to consider.
    pub const MAX_NEIGHBORS: usize = 10;
}

/// Configuration for ORCA collision avoidance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ORCAConfig {
    /// Time horizon for collision prediction (seconds).
    /// Longer = more conservative avoidance, starts earlier.
    pub time_horizon: f32,

    /// Combined radius for collision detection (meters).
    /// Should be slightly larger than physical drone radius.
    pub agent_radius: f32,

    /// Maximum speed the agent can achieve (m/s).
    pub max_speed: f32,

    /// Neighbor distance - only consider agents within this range (meters).
    pub neighbor_dist: f32,

    /// Maximum number of neighbors to consider.
    pub max_neighbors: usize,

    /// Avoidance radius for enemy (different-group) drones (meters).
    /// This is the total separation distance maintained from enemies.
    /// When 0.0, falls back to `2 * agent_radius`.
    pub enemy_avoidance_radius: f32,
}

impl Default for ORCAConfig {
    fn default() -> Self {
        ORCAConfig {
            time_horizon: defaults::TIME_HORIZON,
            agent_radius: defaults::AGENT_RADIUS,
            max_speed: units::DEFAULT_MAX_VELOCITY,
            neighbor_dist: defaults::NEIGHBOR_DIST,
            max_neighbors: defaults::MAX_NEIGHBORS,
            enemy_avoidance_radius: 0.0,
        }
    }
}

impl ORCAConfig {
    /// Create a new ORCA configuration.
    pub fn new(
        time_horizon: f32,
        agent_radius: f32,
        max_speed: f32,
        neighbor_dist: f32,
        max_neighbors: usize,
    ) -> Self {
        ORCAConfig {
            time_horizon,
            agent_radius,
            max_speed,
            neighbor_dist,
            max_neighbors,
            enemy_avoidance_radius: 0.0,
        }
    }
}

/// Represents an ORCA half-plane constraint.
/// All velocities v satisfying (v - point) · normal >= 0 are valid.
#[derive(Debug, Clone, Copy)]
pub struct HalfPlane {
    /// A point on the half-plane boundary.
    pub point: Vec2,
    /// Outward normal (points toward valid velocities).
    pub normal: Vec2,
}

impl HalfPlane {
    /// Create a new half-plane.
    pub fn new(point: Vec2, normal: Vec2) -> Self {
        HalfPlane { point, normal }
    }

    /// Check if a velocity satisfies this constraint.
    pub fn contains(&self, velocity: Vec2) -> bool {
        let rel = Vec2::new(velocity.x - self.point.x, velocity.y - self.point.y);
        rel.x * self.normal.x + rel.y * self.normal.y >= -1e-5
    }

    /// Project a velocity onto the half-plane boundary if it violates the constraint.
    /// Returns the closest point on the boundary.
    pub fn project(&self, velocity: Vec2) -> Vec2 {
        let rel = Vec2::new(velocity.x - self.point.x, velocity.y - self.point.y);
        let dot = rel.x * self.normal.x + rel.y * self.normal.y;

        if dot >= 0.0 {
            // Already satisfies constraint
            velocity
        } else {
            // Project onto boundary: v - (v·n - d)n where d = point·n
            Vec2::new(
                velocity.x - dot * self.normal.x,
                velocity.y - dot * self.normal.y,
            )
        }
    }
}

/// Compute the ORCA half-plane for a single neighbor.
///
/// # Arguments
/// * `self_pos` - Current agent position
/// * `self_vel` - Current agent velocity
/// * `other_pos` - Neighbor position
/// * `other_vel` - Neighbor velocity
/// * `responsibility` - Fraction of avoidance this agent takes (0.5 = standard ORCA,
///   1.0 = full responsibility for non-cooperative agents, >1.0 = aggressive evasion)
/// * `bounds` - World bounds for toroidal distance
/// * `config` - ORCA configuration
///
/// # Returns
/// Half-plane constraint, or None if neighbor is too far.
pub fn compute_orca_half_plane(
    self_pos: Position,
    self_vel: Velocity,
    other_pos: Position,
    other_vel: Velocity,
    responsibility: f32,
    bounds: &Bounds,
    config: &ORCAConfig,
) -> Option<HalfPlane> {
    // Relative position and velocity
    let rel_pos = bounds.delta(other_pos.as_vec2(), self_pos.as_vec2());
    let dist_sq = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y;
    let dist = dist_sq.sqrt();

    // Skip if too far
    if dist > config.neighbor_dist {
        return None;
    }

    let self_vel_vec = self_vel.as_vec2();
    let other_vel_vec = other_vel.as_vec2();
    let rel_vel = Vec2::new(
        self_vel_vec.x - other_vel_vec.x,
        self_vel_vec.y - other_vel_vec.y,
    );

    // Combined radius of both agents
    let combined_radius = 2.0 * config.agent_radius;

    // Time to collision if heading straight at each other
    let tau = config.time_horizon;

    if dist < combined_radius {
        // Already colliding - create constraint to move apart
        let normal = if dist > 1e-5 {
            Vec2::new(-rel_pos.x / dist, -rel_pos.y / dist)
        } else {
            // Arbitrary direction if exactly overlapping
            Vec2::new(1.0, 0.0)
        };

        // Move apart with velocity proportional to overlap
        let overlap = combined_radius - dist;
        let escape_speed = overlap / tau + 0.1;

        return Some(HalfPlane::new(
            Vec2::new(normal.x * escape_speed, normal.y * escape_speed),
            normal,
        ));
    }

    // Velocity obstacle computation
    // The VO is a truncated cone with apex at relative velocity origin

    // Leg directions of the velocity obstacle cone
    let leg_dist = (dist_sq - combined_radius * combined_radius).sqrt();

    // Unit vector from self to other
    let rel_pos_norm = Vec2::new(rel_pos.x / dist, rel_pos.y / dist);

    // Perpendicular vectors for cone legs
    let perp = Vec2::new(-rel_pos_norm.y, rel_pos_norm.x);

    // Leg vectors (left and right boundaries of the VO cone)
    let sin_angle = combined_radius / dist;
    let cos_angle = leg_dist / dist;

    let left_leg = Vec2::new(
        rel_pos_norm.x * cos_angle - perp.x * sin_angle,
        rel_pos_norm.y * cos_angle - perp.y * sin_angle,
    );

    let right_leg = Vec2::new(
        rel_pos_norm.x * cos_angle + perp.x * sin_angle,
        rel_pos_norm.y * cos_angle + perp.y * sin_angle,
    );

    // Determine which part of the VO boundary the relative velocity projects to
    // and compute the smallest change `u` to escape the VO

    // Cutoff circle center (the truncation of the VO at time horizon)
    let cutoff_center = Vec2::new(rel_pos.x / tau, rel_pos.y / tau);
    let cutoff_radius = combined_radius / tau;

    // Check if relative velocity is in the cutoff region or leg region
    let rel_to_cutoff = Vec2::new(
        rel_vel.x - cutoff_center.x,
        rel_vel.y - cutoff_center.y,
    );
    let cutoff_dist_sq = rel_to_cutoff.x * rel_to_cutoff.x + rel_to_cutoff.y * rel_to_cutoff.y;

    // Determine the escape vector `u`
    let (u, normal) = if cutoff_dist_sq <= cutoff_radius * cutoff_radius {
        // Inside cutoff circle - project radially outward
        escape_from_cutoff(rel_to_cutoff, cutoff_dist_sq, cutoff_radius, rel_pos_norm)
    } else {
        // Check which leg the velocity is closest to
        let left_dot = rel_vel.x * left_leg.x + rel_vel.y * left_leg.y;
        let right_dot = rel_vel.x * right_leg.x + rel_vel.y * right_leg.y;

        // Cross products to determine which side of each leg
        let left_cross = rel_vel.x * left_leg.y - rel_vel.y * left_leg.x;
        let right_cross = rel_vel.x * right_leg.y - rel_vel.y * right_leg.x;

        if left_cross > 0.0 && right_cross < 0.0 {
            // Inside the cone - project to nearest leg
            if left_cross < -right_cross {
                escape_from_leg(rel_vel, left_leg, true)
            } else {
                escape_from_leg(rel_vel, right_leg, false)
            }
        } else if left_cross > 0.0 {
            if left_dot > 0.0 {
                escape_from_leg(rel_vel, left_leg, true)
            } else {
                escape_from_cutoff(rel_to_cutoff, cutoff_dist_sq, cutoff_radius, rel_pos_norm)
            }
        } else if right_cross < 0.0 {
            if right_dot > 0.0 {
                escape_from_leg(rel_vel, right_leg, false)
            } else {
                escape_from_cutoff(rel_to_cutoff, cutoff_dist_sq, cutoff_radius, rel_pos_norm)
            }
        } else {
            // Outside the VO entirely - no constraint needed
            return None;
        }
    };

    // ORCA constraint: scale the escape vector by the responsibility factor.
    // 0.5 = standard reciprocity, 1.0 = full responsibility, >1.0 = aggressive evasion.
    let scaled_u = Vec2::new(u.x * responsibility, u.y * responsibility);
    let constraint_point = Vec2::new(
        self_vel_vec.x + scaled_u.x,
        self_vel_vec.y + scaled_u.y,
    );

    Some(HalfPlane::new(constraint_point, normal))
}

/// Compute escape vector when relative velocity is inside the cutoff circle.
///
/// Returns `(escape_u, normal)` - the smallest change to escape the VO
/// and the outward normal at the escape point.
fn escape_from_cutoff(
    rel_to_cutoff: Vec2,
    cutoff_dist_sq: f32,
    cutoff_radius: f32,
    rel_pos_norm: Vec2,
) -> (Vec2, Vec2) {
    let cutoff_dist = cutoff_dist_sq.sqrt();
    if cutoff_dist > 1e-5 {
        let n = Vec2::new(
            rel_to_cutoff.x / cutoff_dist,
            rel_to_cutoff.y / cutoff_dist,
        );
        let penetration = cutoff_radius - cutoff_dist;
        (Vec2::new(n.x * penetration, n.y * penetration), n)
    } else {
        // At center of cutoff - push in direction away from neighbor
        let n = Vec2::new(-rel_pos_norm.x, -rel_pos_norm.y);
        (Vec2::new(n.x * cutoff_radius, n.y * cutoff_radius), n)
    }
}

/// Compute escape vector by projecting relative velocity onto a cone leg.
///
/// Returns `(escape_u, normal)` for the given leg direction.
fn escape_from_leg(rel_vel: Vec2, leg: Vec2, is_left: bool) -> (Vec2, Vec2) {
    let cross = rel_vel.x * leg.y - rel_vel.y * leg.x;
    let penetration = if is_left { cross } else { -cross };

    let n = if is_left {
        Vec2::new(leg.y, -leg.x)  // Perpendicular pointing right
    } else {
        Vec2::new(-leg.y, leg.x)  // Perpendicular pointing left
    };

    (
        Vec2::new(-n.x * penetration, -n.y * penetration),
        Vec2::new(-n.x, -n.y),
    )
}

/// Compute all ORCA half-planes for a drone in the swarm.
///
/// # Arguments
/// * `self_pos` - Current drone position
/// * `self_vel` - Current drone velocity
/// * `self_id` - Current drone's ID (excluded from calculation)
/// * `swarm` - All drones in the swarm
/// * `bounds` - World bounds for toroidal distance
/// * `config` - ORCA configuration
///
/// # Returns
/// Vector of half-plane constraints.
pub fn compute_orca_constraints(
    self_pos: Position,
    self_vel: Velocity,
    self_id: usize,
    swarm: &[DroneInfo],
    bounds: &Bounds,
    config: &ORCAConfig,
) -> Vec<HalfPlane> {
    let mut constraints = Vec::with_capacity(config.max_neighbors);

    // Look up self group for friend/foe distinction
    let self_group = swarm.iter()
        .find(|d| d.uid == self_id)
        .map(|d| d.group)
        .unwrap_or(0);

    // Detection range must cover the forecast distance for enemy collision courses.
    // Two drones at max_speed close at 2*max_speed; with 5x time_horizon forecast,
    // threats can be detected at 2 * max_speed * 5 * time_horizon meters out.
    let enemy_forecast_dist = if config.enemy_avoidance_radius > 0.0 {
        2.0 * config.max_speed * config.time_horizon * 5.0
    } else {
        0.0
    };
    let effective_neighbor_dist = config.neighbor_dist
        .max(enemy_forecast_dist);

    // Collect neighbors sorted by distance
    let mut neighbors: Vec<(usize, f32)> = crate::types::neighbors_excluding(swarm, self_id)
        .map(|d| {
            let delta = bounds.delta(d.pos.as_vec2(), self_pos.as_vec2());
            let dist_sq = delta.x * delta.x + delta.y * delta.y;
            (d.uid, dist_sq)
        })
        .filter(|(_, dist_sq)| *dist_sq < effective_neighbor_dist * effective_neighbor_dist)
        .collect();

    neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    neighbors.truncate(config.max_neighbors);

    for (neighbor_id, _) in neighbors {
        if let Some(neighbor) = swarm.iter().find(|d| d.uid == neighbor_id) {
            let is_enemy = neighbor.group != self_group;

            // Responsibility factor:
            //   0.5 = standard ORCA reciprocity (both sides cooperate)
            //   1.0 = full responsibility (formation leaders)
            //   1.5 = aggressive evasion (enemy drones — overcompensate to ensure clearance)
            let responsibility = if is_enemy {
                1.5
            } else if neighbor.is_formation_leader {
                1.0
            } else {
                0.5
            };

            // For enemy drones: use the blast radius as the collision zone and
            // forecast much further ahead to detect collision courses early.
            // The VO cone only triggers when the enemy's trajectory actually leads
            // into the blast radius within the time horizon — not a static bubble.
            let effective_config = if is_enemy && config.enemy_avoidance_radius > 0.0 {
                let mut c = *config;
                c.agent_radius = config.enemy_avoidance_radius / 2.0;
                // Long time horizon to forecast collision courses well in advance.
                // At 240 m/s closing speed, 10s horizon detects threats ~2400m out.
                c.time_horizon = config.time_horizon * 5.0;
                c
            } else {
                *config
            };

            if let Some(half_plane) = compute_orca_half_plane(
                self_pos,
                self_vel,
                neighbor.pos,
                neighbor.vel,
                responsibility,
                bounds,
                &effective_config,
            ) {
                constraints.push(half_plane);
            }
        }
    }

    constraints
}

/// Find the optimal velocity satisfying all ORCA constraints.
///
/// Uses iterative projection: start with preferred velocity, then
/// project onto each violated constraint iteratively.
///
/// # Arguments
/// * `preferred_vel` - Desired velocity (toward goal)
/// * `constraints` - ORCA half-plane constraints
/// * `max_speed` - Maximum allowed speed
///
/// # Returns
/// Optimal velocity satisfying all constraints (or closest approximation).
pub fn find_optimal_velocity(
    preferred_vel: Vec2,
    constraints: &[HalfPlane],
    max_speed: f32,
) -> Vec2 {
    if constraints.is_empty() {
        // No constraints - just clamp to max speed
        return clamp_velocity(preferred_vel, max_speed);
    }

    let mut velocity = preferred_vel;

    // Iterative projection - multiple passes for convergence
    for _ in 0..10 {
        let mut changed = false;

        for constraint in constraints {
            if !constraint.contains(velocity) {
                velocity = constraint.project(velocity);
                changed = true;
            }
        }

        // Speed limit constraint
        velocity = clamp_velocity(velocity, max_speed);

        if !changed {
            break;
        }
    }

    // Final check - if still violating, try projecting from origin
    for constraint in constraints {
        if !constraint.contains(velocity) {
            // Fall back to a safe but slow velocity
            velocity = constraint.project(Vec2::new(0.0, 0.0));
            velocity = clamp_velocity(velocity, max_speed);
        }
    }

    velocity
}

/// Clamp velocity to maximum speed while preserving direction.
fn clamp_velocity(vel: Vec2, max_speed: f32) -> Vec2 {
    let speed_sq = vel.x * vel.x + vel.y * vel.y;
    if speed_sq > max_speed * max_speed {
        let speed = speed_sq.sqrt();
        Vec2::new(vel.x * max_speed / speed, vel.y * max_speed / speed)
    } else {
        vel
    }
}

/// Compute ORCA-based velocity for a drone.
///
/// This is the main entry point for ORCA collision avoidance.
///
/// # Arguments
/// * `self_pos` - Current drone position
/// * `self_vel` - Current drone velocity
/// * `self_id` - Current drone's ID
/// * `preferred_vel` - Desired velocity (toward goal)
/// * `swarm` - All drones in the swarm
/// * `bounds` - World bounds
/// * `config` - ORCA configuration
///
/// # Returns
/// Collision-free velocity.
pub fn compute_orca_velocity(
    self_pos: Position,
    self_vel: Velocity,
    self_id: usize,
    preferred_vel: Vec2,
    swarm: &[DroneInfo],
    bounds: &Bounds,
    config: &ORCAConfig,
) -> Vec2 {
    let constraints = compute_orca_constraints(
        self_pos, self_vel, self_id, swarm, bounds, config,
    );

    find_optimal_velocity(preferred_vel, &constraints, config.max_speed)
}

/// Convert ORCA velocity to heading and speed.
///
/// # Arguments
/// * `orca_velocity` - Velocity computed by ORCA
///
/// # Returns
/// (heading_radians, speed)
pub fn velocity_to_heading_speed(orca_velocity: Vec2) -> (f32, f32) {
    let speed = (orca_velocity.x * orca_velocity.x + orca_velocity.y * orca_velocity.y).sqrt();
    let heading = orca_velocity.y.atan2(orca_velocity.x);
    (heading, speed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::fixtures::create_test_bounds;
    use crate::types::Heading;

    #[test]
    fn test_half_plane_contains() {
        let hp = HalfPlane::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));

        // Points with positive x should be valid
        assert!(hp.contains(Vec2::new(1.0, 0.0)));
        assert!(hp.contains(Vec2::new(0.5, 1.0)));
        assert!(hp.contains(Vec2::new(0.0, 0.0))); // On boundary

        // Points with negative x should be invalid
        assert!(!hp.contains(Vec2::new(-1.0, 0.0)));
        assert!(!hp.contains(Vec2::new(-0.1, 0.0)));
    }

    #[test]
    fn test_half_plane_project() {
        let hp = HalfPlane::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));

        // Valid point - no change
        let valid = Vec2::new(1.0, 2.0);
        let projected = hp.project(valid);
        assert!((projected.x - 1.0).abs() < 0.001);
        assert!((projected.y - 2.0).abs() < 0.001);

        // Invalid point - project to boundary
        let invalid = Vec2::new(-1.0, 2.0);
        let projected = hp.project(invalid);
        assert!(projected.x.abs() < 0.001); // Should be on boundary (x=0)
        assert!((projected.y - 2.0).abs() < 0.001); // y unchanged
    }

    #[test]
    fn test_no_constraint_when_far() {
        let bounds = create_test_bounds();
        let config = ORCAConfig::default();

        let self_pos = Position::new(100.0, 100.0);
        let self_vel = Velocity::new(10.0, 0.0);
        let other_pos = Position::new(500.0, 500.0); // Far away
        let other_vel = Velocity::new(0.0, 0.0);

        let result = compute_orca_half_plane(
            self_pos, self_vel, other_pos, other_vel, 0.5, &bounds, &config,
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_constraint_when_close() {
        let bounds = create_test_bounds();
        let config = ORCAConfig::default();

        let self_pos = Position::new(100.0, 100.0);
        let self_vel = Velocity::new(10.0, 0.0);
        let other_pos = Position::new(103.0, 100.0); // 3 units away, within NEIGHBOR_DIST (5m)
        let other_vel = Velocity::new(-10.0, 0.0); // Coming toward us

        let result = compute_orca_half_plane(
            self_pos, self_vel, other_pos, other_vel, 0.5, &bounds, &config,
        );

        assert!(result.is_some());
    }

    #[test]
    fn test_optimal_velocity_no_constraints() {
        let preferred = Vec2::new(50.0, 0.0);
        let constraints: Vec<HalfPlane> = vec![];

        let result = find_optimal_velocity(preferred, &constraints, 100.0);

        assert!((result.x - 50.0).abs() < 0.001);
        assert!(result.y.abs() < 0.001);
    }

    #[test]
    fn test_optimal_velocity_speed_clamp() {
        let preferred = Vec2::new(200.0, 0.0); // Over max speed
        let constraints: Vec<HalfPlane> = vec![];

        let result = find_optimal_velocity(preferred, &constraints, 100.0);

        let speed = (result.x * result.x + result.y * result.y).sqrt();
        assert!((speed - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_optimal_velocity_with_constraint() {
        // Constraint that forbids rightward movement
        let hp = HalfPlane::new(Vec2::new(0.0, 0.0), Vec2::new(-1.0, 0.0));
        let constraints = vec![hp];

        let preferred = Vec2::new(50.0, 0.0); // Wants to go right

        let result = find_optimal_velocity(preferred, &constraints, 100.0);

        // Should be projected to x <= 0
        assert!(result.x <= 0.001);
    }

    #[test]
    fn test_excludes_self_from_constraints() {
        let bounds = create_test_bounds();
        let config = ORCAConfig::default();

        let self_pos = Position::new(100.0, 100.0);
        let self_vel = Velocity::new(10.0, 0.0);

        // Swarm with only self
        let swarm = vec![DroneInfo {
            uid: 0,
            pos: Position::new(100.0, 100.0),
            hdg: Heading::new(0.0),
            vel: Velocity::new(10.0, 0.0),
            is_formation_leader: false,
            group: 0,
        }];

        let constraints = compute_orca_constraints(
            self_pos, self_vel, 0, &swarm, &bounds, &config,
        );

        assert!(constraints.is_empty());
    }

    #[test]
    fn test_velocity_to_heading_speed() {
        let vel = Vec2::new(30.0, 40.0); // 3-4-5 triangle
        let (heading, speed) = velocity_to_heading_speed(vel);

        assert!((speed - 50.0).abs() < 0.001);
        assert!((heading - (40.0_f32).atan2(30.0)).abs() < 0.001);
    }

    #[test]
    fn test_head_on_collision_avoidance() {
        let bounds = create_test_bounds();
        let config = ORCAConfig {
            time_horizon: 5.0,
            agent_radius: 15.0,
            max_speed: 100.0,
            neighbor_dist: 200.0,
            max_neighbors: 10,
            enemy_avoidance_radius: 0.0,
        };

        // Two drones heading straight at each other
        let self_pos = Position::new(100.0, 100.0);
        let self_vel = Velocity::new(30.0, 0.0); // Going right
        let other_pos = Position::new(200.0, 100.0);
        let other_vel = Velocity::new(-30.0, 0.0); // Coming left

        let swarm = vec![
            DroneInfo {
                uid: 0,
                pos: self_pos,
                hdg: Heading::new(0.0),
                vel: self_vel,
                is_formation_leader: false,
                group: 0,
            },
            DroneInfo {
                uid: 1,
                pos: other_pos,
                hdg: Heading::new(std::f32::consts::PI),
                vel: other_vel,
                is_formation_leader: false,
                group: 0,
            },
        ];

        let preferred = Vec2::new(30.0, 0.0); // Still wants to go right
        let orca_vel = compute_orca_velocity(
            self_pos, self_vel, 0, preferred, &swarm, &bounds, &config,
        );

        // ORCA should deflect the velocity to avoid collision
        // The y component should be non-zero (avoiding by going up or down)
        // or the x component should be reduced
        let deflection = orca_vel.y.abs() > 0.1 || orca_vel.x < preferred.x - 0.1;
        assert!(
            deflection,
            "Expected deflection, got vel: ({}, {})",
            orca_vel.x, orca_vel.y
        );
    }

    #[test]
    fn test_enemy_group_takes_full_avoidance_responsibility() {
        let bounds = create_test_bounds();
        let config = ORCAConfig {
            time_horizon: 5.0,
            agent_radius: 15.0,
            max_speed: 100.0,
            neighbor_dist: 200.0,
            max_neighbors: 10,
            enemy_avoidance_radius: 0.0,
        };

        let self_pos = Position::new(100.0, 100.0);
        let self_vel = Velocity::new(30.0, 0.0);
        let other_pos = Position::new(200.0, 100.0);
        let other_vel = Velocity::new(-30.0, 0.0);

        // Same group: standard reciprocity (0.5 responsibility each)
        let swarm_same_group = vec![
            DroneInfo {
                uid: 0, pos: self_pos, hdg: Heading::new(0.0),
                vel: self_vel, is_formation_leader: false, group: 0,
            },
            DroneInfo {
                uid: 1, pos: other_pos, hdg: Heading::new(std::f32::consts::PI),
                vel: other_vel, is_formation_leader: false, group: 0,
            },
        ];

        // Different group: non-cooperative (full responsibility)
        let swarm_diff_group = vec![
            DroneInfo {
                uid: 0, pos: self_pos, hdg: Heading::new(0.0),
                vel: self_vel, is_formation_leader: false, group: 0,
            },
            DroneInfo {
                uid: 1, pos: other_pos, hdg: Heading::new(std::f32::consts::PI),
                vel: other_vel, is_formation_leader: false, group: 1,
            },
        ];

        let preferred = Vec2::new(30.0, 0.0);

        let vel_same = compute_orca_velocity(
            self_pos, self_vel, 0, preferred, &swarm_same_group, &bounds, &config,
        );
        let vel_diff = compute_orca_velocity(
            self_pos, self_vel, 0, preferred, &swarm_diff_group, &bounds, &config,
        );

        // With enemy drone, avoidance should be stronger (larger deflection)
        let deflection_same = (vel_same.y.abs()).max((preferred.x - vel_same.x).abs());
        let deflection_diff = (vel_diff.y.abs()).max((preferred.x - vel_diff.x).abs());

        assert!(
            deflection_diff > deflection_same,
            "Enemy group should cause stronger avoidance. Same group deflection: {}, diff group: {}",
            deflection_same, deflection_diff
        );
    }

    #[test]
    fn test_enemy_avoidance_radius_increases_deflection() {
        let bounds = create_test_bounds();

        // Config WITHOUT enemy avoidance radius
        let config_no_radius = ORCAConfig {
            time_horizon: 5.0,
            agent_radius: 15.0,
            max_speed: 100.0,
            neighbor_dist: 300.0,
            max_neighbors: 10,
            enemy_avoidance_radius: 0.0,
        };

        // Config WITH enemy avoidance radius (simulating 1.5x blast radius)
        let config_with_radius = ORCAConfig {
            enemy_avoidance_radius: 100.0,
            ..config_no_radius
        };

        let self_pos = Position::new(100.0, 100.0);
        let self_vel = Velocity::new(30.0, 0.0);
        // Enemy at 150m — beyond enemy_avoidance_radius (100m) so VO cone logic applies,
        // but close enough that the larger radius creates a wider cone.
        let other_pos = Position::new(250.0, 110.0);
        let other_vel = Velocity::new(-30.0, 0.0);

        let swarm = vec![
            DroneInfo {
                uid: 0, pos: self_pos, hdg: Heading::new(0.0),
                vel: self_vel, is_formation_leader: false, group: 0,
            },
            DroneInfo {
                uid: 1, pos: other_pos, hdg: Heading::new(std::f32::consts::PI),
                vel: other_vel, is_formation_leader: false, group: 1,
            },
        ];

        let preferred = Vec2::new(30.0, 0.0);

        let vel_no_radius = compute_orca_velocity(
            self_pos, self_vel, 0, preferred, &swarm, &bounds, &config_no_radius,
        );
        let vel_with_radius = compute_orca_velocity(
            self_pos, self_vel, 0, preferred, &swarm, &bounds, &config_with_radius,
        );

        // With the larger enemy avoidance radius, the drone should deflect more
        let deflection_no = vel_no_radius.y.abs();
        let deflection_with = vel_with_radius.y.abs();

        assert!(
            deflection_with > deflection_no,
            "Enemy avoidance radius should increase deflection. Without: {}, with: {}",
            deflection_no, deflection_with
        );
    }
}
