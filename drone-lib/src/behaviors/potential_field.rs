//! Artificial Potential Field (APF) for obstacle avoidance.
//!
//! Implements repulsive potential fields with inverse-square falloff,
//! providing stronger close-range forces compared to linear separation.
//!
//! # Standard APF Formulation
//!
//! For a repulsive potential field around an obstacle:
//! - `U_rep = 0.5 * η * (1/d - 1/d0)²` for `d < d0`
//! - `U_rep = 0` for `d >= d0`
//!
//! The repulsive force is the negative gradient:
//! - `F_rep = η * (1/d - 1/d0) * (1/d²) * direction`
//!
//! Where:
//! - `d` = distance to obstacle
//! - `d0` = influence distance (range of potential field)
//! - `η` = repulsion strength coefficient

use crate::types::{Bounds, DroneInfo, Position, Vec2};

/// Artificial Potential Field default constants.
pub(crate) mod defaults {
    /// Influence range of repulsive field (meters).
    pub const INFLUENCE_DISTANCE: f32 = 5.0;

    /// Hard minimum distance (meters).
    pub const MIN_DISTANCE: f32 = 1.0;

    /// Repulsion strength coefficient.
    pub const REPULSION_STRENGTH: f32 = 50000.0;

    /// Maximum force magnitude.
    pub const MAX_FORCE: f32 = 40.0;
}

/// Configuration for APF repulsive field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct APFConfig {
    /// Influence distance - obstacles beyond this don't affect the drone (meters).
    pub influence_distance: f32,

    /// Repulsion strength coefficient (η).
    /// Higher values = stronger repulsion.
    pub repulsion_strength: f32,

    /// Minimum allowed distance (hard boundary, meters).
    /// Forces become very large as distance approaches this value.
    /// Should be slightly larger than physical collision radius.
    pub min_distance: f32,

    /// Maximum force magnitude (prevents instability from very close encounters).
    pub max_force: f32,

    /// Influence distance for enemy (different-group) drones (meters).
    /// When > 0, enemy drones use this instead of `influence_distance`.
    pub enemy_influence_distance: f32,
}

impl Default for APFConfig {
    fn default() -> Self {
        APFConfig {
            influence_distance: defaults::INFLUENCE_DISTANCE,
            repulsion_strength: defaults::REPULSION_STRENGTH,
            min_distance: defaults::MIN_DISTANCE,
            max_force: defaults::MAX_FORCE,
            enemy_influence_distance: 0.0,
        }
    }
}

impl APFConfig {
    /// Create a new APF configuration.
    pub fn new(
        influence_distance: f32,
        repulsion_strength: f32,
        min_distance: f32,
        max_force: f32,
    ) -> Self {
        APFConfig {
            influence_distance,
            repulsion_strength,
            min_distance,
            max_force,
            enemy_influence_distance: 0.0,
        }
    }
}

/// Represents an obstacle in the potential field.
#[derive(Debug, Clone, Copy)]
pub struct Obstacle {
    /// Center position of the obstacle.
    pub position: Position,
    /// Radius of the obstacle (adds to effective distance calculation).
    pub radius: f32,
}

impl Obstacle {
    /// Create a new point obstacle (radius = 0).
    pub fn point(position: Position) -> Self {
        Obstacle { position, radius: 0.0 }
    }

    /// Create a new circular obstacle.
    pub fn circle(position: Position, radius: f32) -> Self {
        Obstacle { position, radius }
    }
}

/// Calculate repulsive force from a single obstacle using APF.
///
/// Returns force vector pointing away from obstacle, with magnitude
/// following inverse-square falloff.
///
/// # Arguments
/// * `self_pos` - Current drone position
/// * `obstacle` - Obstacle to avoid
/// * `bounds` - World bounds for toroidal distance
/// * `config` - APF configuration
///
/// # Returns
/// Repulsive force vector (zero if outside influence range).
pub fn calculate_apf_repulsion(
    self_pos: Position,
    obstacle: &Obstacle,
    bounds: &Bounds,
    config: &APFConfig,
) -> Vec2 {
    let delta = bounds.delta(self_pos.as_vec2(), obstacle.position.as_vec2());
    let raw_distance = delta.magnitude();

    // Effective distance (accounting for obstacle radius)
    let distance = (raw_distance - obstacle.radius).max(config.min_distance);

    // Outside influence range - no force
    if distance >= config.influence_distance {
        return Vec2::new(0.0, 0.0);
    }

    // Too close to compute (at obstacle center)
    if raw_distance < f32::EPSILON {
        return Vec2::new(0.0, 0.0);
    }

    // Direction away from obstacle (normalized)
    let away = Vec2::new(-delta.x / raw_distance, -delta.y / raw_distance);

    // APF repulsive force: F = η * (1/d - 1/d0) * (1/d²)
    let d = distance;
    let d0 = config.influence_distance;
    let eta = config.repulsion_strength;

    let term = (1.0 / d) - (1.0 / d0);
    let force_magnitude = eta * term / (d * d);

    // Clamp to prevent instability
    let clamped_magnitude = force_magnitude.min(config.max_force);

    Vec2::new(away.x * clamped_magnitude, away.y * clamped_magnitude)
}

/// Calculate total repulsive force from multiple obstacles.
///
/// Sums the repulsive forces from all obstacles within influence range.
///
/// # Arguments
/// * `self_pos` - Current drone position
/// * `obstacles` - List of obstacles to avoid
/// * `bounds` - World bounds for toroidal distance
/// * `config` - APF configuration
///
/// # Returns
/// Combined repulsive force vector.
pub fn calculate_apf_from_obstacles(
    self_pos: Position,
    obstacles: &[Obstacle],
    bounds: &Bounds,
    config: &APFConfig,
) -> Vec2 {
    let mut total_force = Vec2::new(0.0, 0.0);

    for obstacle in obstacles {
        let force = calculate_apf_repulsion(self_pos, obstacle, bounds, config);
        total_force.x += force.x;
        total_force.y += force.y;
    }

    total_force
}

/// Calculate repulsive force from other drones using APF.
///
/// Treats each drone as a point obstacle and sums repulsive forces.
/// Enemy drones (different group) use `enemy_influence_distance` if configured,
/// producing a much larger repulsive field.
///
/// # Arguments
/// * `self_pos` - Current drone position
/// * `self_id` - Current drone's ID (excluded from calculation)
/// * `swarm` - All drones in the swarm
/// * `bounds` - World bounds for toroidal distance
/// * `config` - APF configuration
///
/// # Returns
/// Combined repulsive force vector from all nearby drones.
pub fn calculate_apf_from_swarm(
    self_pos: Position,
    self_id: usize,
    swarm: &[DroneInfo],
    bounds: &Bounds,
    config: &APFConfig,
) -> Vec2 {
    let mut total_force = Vec2::new(0.0, 0.0);

    let self_group = swarm.iter()
        .find(|d| d.uid == self_id)
        .map(|d| d.group)
        .unwrap_or(0);

    for other in crate::types::neighbors_excluding(swarm, self_id) {
        let is_enemy = other.group != self_group;

        let effective_config = if is_enemy && config.enemy_influence_distance > 0.0 {
            APFConfig {
                influence_distance: config.enemy_influence_distance,
                ..*config
            }
        } else {
            *config
        };

        let obstacle = Obstacle::point(other.pos);
        let force = calculate_apf_repulsion(self_pos, &obstacle, bounds, &effective_config);
        total_force.x += force.x;
        total_force.y += force.y;
    }

    total_force
}

/// Combined APF calculation from both obstacles and other drones.
///
/// # Arguments
/// * `self_pos` - Current drone position
/// * `self_id` - Current drone's ID
/// * `swarm` - All drones in the swarm
/// * `obstacles` - Static obstacles to avoid
/// * `bounds` - World bounds for toroidal distance
/// * `config` - APF configuration
///
/// # Returns
/// Combined repulsive force vector from all sources.
pub fn calculate_apf_combined(
    self_pos: Position,
    self_id: usize,
    swarm: &[DroneInfo],
    obstacles: &[Obstacle],
    bounds: &Bounds,
    config: &APFConfig,
) -> Vec2 {
    let swarm_force = calculate_apf_from_swarm(self_pos, self_id, swarm, bounds, config);
    let obstacle_force = calculate_apf_from_obstacles(self_pos, obstacles, bounds, config);

    Vec2::new(
        swarm_force.x + obstacle_force.x,
        swarm_force.y + obstacle_force.y,
    )
}

/// Convert APF force to heading adjustment.
///
/// Given a desired heading and an APF force, returns an adjusted heading
/// that incorporates the repulsive force while maintaining forward progress.
///
/// # Arguments
/// * `current_heading` - Current heading in radians
/// * `desired_heading` - Desired heading (toward goal) in radians
/// * `apf_force` - Repulsive force from APF calculation
/// * `force_weight` - How much to weight APF force vs desired heading (0.0-1.0)
/// * `current_speed` - Current speed (for force normalization)
///
/// # Returns
/// Adjusted heading in radians.
pub fn apply_apf_to_heading(
    _current_heading: f32,
    desired_heading: f32,
    apf_force: Vec2,
    force_weight: f32,
    current_speed: f32,
) -> f32 {
    let force_magnitude = apf_force.magnitude();

    // No force = no adjustment needed
    if force_magnitude < 0.1 {
        return desired_heading;
    }

    // At low/zero speed, use force direction directly with weight
    // This is critical for collision avoidance when drones are stopped
    let effective_speed = current_speed.max(1.0); // Use minimum speed of 1.0 for blending

    // Convert desired heading to velocity vector
    let desired_x = desired_heading.cos() * effective_speed;
    let desired_y = desired_heading.sin() * effective_speed;

    // Normalize APF force to same scale as desired velocity for balanced blending
    let normalized_force = Vec2::new(
        apf_force.x / force_magnitude * effective_speed,
        apf_force.y / force_magnitude * effective_speed,
    );

    // Blend desired velocity with APF force
    let weight = force_weight.clamp(0.0, 1.0);
    let blended_x = desired_x * (1.0 - weight) + normalized_force.x * weight;
    let blended_y = desired_y * (1.0 - weight) + normalized_force.y * weight;

    // Convert back to heading
    blended_y.atan2(blended_x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::fixtures::create_test_bounds;

    #[test]
    fn test_no_force_outside_influence() {
        let bounds = create_test_bounds();
        let config = APFConfig::new(100.0, 10000.0, 10.0, 200.0);

        let self_pos = Position::new(500.0, 500.0);
        let obstacle = Obstacle::point(Position::new(700.0, 500.0)); // 200 units away

        let force = calculate_apf_repulsion(self_pos, &obstacle, &bounds, &config);

        assert_eq!(force.x, 0.0);
        assert_eq!(force.y, 0.0);
    }

    #[test]
    fn test_force_within_influence() {
        let bounds = create_test_bounds();
        let config = APFConfig::new(100.0, 10000.0, 10.0, 200.0);

        let self_pos = Position::new(500.0, 500.0);
        let obstacle = Obstacle::point(Position::new(550.0, 500.0)); // 50 units away (right)

        let force = calculate_apf_repulsion(self_pos, &obstacle, &bounds, &config);

        // Force should point left (away from obstacle)
        assert!(force.x < 0.0, "Force should point left, got {}", force.x);
        assert!(force.y.abs() < 0.001, "Force should have no y component");
    }

    #[test]
    fn test_force_increases_with_proximity() {
        let bounds = create_test_bounds();
        let config = APFConfig::new(100.0, 10000.0, 10.0, 500.0); // Higher max for this test

        let self_pos = Position::new(500.0, 500.0);

        // Close obstacle
        let close_obstacle = Obstacle::point(Position::new(530.0, 500.0)); // 30 units
        let close_force = calculate_apf_repulsion(self_pos, &close_obstacle, &bounds, &config);

        // Far obstacle
        let far_obstacle = Obstacle::point(Position::new(570.0, 500.0)); // 70 units
        let far_force = calculate_apf_repulsion(self_pos, &far_obstacle, &bounds, &config);

        assert!(
            close_force.x.abs() > far_force.x.abs(),
            "Close force {} should be stronger than far force {}",
            close_force.x.abs(),
            far_force.x.abs()
        );
    }

    #[test]
    fn test_inverse_square_falloff() {
        let bounds = create_test_bounds();
        let config = APFConfig::new(200.0, 10000.0, 5.0, 1000.0);

        let self_pos = Position::new(500.0, 500.0);

        // Force at d=20
        let obs_20 = Obstacle::point(Position::new(520.0, 500.0));
        let force_20 = calculate_apf_repulsion(self_pos, &obs_20, &bounds, &config);

        // Force at d=40 (2x distance)
        let obs_40 = Obstacle::point(Position::new(540.0, 500.0));
        let force_40 = calculate_apf_repulsion(self_pos, &obs_40, &bounds, &config);

        // Due to 1/d² term, force ratio should be roughly 4:1 for 2x distance
        // (though the (1/d - 1/d0) term affects this)
        let ratio = force_20.x.abs() / force_40.x.abs();
        assert!(ratio > 2.0, "Force ratio {} should be > 2 for inverse-square", ratio);
    }

    #[test]
    fn test_excludes_self_from_swarm() {
        let bounds = create_test_bounds();
        let config = APFConfig::default();

        let self_pos = Position::new(500.0, 500.0);
        let swarm = vec![
            DroneInfo {
                uid: 0,
                pos: Position::new(500.0, 500.0), // Self
                hdg: crate::types::Heading::new(0.0),
                vel: crate::types::Velocity::zero(),
                is_formation_leader: false,
                group: 0,
            },
        ];

        let force = calculate_apf_from_swarm(self_pos, 0, &swarm, &bounds, &config);

        assert_eq!(force.x, 0.0);
        assert_eq!(force.y, 0.0);
    }

    #[test]
    fn test_multiple_obstacles_sum() {
        let bounds = create_test_bounds();
        let config = APFConfig::new(100.0, 10000.0, 10.0, 200.0);

        let self_pos = Position::new(500.0, 500.0);

        // Obstacles on opposite sides
        let obstacles = vec![
            Obstacle::point(Position::new(550.0, 500.0)), // Right
            Obstacle::point(Position::new(450.0, 500.0)), // Left (same distance)
        ];

        let force = calculate_apf_from_obstacles(self_pos, &obstacles, &bounds, &config);

        // Equal and opposite forces should cancel
        assert!(force.x.abs() < 0.001, "X forces should cancel, got {}", force.x);
    }

    #[test]
    fn test_obstacle_radius_affects_force() {
        let bounds = create_test_bounds();
        let config = APFConfig::new(100.0, 10000.0, 10.0, 200.0);

        let self_pos = Position::new(500.0, 500.0);

        // Point obstacle at 50 units
        let point = Obstacle::point(Position::new(550.0, 500.0));
        let point_force = calculate_apf_repulsion(self_pos, &point, &bounds, &config);

        // Circular obstacle at 50 units with 20 radius (effective distance = 30)
        let circle = Obstacle::circle(Position::new(550.0, 500.0), 20.0);
        let circle_force = calculate_apf_repulsion(self_pos, &circle, &bounds, &config);

        // Circle should produce stronger force (closer effective distance)
        assert!(
            circle_force.x.abs() > point_force.x.abs(),
            "Circle force {} should be stronger than point force {}",
            circle_force.x.abs(),
            point_force.x.abs()
        );
    }

    #[test]
    fn test_apply_apf_to_heading_no_force() {
        let desired = 0.5;
        let force = Vec2::new(0.0, 0.0);

        let result = apply_apf_to_heading(0.0, desired, force, 0.5, 50.0);

        assert!((result - desired).abs() < 0.001);
    }

    #[test]
    fn test_apply_apf_to_heading_with_force() {
        let desired = 0.0; // Heading right
        let force = Vec2::new(0.0, 50.0); // Force pushing up

        let result = apply_apf_to_heading(0.0, desired, force, 0.5, 50.0);

        // Result should be between 0 and π/2 (biased upward)
        assert!(result > 0.0, "Heading should turn up, got {}", result);
        assert!(result < std::f32::consts::FRAC_PI_2, "Heading shouldn't exceed π/2");
    }

    #[test]
    fn test_force_clamped_to_max() {
        let bounds = create_test_bounds();
        let config = APFConfig::new(100.0, 10000.0, 10.0, 50.0); // Low max

        let self_pos = Position::new(500.0, 500.0);
        let obstacle = Obstacle::point(Position::new(515.0, 500.0)); // Very close

        let force = calculate_apf_repulsion(self_pos, &obstacle, &bounds, &config);

        assert!(
            force.magnitude() <= config.max_force + 0.01,
            "Force {} should be clamped to {}",
            force.magnitude(),
            config.max_force
        );
    }
}
