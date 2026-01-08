//! Separation behavior for collision avoidance.
//!
//! Implements a repulsive force that steers drones away from nearby neighbors,
//! preventing collisions while maintaining smooth flight paths.

use crate::types::{Bounds, DroneInfo, Position, Vec2};

/// Physical collision radius of a drone (world units).
/// Two drones are considered colliding if their centers are within 2x this distance.
pub const COLLISION_RADIUS: f32 = 10.0;

/// Default multiplier for minimum separation distance.
/// `min_distance = COLLISION_RADIUS * MIN_DISTANCE_MULTIPLIER`
const DEFAULT_MIN_DISTANCE_MULTIPLIER: f32 = 2.0;

/// Configuration for separation behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeparationConfig {
    /// Detection radius - drones within this distance trigger avoidance.
    /// Units: world units (same as position).
    pub radius: f32,

    /// Force multiplier - higher values create stronger avoidance.
    /// Typical range: 1.0 - 100.0
    pub strength: f32,

    /// Minimum distance to maintain from other drones.
    /// Should be a multiple of [`COLLISION_RADIUS`] to ensure safe separation.
    /// Below this distance, avoidance force increases dramatically.
    pub min_distance: f32,
}

impl Default for SeparationConfig {
    fn default() -> Self {
        SeparationConfig {
            radius: 100.0,
            strength: 150.0,
            min_distance: COLLISION_RADIUS * DEFAULT_MIN_DISTANCE_MULTIPLIER,
        }
    }
}

impl SeparationConfig {
    /// Create a new separation configuration.
    ///
    /// # Arguments
    /// * `radius` - Detection radius for nearby drones
    /// * `strength` - Force multiplier for avoidance steering
    /// * `min_distance` - Minimum safe distance between drones
    pub fn new(radius: f32, strength: f32, min_distance: f32) -> Self {
        SeparationConfig {
            radius,
            strength,
            min_distance,
        }
    }

    /// Create a configuration with min_distance as a multiple of [`COLLISION_RADIUS`].
    ///
    /// # Arguments
    /// * `radius` - Detection radius for nearby drones
    /// * `strength` - Force multiplier for avoidance steering
    /// * `min_distance_multiplier` - Multiplier for `COLLISION_RADIUS` (e.g., 2.0 = 2x collision radius)
    pub fn with_min_distance_multiplier(radius: f32, strength: f32, min_distance_multiplier: f32) -> Self {
        SeparationConfig {
            radius,
            strength,
            min_distance: COLLISION_RADIUS * min_distance_multiplier,
        }
    }
}

/// Calculate separation steering vector to avoid nearby drones.
///
/// Returns a steering vector pointing away from nearby drones, with magnitude
/// proportional to proximity. Closer drones produce stronger avoidance forces.
///
/// Uses toroidal (wraparound) distance calculations to work correctly at
/// world boundaries.
///
/// # Arguments
/// * `self_pos` - Current drone's position
/// * `self_id` - Current drone's ID (to exclude self from calculations)
/// * `swarm` - Slice of all drone info in the swarm
/// * `bounds` - World bounds for toroidal distance calculations
/// * `config` - Separation behavior configuration
///
/// # Returns
/// A `Vec2` steering vector. Zero vector if no drones are within range.
///
/// # Example
/// ```
/// use drone_lib::behaviors::{calculate_separation, SeparationConfig};
/// use drone_lib::{Bounds, DroneInfo, Position};
///
/// let bounds = Bounds::new(1000.0, 1000.0).unwrap();
/// let config = SeparationConfig::default();
/// let self_pos = Position::new(100.0, 100.0);
///
/// // Empty swarm returns zero vector
/// let steering = calculate_separation(self_pos, 0, &[], &bounds, &config);
/// assert_eq!(steering.x, 0.0);
/// assert_eq!(steering.y, 0.0);
/// ```
pub fn calculate_separation(
    self_pos: Position,
    self_id: usize,
    swarm: &[DroneInfo],
    bounds: &Bounds,
    config: &SeparationConfig,
) -> Vec2 {
    let mut steering = Vec2::new(0.0, 0.0);
    let mut neighbor_count = 0;

    for other in swarm {
        // Skip self
        if other.uid == self_id {
            continue;
        }

        // Calculate distance to neighbor
        let delta = bounds.delta(self_pos.as_vec2(), other.pos.as_vec2());
        let distance = delta.magnitude();

        // Skip if outside detection radius
        if distance >= config.radius || distance < f32::EPSILON {
            continue;
        }

        // Calculate repulsion vector (pointing away from neighbor)
        // Normalize delta and invert direction
        let away = Vec2::new(-delta.x / distance, -delta.y / distance);

        // Force increases as distance decreases
        // Linear falloff: force = strength * (1 - distance/radius)
        // This gives more consistent avoidance across the detection range
        let normalized_dist = (distance / config.radius).clamp(0.0, 1.0);
        let force_magnitude = config.strength * (1.0 - normalized_dist);

        // Extra boost when very close (below min_distance)
        let force_magnitude = if distance < config.min_distance {
            force_magnitude * 2.0
        } else {
            force_magnitude
        };

        // Accumulate steering (not averaged - we want cumulative repulsion)
        steering.x += away.x * force_magnitude;
        steering.y += away.y * force_magnitude;
        neighbor_count += 1;
    }

    // Average the steering if we had neighbors
    if neighbor_count > 0 {
        steering.x /= neighbor_count as f32;
        steering.y /= neighbor_count as f32;
    }

    steering
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Heading, Velocity};

    fn create_drone_info(uid: usize, x: f32, y: f32) -> DroneInfo {
        DroneInfo {
            uid,
            pos: Position::new(x, y),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
        }
    }

    fn create_bounds() -> Bounds {
        Bounds::new(1000.0, 1000.0).unwrap()
    }

    #[test]
    fn test_empty_swarm_returns_zero() {
        let bounds = create_bounds();
        let config = SeparationConfig::default();
        let self_pos = Position::new(100.0, 100.0);

        let steering = calculate_separation(self_pos, 0, &[], &bounds, &config);

        assert_eq!(steering.x, 0.0);
        assert_eq!(steering.y, 0.0);
    }

    #[test]
    fn test_excludes_self() {
        let bounds = create_bounds();
        let config = SeparationConfig::default();
        let self_pos = Position::new(100.0, 100.0);

        // Swarm contains only self
        let swarm = vec![create_drone_info(0, 100.0, 100.0)];

        let steering = calculate_separation(self_pos, 0, &swarm, &bounds, &config);

        assert_eq!(steering.x, 0.0);
        assert_eq!(steering.y, 0.0);
    }

    #[test]
    fn test_drone_outside_radius_ignored() {
        let bounds = create_bounds();
        let config = SeparationConfig::new(50.0, 100.0, 10.0);
        let self_pos = Position::new(100.0, 100.0);

        // Other drone at distance 60 (outside radius of 50)
        let swarm = vec![create_drone_info(1, 160.0, 100.0)];

        let steering = calculate_separation(self_pos, 0, &swarm, &bounds, &config);

        assert_eq!(steering.x, 0.0);
        assert_eq!(steering.y, 0.0);
    }

    #[test]
    fn test_steers_away_from_neighbor() {
        let bounds = create_bounds();
        let config = SeparationConfig::new(100.0, 50.0, 10.0);
        let self_pos = Position::new(100.0, 100.0);

        // Other drone directly to the right
        let swarm = vec![create_drone_info(1, 150.0, 100.0)];

        let steering = calculate_separation(self_pos, 0, &swarm, &bounds, &config);

        // Should steer to the left (negative x)
        assert!(steering.x < 0.0, "Expected negative x steering, got {}", steering.x);
        assert!(steering.y.abs() < 1e-6, "Expected near-zero y steering, got {}", steering.y);
    }

    #[test]
    fn test_closer_drone_stronger_force() {
        let bounds = create_bounds();
        let config = SeparationConfig::new(100.0, 50.0, 5.0);
        let self_pos = Position::new(100.0, 100.0);

        // Close drone
        let close_swarm = vec![create_drone_info(1, 120.0, 100.0)];
        let close_steering = calculate_separation(self_pos, 0, &close_swarm, &bounds, &config);

        // Far drone
        let far_swarm = vec![create_drone_info(1, 180.0, 100.0)];
        let far_steering = calculate_separation(self_pos, 0, &far_swarm, &bounds, &config);

        // Close drone should produce stronger steering
        assert!(
            close_steering.x.abs() > far_steering.x.abs(),
            "Close steering {} should be stronger than far steering {}",
            close_steering.x.abs(),
            far_steering.x.abs()
        );
    }

    #[test]
    fn test_multiple_neighbors_averaged() {
        let bounds = create_bounds();
        let config = SeparationConfig::new(100.0, 50.0, 10.0);
        let self_pos = Position::new(100.0, 100.0);

        // Drones on opposite sides should roughly cancel out
        let swarm = vec![
            create_drone_info(1, 150.0, 100.0), // Right
            create_drone_info(2, 50.0, 100.0),  // Left (same distance)
        ];

        let steering = calculate_separation(self_pos, 0, &swarm, &bounds, &config);

        // Forces should approximately cancel
        assert!(steering.x.abs() < 1e-3, "X forces should cancel, got {}", steering.x);
    }

}
