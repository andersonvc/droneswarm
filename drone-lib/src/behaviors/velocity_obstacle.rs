//! Velocity Obstacle (VO) based collision avoidance for fixed-wing drones.
//!
//! This module implements predictive collision avoidance that respects
//! kinematic constraints (turn rate limits). Instead of reactive forces,
//! it samples reachable headings and selects the one that best avoids
//! future collisions while progressing toward the goal.

use crate::types::{Bounds, DroneInfo, Heading, Position, State};

/// Velocity Obstacle algorithm default constants.
pub(crate) mod defaults {
    /// Lookahead time for collision prediction (seconds).
    pub const LOOKAHEAD_TIME: f32 = 2.5;

    /// Safe separation distance (meters).
    pub const SAFE_DISTANCE: f32 = 1.5;

    /// Detection range for early awareness (meters).
    pub const DETECTION_RANGE: f32 = 5.0;

    /// Avoidance weight (0.0-1.0).
    pub const AVOIDANCE_WEIGHT: f32 = 0.9;

    /// Number of velocity samples for optimization.
    pub const NUM_SAMPLES: usize = 25;

    /// Number of time samples for trajectory prediction.
    pub const TIME_SAMPLES: usize = 10;
}

/// Configuration for velocity obstacle avoidance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityObstacleConfig {
    /// Lookahead time for collision prediction (seconds).
    /// Longer = more conservative, shorter = more reactive.
    pub lookahead_time: f32,

    /// Number of heading samples to evaluate.
    /// More samples = better solutions but slower.
    pub num_samples: usize,

    /// Number of time samples along trajectory to check.
    /// More samples = catches crossing paths better.
    pub time_samples: usize,

    /// Minimum safe separation distance between drones (meters).
    pub safe_distance: f32,

    /// How strongly to penalize potential collisions (0.0 - 1.0).
    /// Higher = more aggressive avoidance.
    pub avoidance_weight: f32,

    /// Maximum detection range for considering other drones (meters).
    pub detection_range: f32,
}

impl Default for VelocityObstacleConfig {
    fn default() -> Self {
        VelocityObstacleConfig {
            lookahead_time: defaults::LOOKAHEAD_TIME,
            num_samples: defaults::NUM_SAMPLES,
            time_samples: defaults::TIME_SAMPLES,
            safe_distance: defaults::SAFE_DISTANCE,
            avoidance_weight: defaults::AVOIDANCE_WEIGHT,
            detection_range: defaults::DETECTION_RANGE,
        }
    }
}

impl VelocityObstacleConfig {
    /// Create a new VO configuration.
    pub fn new(
        lookahead_time: f32,
        num_samples: usize,
        time_samples: usize,
        safe_distance: f32,
        avoidance_weight: f32,
        detection_range: f32,
    ) -> Self {
        VelocityObstacleConfig {
            lookahead_time,
            num_samples,
            time_samples,
            safe_distance,
            avoidance_weight,
            detection_range,
        }
    }
}

/// Result of velocity obstacle computation.
#[derive(Debug, Clone, Copy)]
pub struct AvoidanceResult {
    /// Recommended heading adjustment (radians, positive = counterclockwise).
    pub heading_adjustment: f32,

    /// Confidence in the recommendation (0.0 = no threat, 1.0 = imminent collision).
    pub urgency: f32,

    /// Whether any collision threat was detected.
    pub threat_detected: bool,
}

impl Default for AvoidanceResult {
    fn default() -> Self {
        AvoidanceResult {
            heading_adjustment: 0.0,
            urgency: 0.0,
            threat_detected: false,
        }
    }
}

/// Calculate optimal heading adjustment using velocity obstacles.
///
/// Samples reachable headings (constrained by turn rate) and evaluates
/// each for collision risk over the lookahead horizon. Returns the
/// adjustment that best balances goal-seeking with collision avoidance.
///
/// # Arguments
/// * `self_state` - Current drone state (position, velocity, heading)
/// * `self_id` - Drone's unique ID (to exclude self from swarm)
/// * `desired_heading` - Heading toward the goal/waypoint
/// * `max_turn_rate` - Maximum turn rate in rad/s
/// * `swarm` - All drones in the swarm
/// * `bounds` - World bounds for toroidal distance
/// * `config` - VO algorithm configuration
///
/// # Returns
/// An `AvoidanceResult` with the recommended heading adjustment.
pub fn calculate_velocity_obstacle(
    self_state: &State,
    self_id: usize,
    desired_heading: Heading,
    max_turn_rate: f32,
    swarm: &[DroneInfo],
    bounds: &Bounds,
    config: &VelocityObstacleConfig,
) -> AvoidanceResult {
    let current_speed = self_state.vel.as_vec2().magnitude();

    // If not moving, can't do predictive avoidance effectively
    if current_speed < 1.0 {
        return AvoidanceResult::default();
    }

    // Maximum heading change possible in lookahead time
    let max_heading_change = max_turn_rate * config.lookahead_time;

    // Filter to nearby drones only
    let nearby: Vec<&DroneInfo> = swarm
        .iter()
        .filter(|d| {
            if d.uid == self_id {
                return false;
            }
            let dist = bounds.distance(
                self_state.pos.as_vec2(),
                d.pos.as_vec2(),
            );
            dist < config.detection_range
        })
        .collect();

    // No nearby drones = no threat
    if nearby.is_empty() {
        return AvoidanceResult::default();
    }

    // Sample headings from -max_change to +max_change
    let mut best_heading_offset: f32 = 0.0;
    let mut best_score = f32::NEG_INFINITY;
    let mut min_separation_found = f32::MAX;
    let mut threat_detected = false;

    let step = if config.num_samples > 1 {
        (2.0 * max_heading_change) / (config.num_samples - 1) as f32
    } else {
        0.0
    };

    for i in 0..config.num_samples {
        let heading_offset = -max_heading_change + step * i as f32;
        let candidate_heading = Heading::new(self_state.hdg.radians() + heading_offset);

        let min_separation = compute_min_separation(
            self_state.pos, candidate_heading, current_speed, &nearby, bounds, config,
        );

        min_separation_found = min_separation_found.min(min_separation);

        if min_separation < config.safe_distance {
            threat_detected = true;
        }

        let score = score_heading_candidate(
            heading_offset, min_separation, desired_heading, candidate_heading, config,
        );

        if score > best_score {
            best_score = score;
            best_heading_offset = heading_offset;
        }
    }

    // Calculate urgency based on how close the nearest threat is
    let urgency = if min_separation_found < config.safe_distance {
        1.0 - (min_separation_found / config.safe_distance)
    } else if min_separation_found < config.safe_distance * 2.0 {
        0.5 * (1.0 - (min_separation_found - config.safe_distance) / config.safe_distance)
    } else {
        0.0
    };

    AvoidanceResult {
        heading_adjustment: best_heading_offset,
        urgency,
        threat_detected,
    }
}

/// Compute the minimum separation distance between a candidate heading
/// and all nearby drones over the lookahead horizon.
fn compute_min_separation(
    self_pos: Position,
    candidate_heading: Heading,
    current_speed: f32,
    nearby: &[&DroneInfo],
    bounds: &Bounds,
    config: &VelocityObstacleConfig,
) -> f32 {
    let mut min_separation = f32::MAX;

    for neighbor in nearby {
        let neighbor_speed = neighbor.vel.as_vec2().magnitude();

        for t_idx in 1..=config.time_samples {
            let t = config.lookahead_time * (t_idx as f32 / config.time_samples as f32);

            let predicted_self = predict_position(self_pos, candidate_heading, current_speed, t);
            let predicted_neighbor = predict_position(neighbor.pos, neighbor.hdg, neighbor_speed, t);

            let separation = bounds.distance(predicted_self.as_vec2(), predicted_neighbor.as_vec2());
            if separation < min_separation {
                min_separation = separation;
            }
        }
    }

    min_separation
}

/// Score a heading candidate balancing goal-seeking vs collision avoidance.
///
/// Returns the combined score (higher = better). Includes a right-turn bias
/// for threatened situations (ICAO-style rules for deterministic avoidance).
fn score_heading_candidate(
    heading_offset: f32,
    min_separation: f32,
    desired_heading: Heading,
    candidate_heading: Heading,
    config: &VelocityObstacleConfig,
) -> f32 {
    let heading_error = (candidate_heading.radians() - desired_heading.radians()).abs();
    let normalized_error = heading_error / std::f32::consts::PI;

    let separation_score = if min_separation < config.safe_distance {
        -1.0 + (min_separation / config.safe_distance)
    } else {
        (min_separation / config.safe_distance).min(2.0) / 2.0
    };

    let goal_score = 1.0 - normalized_error;
    let mut score = goal_score * (1.0 - config.avoidance_weight)
        + separation_score * config.avoidance_weight;

    // CONSISTENT TURN DIRECTION: Use deterministic rules to avoid "reciprocal dance"
    // Rule: ALWAYS turn RIGHT (negative heading offset) when threatened
    // Both drones turning right means they pass each other safely (left-to-left)
    /// Score bonus for right turns during threat (ICAO-style pass-left-to-left).
    const RIGHT_TURN_BONUS: f32 = 0.3;
    /// Score penalty for left turns during threat (discourages left-to-right passes).
    const LEFT_TURN_PENALTY: f32 = -0.2;
    /// Threat detection multiplier for safe distance.
    const THREAT_DETECTION_MULTIPLIER: f32 = 2.0;
    if min_separation < config.safe_distance * THREAT_DETECTION_MULTIPLIER {
        let threat_proximity = 1.0 - (min_separation / (config.safe_distance * THREAT_DETECTION_MULTIPLIER));
        let turn_right_bias = if heading_offset < 0.0 {
            RIGHT_TURN_BONUS * threat_proximity
        } else if heading_offset > 0.0 {
            LEFT_TURN_PENALTY * threat_proximity
        } else {
            0.0
        };
        score += turn_right_bias;
    }

    score
}

/// Predict position after traveling for a given time at constant heading/speed.
fn predict_position(
    start: Position,
    heading: Heading,
    speed: f32,
    time: f32,
) -> Position {
    let distance = speed * time;
    let dx = heading.radians().cos() * distance;
    let dy = heading.radians().sin() * distance;

    Position::new(
        start.x() + dx,
        start.y() + dy,
    )
}

/// Convenience function to get recommended heading directly.
///
/// Combines current heading with the VO adjustment.
pub fn get_recommended_heading(
    self_state: &State,
    self_id: usize,
    desired_heading: Heading,
    max_turn_rate: f32,
    swarm: &[DroneInfo],
    bounds: &Bounds,
    config: &VelocityObstacleConfig,
) -> Heading {
    let result = calculate_velocity_obstacle(
        self_state,
        self_id,
        desired_heading,
        max_turn_rate,
        swarm,
        bounds,
        config,
    );

    Heading::new(self_state.hdg.radians() + result.heading_adjustment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::fixtures::{create_test_bounds_small, create_test_state_with_vel};
    use crate::types::Velocity;

    fn create_drone_info(uid: usize, x: f32, y: f32, hdg: f32, speed: f32) -> DroneInfo {
        DroneInfo {
            uid,
            pos: Position::new(x, y),
            hdg: Heading::new(hdg),
            vel: Velocity::from_heading_and_speed(Heading::new(hdg), speed),
            is_formation_leader: false,
            group: 0,
        }
    }

    #[test]
    fn test_no_threat_returns_zero_adjustment() {
        let bounds = create_test_bounds_small();
        let config = VelocityObstacleConfig::default();

        // Drone at center moving right (7.5 m/s)
        let state = create_test_state_with_vel(250.0, 250.0, 0.0, 7.5);
        let desired = Heading::new(0.0);

        // Empty swarm
        let result = calculate_velocity_obstacle(
            &state, 0, desired, 2.0, &[], &bounds, &config
        );

        assert!(!result.threat_detected);
        assert!(result.urgency < 0.01);
    }

    #[test]
    fn test_threat_detected_when_collision_imminent() {
        let bounds = create_test_bounds_small();
        let config = VelocityObstacleConfig::default();

        // Two drones where one will catch up to the other
        // Self at 200m, speed 10 m/s; Other at 203m, speed 2 m/s (3m apart, within detection 5m)
        let state = create_test_state_with_vel(200.0, 250.0, 0.0, 10.0); // Moving right at 10 m/s
        let desired = Heading::new(0.0);

        let swarm = vec![
            create_drone_info(0, 200.0, 250.0, 0.0, 10.0), // Self
            create_drone_info(1, 203.0, 250.0, 0.0, 2.0),  // Slow drone ahead, same direction
        ];

        let result = calculate_velocity_obstacle(
            &state, 0, desired, 2.0, &swarm, &bounds, &config
        );

        assert!(result.threat_detected);
        assert!(result.urgency > 0.0);
    }

    #[test]
    fn test_suggests_avoidance_maneuver() {
        let bounds = create_test_bounds_small();
        let config = VelocityObstacleConfig::default();

        // Drone heading right, obstacle directly ahead
        let state = create_test_state_with_vel(200.0, 250.0, 0.0, 10.0); // 10 m/s
        let desired = Heading::new(0.0);

        let swarm = vec![
            create_drone_info(0, 200.0, 250.0, 0.0, 10.0),
            create_drone_info(1, 203.0, 250.0, 0.0, 2.0), // Slow drone 3m ahead (within detection 5m)
        ];

        let result = calculate_velocity_obstacle(
            &state, 0, desired, 2.0, &swarm, &bounds, &config
        );

        // Should suggest turning (non-zero adjustment)
        assert!(result.heading_adjustment.abs() > 0.1);
    }

    #[test]
    fn test_stationary_drone_returns_default() {
        let bounds = create_test_bounds_small();
        let config = VelocityObstacleConfig::default();

        // Stationary drone
        let state = create_test_state_with_vel(250.0, 250.0, 0.0, 0.0);
        let desired = Heading::new(0.0);

        let swarm = vec![
            create_drone_info(1, 260.0, 250.0, 0.0, 7.5), // 10m away
        ];

        let result = calculate_velocity_obstacle(
            &state, 0, desired, 2.0, &swarm, &bounds, &config
        );

        // Can't do predictive avoidance when stationary
        assert_eq!(result.heading_adjustment, 0.0);
    }

    #[test]
    fn test_distant_drone_ignored() {
        let bounds = create_test_bounds_small();
        let config = VelocityObstacleConfig {
            detection_range: 50.0, // 50 meters detection range
            ..Default::default()
        };

        let state = create_test_state_with_vel(250.0, 250.0, 0.0, 7.5);
        let desired = Heading::new(0.0);

        // Drone far away (beyond detection range - 100m away)
        let swarm = vec![
            create_drone_info(1, 350.0, 250.0, std::f32::consts::PI, 7.5),
        ];

        let result = calculate_velocity_obstacle(
            &state, 0, desired, 2.0, &swarm, &bounds, &config
        );

        assert!(!result.threat_detected);
    }
}
