//! Safety layer for collision avoidance.
//!
//! The safety layer wraps any behavior's output velocity, adjusting it
//! for collision avoidance using ORCA and APF. It runs independently of
//! the behavior — following the subsumption principle, it can override
//! any behavior when survival requires it.
//!
//! This replaces the old approach where ORCA and APF were behavior tree
//! nodes in a Sequence with SeekWaypoint. That architecture caused
//! SeekWaypoint to fight ORCA every tick. The safety layer approach
//! cleanly separates "what I want to do" from "what I must do to survive."

use crate::behaviors::orca::{compute_orca_velocity, ORCAConfig};
use crate::behaviors::potential_field::{calculate_apf_from_swarm, APFConfig};
use crate::tasks::SafetyFeedback;
use crate::types::{Bounds, DroneInfo, State, Vec2};

/// Safety layer that wraps desired velocity with collision avoidance.
#[derive(Debug, Clone)]
pub struct SafetyLayer {
    pub orca_config: ORCAConfig,
    pub apf_config: APFConfig,
}

impl SafetyLayer {
    /// Create a new safety layer with the given configurations.
    pub fn new(orca_config: ORCAConfig, apf_config: APFConfig) -> Self {
        SafetyLayer {
            orca_config,
            apf_config,
        }
    }

    /// Apply collision avoidance to a desired velocity.
    ///
    /// Returns the adjusted (safe) velocity and feedback about threats.
    pub fn apply(
        &self,
        desired_vel: Vec2,
        state: &State,
        drone_id: usize,
        swarm: &[DroneInfo],
        bounds: &Bounds,
    ) -> (Vec2, SafetyFeedback) {
        // 1. ORCA: compute collision-free velocity
        let orca_vel = compute_orca_velocity(
            state.pos,
            state.vel,
            drone_id,
            desired_vel,
            swarm,
            bounds,
            &self.orca_config,
        );

        // 2. APF: apply repulsive forces on top of ORCA result
        let apf_force = calculate_apf_from_swarm(
            state.pos,
            drone_id,
            swarm,
            bounds,
            &self.apf_config,
        );

        let force_magnitude = apf_force.magnitude();
        let safe_vel = if force_magnitude > 1.0 {
            // Blend APF force into ORCA velocity
            let apf_urgency = (force_magnitude / self.apf_config.max_force).min(1.0);
            let weight = apf_blend_weight(apf_urgency);

            let orca_speed = orca_vel.magnitude().max(0.1);
            let orca_dir = if orca_speed > f32::EPSILON {
                Vec2::new(orca_vel.x / orca_speed, orca_vel.y / orca_speed)
            } else {
                Vec2::new(apf_force.x / force_magnitude, apf_force.y / force_magnitude)
            };

            let escape_dir = Vec2::new(apf_force.x / force_magnitude, apf_force.y / force_magnitude);

            let blended_dir = Vec2::new(
                orca_dir.x * (1.0 - weight) + escape_dir.x * weight,
                orca_dir.y * (1.0 - weight) + escape_dir.y * weight,
            );
            let blended_mag = blended_dir.magnitude();
            let final_dir = if blended_mag > f32::EPSILON {
                Vec2::new(blended_dir.x / blended_mag, blended_dir.y / blended_mag)
            } else {
                escape_dir
            };

            // Maintain speed, boosting if urgency is high
            let effective_speed = if apf_urgency > 0.15 {
                let min_escape = (0.2 + apf_urgency * 0.4) * self.orca_config.max_speed;
                orca_speed.max(min_escape)
            } else {
                orca_speed
            };

            Vec2::new(final_dir.x * effective_speed, final_dir.y * effective_speed)
        } else {
            orca_vel
        };

        // 3. Compute feedback
        let delta_x = safe_vel.x - desired_vel.x;
        let delta_y = safe_vel.y - desired_vel.y;
        let deflection = (delta_x * delta_x + delta_y * delta_y).sqrt();
        let urgency = (deflection / self.orca_config.max_speed).min(1.0);

        // Determine primary threat direction from APF force (points away from threat)
        let threat_direction = if force_magnitude > 1.0 {
            // APF force points AWAY from threats, so negate for threat direction
            Some(Vec2::new(-apf_force.x / force_magnitude, -apf_force.y / force_magnitude))
        } else {
            None
        };

        let feedback = SafetyFeedback {
            urgency,
            threat_direction,
            safe_velocity: safe_vel,
        };

        (safe_vel, feedback)
    }
}

/// Compute blend weight for APF based on urgency.
fn apf_blend_weight(urgency: f32) -> f32 {
    if urgency > 0.7 {
        1.0
    } else if urgency > 0.4 {
        0.7 + (urgency - 0.4) * 1.0
    } else {
        0.9 * (0.5 + urgency * 1.25)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::fixtures::{create_test_bounds, create_test_state_with_vel};
    use crate::types::{Heading, Position, Velocity};

    #[test]
    fn test_safety_layer_no_threats() {
        let layer = SafetyLayer::new(ORCAConfig::default(), APFConfig::default());
        let state = create_test_state_with_vel(500.0, 500.0, 0.0, 10.0);
        let bounds = create_test_bounds();

        let desired = Vec2::new(10.0, 0.0);
        let (safe_vel, feedback) = layer.apply(desired, &state, 0, &[], &bounds);

        // No threats: output should match input
        assert!((safe_vel.x - desired.x).abs() < 0.1);
        assert!((safe_vel.y - desired.y).abs() < 0.1);
        assert!(feedback.urgency < 0.01);
        assert!(feedback.threat_direction.is_none());
    }

    #[test]
    fn test_safety_layer_with_threat() {
        let orca_config = ORCAConfig {
            time_horizon: 5.0,
            agent_radius: 15.0,
            max_speed: 100.0,
            neighbor_dist: 200.0,
            max_neighbors: 10,
            enemy_avoidance_radius: 0.0,
        };
        let layer = SafetyLayer::new(orca_config, APFConfig::default());
        let state = create_test_state_with_vel(100.0, 100.0, 0.0, 30.0);
        let bounds = create_test_bounds();

        let threat = DroneInfo {
            uid: 1,
            pos: Position::new(130.0, 100.0),
            hdg: Heading::new(std::f32::consts::PI),
            vel: Velocity::from_heading_and_speed(Heading::new(std::f32::consts::PI), 30.0),
            is_formation_leader: false,
            group: 0,
        };

        let desired = Vec2::new(30.0, 0.0); // heading straight at threat
        let (safe_vel, feedback) = layer.apply(desired, &state, 0, &[threat], &bounds);

        // Should deflect
        let deflected = safe_vel.y.abs() > 0.1 || safe_vel.x < desired.x - 0.1;
        assert!(deflected, "Expected deflection, got ({}, {})", safe_vel.x, safe_vel.y);
    }
}
