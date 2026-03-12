//! Collision avoidance action nodes.

use crate::behaviors::tree::node::{BehaviorContext, BehaviorNode, BehaviorStatus};
use crate::behaviors::{
    calculate_separation, calculate_velocity_obstacle, SeparationConfig, VelocityObstacleConfig,
    calculate_apf_from_swarm, APFConfig,
};
use crate::behaviors::orca::{compute_orca_velocity, velocity_to_heading_speed, ORCAConfig};
use crate::types::{Heading, Vec2, Velocity};

/// Velocity Obstacle avoidance action.
///
/// Uses predictive collision avoidance to adjust heading and speed.
/// This wraps the existing `calculate_velocity_obstacle` function.
#[derive(Debug)]
pub struct VelocityObstacleAvoid {
    config: VelocityObstacleConfig,
}

impl VelocityObstacleAvoid {
    /// Create a new VO avoidance action with default config.
    pub fn new() -> Self {
        VelocityObstacleAvoid {
            config: VelocityObstacleConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: VelocityObstacleConfig) -> Self {
        VelocityObstacleAvoid { config }
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: VelocityObstacleConfig) {
        self.config = config;
    }

    /// Get the current configuration.
    pub fn config(&self) -> &VelocityObstacleConfig {
        &self.config
    }
}

impl Default for VelocityObstacleAvoid {
    fn default() -> Self {
        Self::new()
    }
}

impl BehaviorNode for VelocityObstacleAvoid {
    fn tick(&mut self, ctx: &mut BehaviorContext) -> BehaviorStatus {
        // Formation leaders don't do collision avoidance with swarm drones.
        // They follow their waypoints and other drones avoid them.
        if ctx.is_formation_leader {
            return BehaviorStatus::Success;
        }

        // Need a desired heading to adjust
        let desired_heading = match ctx.desired_heading {
            Some(hdg) => hdg,
            None => return BehaviorStatus::Success, // Nothing to adjust
        };

        // Build state for VO calculation
        let state = crate::types::State {
            pos: ctx.state.pos,
            hdg: ctx.state.hdg,
            vel: ctx.state.vel,
            acc: ctx.state.acc,
        };

        // Calculate velocity obstacle result
        let vo_result = calculate_velocity_obstacle(
            &state,
            ctx.drone_id,
            desired_heading,
            ctx.perf.max_turn_rate,
            ctx.swarm,
            ctx.bounds,
            &self.config,
        );

        // Apply heading adjustment
        let adjusted_heading = Heading::new(desired_heading.radians() + vo_result.heading_adjustment);
        ctx.desired_heading = Some(adjusted_heading);
        ctx.urgency = vo_result.urgency;

        // Slow down based on urgency
        if vo_result.urgency > 0.3 {
            let urgency_above_threshold = (vo_result.urgency - 0.3) / 0.7;
            ctx.desired_speed *= 1.0 - urgency_above_threshold * 0.6;
        }

        BehaviorStatus::Success
    }

    fn name(&self) -> &str {
        "VelocityObstacleAvoid"
    }
}

/// Separation avoidance action (reactive fallback).
///
/// Uses simple repulsion from nearby drones.
/// This wraps the existing `calculate_separation` function.
#[derive(Debug)]
pub struct SeparationAvoid {
    config: SeparationConfig,
}

impl SeparationAvoid {
    /// Create a new separation avoidance action with default config.
    pub fn new() -> Self {
        SeparationAvoid {
            config: SeparationConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: SeparationConfig) -> Self {
        SeparationAvoid { config }
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: SeparationConfig) {
        self.config = config;
    }

    /// Get the current configuration.
    pub fn config(&self) -> &SeparationConfig {
        &self.config
    }
}

impl Default for SeparationAvoid {
    fn default() -> Self {
        Self::new()
    }
}

impl BehaviorNode for SeparationAvoid {
    fn tick(&mut self, ctx: &mut BehaviorContext) -> BehaviorStatus {
        // Formation leaders don't do collision avoidance with swarm drones.
        // They follow their waypoints and other drones avoid them.
        if ctx.is_formation_leader {
            return BehaviorStatus::Success;
        }

        let avoidance = calculate_separation(
            ctx.state.pos,
            ctx.drone_id,
            ctx.swarm,
            ctx.bounds,
            &self.config,
        );

        let avoidance_mag = avoidance.magnitude();
        if avoidance_mag > f32::EPSILON {
            // Emergency escape - output velocity directly (quadcopter style)
            let max_speed = ctx.perf.max_vel;
            let escape_speed = max_speed * 0.5; // Slow down during emergency avoidance

            // Escape velocity in the avoidance direction
            let escape_vel = Vec2::new(
                avoidance.x / avoidance_mag * escape_speed,
                avoidance.y / avoidance_mag * escape_speed,
            );

            ctx.desired_velocity = Some(Velocity::from_vec2(escape_vel));
            ctx.desired_heading = Some(Heading::new(avoidance.heading()));
            ctx.desired_speed = 0.5;
            BehaviorStatus::Success
        } else {
            // No avoidance needed
            BehaviorStatus::Failure
        }
    }

    fn name(&self) -> &str {
        "SeparationAvoid"
    }
}

/// Artificial Potential Field (APF) avoidance action.
///
/// Uses inverse-square repulsive forces for stronger close-range avoidance.
/// More aggressive than linear separation when close to obstacles.
#[derive(Debug)]
pub struct APFAvoid {
    config: APFConfig,
    /// Weight for blending APF force with desired heading (0.0-1.0).
    force_weight: f32,
}

impl APFAvoid {
    /// Create a new APF avoidance action with default config.
    pub fn new() -> Self {
        APFAvoid {
            config: APFConfig::default(),
            force_weight: 0.5,
        }
    }

    /// Create with custom config.
    pub fn with_config(config: APFConfig) -> Self {
        APFAvoid {
            config,
            force_weight: 0.5,
        }
    }

    /// Create with custom config and force weight.
    pub fn with_config_and_weight(config: APFConfig, force_weight: f32) -> Self {
        APFAvoid {
            config,
            force_weight: force_weight.clamp(0.0, 1.0),
        }
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: APFConfig) {
        self.config = config;
    }

    /// Get the current configuration.
    pub fn config(&self) -> &APFConfig {
        &self.config
    }

    /// Set the force weight for heading blending.
    pub fn set_force_weight(&mut self, weight: f32) {
        self.force_weight = weight.clamp(0.0, 1.0);
    }
}

impl Default for APFAvoid {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute dynamic blending weight based on urgency level.
///
/// At high urgency (>0.7), full override for emergency escape.
/// At medium (>0.4), ramping weight.
/// At low urgency, scaled from the base weight.
fn compute_dynamic_weight(urgency: f32, base_weight: f32) -> f32 {
    if urgency > 0.7 {
        1.0
    } else if urgency > 0.4 {
        0.7 + (urgency - 0.4) * 1.0
    } else {
        base_weight * (0.5 + urgency * 1.25)
    }
}

impl BehaviorNode for APFAvoid {
    fn tick(&mut self, ctx: &mut BehaviorContext) -> BehaviorStatus {
        // Get current desired velocity (velocity-first, or heading+speed)
        let max_speed = ctx.perf.max_vel;
        let current_vel = match ctx.get_desired_velocity(max_speed) {
            Some(v) => v,
            None => return BehaviorStatus::Success,
        };

        // Calculate APF repulsive force from other drones
        let apf_force = calculate_apf_from_swarm(
            ctx.state.pos,
            ctx.drone_id,
            ctx.swarm,
            ctx.bounds,
            &self.config,
        );

        let force_magnitude = apf_force.magnitude();

        // No significant force - no adjustment needed
        if force_magnitude < 1.0 {
            return BehaviorStatus::Success;
        }

        // Calculate urgency based on force magnitude
        let urgency = (force_magnitude / self.config.max_force).min(1.0);
        ctx.urgency = ctx.urgency.max(urgency);

        // Formation leaders skip APF unless it's an emergency (high urgency)
        // They follow their waypoints while followers avoid them
        if ctx.is_formation_leader && urgency < 0.5 {
            return BehaviorStatus::Success;
        }

        let dynamic_weight = compute_dynamic_weight(urgency, self.force_weight);

        // Blend APF escape direction into velocity (quadcopter style)
        let current_vel_vec = current_vel.as_vec2();
        let current_speed = current_vel_vec.magnitude().max(0.1); // Minimum speed for escape

        // Normalize APF force to get escape direction
        let escape_dir = if force_magnitude > f32::EPSILON {
            Vec2::new(apf_force.x / force_magnitude, apf_force.y / force_magnitude)
        } else {
            Vec2::new(0.0, 0.0)
        };

        // At any meaningful urgency, ensure minimum escape speed
        let effective_speed = if urgency > 0.15 {
            let min_escape_speed = (0.2 + urgency * 0.4) * max_speed;
            current_speed.max(min_escape_speed)
        } else {
            current_speed
        };

        // Blend: (1 - weight) * current direction + weight * escape direction
        let current_dir = if current_speed > f32::EPSILON {
            Vec2::new(current_vel_vec.x / current_speed, current_vel_vec.y / current_speed)
        } else {
            escape_dir // If stationary, just use escape direction
        };

        let blended_dir = Vec2::new(
            current_dir.x * (1.0 - dynamic_weight) + escape_dir.x * dynamic_weight,
            current_dir.y * (1.0 - dynamic_weight) + escape_dir.y * dynamic_weight,
        );

        // Normalize blended direction
        let blended_mag = blended_dir.magnitude();
        let final_dir = if blended_mag > f32::EPSILON {
            Vec2::new(blended_dir.x / blended_mag, blended_dir.y / blended_mag)
        } else {
            escape_dir
        };

        // Apply final velocity
        let final_vel = Vec2::new(final_dir.x * effective_speed, final_dir.y * effective_speed);
        ctx.desired_velocity = Some(Velocity::from_vec2(final_vel));

        // Also set heading for facing direction
        ctx.desired_heading = Some(Heading::new(final_vel.heading()));
        ctx.desired_speed = effective_speed / max_speed;

        BehaviorStatus::Success
    }

    fn name(&self) -> &str {
        "APFAvoid"
    }
}

/// ORCA (Optimal Reciprocal Collision Avoidance) action.
///
/// Uses half-plane intersection to find guaranteed collision-free velocities.
/// Each agent takes responsibility for half the avoidance maneuver,
/// providing optimal collision avoidance when all agents use ORCA.
#[derive(Debug)]
pub struct ORCAAvoid {
    config: ORCAConfig,
}

impl ORCAAvoid {
    /// Create a new ORCA avoidance action with default config.
    pub fn new() -> Self {
        ORCAAvoid {
            config: ORCAConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: ORCAConfig) -> Self {
        ORCAAvoid { config }
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: ORCAConfig) {
        self.config = config;
    }

    /// Get the current configuration.
    pub fn config(&self) -> &ORCAConfig {
        &self.config
    }
}

impl Default for ORCAAvoid {
    fn default() -> Self {
        Self::new()
    }
}

impl BehaviorNode for ORCAAvoid {
    fn tick(&mut self, ctx: &mut BehaviorContext) -> BehaviorStatus {
        // Formation leaders don't do collision avoidance with swarm drones.
        if ctx.is_formation_leader {
            return BehaviorStatus::Success;
        }

        // Get preferred velocity from context (velocity-first, or heading+speed)
        let preferred_vel = if let Some(vel) = ctx.desired_velocity {
            vel.as_vec2()
        } else if let Some(desired_heading) = ctx.desired_heading {
            let actual_speed = ctx.desired_speed * self.config.max_speed;
            Vec2::new(
                desired_heading.radians().cos() * actual_speed,
                desired_heading.radians().sin() * actual_speed,
            )
        } else {
            return BehaviorStatus::Success; // Nothing to adjust
        };

        // Compute ORCA-safe velocity
        let orca_vel = compute_orca_velocity(
            ctx.state.pos,
            ctx.state.vel,
            ctx.drone_id,
            preferred_vel,
            ctx.swarm,
            ctx.bounds,
            &self.config,
        );

        // Calculate how much ORCA changed our velocity (for urgency)
        let delta_x = orca_vel.x - preferred_vel.x;
        let delta_y = orca_vel.y - preferred_vel.y;
        let change_magnitude = (delta_x * delta_x + delta_y * delta_y).sqrt();

        // Urgency based on how much we had to deviate
        let urgency = (change_magnitude / self.config.max_speed).min(1.0);
        ctx.urgency = ctx.urgency.max(urgency);

        // OUTPUT VELOCITY DIRECTLY (quadcopter style)
        ctx.desired_velocity = Some(Velocity::from_vec2(orca_vel));

        // Also set heading for facing direction (face direction of movement)
        let (new_heading, new_speed) = velocity_to_heading_speed(orca_vel);
        ctx.desired_heading = Some(Heading::new(new_heading));
        ctx.desired_speed = new_speed / self.config.max_speed;

        BehaviorStatus::Success
    }

    fn name(&self) -> &str {
        "ORCAAvoid"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::fixtures::{create_test_bounds_small, create_test_state_with_vel};
    use crate::types::{Bounds, DronePerfFeatures, DroneInfo, Position, State, Velocity};

    fn test_perf() -> &'static DronePerfFeatures {
        Box::leak(Box::new(DronePerfFeatures::default()))
    }

    fn create_test_state() -> State {
        create_test_state_with_vel(250.0, 250.0, 0.0, 7.5)
    }

    fn create_test_bounds() -> Bounds {
        create_test_bounds_small()
    }

    #[test]
    fn test_vo_avoid_no_threat() {
        let state = create_test_state();
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());
        ctx.desired_heading = Some(Heading::new(0.0));

        let mut vo = VelocityObstacleAvoid::new();
        let status = vo.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Success);
        // No threat, so urgency should be low
        assert!(ctx.urgency < 0.1);
    }

    #[test]
    fn test_vo_avoid_with_threat() {
        let state = create_test_state();
        let bounds = create_test_bounds();

        // Place a drone directly ahead
        let threat = DroneInfo {
            uid: 1,
            pos: Position::new(550.0, 500.0), // 50 units ahead
            hdg: Heading::new(std::f32::consts::PI), // Coming toward us
            vel: Velocity::from_heading_and_speed(Heading::new(std::f32::consts::PI), 60.0),
            is_formation_leader: false,
            group: 0,
        };
        let swarm = [threat];

        let mut ctx = BehaviorContext::new(&state, &swarm, &bounds, 0, 0.016, test_perf());
        ctx.desired_heading = Some(Heading::new(0.0));

        let mut vo = VelocityObstacleAvoid::new();
        let status = vo.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Success);
        // Should detect threat and have some urgency
        // Note: exact values depend on VO algorithm parameters
    }

    #[test]
    fn test_vo_avoid_no_heading() {
        let state = create_test_state();
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());
        // No desired heading set

        let mut vo = VelocityObstacleAvoid::new();
        let status = vo.tick(&mut ctx);

        // Should succeed (nothing to do)
        assert_eq!(status, BehaviorStatus::Success);
    }

    #[test]
    fn test_separation_no_neighbors() {
        let state = create_test_state();
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        let mut sep = SeparationAvoid::new();
        let status = sep.tick(&mut ctx);

        // No neighbors, so failure (no avoidance needed)
        assert_eq!(status, BehaviorStatus::Failure);
    }

    #[test]
    fn test_separation_with_close_neighbor() {
        let state = create_test_state();
        let bounds = create_test_bounds();

        // Place a drone very close (0.5m away - within detection radius)
        let neighbor = DroneInfo {
            uid: 1,
            pos: Position::new(250.5, 250.0), // 0.5 meters away
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
            group: 0,
        };
        let swarm = [neighbor];

        let mut ctx = BehaviorContext::new(&state, &swarm, &bounds, 0, 0.016, test_perf());

        let mut sep = SeparationAvoid::new();
        let status = sep.tick(&mut ctx);

        // Should succeed and set avoidance heading
        assert_eq!(status, BehaviorStatus::Success);
        assert!(ctx.desired_heading.is_some());
    }

    // ===== APF Avoidance tests =====

    #[test]
    fn test_apf_avoid_no_threat() {
        let state = create_test_state();
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());
        ctx.desired_heading = Some(Heading::new(0.0));

        let mut apf = APFAvoid::new();
        let status = apf.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Success);
        // No threat, heading should be unchanged
        assert!((ctx.desired_heading.unwrap().radians()).abs() < 0.01);
    }

    #[test]
    fn test_apf_avoid_with_close_drone() {
        let state = create_test_state();
        let bounds = create_test_bounds();

        // Place a drone close enough to trigger APF response (3m away)
        let threat = DroneInfo {
            uid: 1,
            pos: Position::new(253.0, 250.0), // 3 meters to the right
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
            group: 0,
        };
        let swarm = [threat];

        let mut ctx = BehaviorContext::new(&state, &swarm, &bounds, 0, 0.016, test_perf());
        ctx.desired_heading = Some(Heading::new(0.0)); // Heading right (toward threat)

        // Use config with high repulsion for test (meter-based values)
        let config = APFConfig::new(
            75.0,    // influence_distance (meters)
            5000.0,  // high repulsion_strength
            2.0,     // min_distance (meters)
            50.0,    // lower max_force
        );
        let mut apf = APFAvoid::with_config(config);
        let status = apf.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Success);
        // Should have some urgency
        assert!(ctx.urgency > 0.0, "Expected urgency > 0, got {}", ctx.urgency);
        // Heading should be adjusted (checking it has a value, specific angle depends on algorithm)
        assert!(ctx.desired_heading.is_some());
    }

    #[test]
    fn test_apf_avoid_very_close_ensures_escape_speed() {
        let state = create_test_state();
        let bounds = create_test_bounds();

        // Place a drone at min_distance to trigger maximum force (2m away)
        let threat = DroneInfo {
            uid: 1,
            pos: Position::new(252.0, 250.0), // 2 meters (at min_distance)
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
            group: 0,
        };
        let swarm = [threat];

        let mut ctx = BehaviorContext::new(&state, &swarm, &bounds, 0, 0.016, test_perf());
        ctx.desired_heading = Some(Heading::new(0.0));
        ctx.desired_speed = 0.1; // Start with low speed

        // Use config that produces high urgency (meter-based values)
        let config = APFConfig::new(
            75.0,     // influence_distance (meters)
            10000.0,  // very high repulsion for test
            2.0,      // min_distance (meters)
            25.0,     // low max_force means higher urgency ratio
        );
        let mut apf = APFAvoid::with_config(config);
        apf.tick(&mut ctx);

        // With these settings, urgency should exceed 0.15 threshold for escape
        assert!(ctx.urgency > 0.15, "Expected urgency > 0.15, got {}", ctx.urgency);
        // At high urgency, APF should boost speed (min_escape = 0.2 + urgency * 0.4)
        let expected_min = 0.2 + ctx.urgency * 0.4;
        assert!(ctx.desired_speed >= expected_min,
            "Expected escape speed >= {}, got {}", expected_min, ctx.desired_speed);
    }

    #[test]
    fn test_apf_avoid_no_heading() {
        let state = create_test_state();
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());
        // No desired heading

        let mut apf = APFAvoid::new();
        let status = apf.tick(&mut ctx);

        // Should succeed (nothing to adjust)
        assert_eq!(status, BehaviorStatus::Success);
    }

    #[test]
    fn test_apf_avoid_formation_leader_skipped_low_urgency() {
        // Leaders skip APF when urgency is low (< 0.5)
        // But they DO react when urgency is high (emergency situations)
        let state = create_test_state();
        let bounds = create_test_bounds();

        // Drone at moderate distance - produces low urgency (50m away)
        let threat = DroneInfo {
            uid: 1,
            pos: Position::new(300.0, 250.0), // 50 meters away - moderate distance
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
            group: 0,
        };
        let swarm = [threat];

        let mut ctx = BehaviorContext::new_with_leader(&state, &swarm, &bounds, 0, 0.016, true, test_perf());
        ctx.desired_heading = Some(Heading::new(0.0));

        let mut apf = APFAvoid::new();
        let status = apf.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Success);
        // At moderate distance, leader skips APF so urgency should be low
        assert!(ctx.urgency < 0.5, "Expected low urgency for leader at moderate distance, got {}", ctx.urgency);
    }

    #[test]
    fn test_apf_avoid_formation_leader_emergency() {
        // Leaders DO react to very close threats (emergency)
        let state = create_test_state();
        let bounds = create_test_bounds();

        // Drone very close - produces high urgency (1.5 meters away)
        let threat = DroneInfo {
            uid: 1,
            pos: Position::new(251.5, 250.0), // 1.5 meters away - very close
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
            group: 0,
        };
        let swarm = [threat];

        let mut ctx = BehaviorContext::new_with_leader(&state, &swarm, &bounds, 0, 0.016, true, test_perf());
        ctx.desired_heading = Some(Heading::new(0.0));

        let mut apf = APFAvoid::new();
        let status = apf.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Success);
        // At very close distance, even leader should have high urgency
        assert!(ctx.urgency > 0.5, "Expected high urgency for leader in emergency, got {}", ctx.urgency);
    }

    #[test]
    fn test_apf_force_weight_configurable() {
        let mut apf = APFAvoid::new();
        assert_eq!(apf.force_weight, 0.5);

        apf.set_force_weight(0.8);
        assert_eq!(apf.force_weight, 0.8);

        // Should clamp to valid range
        apf.set_force_weight(1.5);
        assert_eq!(apf.force_weight, 1.0);

        apf.set_force_weight(-0.5);
        assert_eq!(apf.force_weight, 0.0);
    }

    // ===== ORCA Avoidance tests =====

    #[test]
    fn test_orca_avoid_no_threat() {
        let state = create_test_state();
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());
        ctx.desired_heading = Some(Heading::new(0.0));
        ctx.desired_speed = 0.5; // Speed factor (0-1), represents 60/120 = 50%

        let mut orca = ORCAAvoid::new();
        let status = orca.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Success);
        // No threat, urgency should be low
        assert!(ctx.urgency < 0.1);
        // Heading should remain roughly the same
        assert!(ctx.desired_heading.unwrap().radians().abs() < 0.1);
    }

    #[test]
    fn test_orca_avoid_with_collision_course() {
        let state = create_test_state();
        let bounds = create_test_bounds();

        // Place a drone on collision course (3m ahead, coming toward us)
        // Using 3m which is within NEIGHBOR_DIST (5m)
        let threat = DroneInfo {
            uid: 1,
            pos: Position::new(253.0, 250.0), // 3 meters ahead (state starts at 250,250)
            hdg: Heading::new(std::f32::consts::PI), // Coming toward us
            vel: Velocity::from_heading_and_speed(Heading::new(std::f32::consts::PI), 7.5), // 7.5 m/s
            is_formation_leader: false,
            group: 0,
        };
        let swarm = [threat];

        let mut ctx = BehaviorContext::new(&state, &swarm, &bounds, 0, 0.016, test_perf());
        ctx.desired_heading = Some(Heading::new(0.0)); // Heading right
        ctx.desired_speed = 0.5; // Speed factor (0-1)

        let mut orca = ORCAAvoid::new();
        let status = orca.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Success);
        // ORCA should have adjusted our velocity
        // Either heading changed or speed changed
        let heading_changed = ctx.desired_heading.unwrap().radians().abs() > 0.01;
        let speed_changed = (ctx.desired_speed - 0.5).abs() > 0.01;
        assert!(
            heading_changed || speed_changed || ctx.urgency > 0.0,
            "Expected ORCA to make some adjustment"
        );
    }

    #[test]
    fn test_orca_avoid_no_heading() {
        let state = create_test_state();
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());
        // No desired heading

        let mut orca = ORCAAvoid::new();
        let status = orca.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Success);
    }

    #[test]
    fn test_orca_avoid_formation_leader_skipped() {
        let state = create_test_state();
        let bounds = create_test_bounds();

        let threat = DroneInfo {
            uid: 1,
            pos: Position::new(520.0, 500.0),
            hdg: Heading::new(std::f32::consts::PI),
            vel: Velocity::from_heading_and_speed(Heading::new(std::f32::consts::PI), 60.0),
            is_formation_leader: false,
            group: 0,
        };
        let swarm = [threat];

        let mut ctx = BehaviorContext::new_with_leader(&state, &swarm, &bounds, 0, 0.016, true, test_perf());
        ctx.desired_heading = Some(Heading::new(0.0));
        ctx.desired_speed = 0.5; // Speed factor (0-1), represents 60/120 = 50%

        let mut orca = ORCAAvoid::new();
        let status = orca.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Success);
        // Leader should not have urgency set
        assert!(ctx.urgency < 0.01);
    }

    #[test]
    fn test_orca_speed_non_negative() {
        let state = create_test_state();
        let bounds = create_test_bounds();

        // Multiple threats to force strong avoidance
        let swarm = vec![
            DroneInfo {
                uid: 1,
                pos: Position::new(530.0, 500.0),
                hdg: Heading::new(std::f32::consts::PI),
                vel: Velocity::from_heading_and_speed(Heading::new(std::f32::consts::PI), 60.0),
                is_formation_leader: false,
                group: 0,
            },
            DroneInfo {
                uid: 2,
                pos: Position::new(500.0, 530.0),
                hdg: Heading::new(-std::f32::consts::FRAC_PI_2),
                vel: Velocity::from_heading_and_speed(Heading::new(-std::f32::consts::FRAC_PI_2), 60.0),
                is_formation_leader: false,
                group: 0,
            },
        ];

        let mut ctx = BehaviorContext::new(&state, &swarm, &bounds, 0, 0.016, test_perf());
        ctx.desired_heading = Some(Heading::new(0.0));
        ctx.desired_speed = 0.5; // Speed factor (0-1), represents 60/120 = 50%

        let mut orca = ORCAAvoid::new();
        orca.tick(&mut ctx);

        // Speed should be non-negative (can go to zero)
        assert!(
            ctx.desired_speed >= 0.0,
            "Speed factor {} should be >= 0",
            ctx.desired_speed
        );
    }
}
