//! Generic platform implementation with realistic kinematics.

use crate::types::{
    units, Acceleration, Bounds, DronePerfFeatures, Heading, Position, State, Velocity,
};

use super::traits::Platform;

/// Default maximum velocity in meters/second.
pub const DEFAULT_MAX_VELOCITY: f32 = units::DEFAULT_MAX_VELOCITY;
/// Default maximum acceleration in meters/second².
pub const DEFAULT_MAX_ACCELERATION: f32 = units::DEFAULT_MAX_ACCELERATION;
/// Default maximum turn rate in radians/second (at max velocity).
pub const DEFAULT_MAX_TURN_RATE: f32 = units::DEFAULT_MAX_TURN_RATE;
/// Minimum turn rate even at zero velocity (rad/s).
pub const MIN_TURN_RATE: f32 = units::MIN_TURN_RATE;

/// Generic platform with realistic kinematic constraints.
///
/// Key characteristics:
/// - Turn rate scales with velocity squared (need momentum to turn)
/// - Minimum turn rate ensures some control even when slow
/// - Acceleration is bounded
#[derive(Debug, Clone)]
pub struct GenericPlatform {
    state: State,
    perf: DronePerfFeatures,
    bounds: Bounds,
}

impl GenericPlatform {
    /// Create a new fixed-wing platform with default performance parameters.
    pub fn new(pos: Position, hdg: Heading, bounds: Bounds) -> Self {
        let state = State {
            hdg,
            pos,
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };

        GenericPlatform {
            state,
            perf: DronePerfFeatures::new_unchecked(
                DEFAULT_MAX_VELOCITY,
                DEFAULT_MAX_ACCELERATION,
                DEFAULT_MAX_TURN_RATE,
            ),
            bounds,
        }
    }

    /// Create a fixed-wing platform with custom performance parameters.
    pub fn with_perf(pos: Position, hdg: Heading, bounds: Bounds, perf: DronePerfFeatures) -> Self {
        let state = State {
            hdg,
            pos,
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };

        GenericPlatform { state, bounds, perf }
    }

    /// Create a fixed-wing platform with an existing state.
    pub fn from_state(state: State, bounds: Bounds) -> Self {
        GenericPlatform {
            state,
            perf: DronePerfFeatures::new_unchecked(
                DEFAULT_MAX_VELOCITY,
                DEFAULT_MAX_ACCELERATION,
                DEFAULT_MAX_TURN_RATE,
            ),
            bounds,
        }
    }
}

impl Platform for GenericPlatform {
    /// Quadcopter-style velocity steering: velocity and heading are independent.
    ///
    /// The drone accelerates toward the desired velocity (any direction) while
    /// optionally rotating to face a desired heading. This allows strafing,
    /// flying backwards, and rotating while hovering.
    fn apply_velocity_steering(
        &mut self,
        desired_velocity: Velocity,
        desired_heading: Option<Heading>,
        dt: f32,
    ) {
        use crate::types::Vec2;

        let current_vel = self.state.vel.as_vec2();
        let desired_vel = desired_velocity.clamp_speed(self.perf.max_vel).as_vec2();

        // 1. Velocity control: accelerate toward desired velocity (any direction)
        let vel_error = desired_vel - current_vel;
        let error_magnitude = vel_error.magnitude();

        let new_vel = if error_magnitude > f32::EPSILON {
            // Calculate acceleration needed (limited by max_acc)
            let max_accel = self.perf.max_acc * dt;
            let accel_magnitude = error_magnitude.min(max_accel);
            let accel_dir = Vec2::new(vel_error.x / error_magnitude, vel_error.y / error_magnitude);
            let accel = accel_dir * accel_magnitude;

            // Store acceleration for state
            self.state.acc = Acceleration::from_vec2(accel / dt);

            // Apply acceleration to velocity
            Velocity::from_vec2(current_vel + accel).clamp_speed(self.perf.max_vel)
        } else {
            self.state.acc = Acceleration::zero();
            desired_velocity.clamp_speed(self.perf.max_vel)
        };

        self.state.vel = new_vel;

        // 2. Heading control: rotate toward desired heading (INDEPENDENT of velocity)
        if let Some(target_hdg) = desired_heading {
            let hdg_error = self.state.hdg.difference(target_hdg);
            let max_turn = self.perf.max_turn_rate * dt;
            let turn = hdg_error.clamp(-max_turn, max_turn);
            self.state.hdg += turn;
        }
        // If no desired_heading, heading stays unchanged (hover rotation)

        // 3. Integrate position
        self.state.pos += self.state.vel.scaled(dt);
    }

    fn state(&self) -> &State {
        &self.state
    }

    fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    fn perf(&self) -> &DronePerfFeatures {
        &self.perf
    }

    fn set_perf(&mut self, perf: DronePerfFeatures) {
        self.perf = perf;
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn effective_turn_rate(&self) -> f32 {
        self.perf.max_turn_rate
    }

    fn min_turn_rate(&self) -> f32 {
        MIN_TURN_RATE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn create_test_bounds() -> Bounds {
        Bounds::new(1000.0, 1000.0).unwrap()
    }

    fn create_test_platform() -> GenericPlatform {
        GenericPlatform::new(
            Position::new(500.0, 500.0),
            Heading::new(0.0),
            create_test_bounds(),
        )
    }

    #[test]
    fn test_new_platform_defaults() {
        let platform = create_test_platform();
        assert_eq!(platform.state().pos.x(), 500.0);
        assert_eq!(platform.state().pos.y(), 500.0);
        assert!((platform.state().hdg.radians() - 0.0).abs() < f32::EPSILON);
        assert_eq!(platform.perf().max_vel, DEFAULT_MAX_VELOCITY);
        assert_eq!(platform.perf().max_acc, DEFAULT_MAX_ACCELERATION);
        assert_eq!(platform.perf().max_turn_rate, DEFAULT_MAX_TURN_RATE);
    }

    #[test]
    fn test_apply_steering_straight_ahead() {
        let mut platform = create_test_platform();
        // Give it some initial velocity
        platform.state.vel = Velocity::from_heading_and_speed(Heading::new(0.0), 60.0);

        let initial_heading = platform.state().hdg.radians();
        platform.apply_steering(Heading::new(0.0), 60.0, 0.016);

        // Heading should not change when already aligned
        assert!((platform.state().hdg.radians() - initial_heading).abs() < 0.01);
    }

    #[test]
    fn test_apply_steering_turn_rate_limit() {
        let mut platform = create_test_platform();
        // Give it max velocity for max turn rate
        platform.state.vel = Velocity::from_heading_and_speed(Heading::new(0.0), DEFAULT_MAX_VELOCITY);

        // Try to turn 180 degrees instantly
        platform.apply_steering(Heading::new(PI), DEFAULT_MAX_VELOCITY, 0.016);

        // Should only turn by max_turn_rate * dt
        let max_expected_turn = DEFAULT_MAX_TURN_RATE * 0.016;
        let actual_turn = platform.state().hdg.radians().abs();
        assert!(actual_turn <= max_expected_turn + 0.001);
    }

    #[test]
    fn test_apply_steering_speed_limit() {
        let mut platform = create_test_platform();

        // Try to go faster than max velocity
        platform.apply_steering(Heading::new(0.0), DEFAULT_MAX_VELOCITY * 2.0, 1.0);

        // Speed should be clamped to max
        let speed = platform.state().vel.as_vec2().magnitude();
        assert!(speed <= DEFAULT_MAX_VELOCITY);
    }

    #[test]
    fn test_apply_steering_acceleration_limit() {
        let mut platform = create_test_platform();

        // Try to instantly reach max velocity from standstill
        platform.apply_steering(Heading::new(0.0), DEFAULT_MAX_VELOCITY, 0.016);

        // Should only accelerate by max_acc * dt
        let speed = platform.state().vel.as_vec2().magnitude();
        let max_expected_speed = DEFAULT_MAX_ACCELERATION * 0.016;
        assert!(speed <= max_expected_speed + 0.001);
    }

    #[test]
    fn test_effective_turn_rate_at_zero_speed() {
        let platform = create_test_platform();
        // With speed-based turn rate DISABLED, always returns max turn rate
        assert!((platform.effective_turn_rate() - DEFAULT_MAX_TURN_RATE).abs() < f32::EPSILON);
    }

    #[test]
    fn test_effective_turn_rate_at_max_speed() {
        let mut platform = create_test_platform();
        platform.state.vel = Velocity::from_heading_and_speed(Heading::new(0.0), DEFAULT_MAX_VELOCITY);

        // At max velocity, should return max turn rate
        assert!((platform.effective_turn_rate() - DEFAULT_MAX_TURN_RATE).abs() < 0.01);
    }

    #[test]
    fn test_effective_turn_rate_constant() {
        let mut platform = create_test_platform();

        // With speed-based turn rate DISABLED, turn rate is constant regardless of speed
        let half_speed = DEFAULT_MAX_VELOCITY / 2.0;
        platform.state.vel = Velocity::from_heading_and_speed(Heading::new(0.0), half_speed);

        // Should always be max turn rate
        assert!((platform.effective_turn_rate() - DEFAULT_MAX_TURN_RATE).abs() < 0.01);
    }

    #[test]
    fn test_position_integrates_correctly() {
        let mut platform = create_test_platform();
        // Use 10 m/s (reasonable speed in meters)
        platform.state.vel = Velocity::from_heading_and_speed(Heading::new(0.0), 10.0);

        let initial_x = platform.state().pos.x();
        platform.apply_steering(Heading::new(0.0), 10.0, 1.0);

        // Should move 10 meters in 1 second at 10 m/s
        let expected_x = initial_x + 10.0;
        assert!((platform.state().pos.x() - expected_x).abs() < 0.5);
    }
}
