//! Platform trait definitions for drone physical characteristics and kinematics.

use crate::types::{Bounds, DronePerfFeatures, Heading, State, Velocity};

/// Platform trait - defines physical drone characteristics and kinematics.
///
/// A platform represents the physical vehicle with its kinematic constraints.
/// It handles steering commands and integrates physics, but does not make
/// decisions about where to go (that's the job of behaviors and missions).
pub trait Platform: Send + Sync + std::fmt::Debug {
    /// Apply a velocity-based steering command (quadcopter style).
    ///
    /// The drone will accelerate toward the desired velocity while
    /// optionally rotating to face desired_heading (if provided).
    /// Velocity and heading are controlled independently.
    ///
    /// # Arguments
    /// * `desired_velocity` - Target velocity vector (any direction)
    /// * `desired_heading` - Optional heading to face (for camera/weapon pointing)
    /// * `dt` - Delta time in seconds
    fn apply_velocity_steering(
        &mut self,
        desired_velocity: Velocity,
        desired_heading: Option<Heading>,
        dt: f32,
    );

    /// Apply a heading-based steering command (legacy fixed-wing style).
    ///
    /// Takes a desired heading and speed, applies physical constraints
    /// (turn rate limits, acceleration limits), and integrates position.
    /// Velocity is coupled to heading direction.
    ///
    /// # Arguments
    /// * `desired_heading` - The heading we want to achieve
    /// * `desired_speed` - The speed we want to achieve
    /// * `dt` - Delta time in seconds
    fn apply_steering(&mut self, desired_heading: Heading, desired_speed: f32, dt: f32) {
        // Default: convert heading+speed to velocity and call velocity steering
        let vel = Velocity::from_heading_and_speed(desired_heading, desired_speed);
        self.apply_velocity_steering(vel, Some(desired_heading), dt);
    }

    /// Get current kinematic state (position, heading, velocity, acceleration).
    fn state(&self) -> &State;

    /// Get mutable reference to state (for direct manipulation if needed).
    fn state_mut(&mut self) -> &mut State;

    /// Get performance parameters (max velocity, acceleration, turn rate).
    fn perf(&self) -> &DronePerfFeatures;

    /// Set performance parameters.
    fn set_perf(&mut self, perf: DronePerfFeatures);

    /// Get world bounds reference.
    fn bounds(&self) -> &Bounds;

    /// Calculate effective turn rate at current speed.
    ///
    /// For fixed-wing aircraft, turn rate scales with velocity squared
    /// because control authority depends on airspeed.
    fn effective_turn_rate(&self) -> f32;

    /// Get minimum turn rate (even at zero velocity).
    fn min_turn_rate(&self) -> f32;
}
