//! Physical constants and realistic drone parameters in SI units (meters, m/s, m/s²).
//!
//! These values represent a small quadcopter swarm operating in a real-world environment.
//! The wasm-lib layer translates between these units and pixel coordinates for rendering.

// =============================================================================
// Platform Defaults
// =============================================================================

/// Default physical radius of a drone (meters).
/// Represents the collision envelope for a small quadcopter.
pub const DRONE_RADIUS: f32 = 0.4;

/// Default maximum velocity in meters per second.
/// Typical for a small quadcopter in coordinated flight.
pub const DEFAULT_MAX_VELOCITY: f32 = 20.0;

/// Default maximum acceleration in meters per second squared.
/// Quadcopters can accelerate quickly.
pub const DEFAULT_MAX_ACCELERATION: f32 = 12.0;

/// Default maximum turn rate in radians per second.
/// Quadcopters can turn very quickly (about 180°/s).
pub const DEFAULT_MAX_TURN_RATE: f32 = 3.0;

/// Minimum turn rate (radians/second) - for legacy fixed-wing mode.
pub const MIN_TURN_RATE: f32 = 1.0;

// =============================================================================
// Navigation
// =============================================================================

/// Default waypoint arrival threshold (meters).
/// Drone considers waypoint reached when within this distance.
pub const WAYPOINT_CLEARANCE: f32 = 5.0;

// =============================================================================
// World Defaults
// =============================================================================

/// Default world dimensions.
pub mod world {
    /// Default world width (meters).
    pub const DEFAULT_WIDTH: f32 = 2500.0;

    /// Default world height (meters).
    pub const DEFAULT_HEIGHT: f32 = 2500.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_are_positive() {
        assert!(DRONE_RADIUS > 0.0);
        assert!(DEFAULT_MAX_VELOCITY > 0.0);
        assert!(DEFAULT_MAX_ACCELERATION > 0.0);
        assert!(DEFAULT_MAX_TURN_RATE > 0.0);
        assert!(WAYPOINT_CLEARANCE > 0.0);
    }
}
