//! Shared test fixtures for drone-lib tests.

#[cfg(test)]
pub(crate) mod fixtures {
    use crate::types::{Acceleration, Bounds, DroneInfo, Heading, Position, State, Velocity};

    /// Create standard 1000x1000 test bounds.
    pub fn create_test_bounds() -> Bounds {
        Bounds::new(1000.0, 1000.0).unwrap()
    }

    /// Create smaller 500x500 test bounds.
    pub fn create_test_bounds_small() -> Bounds {
        Bounds::new(500.0, 500.0).unwrap()
    }

    /// Create a test state at position (x, y) with heading 0 and zero velocity.
    pub fn create_test_state(x: f32, y: f32) -> State {
        State {
            pos: Position::new(x, y),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        }
    }

    /// Create a test state with specified heading and speed.
    pub fn create_test_state_with_vel(x: f32, y: f32, hdg: f32, speed: f32) -> State {
        State {
            pos: Position::new(x, y),
            hdg: Heading::new(hdg),
            vel: Velocity::from_heading_and_speed(Heading::new(hdg), speed),
            acc: Acceleration::zero(),
        }
    }

    /// Create a DroneInfo at position (x, y) with zero velocity.
    pub fn create_drone_info(uid: usize, x: f32, y: f32) -> DroneInfo {
        DroneInfo {
            uid,
            pos: Position::new(x, y),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
        }
    }
}
