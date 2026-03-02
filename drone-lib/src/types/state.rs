use std::collections::VecDeque;
use std::sync::Arc;

use super::{units, Acceleration, Heading, Position, Velocity};
use crate::types::error::{DroneError, DroneResult};

/// Drone objective – each variant carries only the data it needs.
#[derive(Debug, Clone, Default)]
pub enum Objective {
    /// Inactive state, no movement.
    #[default]
    Sleep,
    /// Navigate directly to waypoints, stop when done.
    ReachWaypoint { waypoints: VecDeque<Position> },
    /// Follow a route indefinitely, looping back to start.
    FollowRoute { waypoints: VecDeque<Position>, route: Arc<[Position]> },
    /// Track moving targets.
    FollowTarget { targets: Vec<Position> },
    /// Hold position or patrol area.
    Loiter { center: Option<Position> },
}

/// Drone state containing position, heading, velocity, and acceleration
#[derive(Debug, Clone, Copy)]
pub struct State {
    pub hdg: Heading,
    pub pos: Position,
    pub vel: Velocity,
    pub acc: Acceleration,
}

impl Default for State {
    fn default() -> Self {
        State {
            hdg: Heading::default(),
            pos: Position::default(),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        }
    }
}

/// Information about a drone that can be shared with other drones
#[derive(Debug, Clone, Copy)]
pub struct DroneInfo {
    pub uid: usize,
    pub pos: Position,
    pub hdg: Heading,
    pub vel: Velocity,
    /// Whether this drone is a formation leader (doesn't do ORCA avoidance)
    pub is_formation_leader: bool,
}

impl DroneInfo {
    pub fn new(uid: usize, state: &State) -> Self {
        DroneInfo {
            uid,
            pos: state.pos,
            hdg: state.hdg,
            vel: state.vel,
            is_formation_leader: false,
        }
    }

    /// Create DroneInfo with leader status
    pub fn new_with_leader(uid: usize, state: &State, is_formation_leader: bool) -> Self {
        DroneInfo {
            uid,
            pos: state.pos,
            hdg: state.hdg,
            vel: state.vel,
            is_formation_leader,
        }
    }
}

/// Filter swarm to exclude a specific drone by ID.
///
/// Returns an iterator over all `DroneInfo` entries except the one with `self_id`.
pub fn neighbors_excluding(swarm: &[DroneInfo], self_id: usize) -> impl Iterator<Item = &DroneInfo> {
    swarm.iter().filter(move |d| d.uid != self_id)
}

/// Drone performance features (flight parameters)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DronePerfFeatures {
    pub max_vel: f32,
    pub max_acc: f32,
    pub max_turn_rate: f32,
}

impl DronePerfFeatures {
    /// Create new performance features with validation
    ///
    /// # Errors
    /// Returns `DroneError::InvalidFlightParam` if any parameter is <= 0
    pub fn new(max_vel: f32, max_acc: f32, max_turn_rate: f32) -> DroneResult<Self> {
        if max_vel <= 0.0 {
            return Err(DroneError::InvalidFlightParam {
                param: "max_vel",
                value: max_vel,
            });
        }
        if max_acc <= 0.0 {
            return Err(DroneError::InvalidFlightParam {
                param: "max_acc",
                value: max_acc,
            });
        }
        if max_turn_rate <= 0.0 {
            return Err(DroneError::InvalidFlightParam {
                param: "max_turn_rate",
                value: max_turn_rate,
            });
        }
        Ok(DronePerfFeatures {
            max_vel,
            max_acc,
            max_turn_rate,
        })
    }

    /// Create without validation (for internal use with known-good values).
    ///
    /// # Panics
    /// This function does not panic, but passing non-positive values will cause
    /// undefined simulation behavior (division by zero, incorrect physics).
    #[inline]
    pub fn new_unchecked(max_vel: f32, max_acc: f32, max_turn_rate: f32) -> Self {
        DronePerfFeatures {
            max_vel,
            max_acc,
            max_turn_rate,
        }
    }
}

impl Default for DronePerfFeatures {
    fn default() -> Self {
        DronePerfFeatures {
            max_vel: units::DEFAULT_MAX_VELOCITY,
            max_acc: units::DEFAULT_MAX_ACCELERATION,
            max_turn_rate: units::DEFAULT_MAX_TURN_RATE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drone_perf_features_valid() {
        let perf = DronePerfFeatures::new(100.0, 10.0, 2.0);
        assert!(perf.is_ok());
    }

    #[test]
    fn test_drone_perf_features_invalid() {
        assert!(DronePerfFeatures::new(0.0, 10.0, 2.0).is_err());
        assert!(DronePerfFeatures::new(100.0, -1.0, 2.0).is_err());
        assert!(DronePerfFeatures::new(100.0, 10.0, 0.0).is_err());
    }

    #[test]
    fn test_objective_default() {
        let obj = Objective::default();
        assert!(matches!(obj, Objective::Sleep));
    }
}
