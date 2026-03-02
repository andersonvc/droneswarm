//! Mission trait definitions for task execution and waypoint management.

use crate::types::{Heading, Position, State};

/// Status of mission execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionStatus {
    /// Mission is actively being executed.
    Active,
    /// Mission has completed successfully.
    Completed,
    /// No active mission (idle state).
    Idle,
}

/// Mission trait - defines task execution and waypoint management.
///
/// A mission represents a high-level goal for the drone. It manages
/// waypoints, determines desired headings, and tracks completion status.
pub trait Mission: Send + Sync + std::fmt::Debug {
    /// Get desired heading for current mission state.
    ///
    /// Returns `None` if the mission has no current target (idle/completed).
    fn get_desired_heading(&self, current_state: &State) -> Option<Heading>;

    /// Get desired speed factor for current mission state.
    ///
    /// Returns a value from 0.0 (stopped) to 1.0 (full speed).
    fn get_desired_speed_factor(&self) -> f32;

    /// Update mission state based on current position.
    ///
    /// This should be called each tick to process waypoint arrivals
    /// and update mission status.
    ///
    /// # Arguments
    /// * `current_state` - Current drone state
    /// * `clearance` - Distance threshold for waypoint arrival
    fn update(&mut self, current_state: &State, clearance: f32);

    /// Get mission status.
    fn status(&self) -> MissionStatus;

    /// Get current target position (for visualization).
    fn current_target(&self) -> Option<Position>;

    /// Is path smoothing enabled for this mission?
    fn uses_path_smoothing(&self) -> bool;

    /// Reset the mission to idle state.
    fn reset(&mut self);
}
