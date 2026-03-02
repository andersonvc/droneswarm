//! Core drone trait defining the interface for autonomous agents.

use crate::types::{DroneInfo, DronePerfFeatures, Objective, State};

/// Core trait for autonomous drone agents.
///
/// Defines the interface that all drone implementations must satisfy.
/// Currently implemented by [`super::DroneAgent`].
pub trait Drone: Send + Sync + std::fmt::Debug {
    /// Returns this drone's UID.
    fn uid(&self) -> usize;

    /// Update drone state, with access to all other drones' info.
    fn state_update(&mut self, dt: f32, swarm: &[DroneInfo]);

    /// Update the drone's objective.
    fn set_objective(&mut self, objective: Objective);

    /// Clear the drone's current objective.
    fn clear_objective(&mut self);

    /// Get immutable reference to drone state.
    fn state(&self) -> &State;

    /// Get this drone's info for sharing with others.
    fn get_info(&self) -> DroneInfo;

    /// Set flight parameters.
    ///
    /// Parameters should be created via [`DronePerfFeatures::new`] to ensure
    /// validation. Using [`DronePerfFeatures::new_unchecked`] with invalid
    /// values may cause undefined simulation behavior.
    fn set_flight_params(&mut self, params: DronePerfFeatures);
}
