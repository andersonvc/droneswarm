//! Swarm module - swarm-level coordination behaviors.
//!
//! This module provides `SwarmBehavior` trait and implementations
//! for coordinating multiple agents at the swarm level.

pub mod formation;
pub mod traits;

pub use formation::{
    DroneStabilityMetrics, FormationBehavior, FormationCoordinator, FormationStabilityMetrics,
    FormationStatus, FormationType, compute_offset_route,
};
pub use traits::{DefaultSwarmBehavior, SwarmBehavior, WaypointPriorityBehavior};
