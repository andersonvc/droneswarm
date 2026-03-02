//! Agent module - composed autonomous agents.
//!
//! This module provides `DroneAgent`, which composes Platform + Mission + BehaviorTree
//! into a complete autonomous agent following MOSA principles.
//!
//! The [`Drone`] trait defines the core interface for all drone implementations.

pub mod drone_agent;
pub mod formation_navigator;
pub mod traits;

pub use drone_agent::{DroneAgent, FormationApproachMode};
pub use formation_navigator::FormationNavigator;
pub use traits::Drone;
