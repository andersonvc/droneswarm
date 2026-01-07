//! Steering behaviors for drone swarm coordination.
//!
//! This module provides composable steering behaviors that can be combined
//! to create complex swarm dynamics. Each behavior returns a steering vector
//! that influences the drone's movement.
//!
//! # Available Behaviors
//!
//! - [`separation`] - Collision avoidance by steering away from nearby drones
//! - [`velocity_obstacle`] - Predictive collision avoidance using velocity obstacles

pub mod separation;
pub mod velocity_obstacle;

pub use separation::{calculate_separation, SeparationConfig, COLLISION_RADIUS};
pub use velocity_obstacle::{
    calculate_velocity_obstacle, get_recommended_heading, AvoidanceResult, VelocityObstacleConfig,
};
