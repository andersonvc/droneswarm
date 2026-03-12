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
//!
//! # Behavior Tree Support
//!
//! - [`tree`] - Behavior tree infrastructure (nodes, composites)
//! - [`actions`] - Action nodes for seeking and avoiding
//! - [`conditions`] - Condition nodes for checking state
//! - [`factory`] - Factory functions for creating standard behavior trees

pub mod actions;
pub mod conditions;
pub mod factory;
pub mod orca;
pub mod potential_field;
pub mod separation;
pub mod tree;
pub mod velocity_obstacle;

// Core behavior calculations
pub use separation::{calculate_separation, SeparationConfig, COLLISION_RADIUS};
pub use velocity_obstacle::{
    calculate_velocity_obstacle, get_recommended_heading, AvoidanceResult, VelocityObstacleConfig,
};
pub use potential_field::{
    calculate_apf_combined, calculate_apf_from_obstacles, calculate_apf_from_swarm,
    calculate_apf_repulsion, apply_apf_to_heading, APFConfig, Obstacle,
};
pub use orca::{
    compute_orca_constraints, compute_orca_velocity, find_optimal_velocity,
    velocity_to_heading_speed, HalfPlane, ORCAConfig,
};

// Behavior tree infrastructure
pub use tree::{BehaviorContext, BehaviorNode, BehaviorStatus, Parallel, Selector, Sequence};

// Action nodes
pub use actions::{APFAvoid, ORCAAvoid, SeekWaypoint, SeparationAvoid, VelocityObstacleAvoid};

// Condition nodes
pub use conditions::{CollisionImminent, HasTarget};

// Factory functions
pub use factory::{
    create_avoidance_only_bt, create_fixed_wing_bt, create_orca_avoidance_only_bt,
    create_orca_bt, create_orca_bt_with_apf, create_seek_only_bt,
};
