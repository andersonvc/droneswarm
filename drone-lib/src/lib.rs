//! Drone swarm simulation library.
//!
//! Provides modular, composable components for autonomous drone swarm simulation
//! following MOSA (Modular Open Systems Approach) principles:
//!
//! - **[`agent`]** — Composed autonomous agents (`DroneAgent`) with pluggable
//!   platform, mission, and behavior tree components.
//! - **[`behaviors`]** — Steering behaviors (separation, velocity obstacles, ORCA,
//!   APF) and a behavior tree framework for composing them.
//! - **[`missions`]** — Task management, waypoint navigation, and Hermite-spline
//!   path planning.
//! - **[`platform`]** — Kinematic platform abstraction (generic quadcopter model).
//! - **[`swarm`]** — Swarm-level coordination: formation flying, stability metrics,
//!   and slot assignment.
//! - **[`messages`]** — Inter-drone message types for formation commands and
//!   velocity consensus.
//! - **[`types`]** — Core value types (position, heading, velocity, bounds, errors).

#[cfg(test)]
mod test_utils;

pub mod agent;
pub mod behaviors;
pub mod messages;
pub mod missions;
pub mod platform;
pub mod swarm;
pub mod types;

// Agent module exports
pub use agent::{Drone, DroneAgent, FormationApproachMode};

// Behavior module exports
pub use behaviors::{
    calculate_separation, calculate_velocity_obstacle, get_recommended_heading, APFConfig,
    AvoidanceResult, SeparationConfig, VelocityObstacleConfig, COLLISION_RADIUS,
};

// Type exports
pub use types::{
    Acceleration, Bounds, Heading, Position, Vec2, Velocity,
    DroneInfo, DronePerfFeatures, Objective, State,
    DroneError, DroneResult,
    normalize_angle,
};

// Platform module exports
pub use platform::{
    GenericPlatform, Platform, DEFAULT_MAX_ACCELERATION, DEFAULT_MAX_TURN_RATE,
    DEFAULT_MAX_VELOCITY, MIN_TURN_RATE,
};

// Missions module exports
pub use missions::{
    AgentCommand, CommandQueue, Mission, MissionStatus, ParameterizedPath, PathPlanner, Task,
    WaypointMission,
};

// Swarm module exports
pub use swarm::{
    DefaultSwarmBehavior, DroneStabilityMetrics, FormationBehavior, FormationCoordinator,
    FormationStabilityMetrics, FormationStatus, FormationType, SwarmBehavior,
    WaypointPriorityBehavior, compute_offset_route,
};

// Comms module exports
pub use messages::{FormationCommand, FormationSlot, PathProgress, VelocityConsensus};
