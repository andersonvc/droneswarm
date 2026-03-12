//! Missions module - task management and path planning.
//!
//! This module provides:
//! - `Mission` trait for task execution
//! - `WaypointMission` for waypoint-based navigation
//! - `PathPlanner` for smoothed path following
//! - `CommandQueue` for decoupled command processing

pub mod command;
pub mod parameterized_path;
pub mod planner;
pub mod task;
pub mod traits;

pub use command::{AgentCommand, CommandQueue};
pub use parameterized_path::ParameterizedPath;
pub use planner::{ApproachGateConfig, PathPlanner};
pub use task::{Task, WaypointMission};
pub use traits::{Mission, MissionStatus};
