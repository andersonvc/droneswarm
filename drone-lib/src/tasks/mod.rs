//! Task state machines for the operational control hierarchy.
//!
//! Tasks are the "sequencer" layer (3T architecture) — they select which
//! behavior to activate based on the current phase of execution, and
//! transition phases based on feedback from the safety layer.
//!
//! # Task → Behavior → Safety → Platform
//!
//! 1. Task produces a desired velocity (via its active behavior)
//! 2. Safety layer adjusts velocity for collision avoidance (ORCA + APF)
//! 3. Safety layer returns feedback (urgency, threat direction)
//! 4. Task uses feedback to transition phases (e.g., pursue → evade)
//! 5. Platform applies the safe velocity

pub mod intercept;
pub mod intercept_group;
pub mod attack;
pub mod defend;
pub mod evade;
pub mod loiter;
pub mod patrol;

use crate::types::{Bounds, DroneInfo, Heading, State, Vec2};

/// Feedback from the safety layer to the task after collision avoidance.
#[derive(Debug, Clone, Copy)]
pub struct SafetyFeedback {
    /// Urgency from collision avoidance (0.0 = no threat, 1.0 = maximum).
    pub urgency: f32,
    /// Direction toward the primary threat (if any), in world coordinates.
    pub threat_direction: Option<Vec2>,
    /// The velocity after safety layer adjustment.
    pub safe_velocity: Vec2,
}

/// Output from a task's tick — what the task wants the drone to do.
#[derive(Debug, Clone)]
pub struct TaskOutput {
    /// Desired velocity vector before safety layer adjustment.
    pub desired_velocity: Vec2,
    /// Desired facing heading (may differ from velocity direction).
    pub desired_heading: Option<Heading>,
    /// Whether the drone should detonate this tick.
    pub detonate: bool,
    /// Drone IDs to exclude from collision avoidance.
    /// Used by intercept tasks so the interceptor can close on targets.
    pub exclude_from_ca: Vec<usize>,
}

/// Status of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is actively executing.
    Active,
    /// Task completed successfully (e.g., target destroyed).
    Complete,
    /// Task failed (e.g., target lost).
    Failed,
}

/// Trait for task state machines.
///
/// A task selects which behavior to run based on its current phase,
/// produces a desired velocity, and transitions phases based on
/// feedback from the safety layer.
pub trait DroneTask: Send + Sync + std::fmt::Debug {
    /// Tick the task: assess situation, select behavior, produce desired velocity.
    ///
    /// The task has access to the drone's own state and the full swarm info
    /// so it can look up targets, scan for threats, etc.
    fn tick(
        &mut self,
        state: &State,
        swarm: &[DroneInfo],
        bounds: &Bounds,
        perf: &crate::types::DronePerfFeatures,
        dt: f32,
    ) -> TaskOutput;

    /// Process feedback from the safety layer to drive phase transitions.
    fn process_feedback(&mut self, feedback: &SafetyFeedback);

    /// Current phase name (for debugging and rendering).
    fn phase_name(&self) -> &str;

    /// Current task status.
    fn status(&self) -> TaskStatus;

    /// Task name.
    fn name(&self) -> &str;
}
