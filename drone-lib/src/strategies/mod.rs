//! Swarm-level strategies that decompose goals into drone-level task assignments.
//!
//! Strategies are the middle of the operational hierarchy:
//! **Objective → Strategy → Task → Safety → Platform**
//!
//! They monitor swarm state, allocate drones to tasks, and reallocate when
//! tasks complete, fail, or the situation changes.
//!
//! # Available Strategies
//!
//! - [`defend_area`] — Protect a zone with orbiting defenders and proactive interceptors
//! - [`attack_zone`] — Coordinated assault on a target area
//! - [`patrol_perimeter`] — Distributed surveillance coverage along a route

pub mod attack_zone;
pub mod defend_area;
pub mod patrol_perimeter;

use crate::doctrine::DoctrineMode;
use crate::types::{Bounds, DroneInfo, Position, Velocity};

/// State of a drone as seen by a strategy.
#[derive(Debug, Clone)]
pub struct StrategyDroneState {
    /// Drone's unique ID.
    pub id: usize,
    /// Current position.
    pub pos: Position,
    /// Current velocity.
    pub vel: Velocity,
    /// Whether the drone is available for (re)assignment.
    /// True when the drone has no active task, or its task completed/failed.
    pub available: bool,
}

/// A task assignment produced by a strategy.
#[derive(Debug, Clone)]
pub enum TaskAssignment {
    /// Assign an intercept task to pursue a target drone.
    Intercept { drone_id: usize, target_id: usize },
    /// Assign an intercept-group task: fly into a cluster and detonate for splash damage.
    InterceptGroup { drone_id: usize },
    /// Assign an attack task toward a fixed position.
    Attack { drone_id: usize, target: Position },
    /// Assign a defend task (orbit center, engage threats in zone).
    Defend {
        drone_id: usize,
        center: Position,
        orbit_radius: f32,
        engage_radius: f32,
    },
    /// Assign a patrol task along waypoints.
    Patrol {
        drone_id: usize,
        waypoints: Vec<Position>,
        loiter_duration: f32,
    },
    /// Assign a loiter (hold position) task.
    Loiter { drone_id: usize, position: Position },
    /// Assign a patrol task to a formation squad.
    /// The leader gets the patrol route; followers form up around the leader.
    PatrolFormation {
        leader_id: usize,
        follower_ids: Vec<usize>,
        waypoints: Vec<Position>,
        loiter_duration: f32,
    },
}

/// Swarm-level strategy that decomposes goals into drone-level task assignments.
///
/// Each tick, the strategy examines its drones' states and the full swarm,
/// then returns task assignments for any drones that need (re)allocation.
/// Assignments are only emitted when something changes — not every tick.
pub trait SwarmStrategy: Send + Sync + std::fmt::Debug {
    /// Tick the strategy. Returns task assignments to apply this tick.
    ///
    /// * `own_drones` — state of drones managed by this strategy (filtered by caller)
    /// * `all_drones` — full swarm info (for threat detection, etc.)
    /// * `bounds` — world bounds for distance calculations
    /// * `dt` — delta time in seconds
    fn tick(
        &mut self,
        own_drones: &[StrategyDroneState],
        all_drones: &[DroneInfo],
        bounds: &Bounds,
        dt: f32,
    ) -> Vec<TaskAssignment>;

    /// The drone IDs this strategy manages.
    fn drone_ids(&self) -> &[usize];

    /// Strategy name for debugging.
    fn name(&self) -> &str;

    /// Whether the strategy has completed its objective.
    fn is_complete(&self) -> bool;

    /// Update target information. Default no-op; used by the Objective layer.
    fn update_targets(
        &mut self,
        _friendly_targets: &[Position],
        _enemy_targets: &[Position],
    ) {}

    /// Group ID this strategy belongs to, if applicable.
    fn group(&self) -> Option<u32> {
        None
    }

    /// Set doctrine mode (only meaningful for doctrine strategies).
    fn set_doctrine_mode(&mut self, _mode: DoctrineMode) {}

    /// Get defend/attack drone counts (only meaningful for doctrine strategies).
    fn defend_attack_counts(&self) -> Option<(usize, usize)> {
        None
    }
}
