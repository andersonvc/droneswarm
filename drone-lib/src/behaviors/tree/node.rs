//! Behavior tree node trait and context definitions.

use crate::types::{Bounds, DroneInfo, DronePerfFeatures, Heading, State, Velocity};

/// Result of a behavior node tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviorStatus {
    /// Behavior completed successfully.
    Success,
    /// Behavior failed.
    Failure,
    /// Behavior is still running.
    Running,
}

/// Context passed to behaviors during tick.
///
/// Contains all information a behavior needs to make decisions,
/// plus mutable output fields for the behavior to set.
#[derive(Debug)]
pub struct BehaviorContext<'a> {
    /// Current drone state (position, heading, velocity).
    pub state: &'a State,
    /// Information about other drones in the swarm.
    pub swarm: &'a [DroneInfo],
    /// Delta time for this tick in seconds.
    pub dt: f32,
    /// World bounds for distance calculations.
    pub bounds: &'a Bounds,
    /// Drone's unique ID.
    pub drone_id: usize,
    /// Whether this drone is the formation leader.
    /// Leaders don't avoid other swarm drones - followers avoid them.
    pub is_formation_leader: bool,
    /// Drone performance features (max velocity, acceleration, turn rate).
    pub perf: &'a DronePerfFeatures,

    // Output fields - behaviors can modify these
    /// Desired velocity vector (preferred for quadcopter control).
    /// If set, this takes precedence over desired_heading/desired_speed.
    pub desired_velocity: Option<Velocity>,
    /// Desired heading (set by navigation behaviors).
    /// For quadcopters: controls where the drone FACES (camera/weapons).
    /// For fixed-wing: controls direction of travel.
    pub desired_heading: Option<Heading>,
    /// Desired speed factor (0.0 to 1.0, set by navigation behaviors).
    /// Ignored if desired_velocity is set.
    pub desired_speed: f32,
    /// Collision urgency (0.0 to 1.0, set by avoidance behaviors).
    pub urgency: f32,
}

impl<'a> BehaviorContext<'a> {
    /// Create a new behavior context.
    pub fn new(
        state: &'a State,
        swarm: &'a [DroneInfo],
        bounds: &'a Bounds,
        drone_id: usize,
        dt: f32,
        perf: &'a DronePerfFeatures,
    ) -> Self {
        BehaviorContext {
            state,
            swarm,
            dt,
            bounds,
            drone_id,
            is_formation_leader: false,
            perf,
            desired_velocity: None,
            desired_heading: None,
            desired_speed: 1.0,
            urgency: 0.0,
        }
    }

    /// Create a new behavior context with formation leader flag.
    pub fn new_with_leader(
        state: &'a State,
        swarm: &'a [DroneInfo],
        bounds: &'a Bounds,
        drone_id: usize,
        dt: f32,
        is_formation_leader: bool,
        perf: &'a DronePerfFeatures,
    ) -> Self {
        BehaviorContext {
            state,
            swarm,
            dt,
            bounds,
            drone_id,
            is_formation_leader,
            perf,
            desired_velocity: None,
            desired_heading: None,
            desired_speed: 1.0,
            urgency: 0.0,
        }
    }

    /// Get effective desired velocity.
    ///
    /// Returns `desired_velocity` if set, otherwise converts `desired_heading`
    /// and `desired_speed` to a velocity vector.
    ///
    /// # Arguments
    /// * `max_speed` - Maximum speed for converting speed factor to actual speed
    pub fn get_desired_velocity(&self, max_speed: f32) -> Option<Velocity> {
        if let Some(vel) = self.desired_velocity {
            Some(vel)
        } else {
            self.desired_heading.map(|hdg| {
                Velocity::from_heading_and_speed(hdg, max_speed * self.desired_speed)
            })
        }
    }
}

/// Core behavior node trait.
///
/// All behavior tree nodes (composites, decorators, actions, conditions)
/// implement this trait.
pub trait BehaviorNode: Send + Sync + std::fmt::Debug {
    /// Tick the behavior node.
    ///
    /// This is called once per frame. The node should:
    /// - Check conditions and/or execute actions
    /// - Optionally modify the context (set desired_heading, etc.)
    /// - Return its status
    fn tick(&mut self, ctx: &mut BehaviorContext) -> BehaviorStatus;

    /// Reset the behavior state.
    ///
    /// Called when the behavior tree is reset or when a parent
    /// composite node needs to restart its children.
    fn reset(&mut self) {}

    /// Get the node's name for debugging.
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::fixtures::{create_test_bounds, create_test_state};

    fn create_test_context<'a>(
        state: &'a State,
        swarm: &'a [DroneInfo],
        bounds: &'a Bounds,
    ) -> BehaviorContext<'a> {
        // Use a leaked reference for test perf to simplify lifetimes
        let perf = Box::leak(Box::new(DronePerfFeatures::default()));
        BehaviorContext::new(state, swarm, bounds, 0, 0.016, perf)
    }

    #[test]
    fn test_context_defaults() {
        let state = create_test_state(500.0, 500.0);
        let bounds = create_test_bounds();
        let ctx = create_test_context(&state, &[], &bounds);

        assert!(ctx.desired_velocity.is_none());
        assert!(ctx.desired_heading.is_none());
        assert_eq!(ctx.desired_speed, 1.0);
        assert_eq!(ctx.urgency, 0.0);
    }

    #[test]
    fn test_get_desired_velocity_from_heading() {
        let state = create_test_state(500.0, 500.0);
        let bounds = create_test_bounds();
        let mut ctx = create_test_context(&state, &[], &bounds);

        // Set heading and speed
        ctx.desired_heading = Some(Heading::new(0.0)); // facing right
        ctx.desired_speed = 0.5;

        let vel = ctx.get_desired_velocity(20.0).unwrap();
        assert!((vel.speed() - 10.0).abs() < 0.01); // 50% of 20
    }

    #[test]
    fn test_get_desired_velocity_prefers_velocity() {
        let state = create_test_state(500.0, 500.0);
        let bounds = create_test_bounds();
        let mut ctx = create_test_context(&state, &[], &bounds);

        // Set both velocity and heading
        ctx.desired_velocity = Some(Velocity::new(5.0, 0.0));
        ctx.desired_heading = Some(Heading::new(std::f32::consts::PI)); // facing left
        ctx.desired_speed = 1.0;

        let vel = ctx.get_desired_velocity(20.0).unwrap();
        // Should use desired_velocity, not heading
        assert!((vel.as_vec2().x - 5.0).abs() < 0.01);
    }
}
