//! Seek waypoint action node.

use crate::behaviors::tree::node::{BehaviorContext, BehaviorNode, BehaviorStatus};
use crate::types::Heading;

/// Seek waypoint action: sets desired heading toward a target.
///
/// This is a simple action that sets the context's desired_heading
/// based on an externally-provided heading (typically from a Mission).
/// It always succeeds if a heading is available.
#[derive(Debug, Default)]
pub struct SeekWaypoint {
    /// Optional target heading (can be set externally or use context).
    target_heading: Option<Heading>,
}

impl SeekWaypoint {
    /// Create a new seek waypoint action.
    pub fn new() -> Self {
        SeekWaypoint {
            target_heading: None,
        }
    }

    /// Create with a specific target heading.
    pub fn with_heading(heading: Heading) -> Self {
        SeekWaypoint {
            target_heading: Some(heading),
        }
    }

    /// Set the target heading.
    pub fn set_heading(&mut self, heading: Option<Heading>) {
        self.target_heading = heading;
    }
}

impl BehaviorNode for SeekWaypoint {
    fn tick(&mut self, ctx: &mut BehaviorContext) -> BehaviorStatus {
        // Use explicitly set heading, or fall back to context's desired heading
        let heading = self.target_heading.or(ctx.desired_heading);

        if let Some(hdg) = heading {
            ctx.desired_heading = Some(hdg);
            // Only set default speed if not already set by caller (e.g., formation logic)
            // This preserves formation speed overrides for smooth station-keeping
            if ctx.desired_speed <= 0.0 {
                ctx.desired_speed = 1.0; // Full speed toward target
            }
            BehaviorStatus::Success
        } else {
            // No heading available - this is expected when idle
            BehaviorStatus::Failure
        }
    }

    fn reset(&mut self) {
        self.target_heading = None;
    }

    fn name(&self) -> &str {
        "SeekWaypoint"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Acceleration, Bounds, DronePerfFeatures, Position, State, Velocity};

    fn create_test_context<'a>(state: &'a State, bounds: &'a Bounds) -> BehaviorContext<'a> {
        let perf = Box::leak(Box::new(DronePerfFeatures::default()));
        BehaviorContext::new(state, &[], bounds, 0, 0.016, perf)
    }

    #[test]
    fn test_seek_with_explicit_heading() {
        let state = State {
            pos: Position::new(500.0, 500.0),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(1000.0, 1000.0).unwrap();
        let mut ctx = create_test_context(&state, &bounds);

        let mut seek = SeekWaypoint::with_heading(Heading::new(1.0));
        let status = seek.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Success);
        assert!(ctx.desired_heading.is_some());
        assert!((ctx.desired_heading.unwrap().radians() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_seek_uses_context_heading() {
        let state = State {
            pos: Position::new(500.0, 500.0),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(1000.0, 1000.0).unwrap();
        let mut ctx = create_test_context(&state, &bounds);
        ctx.desired_heading = Some(Heading::new(2.0));

        let mut seek = SeekWaypoint::new();
        let status = seek.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Success);
        assert!((ctx.desired_heading.unwrap().radians() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_seek_fails_without_heading() {
        let state = State {
            pos: Position::new(500.0, 500.0),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(1000.0, 1000.0).unwrap();
        let mut ctx = create_test_context(&state, &bounds);

        let mut seek = SeekWaypoint::new();
        let status = seek.tick(&mut ctx);

        assert_eq!(status, BehaviorStatus::Failure);
    }
}
