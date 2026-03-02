//! Collision detection condition nodes.

use crate::behaviors::tree::node::{BehaviorContext, BehaviorNode, BehaviorStatus};

/// Condition: is collision imminent?
///
/// Returns Success if any drone is within the threshold distance,
/// Failure otherwise.
#[derive(Debug)]
pub struct CollisionImminent {
    threshold_distance: f32,
}

impl CollisionImminent {
    /// Create a new collision imminent condition.
    ///
    /// # Arguments
    /// * `threshold_distance` - Distance at which collision is considered imminent
    pub fn new(threshold_distance: f32) -> Self {
        CollisionImminent { threshold_distance }
    }
}

impl Default for CollisionImminent {
    fn default() -> Self {
        CollisionImminent {
            threshold_distance: 5.0, // Default: 5 units
        }
    }
}

impl BehaviorNode for CollisionImminent {
    fn tick(&mut self, ctx: &mut BehaviorContext) -> BehaviorStatus {
        for other in crate::types::neighbors_excluding(ctx.swarm, ctx.drone_id) {
            let dist = ctx.bounds.distance(
                ctx.state.pos.as_vec2(),
                other.pos.as_vec2(),
            );

            if dist < self.threshold_distance {
                return BehaviorStatus::Success; // Collision is imminent
            }
        }
        BehaviorStatus::Failure // No imminent collision
    }

    fn name(&self) -> &str {
        "CollisionImminent"
    }
}

/// Condition: has active waypoint/target?
///
/// Returns Success if there's a desired heading set in context,
/// Failure otherwise.
#[derive(Debug, Default)]
pub struct HasTarget;

impl HasTarget {
    /// Create a new has target condition.
    pub fn new() -> Self {
        HasTarget
    }
}

impl BehaviorNode for HasTarget {
    fn tick(&mut self, ctx: &mut BehaviorContext) -> BehaviorStatus {
        if ctx.desired_heading.is_some() {
            BehaviorStatus::Success
        } else {
            BehaviorStatus::Failure
        }
    }

    fn name(&self) -> &str {
        "HasTarget"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::fixtures::{create_test_bounds, create_test_state};
    use crate::types::{DronePerfFeatures, DroneInfo, Heading, Position, Velocity};

    fn test_perf() -> &'static DronePerfFeatures {
        Box::leak(Box::new(DronePerfFeatures::default()))
    }

    #[test]
    fn test_collision_imminent_no_drones() {
        let state = create_test_state(500.0, 500.0);
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        let mut cond = CollisionImminent::new(50.0);
        assert_eq!(cond.tick(&mut ctx), BehaviorStatus::Failure);
    }

    #[test]
    fn test_collision_imminent_drone_far() {
        let state = create_test_state(500.0, 500.0);
        let bounds = create_test_bounds();

        let far_drone = DroneInfo {
            uid: 1,
            pos: Position::new(700.0, 500.0), // 200 units away
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
        };
        let swarm = [far_drone];
        let mut ctx = BehaviorContext::new(&state, &swarm, &bounds, 0, 0.016, test_perf());

        let mut cond = CollisionImminent::new(50.0);
        assert_eq!(cond.tick(&mut ctx), BehaviorStatus::Failure);
    }

    #[test]
    fn test_collision_imminent_drone_close() {
        let state = create_test_state(500.0, 500.0);
        let bounds = create_test_bounds();

        let close_drone = DroneInfo {
            uid: 1,
            pos: Position::new(520.0, 500.0), // 20 units away
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
        };
        let swarm = [close_drone];
        let mut ctx = BehaviorContext::new(&state, &swarm, &bounds, 0, 0.016, test_perf());

        let mut cond = CollisionImminent::new(50.0);
        assert_eq!(cond.tick(&mut ctx), BehaviorStatus::Success);
    }

    #[test]
    fn test_collision_imminent_excludes_self() {
        let state = create_test_state(500.0, 500.0);
        let bounds = create_test_bounds();

        // Same drone ID as context
        let self_drone = DroneInfo {
            uid: 0,
            pos: Position::new(500.0, 500.0),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
        };
        let swarm = [self_drone];
        let mut ctx = BehaviorContext::new(&state, &swarm, &bounds, 0, 0.016, test_perf());

        let mut cond = CollisionImminent::new(50.0);
        assert_eq!(cond.tick(&mut ctx), BehaviorStatus::Failure);
    }

    #[test]
    fn test_has_target_with_heading() {
        let state = create_test_state(500.0, 500.0);
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());
        ctx.desired_heading = Some(Heading::new(0.0));

        let mut cond = HasTarget::new();
        assert_eq!(cond.tick(&mut ctx), BehaviorStatus::Success);
    }

    #[test]
    fn test_has_target_without_heading() {
        let state = create_test_state(500.0, 500.0);
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        let mut cond = HasTarget::new();
        assert_eq!(cond.tick(&mut ctx), BehaviorStatus::Failure);
    }
}
