//! Behavior tree factory for creating standard behavior trees.

use super::actions::{APFAvoid, ORCAAvoid, SeekWaypoint, SeparationAvoid, VelocityObstacleAvoid};
use super::tree::composite::{Selector, Sequence};
use super::tree::node::BehaviorNode;
use super::{APFConfig, ORCAConfig, SeparationConfig, VelocityObstacleConfig};

/// Create the default fixed-wing behavior tree.
///
/// Structure:
/// ```text
/// Selector (root)
/// ├── Sequence (navigate_with_avoidance)
/// │   ├── SeekWaypoint           # Sets ctx.desired_heading from mission
/// │   └── VelocityObstacleAvoid  # Adjusts heading, sets urgency
/// └── SeparationAvoid            # Fallback reactive avoidance
/// ```
///
/// Logic:
/// 1. If there's a waypoint to seek, navigate toward it with VO avoidance
/// 2. If not, fall back to separation-only behavior (for idle drones)
pub fn create_fixed_wing_bt(
    vo_config: VelocityObstacleConfig,
    sep_config: SeparationConfig,
) -> Box<dyn BehaviorNode> {
    Box::new(Selector::new(
        "root",
        vec![
            // Primary: navigate with avoidance
            Box::new(Sequence::new(
                "navigate_with_avoidance",
                vec![
                    Box::new(SeekWaypoint::new()),
                    Box::new(VelocityObstacleAvoid::with_config(vo_config)),
                ],
            )),
            // Fallback: separation only (for idle/emergency)
            Box::new(SeparationAvoid::with_config(sep_config)),
        ],
    ))
}

/// Create a simple seek-only behavior tree (no avoidance).
///
/// Useful for testing or scenarios where avoidance is handled externally.
pub fn create_seek_only_bt() -> Box<dyn BehaviorNode> {
    Box::new(SeekWaypoint::new())
}

/// Create an avoidance-only behavior tree (no seeking).
///
/// Useful for testing or stationary drones that only need collision avoidance.
pub fn create_avoidance_only_bt(
    vo_config: VelocityObstacleConfig,
    sep_config: SeparationConfig,
) -> Box<dyn BehaviorNode> {
    Box::new(Selector::new(
        "avoidance_only",
        vec![
            Box::new(VelocityObstacleAvoid::with_config(vo_config)),
            Box::new(SeparationAvoid::with_config(sep_config)),
        ],
    ))
}

/// Create a fixed-wing behavior tree using ORCA for collision avoidance.
///
/// ORCA (Optimal Reciprocal Collision Avoidance) provides guaranteed
/// collision-free navigation when all agents use ORCA. It computes
/// half-planes of safe velocities and finds the optimal velocity
/// within their intersection.
///
/// APF (Artificial Potential Field) provides a local safety layer that
/// runs after ORCA, applying repulsive forces for very close encounters.
///
/// Structure:
/// ```text
/// Selector (root)
/// ├── Sequence (navigate_with_orca)
/// │   ├── SeekWaypoint   # Sets ctx.desired_heading from mission
/// │   ├── ORCAAvoid      # Computes collision-free velocity
/// │   └── APFAvoid       # Local emergency repulsion (blends with ORCA)
/// └── SeparationAvoid    # Fallback reactive avoidance
/// ```
pub fn create_orca_bt(
    orca_config: ORCAConfig,
    sep_config: SeparationConfig,
) -> Box<dyn BehaviorNode> {
    // APF config for emergency close-range avoidance
    // Very high repulsion to produce meaningful force at distance
    // Force = η * (1/d - 1/d0) / d² must exceed 1.0 to trigger avoidance
    let apf_config = APFConfig {
        influence_distance: 150.0,      // Wide range
        repulsion_strength: 10000000.0, // Very high - needed for 1/d² falloff
        min_distance: 35.0,             // Hard boundary - larger for multi-drone scenarios
        max_force: 80.0,                // Lower cap = higher urgency ratio
    };

    Box::new(Selector::new(
        "root",
        vec![
            // Primary: navigate with ORCA + APF avoidance
            Box::new(Sequence::new(
                "navigate_with_orca",
                vec![
                    Box::new(SeekWaypoint::new()),
                    Box::new(ORCAAvoid::with_config(orca_config)),
                    Box::new(APFAvoid::with_config(apf_config)),
                ],
            )),
            // Fallback: separation only (for idle/emergency)
            Box::new(SeparationAvoid::with_config(sep_config)),
        ],
    ))
}

/// Create an ORCA-only avoidance behavior tree (no seeking).
///
/// Useful for testing or scenarios where navigation is handled externally.
/// Runs ORCA followed by APF for layered collision avoidance.
pub fn create_orca_avoidance_only_bt(
    orca_config: ORCAConfig,
    _sep_config: SeparationConfig,
) -> Box<dyn BehaviorNode> {
    // APF config for emergency close-range avoidance
    let apf_config = APFConfig {
        influence_distance: 150.0,
        repulsion_strength: 10000000.0,
        min_distance: 35.0,
        max_force: 80.0,
    };

    // Run both ORCA and APF in sequence
    Box::new(Sequence::new(
        "orca_avoidance_only",
        vec![
            Box::new(ORCAAvoid::with_config(orca_config)),
            Box::new(APFAvoid::with_config(apf_config)),
        ],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behaviors::tree::node::{BehaviorContext, BehaviorStatus};
    use crate::test_utils::fixtures::{create_test_bounds, create_test_state_with_vel};
    use crate::types::{DronePerfFeatures, Heading, State};

    fn test_perf() -> &'static DronePerfFeatures {
        Box::leak(Box::new(DronePerfFeatures::default()))
    }

    fn create_test_state() -> State {
        create_test_state_with_vel(500.0, 500.0, 0.0, 60.0)
    }

    #[test]
    fn test_create_fixed_wing_bt() {
        let bt = create_fixed_wing_bt(
            VelocityObstacleConfig::default(),
            SeparationConfig::default(),
        );
        assert_eq!(bt.name(), "root");
    }

    #[test]
    fn test_fixed_wing_bt_with_waypoint() {
        let mut bt = create_fixed_wing_bt(
            VelocityObstacleConfig::default(),
            SeparationConfig::default(),
        );

        let state = create_test_state();
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        // Set a desired heading (simulating mission providing target)
        ctx.desired_heading = Some(Heading::new(0.5));

        let status = bt.tick(&mut ctx);

        // Should succeed (SeekWaypoint succeeds, VO succeeds)
        assert_eq!(status, BehaviorStatus::Success);
        // Heading should still be set (possibly adjusted by VO)
        assert!(ctx.desired_heading.is_some());
    }

    #[test]
    fn test_fixed_wing_bt_without_waypoint() {
        let mut bt = create_fixed_wing_bt(
            VelocityObstacleConfig::default(),
            SeparationConfig::default(),
        );

        let state = create_test_state();
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        // No desired heading set

        let status = bt.tick(&mut ctx);

        // SeekWaypoint fails (no heading), falls through to SeparationAvoid
        // SeparationAvoid fails (no neighbors)
        assert_eq!(status, BehaviorStatus::Failure);
    }

    #[test]
    fn test_seek_only_bt() {
        let mut bt = create_seek_only_bt();
        assert_eq!(bt.name(), "SeekWaypoint");

        let state = create_test_state();
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());
        ctx.desired_heading = Some(Heading::new(1.0));

        let status = bt.tick(&mut ctx);
        assert_eq!(status, BehaviorStatus::Success);
    }

    #[test]
    fn test_avoidance_only_bt() {
        let bt = create_avoidance_only_bt(
            VelocityObstacleConfig::default(),
            SeparationConfig::default(),
        );
        assert_eq!(bt.name(), "avoidance_only");
    }

    #[test]
    fn test_create_orca_bt() {
        let bt = create_orca_bt(ORCAConfig::default(), SeparationConfig::default());
        assert_eq!(bt.name(), "root");
    }

    #[test]
    fn test_orca_bt_with_waypoint() {
        let mut bt = create_orca_bt(ORCAConfig::default(), SeparationConfig::default());

        let state = create_test_state();
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        // Set a desired heading (simulating mission providing target)
        ctx.desired_heading = Some(Heading::new(0.5));
        ctx.desired_speed = 60.0;

        let status = bt.tick(&mut ctx);

        // Should succeed (SeekWaypoint succeeds, ORCA succeeds)
        assert_eq!(status, BehaviorStatus::Success);
        // Heading should still be set (possibly adjusted by ORCA)
        assert!(ctx.desired_heading.is_some());
    }

    #[test]
    fn test_orca_bt_without_waypoint() {
        let mut bt = create_orca_bt(ORCAConfig::default(), SeparationConfig::default());

        let state = create_test_state();
        let bounds = create_test_bounds();
        let mut ctx = BehaviorContext::new(&state, &[], &bounds, 0, 0.016, test_perf());

        // No desired heading set

        let status = bt.tick(&mut ctx);

        // SeekWaypoint fails (no heading), falls through to SeparationAvoid
        // SeparationAvoid fails (no neighbors)
        assert_eq!(status, BehaviorStatus::Failure);
    }

    #[test]
    fn test_orca_avoidance_only_bt() {
        let bt = create_orca_avoidance_only_bt(ORCAConfig::default(), SeparationConfig::default());
        assert_eq!(bt.name(), "orca_avoidance_only");
    }
}
