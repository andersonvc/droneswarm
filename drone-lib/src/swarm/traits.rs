//! SwarmBehavior trait for swarm-level coordination.

use crate::agent::DroneAgent;
use crate::missions::Mission;
use crate::types::DroneInfo;

/// Trait for swarm-level coordination behaviors.
///
/// SwarmBehaviors operate at the collective level, coordinating
/// multiple agents to achieve swarm-level goals like formations,
/// area coverage, or consensus protocols.
pub trait SwarmBehavior: Send + Sync + std::fmt::Debug {
    /// Pre-tick hook: called before individual agent updates.
    ///
    /// Can be used to establish priority order, share information,
    /// or compute swarm-level state.
    fn pre_tick(&mut self, agents: &[DroneAgent], dt: f32);

    /// Get agent update order (indices into agents slice).
    ///
    /// This determines which agents update first, which is important
    /// for collision avoidance priority.
    fn get_update_order(&self, agents: &[DroneAgent]) -> Vec<usize>;

    /// Post-tick hook: called after individual agent updates.
    ///
    /// Can be used to check swarm-level constraints or metrics.
    fn post_tick(&mut self, agents: &[DroneAgent], dt: f32);

    /// Get swarm-level information to share with agents.
    ///
    /// Returns a snapshot of all drone positions/velocities.
    fn get_swarm_info(&self, agents: &[DroneAgent]) -> Vec<DroneInfo>;
}

/// Default swarm behavior: priority by ID.
///
/// Lower ID drones update first and have priority in collision avoidance.
#[derive(Debug, Default, Clone)]
pub struct DefaultSwarmBehavior;

impl DefaultSwarmBehavior {
    /// Create a new default swarm behavior.
    pub fn new() -> Self {
        DefaultSwarmBehavior
    }
}

impl SwarmBehavior for DefaultSwarmBehavior {
    fn pre_tick(&mut self, _agents: &[DroneAgent], _dt: f32) {
        // No-op for default behavior
    }

    fn get_update_order(&self, agents: &[DroneAgent]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..agents.len()).collect();
        indices.sort_by_key(|&i| agents[i].uid());
        indices
    }

    fn post_tick(&mut self, _agents: &[DroneAgent], _dt: f32) {
        // No-op for default behavior
    }

    fn get_swarm_info(&self, agents: &[DroneAgent]) -> Vec<DroneInfo> {
        agents.iter().map(|a| a.get_info()).collect()
    }
}

/// Priority by waypoint distance swarm behavior.
///
/// Drones closer to their waypoints update first, giving them
/// priority in collision avoidance.
#[derive(Debug, Default, Clone)]
pub struct WaypointPriorityBehavior;

impl WaypointPriorityBehavior {
    /// Create a new waypoint priority behavior.
    pub fn new() -> Self {
        WaypointPriorityBehavior
    }
}

impl SwarmBehavior for WaypointPriorityBehavior {
    fn pre_tick(&mut self, _agents: &[DroneAgent], _dt: f32) {
        // No-op
    }

    fn get_update_order(&self, agents: &[DroneAgent]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..agents.len()).collect();

        // Sort by distance to current waypoint (closer = higher priority)
        indices.sort_by(|&a, &b| {
            let dist_a = agents[a].mission().current_target()
                .map(|t| {
                    let bounds = agents[a].mission().bounds();
                    bounds.distance(agents[a].state().pos.as_vec2(), t.as_vec2())
                })
                .unwrap_or(f32::MAX);

            let dist_b = agents[b].mission().current_target()
                .map(|t| {
                    let bounds = agents[b].mission().bounds();
                    bounds.distance(agents[b].state().pos.as_vec2(), t.as_vec2())
                })
                .unwrap_or(f32::MAX);

            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        indices
    }

    fn post_tick(&mut self, _agents: &[DroneAgent], _dt: f32) {
        // No-op
    }

    fn get_swarm_info(&self, agents: &[DroneAgent]) -> Vec<DroneInfo> {
        agents.iter().map(|a| a.get_info()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Bounds, Heading, Objective, Position};

    fn create_test_bounds() -> Bounds {
        Bounds::new(1000.0, 1000.0).unwrap()
    }

    fn create_test_agents() -> Vec<DroneAgent> {
        vec![
            DroneAgent::new(2, Position::new(500.0, 500.0), Heading::new(0.0), create_test_bounds()),
            DroneAgent::new(0, Position::new(600.0, 500.0), Heading::new(0.0), create_test_bounds()),
            DroneAgent::new(1, Position::new(400.0, 500.0), Heading::new(0.0), create_test_bounds()),
        ]
    }

    #[test]
    fn test_default_priority_by_id() {
        let agents = create_test_agents();
        let behavior = DefaultSwarmBehavior::new();

        let order = behavior.get_update_order(&agents);

        // Should be sorted by UID: 0, 1, 2
        // Agent at index 1 has UID 0
        // Agent at index 2 has UID 1
        // Agent at index 0 has UID 2
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn test_get_swarm_info() {
        let agents = create_test_agents();
        let behavior = DefaultSwarmBehavior::new();

        let info = behavior.get_swarm_info(&agents);

        assert_eq!(info.len(), 3);
        assert_eq!(info[0].uid, 2); // First agent has UID 2
        assert_eq!(info[1].uid, 0); // Second agent has UID 0
    }

    #[test]
    fn test_waypoint_priority() {
        let mut agents = create_test_agents();

        // Set waypoints at different distances
        // Agent 0 (UID 2): waypoint 100 units away
        agents[0].set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(600.0, 500.0)].into(),
        });

        // Agent 1 (UID 0): waypoint 50 units away
        agents[1].set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(650.0, 500.0)].into(),
        });

        // Agent 2 (UID 1): waypoint 200 units away
        agents[2].set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(600.0, 500.0)].into(),
        });

        let behavior = WaypointPriorityBehavior::new();
        let order = behavior.get_update_order(&agents);

        // Agent 1 is closest (50 units), then Agent 0 (100 units), then Agent 2 (200 units)
        assert_eq!(order[0], 1); // Agent 1 first (closest)
    }
}
