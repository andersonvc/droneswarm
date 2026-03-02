//! Task types and waypoint-based mission implementation.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::types::{Bounds, Heading, Position, State};

use super::planner::PathPlanner;
use super::traits::{Mission, MissionStatus};

/// Task types representing different mission objectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task {
    /// Navigate directly to waypoints, stop when done.
    ReachWaypoint,
    /// Follow a route indefinitely, looping back to start.
    FollowRoute,
    /// Track a moving target.
    FollowTarget,
    /// Hold position or patrol area.
    Loiter,
    /// Inactive state, no movement.
    #[default]
    Sleep,
}

/// Waypoint-based mission implementation.
///
/// Manages a queue of waypoints and determines desired headings
/// using either direct navigation or smoothed path planning.
#[derive(Debug, Clone)]
pub struct WaypointMission {
    /// Current task type.
    task: Task,
    /// Queue of waypoints to visit.
    waypoints: VecDeque<Position>,
    /// Shared route for looping (FollowRoute mode).
    route: Option<Arc<[Position]>>,
    /// Path planner for smoothed navigation.
    planner: PathPlanner,
    /// World bounds for distance calculations.
    bounds: Bounds,
    /// Current mission status.
    status: MissionStatus,
}

impl WaypointMission {
    /// Create a new idle mission.
    pub fn new(bounds: Bounds) -> Self {
        WaypointMission {
            task: Task::Sleep,
            waypoints: VecDeque::new(),
            route: None,
            planner: PathPlanner::new(),
            bounds,
            status: MissionStatus::Idle,
        }
    }

    /// Set waypoints for a new mission.
    ///
    /// # Arguments
    /// * `task` - Task type
    /// * `waypoints` - Queue of waypoints
    /// * `route` - Optional shared route for looping
    pub fn set_waypoints(
        &mut self,
        task: Task,
        waypoints: VecDeque<Position>,
        route: Option<Arc<[Position]>>,
    ) {
        self.task = task;
        self.waypoints = waypoints;
        self.route = route;
        self.status = if self.waypoints.is_empty() && !matches!(task, Task::FollowRoute) {
            MissionStatus::Idle
        } else {
            MissionStatus::Active
        };
    }

    /// Clear the mission and return to idle state.
    pub fn clear(&mut self) {
        self.task = Task::Sleep;
        self.waypoints.clear();
        self.route = None;
        self.status = MissionStatus::Idle;
    }

    /// Get the current waypoint queue.
    pub fn waypoints(&self) -> &VecDeque<Position> {
        &self.waypoints
    }

    /// Get the current (next) waypoint, if any.
    pub fn current_waypoint(&self) -> Option<Position> {
        self.waypoints.front().copied()
    }

    /// Get upcoming waypoints (up to n).
    pub fn upcoming_waypoints(&self, n: usize) -> Vec<Position> {
        self.waypoints.iter().take(n).copied().collect()
    }

    /// Get the current task type.
    pub fn task(&self) -> Task {
        self.task
    }

    /// Get reference to the route (for looping).
    pub fn route(&self) -> Option<&Arc<[Position]>> {
        self.route.as_ref()
    }

    /// Get reference to the path planner.
    pub fn planner(&self) -> &PathPlanner {
        &self.planner
    }

    /// Get the bounds.
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    /// Check if the drone has passed a waypoint along the route direction.
    ///
    /// Returns true if the drone is on the far side of the waypoint's
    /// perpendicular plane, oriented toward the next waypoint. This detects
    /// overshoot situations where the drone flew past the waypoint.
    fn has_passed_waypoint(&self, state: &State, waypoint: &Position) -> bool {
        // Determine the "forward" direction at this waypoint (toward next waypoint)
        let next_wp = if let Some(&next) = self.waypoints.get(1) {
            next
        } else if let Some(route) = &self.route {
            // For looping routes, next is route[0]
            if !route.is_empty() {
                route[0]
            } else {
                return false;
            }
        } else {
            return false;
        };

        // Direction from current waypoint to next waypoint
        let segment = self.bounds.delta(waypoint.as_vec2(), next_wp.as_vec2());
        if segment.magnitude_squared() < f32::EPSILON {
            return false;
        }

        // Vector from waypoint to drone
        let to_drone = self.bounds.delta(waypoint.as_vec2(), state.pos.as_vec2());

        // Positive dot product means drone is on the "far side" of the waypoint
        // (past it in the direction of the next waypoint)
        to_drone.dot(segment) > 0.0
    }

    /// Get waypoints as a slice for spline calculations.
    /// Returns up to 2 waypoints, potentially supplemented from route.
    pub fn get_waypoints_for_spline(&self) -> Vec<Position> {
        let mut waypoints: Vec<Position> = self.waypoints.iter().take(2).copied().collect();

        // If only one waypoint but we have a route, use route[0] as second waypoint
        if waypoints.len() == 1 {
            if let Some(route) = &self.route {
                if !route.is_empty() {
                    waypoints.push(route[0]);
                }
            }
        }

        waypoints
    }
}

impl Mission for WaypointMission {
    fn get_desired_heading(&self, current_state: &State) -> Option<Heading> {
        if self.waypoints.is_empty() {
            return None;
        }

        let waypoint = self.waypoints.front()?;

        if self.uses_path_smoothing() {
            let waypoints = self.get_waypoints_for_spline();
            if waypoints.len() >= 2 {
                Some(self.planner.compute_smoothed_heading(
                    current_state,
                    &waypoints,
                    &self.bounds,
                ))
            } else {
                // Fall back to direct heading
                let delta = self.bounds.delta(
                    current_state.pos.as_vec2(),
                    waypoint.as_vec2(),
                );
                Some(Heading::new(delta.heading()))
            }
        } else {
            // Direct heading to waypoint
            let delta = self.bounds.delta(
                current_state.pos.as_vec2(),
                waypoint.as_vec2(),
            );
            Some(Heading::new(delta.heading()))
        }
    }

    fn get_desired_speed_factor(&self) -> f32 {
        match self.task {
            Task::Sleep | Task::Loiter => 0.0,
            Task::ReachWaypoint | Task::FollowRoute => 0.5, // Half speed for waypoint following
            _ => 1.0,
        }
    }

    fn update(&mut self, current_state: &State, clearance: f32) {
        // Pop all waypoints within clearance or that have been passed
        while let Some(&waypoint) = self.waypoints.front() {
            let delta = self.bounds.delta(
                current_state.pos.as_vec2(),
                waypoint.as_vec2(),
            );
            let dist = delta.magnitude();

            if dist < clearance {
                self.waypoints.pop_front();
                continue;
            }

            // Waypoint passage check: if the drone has passed the perpendicular
            // plane at the waypoint (along the route direction), pop it even if
            // distance > clearance. Prevents 360-degree turns when overshooting.
            if dist < clearance * 5.0
                && self.has_passed_waypoint(current_state, &waypoint)
            {
                self.waypoints.pop_front();
                continue;
            }

            break;
        }

        // Handle route looping and task completion
        if self.waypoints.is_empty() {
            match self.task {
                Task::FollowRoute => {
                    // Reload waypoints from route
                    if let Some(route) = &self.route {
                        if !route.is_empty() {
                            self.waypoints = route.iter().copied().collect();
                        } else {
                            self.status = MissionStatus::Completed;
                        }
                    } else {
                        self.status = MissionStatus::Completed;
                    }
                }
                Task::ReachWaypoint => {
                    self.task = Task::Sleep;
                    self.status = MissionStatus::Completed;
                }
                _ => {
                    self.status = MissionStatus::Idle;
                }
            }
        }
    }

    fn status(&self) -> MissionStatus {
        self.status
    }

    fn current_target(&self) -> Option<Position> {
        self.waypoints.front().copied()
    }

    fn uses_path_smoothing(&self) -> bool {
        matches!(self.task, Task::FollowRoute)
    }

    fn reset(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Acceleration, Velocity};

    fn create_test_bounds() -> Bounds {
        Bounds::new(1000.0, 1000.0).unwrap()
    }

    fn create_test_state(x: f32, y: f32) -> State {
        State {
            pos: Position::new(x, y),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        }
    }

    #[test]
    fn test_new_mission_is_idle() {
        let mission = WaypointMission::new(create_test_bounds());
        assert_eq!(mission.status(), MissionStatus::Idle);
        assert_eq!(mission.task(), Task::Sleep);
        assert!(mission.waypoints().is_empty());
    }

    #[test]
    fn test_set_waypoints_activates_mission() {
        let mut mission = WaypointMission::new(create_test_bounds());
        let waypoints: VecDeque<Position> = vec![
            Position::new(100.0, 100.0),
            Position::new(200.0, 200.0),
        ].into();

        mission.set_waypoints(Task::ReachWaypoint, waypoints, None);

        assert_eq!(mission.status(), MissionStatus::Active);
        assert_eq!(mission.task(), Task::ReachWaypoint);
        assert_eq!(mission.waypoints().len(), 2);
    }

    #[test]
    fn test_waypoint_arrival_pops_waypoint() {
        let mut mission = WaypointMission::new(create_test_bounds());
        let waypoints: VecDeque<Position> = vec![
            Position::new(100.0, 100.0),
            Position::new(200.0, 200.0),
        ].into();
        mission.set_waypoints(Task::ReachWaypoint, waypoints, None);

        // State is at (95, 95), within clearance of (100, 100)
        let state = create_test_state(95.0, 95.0);
        mission.update(&state, 10.0);

        // First waypoint should be popped
        assert_eq!(mission.waypoints().len(), 1);
        assert_eq!(mission.current_target().unwrap().x(), 200.0);
    }

    #[test]
    fn test_reach_waypoint_completes_when_done() {
        let mut mission = WaypointMission::new(create_test_bounds());
        let waypoints: VecDeque<Position> = vec![
            Position::new(100.0, 100.0),
        ].into();
        mission.set_waypoints(Task::ReachWaypoint, waypoints, None);

        // State is at the waypoint
        let state = create_test_state(100.0, 100.0);
        mission.update(&state, 10.0);

        assert_eq!(mission.status(), MissionStatus::Completed);
        assert_eq!(mission.task(), Task::Sleep);
    }

    #[test]
    fn test_follow_route_loops() {
        let mut mission = WaypointMission::new(create_test_bounds());
        let route: Arc<[Position]> = vec![
            Position::new(100.0, 100.0),
            Position::new(200.0, 200.0),
        ].into();
        let waypoints: VecDeque<Position> = route.iter().copied().collect();
        mission.set_waypoints(Task::FollowRoute, waypoints, Some(route));

        // Visit both waypoints
        let state1 = create_test_state(100.0, 100.0);
        mission.update(&state1, 10.0);
        assert_eq!(mission.waypoints().len(), 1);

        let state2 = create_test_state(200.0, 200.0);
        mission.update(&state2, 10.0);

        // Should have reloaded from route
        assert_eq!(mission.waypoints().len(), 2);
        assert_eq!(mission.status(), MissionStatus::Active);
    }

    #[test]
    fn test_clear_resets_mission() {
        let mut mission = WaypointMission::new(create_test_bounds());
        let waypoints: VecDeque<Position> = vec![Position::new(100.0, 100.0)].into();
        mission.set_waypoints(Task::ReachWaypoint, waypoints, None);

        mission.clear();

        assert_eq!(mission.status(), MissionStatus::Idle);
        assert_eq!(mission.task(), Task::Sleep);
        assert!(mission.waypoints().is_empty());
    }

    #[test]
    fn test_uses_path_smoothing_only_for_follow_route() {
        let mut mission = WaypointMission::new(create_test_bounds());

        mission.set_waypoints(Task::ReachWaypoint, VecDeque::new(), None);
        assert!(!mission.uses_path_smoothing());

        mission.set_waypoints(Task::FollowRoute, VecDeque::new(), None);
        assert!(mission.uses_path_smoothing());
    }

    #[test]
    fn test_get_desired_speed_factor() {
        let mut mission = WaypointMission::new(create_test_bounds());

        mission.set_waypoints(Task::Sleep, VecDeque::new(), None);
        assert_eq!(mission.get_desired_speed_factor(), 0.0);

        mission.set_waypoints(Task::Loiter, VecDeque::new(), None);
        assert_eq!(mission.get_desired_speed_factor(), 0.0);

        // Waypoint following uses half speed for smoother control
        mission.set_waypoints(Task::ReachWaypoint, VecDeque::new(), None);
        assert_eq!(mission.get_desired_speed_factor(), 0.5);

        mission.set_waypoints(Task::FollowRoute, VecDeque::new(), None);
        assert_eq!(mission.get_desired_speed_factor(), 0.5);
    }
}
