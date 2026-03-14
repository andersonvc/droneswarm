//! Patrol task: transit between waypoints with loiter at each.

use crate::tasks::{DroneTask, SafetyFeedback, TaskOutput, TaskStatus};
use crate::types::{Bounds, DroneInfo, DronePerfFeatures, Heading, Position, State, Vec2};

/// Phase of the patrol task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatrolPhase {
    /// Navigating toward the next waypoint.
    Transit,
    /// Holding position at the current waypoint.
    Loiter,
}

/// Patrol task: follow a route of waypoints, loitering at each one.
///
/// The drone transits to each waypoint, loiters for a configurable duration,
/// then advances to the next. After the last waypoint it cycles back to the first.
#[derive(Debug)]
pub struct PatrolTask {
    /// Waypoints to patrol between.
    waypoints: Vec<Position>,
    /// Index of the current target waypoint.
    current_idx: usize,
    /// Current phase.
    phase: PatrolPhase,
    /// Task status.
    status: TaskStatus,
    /// Arrival radius — transition to loiter when this close.
    arrival_radius: f32,
    /// How long to loiter at each waypoint (seconds).
    loiter_duration: f32,
    /// Time spent loitering at the current waypoint.
    loiter_timer: f32,
}

impl PatrolTask {
    /// Create a new patrol task.
    ///
    /// * `waypoints` — positions to patrol (must have at least 1)
    /// * `arrival_radius` — distance to waypoint that triggers loiter
    /// * `loiter_duration` — seconds to hold at each waypoint (0.0 = no loiter)
    pub fn new(waypoints: Vec<Position>, arrival_radius: f32, loiter_duration: f32) -> Self {
        PatrolTask {
            waypoints,
            current_idx: 0,
            phase: PatrolPhase::Transit,
            status: TaskStatus::Active,
            arrival_radius,
            loiter_duration,
            loiter_timer: 0.0,
        }
    }

    /// Get the current phase.
    pub fn phase(&self) -> PatrolPhase {
        self.phase
    }

    /// Get the index of the current target waypoint.
    pub fn current_waypoint_index(&self) -> usize {
        self.current_idx
    }

    /// Current target waypoint position.
    fn current_waypoint(&self) -> Position {
        self.waypoints[self.current_idx]
    }

    /// Advance to the next waypoint (wrapping around).
    fn advance(&mut self) {
        self.current_idx = (self.current_idx + 1) % self.waypoints.len();
        self.phase = PatrolPhase::Transit;
        self.loiter_timer = 0.0;
    }

    /// Compute transit velocity toward the current waypoint.
    fn transit_velocity(
        &self,
        state: &State,
        bounds: &Bounds,
        max_speed: f32,
    ) -> Vec2 {
        let target = self.current_waypoint();
        let delta = bounds.delta(state.pos.as_vec2(), target.as_vec2());
        let dist = delta.magnitude();

        if dist < 1.0 {
            return Vec2::new(0.0, 0.0);
        }

        // Only decelerate if we'll actually loiter; otherwise maintain full speed
        let speed = if self.loiter_duration > 0.0 {
            let speed_factor = (dist / (self.arrival_radius * 3.0)).min(1.0);
            max_speed * speed_factor
        } else {
            max_speed
        };

        Vec2::new(
            delta.x / dist * speed,
            delta.y / dist * speed,
        )
    }

    /// Compute hold velocity to station-keep at the current waypoint.
    fn hold_velocity(
        &self,
        state: &State,
        bounds: &Bounds,
        max_speed: f32,
    ) -> Vec2 {
        let target = self.current_waypoint();
        let delta = bounds.delta(state.pos.as_vec2(), target.as_vec2());
        let dist = delta.magnitude();

        if dist < 1.0 {
            return Vec2::new(0.0, 0.0);
        }

        // Gentle correction — proportional to drift
        let speed_factor = (dist / (self.arrival_radius * 3.0)).min(0.5);
        let speed = max_speed * speed_factor;

        Vec2::new(
            delta.x / dist * speed,
            delta.y / dist * speed,
        )
    }
}

impl DroneTask for PatrolTask {
    fn tick(
        &mut self,
        state: &State,
        _swarm: &[DroneInfo],
        bounds: &Bounds,
        perf: &DronePerfFeatures,
        dt: f32,
    ) -> TaskOutput {
        if self.waypoints.is_empty() {
            self.status = TaskStatus::Failed;
            return TaskOutput {
                desired_velocity: Vec2::new(0.0, 0.0),
                desired_heading: None,
                detonate: false,
                exclude_from_ca: vec![],
            };
        }

        let target = self.current_waypoint();
        let delta = bounds.delta(state.pos.as_vec2(), target.as_vec2());
        let dist = delta.magnitude();

        // Phase transitions
        match self.phase {
            PatrolPhase::Transit => {
                if dist <= self.arrival_radius {
                    self.phase = PatrolPhase::Loiter;
                    self.loiter_timer = 0.0;
                }
            }
            PatrolPhase::Loiter => {
                self.loiter_timer += dt;
                if self.loiter_timer >= self.loiter_duration {
                    self.advance();
                }
            }
        }

        let desired_vel = match self.phase {
            PatrolPhase::Transit => self.transit_velocity(state, bounds, perf.max_vel),
            PatrolPhase::Loiter => self.hold_velocity(state, bounds, perf.max_vel),
        };

        let heading = if desired_vel.magnitude() > f32::EPSILON {
            Some(Heading::new(desired_vel.y.atan2(desired_vel.x)))
        } else {
            None
        };

        TaskOutput {
            desired_velocity: desired_vel,
            desired_heading: heading,
            detonate: false,
            exclude_from_ca: vec![],
        }
    }

    fn process_feedback(&mut self, _feedback: &SafetyFeedback) {
        // Patrol tolerates safety adjustments without phase changes
    }

    fn phase_name(&self) -> &str {
        match self.phase {
            PatrolPhase::Transit => "transit",
            PatrolPhase::Loiter => "loiter",
        }
    }

    fn status(&self) -> TaskStatus {
        self.status
    }

    fn name(&self) -> &str {
        "Patrol"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Acceleration, Heading, Velocity};

    fn test_perf() -> DronePerfFeatures {
        DronePerfFeatures {
            max_vel: 120.0,
            max_acc: 100.0,
            max_turn_rate: 0.6,
        }
    }

    fn make_state(x: f32, y: f32) -> State {
        State {
            pos: Position::new(x, y),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        }
    }

    #[test]
    fn test_transit_toward_first_waypoint() {
        let waypoints = vec![Position::new(500.0, 0.0), Position::new(500.0, 500.0)];
        let mut task = PatrolTask::new(waypoints, 50.0, 5.0);
        let state = make_state(0.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        let output = task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        assert!(output.desired_velocity.x > 0.0, "Should transit toward first waypoint");
        assert_eq!(task.phase(), PatrolPhase::Transit);
        assert_eq!(task.current_waypoint_index(), 0);
    }

    #[test]
    fn test_loiter_on_arrival() {
        let waypoints = vec![Position::new(30.0, 0.0), Position::new(500.0, 500.0)];
        let mut task = PatrolTask::new(waypoints, 50.0, 5.0);
        let state = make_state(0.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        assert_eq!(task.phase(), PatrolPhase::Loiter);
        assert_eq!(task.current_waypoint_index(), 0);
    }

    #[test]
    fn test_advance_after_loiter_duration() {
        let waypoints = vec![Position::new(10.0, 0.0), Position::new(500.0, 500.0)];
        let mut task = PatrolTask::new(waypoints, 50.0, 1.0);
        let state = make_state(0.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        // First tick: arrive and start loiter
        task.tick(&state, &[], &bounds, &test_perf(), 0.016);
        assert_eq!(task.phase(), PatrolPhase::Loiter);

        // Tick enough to exceed loiter_duration (1.0 second)
        task.tick(&state, &[], &bounds, &test_perf(), 1.1);

        assert_eq!(task.phase(), PatrolPhase::Transit);
        assert_eq!(task.current_waypoint_index(), 1);
    }

    #[test]
    fn test_wraps_around_to_first_waypoint() {
        let waypoints = vec![Position::new(10.0, 0.0), Position::new(20.0, 0.0)];
        let mut task = PatrolTask::new(waypoints, 50.0, 0.0);
        let state = make_state(0.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        // Arrive at first waypoint, loiter_duration=0 so advance immediately
        task.tick(&state, &[], &bounds, &test_perf(), 0.016);
        assert_eq!(task.phase(), PatrolPhase::Loiter);
        task.tick(&state, &[], &bounds, &test_perf(), 0.016);
        // Should have advanced to waypoint 1
        assert_eq!(task.current_waypoint_index(), 1);

        // Arrive at second waypoint
        let state2 = make_state(20.0, 0.0);
        task.tick(&state2, &[], &bounds, &test_perf(), 0.016);
        task.tick(&state2, &[], &bounds, &test_perf(), 0.016);
        // Should wrap to waypoint 0
        assert_eq!(task.current_waypoint_index(), 0);
    }

    #[test]
    fn test_empty_waypoints_fails() {
        let mut task = PatrolTask::new(vec![], 50.0, 5.0);
        let state = make_state(0.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        assert_eq!(task.status(), TaskStatus::Failed);
    }
}
