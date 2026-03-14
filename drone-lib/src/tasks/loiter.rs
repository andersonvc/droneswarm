//! Loiter task: hold position at a fixed location.

use crate::tasks::{DroneTask, SafetyFeedback, TaskOutput, TaskStatus};
use crate::types::{Bounds, DroneInfo, DronePerfFeatures, Heading, State, Vec2};

/// Phase of the loiter task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoiterPhase {
    /// Approaching the loiter point.
    Approach,
    /// Holding position at the loiter point.
    Hold,
}

/// Loiter task: navigate to a position and station-keep there.
#[derive(Debug)]
pub struct LoiterTask {
    /// Position to hold.
    target: crate::types::Position,
    /// Current phase.
    phase: LoiterPhase,
    /// Task status.
    status: TaskStatus,
    /// Radius within which the drone is considered "on station".
    hold_radius: f32,
}

impl LoiterTask {
    /// Create a new loiter task.
    ///
    /// `hold_radius` — distance threshold to transition from Approach to Hold.
    pub fn new(target: crate::types::Position, hold_radius: f32) -> Self {
        LoiterTask {
            target,
            phase: LoiterPhase::Approach,
            status: TaskStatus::Active,
            hold_radius,
        }
    }

    /// Get the current phase.
    pub fn phase(&self) -> LoiterPhase {
        self.phase
    }

    /// Compute a velocity that approaches the target and decelerates smoothly.
    fn hold_velocity(
        &self,
        state: &State,
        bounds: &Bounds,
        max_speed: f32,
    ) -> Vec2 {
        let delta = bounds.delta(state.pos.as_vec2(), self.target.as_vec2());
        let dist = delta.magnitude();

        if dist < 1.0 {
            return Vec2::new(0.0, 0.0);
        }

        // Proportional speed: ramp down as we approach
        // Full speed beyond 3x hold_radius, linear ramp to near-zero at hold point
        let speed_factor = (dist / (self.hold_radius * 3.0)).min(1.0);
        let speed = max_speed * speed_factor;

        Vec2::new(
            delta.x / dist * speed,
            delta.y / dist * speed,
        )
    }
}

impl DroneTask for LoiterTask {
    fn tick(
        &mut self,
        state: &State,
        _swarm: &[DroneInfo],
        bounds: &Bounds,
        perf: &DronePerfFeatures,
        _dt: f32,
    ) -> TaskOutput {
        let delta = bounds.delta(state.pos.as_vec2(), self.target.as_vec2());
        let dist = delta.magnitude();

        // Phase transition
        match self.phase {
            LoiterPhase::Approach => {
                if dist <= self.hold_radius {
                    self.phase = LoiterPhase::Hold;
                }
            }
            LoiterPhase::Hold => {
                // If drifted too far, go back to approach
                if dist > self.hold_radius * 2.0 {
                    self.phase = LoiterPhase::Approach;
                }
            }
        }

        let desired_vel = self.hold_velocity(state, bounds, perf.max_vel);

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
        // Loiter tolerates safety adjustments without phase changes
    }

    fn phase_name(&self) -> &str {
        match self.phase {
            LoiterPhase::Approach => "approach",
            LoiterPhase::Hold => "hold",
        }
    }

    fn status(&self) -> TaskStatus {
        self.status
    }

    fn name(&self) -> &str {
        "Loiter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Acceleration, Heading, Position, Velocity};

    fn test_perf() -> DronePerfFeatures {
        DronePerfFeatures {
            max_vel: 120.0,
            max_acc: 100.0,
            max_turn_rate: 0.6,
        }
    }

    #[test]
    fn test_approach_phase_navigates_to_target() {
        let mut task = LoiterTask::new(Position::new(500.0, 0.0), 50.0);
        let state = State {
            pos: Position::new(0.0, 0.0),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        let output = task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        assert!(output.desired_velocity.x > 0.0, "Should navigate toward target");
        assert_eq!(task.phase(), LoiterPhase::Approach);
    }

    #[test]
    fn test_transitions_to_hold_when_close() {
        let mut task = LoiterTask::new(Position::new(30.0, 0.0), 50.0);
        let state = State {
            pos: Position::new(0.0, 0.0),
            hdg: Heading::new(0.0),
            vel: Velocity::new(60.0, 0.0),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        assert_eq!(task.phase(), LoiterPhase::Hold);
    }

    #[test]
    fn test_hold_produces_near_zero_velocity_at_target() {
        let mut task = LoiterTask::new(Position::new(5.0, 0.0), 50.0);
        let state = State {
            pos: Position::new(0.0, 0.0),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        let output = task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        // Close to target — velocity should be small
        let speed = output.desired_velocity.magnitude();
        assert!(speed < 10.0, "Expected low speed near target, got {}", speed);
    }

    #[test]
    fn test_reapproach_on_drift() {
        let mut task = LoiterTask::new(Position::new(0.0, 0.0), 50.0);
        // Start in hold phase
        task.phase = LoiterPhase::Hold;

        let state = State {
            pos: Position::new(150.0, 0.0), // drifted 150m > 2x hold_radius (100m)
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        assert_eq!(task.phase(), LoiterPhase::Approach);
    }
}
