//! Attack task: navigate to a fixed position and detonate.
//!
//! Supports two modes:
//! - **Direct**: Full speed beeline to target, ignores safety feedback (terminal dive).
//! - **Evasive**: Dodges threats en route by blending safety feedback into desired
//!   velocity. Commits to direct mode when close to target (< 2x detonation radius).

use crate::tasks::{DroneTask, SafetyFeedback, TaskOutput, TaskStatus};
use crate::types::{Bounds, DroneInfo, DronePerfFeatures, Heading, Position, State, Vec2};

/// Phase of the attack task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttackPhase {
    /// Navigating toward the target position.
    Navigate,
    /// Final approach — committed to the attack run.
    Terminal,
}

/// Attack task: navigate to a fixed target position and detonate on arrival.
#[derive(Debug)]
pub struct AttackTask {
    /// Target position to attack.
    target: Position,
    /// Current phase.
    phase: AttackPhase,
    /// Task status.
    status: TaskStatus,
    /// Blast radius — detonate when this close.
    detonation_radius: f32,
    /// Whether to use evasive mode (dodge threats en route).
    evasive: bool,
    /// Last safety urgency (used in evasive mode).
    last_urgency: f32,
    /// Last threat direction from safety feedback (used in evasive mode).
    last_threat_dir: Option<Vec2>,
}

impl AttackTask {
    /// Create a new direct attack task (ignores safety feedback).
    pub fn new(target: Position, detonation_radius: f32) -> Self {
        AttackTask {
            target,
            phase: AttackPhase::Navigate,
            status: TaskStatus::Active,
            detonation_radius,
            evasive: false,
            last_urgency: 0.0,
            last_threat_dir: None,
        }
    }

    /// Create a new evasive attack task (dodges threats, commits when close).
    pub fn new_evasive(target: Position, detonation_radius: f32) -> Self {
        AttackTask {
            target,
            phase: AttackPhase::Navigate,
            status: TaskStatus::Active,
            detonation_radius,
            evasive: true,
            last_urgency: 0.0,
            last_threat_dir: None,
        }
    }

    /// Whether this is an evasive attack.
    pub fn is_evasive(&self) -> bool {
        self.evasive
    }

    /// Get the target position.
    pub fn target(&self) -> Position {
        self.target
    }

    /// Get the current phase.
    pub fn phase(&self) -> AttackPhase {
        self.phase
    }
}

impl DroneTask for AttackTask {
    fn tick(
        &mut self,
        state: &State,
        _swarm: &[DroneInfo],
        bounds: &Bounds,
        perf: &DronePerfFeatures,
        _dt: f32,
    ) -> TaskOutput {
        let max_speed = perf.max_vel;

        // Vector from self to target
        let delta = bounds.delta(state.pos.as_vec2(), self.target.as_vec2());
        let dist = delta.magnitude();

        // Check if within detonation range
        if dist <= self.detonation_radius {
            self.status = TaskStatus::Complete;
            return TaskOutput {
                desired_velocity: Vec2::new(0.0, 0.0),
                desired_heading: None,
                detonate: true,
                exclude_from_ca: vec![],
            };
        }

        // Transition to terminal phase when close
        if dist < self.detonation_radius * 3.0 {
            self.phase = AttackPhase::Terminal;
        }

        // Evasive mode: dodge threats when far from target, commit when close.
        if self.evasive && self.phase == AttackPhase::Navigate {
            let speed = max_speed * 0.75; // slower to give ORCA more room
            let mut dir = Vec2::new(delta.x / dist, delta.y / dist);

            // Blend away from threat direction when urgency is significant.
            if let Some(threat_dir) = self.last_threat_dir {
                if self.last_urgency > 0.2 {
                    let threat_mag = threat_dir.magnitude();
                    if threat_mag > f32::EPSILON {
                        // Flee direction = opposite of threat direction
                        let avoid = Vec2::new(-threat_dir.x / threat_mag, -threat_dir.y / threat_mag);
                        let blend = (self.last_urgency * 0.6).min(0.8);
                        dir = Vec2::new(
                            dir.x * (1.0 - blend) + avoid.x * blend,
                            dir.y * (1.0 - blend) + avoid.y * blend,
                        );
                        let mag = dir.magnitude();
                        if mag > f32::EPSILON {
                            dir = Vec2::new(dir.x / mag, dir.y / mag);
                        }
                    }
                }
            }

            let desired_vel = Vec2::new(dir.x * speed, dir.y * speed);
            let heading = Heading::new(desired_vel.y.atan2(desired_vel.x));

            return TaskOutput {
                desired_velocity: desired_vel,
                desired_heading: Some(heading),
                detonate: false,
                exclude_from_ca: vec![],
            };
        }

        // Direct mode (or terminal phase): full speed toward target.
        let desired_vel = Vec2::new(
            delta.x / dist * max_speed,
            delta.y / dist * max_speed,
        );

        let heading = Heading::new(desired_vel.y.atan2(desired_vel.x));

        TaskOutput {
            desired_velocity: desired_vel,
            desired_heading: Some(heading),
            detonate: false,
            exclude_from_ca: vec![],
        }
    }

    fn process_feedback(&mut self, feedback: &SafetyFeedback) {
        if self.evasive {
            self.last_urgency = feedback.urgency;
            self.last_threat_dir = feedback.threat_direction;
        }
        // Direct mode ignores safety feedback entirely.
    }

    fn phase_name(&self) -> &str {
        match self.phase {
            AttackPhase::Navigate => "navigate",
            AttackPhase::Terminal => "terminal",
        }
    }

    fn status(&self) -> TaskStatus {
        self.status
    }

    fn name(&self) -> &str {
        if self.evasive { "AttackEvasive" } else { "Attack" }
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

    #[test]
    fn test_navigate_toward_target() {
        let mut task = AttackTask::new(Position::new(1000.0, 500.0), 187.5);
        let state = State {
            pos: Position::new(0.0, 500.0),
            hdg: Heading::new(0.0),
            vel: Velocity::new(60.0, 0.0),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        let output = task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        assert!(output.desired_velocity.x > 0.0, "Should navigate toward target");
        assert!(!output.detonate);
        assert_eq!(task.phase(), AttackPhase::Navigate);
    }

    #[test]
    fn test_detonate_on_arrival() {
        let mut task = AttackTask::new(Position::new(100.0, 0.0), 187.5);
        let state = State {
            pos: Position::new(0.0, 0.0),
            hdg: Heading::new(0.0),
            vel: Velocity::new(60.0, 0.0),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        let output = task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        assert!(output.detonate);
        assert_eq!(task.status(), TaskStatus::Complete);
    }

    #[test]
    fn test_terminal_phase_when_close() {
        let mut task = AttackTask::new(Position::new(400.0, 0.0), 187.5);
        let state = State {
            pos: Position::new(0.0, 0.0),
            hdg: Heading::new(0.0),
            vel: Velocity::new(60.0, 0.0),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        assert_eq!(task.phase(), AttackPhase::Terminal);
    }

    #[test]
    fn test_evasive_responds_to_threat() {
        let mut task = AttackTask::new_evasive(Position::new(2000.0, 0.0), 187.5);
        let state = State {
            pos: Position::new(0.0, 0.0),
            hdg: Heading::new(0.0),
            vel: Velocity::new(60.0, 0.0),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(5000.0, 5000.0).unwrap();

        // First tick without feedback — should head toward target.
        let output1 = task.tick(&state, &[], &bounds, &test_perf(), 0.016);
        assert!(output1.desired_velocity.x > 0.0);

        // Simulate threat from below (positive y direction).
        task.process_feedback(&SafetyFeedback {
            urgency: 0.8,
            threat_direction: Some(Vec2::new(0.0, 1.0)),
            safe_velocity: Vec2::new(60.0, -30.0),
        });

        // Second tick — should deflect away from threat (negative y).
        let output2 = task.tick(&state, &[], &bounds, &test_perf(), 0.016);
        assert!(output2.desired_velocity.y < 0.0, "Should dodge away from threat");
        assert!(output2.desired_velocity.x > 0.0, "Should still make progress toward target");
    }

    #[test]
    fn test_evasive_commits_when_close() {
        let mut task = AttackTask::new_evasive(Position::new(400.0, 0.0), 187.5);
        let state = State {
            pos: Position::new(0.0, 0.0),
            hdg: Heading::new(0.0),
            vel: Velocity::new(60.0, 0.0),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(5000.0, 5000.0).unwrap();

        // Feed threat to ensure evasive logic is active.
        task.process_feedback(&SafetyFeedback {
            urgency: 0.9,
            threat_direction: Some(Vec2::new(0.0, 1.0)),
            safe_velocity: Vec2::new(60.0, -30.0),
        });

        let output = task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        // Should transition to terminal and go full speed direct.
        assert_eq!(task.phase(), AttackPhase::Terminal);
        // In terminal phase, velocity is full speed toward target (ignores evasion).
        let speed = (output.desired_velocity.x.powi(2) + output.desired_velocity.y.powi(2)).sqrt();
        assert!((speed - 120.0).abs() < 1.0, "Terminal phase should be at max speed");
    }

    #[test]
    fn test_evasive_reduced_speed() {
        let mut task = AttackTask::new_evasive(Position::new(2000.0, 0.0), 187.5);
        let state = State {
            pos: Position::new(0.0, 0.0),
            hdg: Heading::new(0.0),
            vel: Velocity::new(60.0, 0.0),
            acc: Acceleration::zero(),
        };
        let bounds = Bounds::new(5000.0, 5000.0).unwrap();

        let output = task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        let speed = (output.desired_velocity.x.powi(2) + output.desired_velocity.y.powi(2)).sqrt();
        // Evasive mode uses 75% max speed
        assert!((speed - 90.0).abs() < 1.0, "Evasive should use 75% speed, got {}", speed);
    }
}
