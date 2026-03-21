//! Intercept task: terminal pursuit of an enemy drone.
//!
//! The interceptor charges straight at its target without evading.
//! Collision avoidance is disabled for the target drone so the
//! interceptor can close and detonate.

use crate::tasks::{DroneTask, SafetyFeedback, TaskOutput, TaskStatus};
use crate::types::{Bounds, DroneInfo, DronePerfFeatures, Heading, State, Vec2};

/// Intercept task — pure pursuit toward a target drone.
///
/// Always pursues with lead pursuit (predicted intercept point).
/// Detonates when within blast radius. The safety layer is told
/// to exclude the target drone from collision avoidance via
/// `TaskOutput::exclude_from_ca`.
#[derive(Debug)]
pub struct InterceptTask {
    /// Target drone ID to intercept.
    target_id: usize,
    /// Task status.
    status: TaskStatus,
    /// Blast radius — detonate when this close.
    detonation_radius: f32,
}

impl InterceptTask {
    /// Create a new intercept task.
    pub fn new(_self_id: usize, target_id: usize, _self_group: u32, detonation_radius: f32) -> Self {
        InterceptTask {
            target_id,
            status: TaskStatus::Active,
            detonation_radius,
        }
    }

    /// Get the target drone ID.
    pub fn target_id(&self) -> usize {
        self.target_id
    }

    /// Find the target drone in the swarm info.
    fn find_target<'a>(&self, swarm: &'a [DroneInfo]) -> Option<&'a DroneInfo> {
        swarm.iter().find(|d| d.uid == self.target_id)
    }

    /// Compute lead pursuit velocity toward predicted target position.
    fn pursue_velocity(
        &self,
        state: &State,
        target: &DroneInfo,
        bounds: &Bounds,
        max_speed: f32,
    ) -> Vec2 {
        let delta = bounds.delta(state.pos.as_vec2(), target.pos.as_vec2());
        let dist = delta.magnitude();

        if dist < 0.1 {
            return Vec2::new(0.0, 0.0);
        }

        // Lead pursuit: predict where target will be
        let target_vel = target.vel.as_vec2();
        let closing_speed = max_speed + target_vel.magnitude();
        let intercept_time = if closing_speed > 0.1 {
            (dist / closing_speed).min(3.0)
        } else {
            0.0
        };

        let predicted = Vec2::new(
            delta.x + target_vel.x * intercept_time,
            delta.y + target_vel.y * intercept_time,
        );

        let pred_dist = predicted.magnitude();
        if pred_dist < 0.1 {
            return Vec2::new(delta.x / dist * max_speed, delta.y / dist * max_speed);
        }

        Vec2::new(
            predicted.x / pred_dist * max_speed,
            predicted.y / pred_dist * max_speed,
        )
    }
}

impl DroneTask for InterceptTask {
    fn tick(
        &mut self,
        state: &State,
        swarm: &[DroneInfo],
        bounds: &Bounds,
        perf: &DronePerfFeatures,
        _dt: f32,
    ) -> TaskOutput {
        let max_speed = perf.max_vel;

        // Find target
        let target = match self.find_target(swarm) {
            Some(t) => t,
            None => {
                self.status = TaskStatus::Failed;
                return TaskOutput {
                    desired_velocity: Vec2::new(0.0, 0.0),
                    desired_heading: None,
                    detonate: false,
                    exclude_from_ca: vec![],
                };
            }
        };

        // Check if within detonation range
        let target_dist = bounds.distance(state.pos.as_vec2(), target.pos.as_vec2());
        if target_dist <= self.detonation_radius {
            self.status = TaskStatus::Complete;
            return TaskOutput {
                desired_velocity: Vec2::new(0.0, 0.0),
                desired_heading: None,
                detonate: true,
                exclude_from_ca: vec![self.target_id],
            };
        }

        // Always pursue — no evasion for terminal intercept
        let desired_vel = self.pursue_velocity(state, target, bounds, max_speed);

        let heading = if desired_vel.magnitude() > f32::EPSILON {
            Some(Heading::new(desired_vel.y.atan2(desired_vel.x)))
        } else {
            None
        };

        TaskOutput {
            desired_velocity: desired_vel,
            desired_heading: heading,
            detonate: false,
            exclude_from_ca: vec![self.target_id],
        }
    }

    fn process_feedback(&mut self, _feedback: &SafetyFeedback) {
        // Terminal intercept ignores safety feedback — stay on target
    }

    fn phase_name(&self) -> &str {
        "pursue"
    }

    fn status(&self) -> TaskStatus {
        self.status
    }

    fn name(&self) -> &str {
        "Intercept"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Acceleration, Heading, Position, Velocity};

    fn make_state(x: f32, y: f32, vx: f32, vy: f32) -> State {
        State {
            pos: Position::new(x, y),
            hdg: Heading::new(vy.atan2(vx)),
            vel: Velocity::new(vx, vy),
            acc: Acceleration::zero(),
        }
    }

    fn make_drone(uid: usize, x: f32, y: f32, vx: f32, vy: f32, group: u32) -> DroneInfo {
        DroneInfo {
            uid,
            pos: Position::new(x, y),
            hdg: Heading::new(vy.atan2(vx)),
            vel: Velocity::new(vx, vy),
            is_formation_leader: false,
            group,
        }
    }

    fn test_perf() -> DronePerfFeatures {
        DronePerfFeatures {
            max_vel: 120.0,
            max_acc: 100.0,
            max_turn_rate: 0.6,
        }
    }

    #[test]
    fn test_pursue_steers_toward_target() {
        let mut task = InterceptTask::new(0, 1, 0, 187.5);
        let state = make_state(0.0, 0.0, 60.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();
        let swarm = vec![
            make_drone(0, 0.0, 0.0, 60.0, 0.0, 0),
            make_drone(1, 1000.0, 0.0, -60.0, 0.0, 1),
        ];

        let output = task.tick(&state, &swarm, &bounds, &test_perf(), 0.016);

        assert!(output.desired_velocity.x > 0.0, "Should pursue toward target");
        assert_eq!(output.exclude_from_ca, vec![1]);
    }

    #[test]
    fn test_target_lost_fails_task() {
        let mut task = InterceptTask::new(0, 99, 0, 187.5);
        let state = make_state(0.0, 0.0, 60.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();
        let swarm = vec![make_drone(0, 0.0, 0.0, 60.0, 0.0, 0)];

        task.tick(&state, &swarm, &bounds, &test_perf(), 0.016);

        assert_eq!(task.status(), TaskStatus::Failed);
    }

    #[test]
    fn test_detonate_when_in_range() {
        let mut task = InterceptTask::new(0, 1, 0, 187.5);
        let state = make_state(0.0, 0.0, 60.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();
        let swarm = vec![
            make_drone(0, 0.0, 0.0, 60.0, 0.0, 0),
            make_drone(1, 100.0, 0.0, -60.0, 0.0, 1), // within 187.5m
        ];

        let output = task.tick(&state, &swarm, &bounds, &test_perf(), 0.016);

        assert!(output.detonate);
        assert_eq!(task.status(), TaskStatus::Complete);
    }

    #[test]
    fn test_no_evasion_on_high_urgency() {
        let mut task = InterceptTask::new(0, 1, 0, 187.5);

        let feedback = SafetyFeedback {
            urgency: 0.8,
            threat_direction: Some(Vec2::new(1.0, 0.0)),
            safe_velocity: Vec2::new(-60.0, 30.0),
        };

        // Terminal intercept ignores feedback — should stay in pursue
        task.process_feedback(&feedback);
        assert_eq!(task.phase_name(), "pursue");
    }

    #[test]
    fn test_exclude_from_ca_set_in_output() {
        let mut task = InterceptTask::new(0, 42, 0, 187.5);
        let state = make_state(0.0, 0.0, 60.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();
        let swarm = vec![
            make_drone(0, 0.0, 0.0, 60.0, 0.0, 0),
            make_drone(42, 1000.0, 0.0, -60.0, 0.0, 1),
        ];

        let output = task.tick(&state, &swarm, &bounds, &test_perf(), 0.016);

        assert_eq!(output.exclude_from_ca, vec![42]);
    }
}
