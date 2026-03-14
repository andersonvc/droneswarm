//! Evade task: flee from the nearest enemy drone at max speed.
//!
//! Scans for the closest enemy each tick and steers directly away from it.
//! Used by RL agents as a strategic withdrawal action.

use crate::tasks::{DroneTask, SafetyFeedback, TaskOutput, TaskStatus};
use crate::types::{Bounds, DroneInfo, DronePerfFeatures, Heading, State, Vec2};

/// Evade task — flee from the nearest enemy drone.
#[derive(Debug)]
pub struct EvadeTask {
    /// The group this drone belongs to (enemies are anything else).
    self_group: u32,
    /// Task status.
    status: TaskStatus,
}

impl EvadeTask {
    pub fn new(self_group: u32) -> Self {
        EvadeTask {
            self_group,
            status: TaskStatus::Active,
        }
    }
}

impl DroneTask for EvadeTask {
    fn tick(
        &mut self,
        state: &State,
        swarm: &[DroneInfo],
        bounds: &Bounds,
        perf: &DronePerfFeatures,
        _dt: f32,
    ) -> TaskOutput {
        let max_speed = perf.max_vel;
        let my_pos = state.pos.as_vec2();

        // Find nearest enemy drone.
        let nearest_enemy = swarm
            .iter()
            .filter(|d| d.group != self.self_group)
            .min_by(|a, b| {
                let da = bounds.distance(my_pos, a.pos.as_vec2());
                let db = bounds.distance(my_pos, b.pos.as_vec2());
                da.partial_cmp(&db).unwrap()
            });

        let desired_vel = match nearest_enemy {
            Some(enemy) => {
                // Flee: move in opposite direction from enemy.
                let delta = bounds.delta(my_pos, enemy.pos.as_vec2());
                let dist = delta.magnitude();
                if dist < f32::EPSILON {
                    // Sitting on top of enemy — pick arbitrary direction.
                    Vec2::new(max_speed, 0.0)
                } else {
                    Vec2::new(-delta.x / dist * max_speed, -delta.y / dist * max_speed)
                }
            }
            None => {
                // No enemies — task is done.
                self.status = TaskStatus::Complete;
                Vec2::new(0.0, 0.0)
            }
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
        // Evade doesn't change behavior based on safety feedback —
        // it's already maximally fleeing.
    }

    fn phase_name(&self) -> &str {
        "flee"
    }

    fn status(&self) -> TaskStatus {
        self.status
    }

    fn name(&self) -> &str {
        "Evade"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Acceleration, Heading, Position, Velocity};

    fn perf() -> DronePerfFeatures {
        DronePerfFeatures {
            max_vel: 120.0,
            max_acc: 100.0,
            max_turn_rate: 0.6,
        }
    }

    fn bounds() -> Bounds {
        Bounds::new(5000.0, 5000.0).unwrap()
    }

    #[test]
    fn test_flees_from_enemy() {
        let mut task = EvadeTask::new(0);
        let state = State {
            pos: Position::new(500.0, 500.0),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };
        let swarm = vec![
            DroneInfo {
                uid: 0,
                pos: Position::new(500.0, 500.0),
                hdg: Heading::new(0.0),
                vel: Velocity::zero(),
                is_formation_leader: false,
                group: 0,
            },
            DroneInfo {
                uid: 1,
                pos: Position::new(600.0, 500.0), // enemy to the right
                hdg: Heading::new(0.0),
                vel: Velocity::zero(),
                is_formation_leader: false,
                group: 1,
            },
        ];

        let output = task.tick(&state, &swarm, &bounds(), &perf(), 0.016);

        // Should flee to the left (negative x)
        assert!(output.desired_velocity.x < 0.0, "Should flee away from enemy");
        assert_eq!(task.status(), TaskStatus::Active);
    }

    #[test]
    fn test_completes_when_no_enemies() {
        let mut task = EvadeTask::new(0);
        let state = State {
            pos: Position::new(500.0, 500.0),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };
        let swarm = vec![DroneInfo {
            uid: 0,
            pos: Position::new(500.0, 500.0),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
            group: 0,
        }];

        task.tick(&state, &swarm, &bounds(), &perf(), 0.016);

        assert_eq!(task.status(), TaskStatus::Complete);
    }

    #[test]
    fn test_flees_at_max_speed() {
        let mut task = EvadeTask::new(0);
        let state = State {
            pos: Position::new(500.0, 500.0),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };
        let swarm = vec![
            DroneInfo {
                uid: 1,
                pos: Position::new(600.0, 500.0),
                hdg: Heading::new(0.0),
                vel: Velocity::zero(),
                is_formation_leader: false,
                group: 1,
            },
        ];

        let output = task.tick(&state, &swarm, &bounds(), &perf(), 0.016);

        let speed = output.desired_velocity.magnitude();
        assert!((speed - 120.0).abs() < 1.0, "Should flee at max speed, got {}", speed);
    }
}
