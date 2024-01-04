use crate::Drone;

use crate::utils::{Objective, ObjectiveType, State};
use std::time::{Duration, Instant};

const MAX_VELOCITY: f32 = 35.0;
const MAX_ACCELERATION: f32 = 7.0;

pub struct QuadCopter {
    pub id: usize,
    state: State,
    objective: Objective,
    clock_time: Instant,
    max_vel: f32,
    max_acc: f32,
}

impl QuadCopter {
    pub fn new(id: usize, pos: (f32, f32)) -> QuadCopter {
        let state = State {
            hdg: 0.0,
            pos: pos,
            vel: (0.0, 0.0),
            acc: (0.0, 0.0),
        };
        let objective = Objective {
            task: ObjectiveType::Sleep,
            waypoints: None,
            targets: None,
        };
        QuadCopter {
            id,
            state,
            objective,
            clock_time: Instant::now(),
            max_vel: MAX_VELOCITY,
            max_acc: MAX_ACCELERATION,
        }
    }
}

impl Drone for QuadCopter {
    fn state_update(&mut self, timestamp: Instant) {
        let dt = timestamp.duration_since(self.clock_time).as_secs_f32();
        self.clock_time = timestamp;

        self.state.pos = (
            self.state.pos.0 + self.state.vel.0 * dt + 0.5 * self.state.acc.0 * dt.powi(2),
            self.state.pos.1 + self.state.vel.1 * dt + 0.5 * self.state.acc.1 * dt.powi(2),
        );

        self.state.vel = (
            self.state.vel.0 + self.state.acc.0 * dt,
            self.state.vel.1 + self.state.acc.1 * dt,
        );
    }

    fn task_update(&mut self, objective: Option<Box<Objective>>) {
        match objective {
            Some(obj) => {
                self.objective = *obj;
            },
            _ => {}
        };
    }

    fn action(&mut self) {
        match self.objective.task {
            ObjectiveType::ReachWaypoint => {
                println!("Reaching waypoint");
            }
            ObjectiveType::FollowTarget => {
                println!("Following target");
            }
            ObjectiveType::Loiter => {
                println!("Loitering");
            }
            ObjectiveType::Sleep => {
                println!("Sleeping");
            }
        };
    }

    fn broadcast_state(&self) -> &State {
        &self.state
    }
}
