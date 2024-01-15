use crate::Drone;

use crate::utils::{Objective, ObjectiveType, State};
use std::time::Instant;

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
            pos,
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

    fn move_to_pos(&mut self, pos: (f32, f32), dt: f32) {
        let dx = pos.0 - self.state.pos.0;
        let dy = pos.1 - self.state.pos.1;

        self.state.acc = (
            ((dx - self.state.vel.0 * dt) / (dt.powi(2)))
                .max(-1.0 * self.max_acc)
                .min(self.max_acc),
            ((dy - self.state.vel.1 * dt) / (dt.powi(2)))
                .max(-1.0 * self.max_acc)
                .min(self.max_acc),
        );

        self.state.vel = (
            (self.state.vel.0 + self.state.acc.0 * dt)
                .max(-1.0 * self.max_vel)
                .min(self.max_vel),
            (self.state.vel.1 + self.state.acc.1 * dt)
                .max(-1.0 * self.max_vel)
                .min(self.max_vel),
        );

        self.state.pos = (
            self.state.pos.0 + self.state.vel.0 * dt,
            self.state.pos.1 + self.state.vel.1 * dt,
        );
    }
}

impl Drone for QuadCopter {
    fn state_update(&mut self, timestamp: Instant) {
        let dt = timestamp.duration_since(self.clock_time).as_secs_f32();
        self.clock_time = timestamp;
        println!("dt: {:?}", dt);

        match self.objective.task {
            ObjectiveType::ReachWaypoint => {
                if let Some(pt) = self.objective.waypoints.as_mut() {
                    let dist_to_waypt = ((self.state.pos.0 - pt[0].0).powi(2)
                        + (self.state.pos.1 - pt[0].1).powi(2))
                    .sqrt();
                    if dist_to_waypt < 5.0 {
                        println!("Reached waypoint");
                        pt.remove(0);
                        panic!("Reached waypoint, {:?}", pt);
                    }
                    if !pt.is_empty() {
                        let next_pos = pt[0];
                        self.move_to_pos(next_pos, dt);
                    }
                }
            }
            _ => {
                //println!("Not moving");
            }
        };
    }

    fn task_update(&mut self, objective: Option<Box<Objective>>) {
        if let Some(obj) = objective {
            self.objective = *obj;
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
