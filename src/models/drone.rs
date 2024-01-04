use crate::utils::{Objective, State};
use std::time::{Duration, Instant};

pub trait Drone {
    fn state_update(&mut self, timestamp: Instant);
    fn task_update(&mut self, objective: Option<Box<Objective>>);
    fn action(&mut self);
    fn broadcast_state(&self) -> &State;
}
