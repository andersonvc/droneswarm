mod models {
    pub mod drone;
    pub mod quadcopter;
}
mod utils;

use std::collections::HashMap;
use std::thread;
use std::time::{Duration, Instant};

use models::{drone::Drone, quadcopter::QuadCopter};
use utils::{Objective, ObjectiveType};

type DroneMap = HashMap<usize, Box<dyn Drone>>;


fn main() {
    // Initialize 5 quadcopters
    let mut drones = (0..5)
        .map(|x| {
            (
                x,
                Box::new(QuadCopter::new(x, (0.0, 0.0))) as Box<dyn Drone>,
            )
        })
        .collect::<DroneMap>();

    // Assign waypoint route to drone 0
    let custom_waypoints = vec![(1.0, 1.0), (50.0, 200.0), (3000.0, 400.0)];
    let new_objective = Some(Box::new(Objective {
        task: ObjectiveType::ReachWaypoint,
        waypoints: Some(custom_waypoints),
        targets: None,
    }));
    drones
        .get_mut(&0)
        .as_mut()
        .unwrap()
        .task_update(new_objective);

    // Run state machine (update cycle every 2s)
    run_state_machine(&mut drones);
}

fn run_state_machine(drones: &mut DroneMap) {
    loop {
        drones.iter_mut().for_each(|(x, v)| {
            v.state_update(Instant::now());
            v.action();
            println!("drone {:?}: {:?}", x, v.broadcast_state());
        });
        thread::sleep(Duration::from_micros(100));
    }
}
