use std::collections::HashMap;
use std::collections::VecDeque;
use std::thread;
use std::time::{Duration, Instant};

use drone_lib::{Bounds, Drone, DroneAgent, DroneInfo, Heading, Objective, Position};
use log::info;

type DroneMap = HashMap<usize, Box<dyn Drone>>;

fn main() {
    // Initialize 5 drones using the modular DroneAgent architecture
    let bounds = Bounds::new(1000.0, 1000.0).expect("Invalid bounds");
    let mut drones = (0..5)
        .map(|x| {
            (
                x,
                Box::new(DroneAgent::new(
                    x,
                    Position::new(0.0, 0.0),
                    Heading::new(0.0),
                    bounds,
                )) as Box<dyn Drone>,
            )
        })
        .collect::<DroneMap>();

    // Assign waypoint route to drone 0
    let custom_waypoints: VecDeque<Position> = vec![
        Position::new(100.0, 25.0),
        Position::new(50.0, 200.0),
        Position::new(300.0, 400.0),
    ]
    .into_iter()
    .collect();

    let new_objective = Objective::ReachWaypoint {
        waypoints: custom_waypoints,
    };

    drones
        .get_mut(&0)
        .expect("Drone 0 not found")
        .set_objective(new_objective);

    // Run state machine (update cycle every 2s)
    run_state_machine(&mut drones);
}

fn run_state_machine(drones: &mut DroneMap) {
    let mut last_time = Instant::now();
    loop {
        let now = Instant::now();
        let dt = now.duration_since(last_time).as_secs_f32();
        last_time = now;

        // Collect all drone info first
        let swarm_info: Vec<DroneInfo> = drones.values().map(|d| d.get_info()).collect();

        for (id, drone) in drones.iter_mut() {
            drone.state_update(dt, &swarm_info);
            info!("drone {}: {:?}", id, drone.state());
        }
        thread::sleep(Duration::from_micros(100));
    }
}
