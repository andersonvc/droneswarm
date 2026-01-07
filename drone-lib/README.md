# drone-lib

A Rust library for simulating fixed-wing drone swarms in a toroidal 2D world. Designed for WASM compilation and real-time visualization.

## Features

- **Type-safe physics**: Newtypes for `Position`, `Velocity`, `Acceleration`, and `Heading` prevent unit confusion
- **Toroidal world**: Positions wrap around world boundaries seamlessly
- **Fixed-wing dynamics**: Realistic turn constraints that require velocity to maneuver
- **WASM-compatible**: No `std::time` or threading dependencies in the library
- **Thread-safe**: `Drone` trait requires `Send + Sync` for parallel simulation support

## Architecture

```
drone-lib/
├── src/
│   ├── lib.rs                 # Crate root with re-exports
│   ├── main.rs                # Example binary (drone-sim)
│   ├── models/
│   │   ├── drone.rs           # Drone trait definition
│   │   └── fixed_wing.rs      # Fixed-wing implementation
│   └── types/
│       ├── vec2.rs            # 2D vector math
│       ├── heading.rs         # Normalized angle type
│       ├── physics.rs         # Position, Velocity, Acceleration
│       ├── bounds.rs          # World bounds with toroidal wrapping
│       ├── state.rs           # State, Objective, DroneInfo
│       └── error.rs           # Error types
```

## Core Types

### Physics Primitives

| Type | Description |
|------|-------------|
| `Vec2` | General-purpose 2D vector with public `x`, `y` fields |
| `Position` | World-space position (newtype over Vec2) |
| `Velocity` | Velocity vector in units/second |
| `Acceleration` | Acceleration vector in units/second² |
| `Heading` | Normalized angle in radians, always in `[-π, π]` |

### Simulation Types

| Type | Description |
|------|-------------|
| `Bounds` | World dimensions with toroidal distance/wrapping |
| `State` | Complete drone state (position, velocity, acceleration, heading) |
| `Objective` | Drone task with waypoints and route data |
| `DronePerfFeatures` | Flight parameters (max velocity, acceleration, turn rate) |
| `DroneInfo` | Lightweight state snapshot for swarm communication |

## Usage

### Creating a Drone

```rust
use drone_lib::{FixedWing, Position, Heading, Bounds};

let bounds = Bounds::new(1000.0, 1000.0).expect("valid bounds");
let drone = FixedWing::new(
    0,                           // unique ID
    Position::new(100.0, 100.0), // starting position
    Heading::new(0.0),           // facing right (+x)
    bounds,
);
```

### Assigning Waypoints

```rust
use drone_lib::{Objective, ObjectiveType, Position};
use std::collections::VecDeque;

let waypoints: VecDeque<Position> = vec![
    Position::new(200.0, 200.0),
    Position::new(300.0, 100.0),
    Position::new(400.0, 300.0),
].into_iter().collect();

drone.set_objective(Objective {
    task: ObjectiveType::ReachWaypoint,
    waypoints,
    route: None,
    targets: None,
});
```

### Running the Simulation

```rust
use drone_lib::DroneInfo;

// Collect swarm info for inter-drone awareness
let swarm_info: Vec<DroneInfo> = drones.iter()
    .map(|d| d.get_info())
    .collect();

// Update each drone with delta time
for drone in &mut drones {
    drone.state_update(dt, &swarm_info);
}
```

### Looping Routes

Use `ObjectiveType::FollowRoute` with an `Arc<[Position]>` for routes that repeat:

```rust
use std::sync::Arc;

let route: Arc<[Position]> = Arc::from(vec![
    Position::new(100.0, 100.0),
    Position::new(900.0, 100.0),
    Position::new(900.0, 900.0),
    Position::new(100.0, 900.0),
]);

drone.set_objective(Objective {
    task: ObjectiveType::FollowRoute,
    waypoints: route.iter().copied().collect(),
    route: Some(route),
    targets: None,
});
```

## Toroidal World

The simulation uses toroidal (wraparound) geometry:

```rust
let bounds = Bounds::new(1000.0, 1000.0).unwrap();

// Position wrapping
let pos = Position::new(1050.0, -50.0);
let wrapped = bounds.wrap_position(pos);
// wrapped = Position(50.0, 950.0)

// Shortest distance calculation
let a = Vec2::new(950.0, 500.0);
let b = Vec2::new(50.0, 500.0);
let dist = bounds.toroidal_distance(a, b);
// dist = 100.0 (wraps around, not 900.0)

// Direction vector for navigation
let delta = bounds.toroidal_delta(a, b);
// delta = Vec2(100.0, 0.0) (points right, through the wrap)
```

## Fixed-Wing Physics

The `FixedWing` drone models realistic fixed-wing constraints:

- **Turn rate scales with speed²**: Cannot turn without forward velocity
- **Continuous forward motion**: Must maintain speed to maneuver
- **Automatic position wrapping**: Stays within world bounds

Default parameters:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_vel` | 120.0 | Maximum velocity (units/s) |
| `max_acc` | 21.0 | Maximum acceleration (units/s²) |
| `max_turn_rate` | 4.0 | Maximum turn rate at max speed (rad/s) |
| Arrival threshold | 10.0 | Distance to consider waypoint reached |

### Custom Flight Parameters

```rust
use drone_lib::DronePerfFeatures;

let params = DronePerfFeatures::new(
    80.0,   // max velocity
    15.0,   // max acceleration
    3.0,    // max turn rate
).expect("valid params");

drone.set_flight_params(params);
```

## The Drone Trait

Implement custom drone types by implementing the `Drone` trait:

```rust
pub trait Drone: Send + Sync + std::fmt::Debug {
    fn uid(&self) -> usize;
    fn state_update(&mut self, dt: f32, swarm: &[DroneInfo]);
    fn set_objective(&mut self, objective: Objective);
    fn clear_objective(&mut self);
    fn state(&self) -> &State;
    fn get_info(&self) -> DroneInfo;
    fn set_flight_params(&mut self, params: DronePerfFeatures);
}
```

## Error Handling

The library uses `DroneResult<T>` for fallible operations:

```rust
use drone_lib::{Bounds, DronePerfFeatures, DroneError};

// Bounds validation
let result = Bounds::new(-100.0, 100.0);
assert!(matches!(result, Err(DroneError::InvalidBounds { .. })));

// Flight param validation
let result = DronePerfFeatures::new(0.0, 10.0, 2.0);
assert!(matches!(result, Err(DroneError::InvalidFlightParam { .. })));

// Unchecked variants for known-good values (no validation overhead)
let bounds = Bounds::new_unchecked(1000.0, 1000.0);
let params = DronePerfFeatures::new_unchecked(120.0, 21.0, 4.0);
```

## Running the Example

```bash
cargo run --bin drone-sim
```

Note: The example binary uses the `log` crate. To see output, initialize a logger like `env_logger`:

```rust
env_logger::init();
```

## Testing

```bash
cargo test
```

The library includes 41 tests covering:
- Vector math operations
- Heading normalization and interpolation
- Toroidal distance and wrapping
- Waypoint navigation
- Route looping
- Flight parameter validation

## WASM Compatibility

The library is designed for WASM compilation:

- No `std::time` usage (caller provides `dt`)
- No threading primitives
- No file I/O
- Minimal dependencies (`log` only)

See the companion `wasm-lib` crate for the WebAssembly bindings.

## License

[Your license here]
