# Business Logic Design Specification
## DroneSwarm - Quadcopter Simulation Engine

**Version:** 1.0
**Date:** 2026-01-06
**Purpose:** Document all simulation entities, physics rules, state transitions, and operations for a 2D quadcopter drone swarm visualization system

---

## 1. Terminology & Definitions

| Term | Definition |
|:-----|:-----------|
| **Drone** | A simulated aerial vehicle with position, velocity, and heading |
| **QuadCopter** | A specific drone type with 4-rotor physics constraints |
| **Swarm** | A collection of drones operating in the same simulation space |
| **Objective** | A task assigned to a drone (e.g., reach waypoint, follow target) |
| **Waypoint** | A 2D coordinate (x, y) that a drone navigates toward |
| **State** | The complete kinematic description of a drone at a point in time |
| **Tick** | A single simulation update cycle |
| **Heading** | The direction a drone is facing (in radians or degrees) |

---

## 2. Core Business Entities

### 2.1 Drone

**Description**: The fundamental simulation unit representing an autonomous aerial vehicle.

**Attributes:**
| Attribute | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `id` | Integer | Yes | Unique identifier for the drone |
| `state` | State | Yes | Current kinematic state (position, velocity, acceleration, heading) |
| `objective` | Objective | No | Currently assigned task, if any |
| `color` | Color | Yes | Visual identifier for rendering (RGB) |

**Relationships:**
- Belongs to one Swarm
- Has one current Objective (or none)
- Has one State at any time

---

### 2.2 State

**Description**: The complete kinematic description of a drone's physical condition.

**Attributes:**
| Attribute | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `position` | (x, y) | Yes | Current 2D coordinates in simulation space |
| `velocity` | (vx, vy) | Yes | Current velocity vector in m/s |
| `acceleration` | (ax, ay) | Yes | Current acceleration vector in m/s² |
| `heading` | Float | Yes | Direction drone is facing (radians, 0 = East, π/2 = North) |

**Derived Values:**
- `speed` = √(vx² + vy²)
- `acceleration_magnitude` = √(ax² + ay²)

---

### 2.3 Objective

**Description**: A task or goal assigned to a drone that determines its behavior.

**Attributes:**
| Attribute | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `type` | ObjectiveType | Yes | The category of objective |
| `waypoints` | [(x, y), ...] | Conditional | List of target positions (for navigation objectives) |
| `target_id` | Integer | Conditional | ID of drone to follow (for FollowTarget) |

**ObjectiveType Enum:**
| Value | Description |
|:------|:------------|
| `ReachWaypoint` | Navigate to one or more waypoints in sequence |
| `FollowTarget` | Track and follow another drone |
| `Loiter` | Hover in place at current position |
| `Sleep` | Inactive, no movement |

---

### 2.4 Swarm

**Description**: A collection of drones operating in shared simulation space.

**Attributes:**
| Attribute | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `drones` | Map<ID, Drone> | Yes | Collection of all drones indexed by ID |
| `simulation_time` | Float | Yes | Current elapsed simulation time in seconds |
| `bounds` | (width, height) | Yes | Simulation space dimensions |
| `speed_multiplier` | Float | Yes | Simulation speed factor (default: 1.0) |
| `selected_drones` | Set<ID> | Yes | Currently selected drone IDs (for multi-select) |

**Relationships:**
- Contains many Drones
- Owns simulation clock
- References SimulationConfig

---

### 2.5 SimulationConfig

**Description**: Configuration settings for simulation initialization and behavior.

**Attributes:**
| Attribute | Type | Required | Description |
|:----------|:-----|:---------|:------------|
| `drone_count` | Integer | Yes | Number of drones to spawn |
| `spawn_pattern` | SpawnPattern | Yes | How drones are initially positioned |
| `bounds` | (width, height) | Yes | Simulation space dimensions |
| `speed_multiplier` | Float | No | Initial speed multiplier (default: 1.0) |
| `predefined_paths` | Map<ID, [Waypoint]> | No | Optional pre-configured drone paths |
| `initial_positions` | [(x, y), ...] | No | Custom spawn positions (for Custom pattern) |
| `cluster_center` | (x, y) | No | Center point for Cluster pattern |
| `cluster_radius` | Float | No | Spread radius for Cluster pattern |

**SpawnPattern Enum:**
| Value | Description |
|:------|:------------|
| `Grid` | Evenly spaced grid pattern across bounds |
| `Random` | Random positions within bounds |
| `Cluster` | Grouped near a center point with configurable density |
| `Custom` | User-specified positions via `initial_positions` |

---

## 3. Entity State Diagrams

### 3.1 Drone Objective State Machine

```
                    ┌─────────────────────┐
                    │                     │
                    ▼                     │
┌─────────┐    ┌─────────┐    ┌──────────────────┐
│  SLEEP  │───▶│ LOITER  │───▶│  REACH_WAYPOINT  │
└─────────┘    └─────────┘    └──────────────────┘
     ▲              ▲                   │
     │              │                   │
     │              │                   ▼
     │              │         ┌──────────────────┐
     │              └─────────│  FOLLOW_TARGET   │
     │                        └──────────────────┘
     │                                  │
     └──────────────────────────────────┘
```

**State Descriptions:**
| State | Description | Entry Condition | Exit Condition |
|:------|:------------|:----------------|:---------------|
| `SLEEP` | Drone is inactive, no physics updates | Explicit assignment or initialization | New objective assigned |
| `LOITER` | Drone hovers at current position | Explicit assignment or waypoint reached | New objective assigned |
| `REACH_WAYPOINT` | Drone navigating to target position(s) | Waypoint objective assigned | All waypoints reached or new objective |
| `FOLLOW_TARGET` | Drone tracking another drone | Follow objective assigned | Target lost or new objective |

---

### 3.2 Waypoint Navigation Sub-States

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ ACCELERATING│───▶│  CRUISING   │───▶│ DECELERATING│
└─────────────┘    └─────────────┘    └─────────────┘
       ▲                                     │
       │                                     │
       │           ┌─────────┐               │
       └───────────│ ARRIVED │◀──────────────┘
                   └─────────┘
```

**Sub-State Descriptions:**
| State | Description | Entry Condition | Exit Condition |
|:------|:------------|:----------------|:---------------|
| `ACCELERATING` | Increasing velocity toward target | Start navigation or new waypoint | Reached max velocity or need to decelerate |
| `CRUISING` | Maintaining max velocity | At max velocity, far from target | Close enough to need deceleration |
| `DECELERATING` | Reducing velocity to stop at target | Within stopping distance | Velocity near zero at target |
| `ARRIVED` | At waypoint position | Within arrival threshold | Next waypoint assigned or objective complete |

---

## 4. Business Events Catalog

### 4.1 Simulation Events

| Event ID | Event Name | Description | Trigger | Side Effects |
|:---------|:-----------|:------------|:--------|:-------------|
| **SIM-001** | SimulationTick | Physics update cycle | Timer interval elapsed | All drone states updated |
| **SIM-002** | SimulationStart | Simulation begins | User initiates | Clock starts, drones activate |
| **SIM-003** | SimulationPause | Simulation pauses | User pauses | Clock stops, states preserved |
| **SIM-004** | SimulationReset | Return to initial state | User resets | All drones return to spawn positions |

### 4.2 Drone Events

| Event ID | Event Name | Description | Trigger | Side Effects |
|:---------|:-----------|:------------|:--------|:-------------|
| **DRN-001** | DroneSpawned | New drone added to swarm | Initialization or user action | Drone added to swarm map |
| **DRN-002** | DroneStateUpdated | Drone physics updated | Simulation tick | Position, velocity, acceleration changed |
| **DRN-003** | ObjectiveAssigned | New task given to drone | User or automation | Drone behavior changes |
| **DRN-004** | WaypointReached | Drone arrived at target | Position within threshold | Next waypoint or objective complete |
| **DRN-005** | ObjectiveCompleted | All waypoints reached | Last waypoint reached | Drone transitions to Loiter |

---

## 5. Business Rules

### 5.1 Physics Constraints

| Rule ID | Description | Enforcement Point |
|:--------|:------------|:------------------|
| **PHY-001** | Maximum velocity is 35.0 m/s | State update calculation |
| **PHY-002** | Maximum acceleration is 7.0 m/s² | State update calculation |
| **PHY-003** | Velocity is clamped to max after acceleration applied | State update calculation |
| **PHY-004** | Acceleration is clamped to max before velocity update | State update calculation |
| **PHY-005** | Position updates via: pos += vel × dt | State update calculation |
| **PHY-006** | Velocity updates via: vel += acc × dt | State update calculation |
| **PHY-007** | Heading is derived from velocity direction when moving | State update calculation |

### 5.2 Navigation Rules

| Rule ID | Description | Enforcement Point |
|:--------|:------------|:------------------|
| **NAV-001** | Waypoint is considered reached when distance < 5.0 units | Waypoint arrival check |
| **NAV-002** | Drone must decelerate to stop at waypoint (no overshoot) | Navigation calculation |
| **NAV-003** | Acceleration direction points toward target waypoint | Navigation calculation |
| **NAV-004** | When multiple waypoints exist, process in sequence | Objective handling |
| **NAV-005** | After last waypoint reached, drone transitions to Loiter | Objective completion |

### 5.3 Simulation Rules

| Rule ID | Description | Enforcement Point |
|:--------|:------------|:------------------|
| **SIM-001** | Simulation tick interval is configurable (default: 16ms for 60fps) | Simulation loop |
| **SIM-002** | All drones update in the same tick (synchronous) | Simulation loop |
| **SIM-003** | Drone IDs must be unique within a swarm | Drone creation |
| **SIM-004** | Simulation space is toroidal: drones wrap to opposite edge when crossing bounds | State update |
| **SIM-005** | Speed multiplier scales effective dt (dt_effective = dt × speed_multiplier) | Simulation loop |
| **SIM-006** | Speed multiplier range: 0.25x to 4.0x (default: 1.0x) | UI control |

### 5.4 Selection Rules

| Rule ID | Description | Enforcement Point |
|:--------|:------------|:------------------|
| **SEL-001** | Click on drone to select it (clears previous selection unless multi-select) | UI interaction |
| **SEL-002** | Ctrl+Click or Shift+Click to add/remove drone from multi-selection | UI interaction |
| **SEL-003** | Click on empty space to deselect all drones | UI interaction |
| **SEL-004** | Selected drones are visually highlighted (ring or glow effect) | Canvas render |
| **SEL-005** | Waypoint assignment applies to all selected drones | Objective assignment |

### 5.5 Visualization Rules

| Rule ID | Description | Enforcement Point |
|:--------|:------------|:------------------|
| **VIS-001** | Each drone rendered as colored circle | Canvas render |
| **VIS-002** | Heading indicator shown as arrow from center | Canvas render |
| **VIS-003** | Drone color must be visually distinct | Drone creation |
| **VIS-004** | Canvas scales to viewport while maintaining aspect ratio | Window resize |
| **VIS-005** | Selected drones show highlight ring | Canvas render |

---

## 6. Business Operations

### 6.1 Initialize Swarm

**Operation:** `initializeSwarm(config: SimulationConfig): Swarm`

**Purpose:** Create a new simulation from configuration.

**Preconditions:**
- config.drone_count > 0
- config.bounds width and height > 0
- If spawn_pattern is Custom: initial_positions.length == drone_count
- If spawn_pattern is Cluster: cluster_center and cluster_radius defined

**Business Logic:**
1. Create empty swarm with given bounds
2. Set speed_multiplier from config (default: 1.0)
3. Calculate spawn positions based on spawn_pattern:
   - **Grid**: Arrange in rows/cols to fit drone_count evenly across bounds
   - **Random**: Generate random (x, y) within bounds for each drone
   - **Cluster**: Distribute around cluster_center within cluster_radius
   - **Custom**: Use provided initial_positions array
4. For each drone (0 to drone_count-1):
   - Generate unique ID
   - Set position from calculated spawn positions
   - Assign distinct color (hue-based: hue = (i / count) × 360°)
   - Initialize state (zero velocity, zero acceleration)
   - If predefined_paths contains ID: set objective to ReachWaypoint with path
   - Else: set objective to Sleep
5. Set simulation time to 0
6. Initialize selected_drones as empty set
7. Return swarm

**Postconditions / Side Effects:**
- Swarm contains drone_count drones
- All drones at rest at spawn positions
- Event DRN-001 emitted for each drone

**Error Conditions:**
| Error Code | Condition | User Message |
|:-----------|:----------|:-------------|
| `INVALID_COUNT` | drone_count <= 0 | "Drone count must be positive" |
| `INVALID_BOUNDS` | bounds contain zero or negative | "Simulation bounds must be positive" |
| `POSITION_MISMATCH` | Custom pattern but positions.length != drone_count | "Position count must match drone count" |
| `MISSING_CLUSTER_CONFIG` | Cluster pattern but center/radius not set | "Cluster pattern requires center and radius" |

---

### 6.2 Update Simulation State

**Operation:** `tick(swarm: Swarm, dt: Float): Swarm`

**Purpose:** Advance simulation by one time step, updating all drone states.

**Preconditions:**
- dt > 0 (positive time delta)
- Simulation is not paused

**Business Logic:**
1. Calculate effective dt: dt_effective = dt × speed_multiplier
2. For each drone in swarm:
   - If objective is Sleep: skip (no state change)
   - If objective is Loiter: maintain position (zero velocity)
   - If objective is ReachWaypoint:
     a. Calculate direction to current waypoint (accounting for toroidal shortest path)
     b. Calculate desired acceleration (toward target)
     c. Apply PHY-002 (clamp acceleration)
     d. Update velocity: vel += acc × dt_effective
     e. Apply PHY-001 (clamp velocity)
     f. Update position: pos += vel × dt_effective
     g. Apply toroidal wrap (SIM-004):
        - If x < 0: x += bounds.width
        - If x >= bounds.width: x -= bounds.width
        - If y < 0: y += bounds.height
        - If y >= bounds.height: y -= bounds.height
     h. Update heading from velocity direction
     i. Check NAV-001 (waypoint reached?)
        - If yes: advance to next waypoint or complete objective
   - If objective is FollowTarget:
     a. Get target drone position
     b. Set current waypoint to target position
     c. Execute ReachWaypoint logic
3. Increment simulation_time by dt_effective
4. Return updated swarm

**Postconditions / Side Effects:**
- All drone states updated
- Event SIM-001 emitted
- Event DRN-002 emitted for each moved drone
- Event DRN-004 emitted for waypoints reached

---

### 6.3 Assign Objective

**Operation:** `assignObjective(swarm: Swarm, droneId: Integer, objective: Objective): Swarm`

**Purpose:** Give a drone a new task to perform.

**Preconditions:**
- Drone with droneId exists in swarm
- Objective is valid for objective type

**Business Logic:**
1. Find drone by ID
2. Replace current objective with new objective
3. If objective is ReachWaypoint: set first waypoint as current target
4. If objective is FollowTarget: validate target drone exists
5. Return updated swarm

**Postconditions / Side Effects:**
- Drone objective updated
- Event DRN-003 emitted

**Error Conditions:**
| Error Code | Condition | User Message |
|:-----------|:----------|:-------------|
| `DRONE_NOT_FOUND` | No drone with given ID | "Drone {id} not found" |
| `INVALID_TARGET` | FollowTarget references non-existent drone | "Target drone {id} not found" |
| `EMPTY_WAYPOINTS` | ReachWaypoint with no waypoints | "Waypoint list cannot be empty" |

---

### 6.4 Get Render State

**Operation:** `getRenderState(swarm: Swarm): [DroneRenderData]`

**Purpose:** Extract minimal data needed to render all drones.

**Preconditions:**
- Swarm exists

**Business Logic:**
1. For each drone in swarm:
   - Extract id, position (x, y), heading, color
   - Package into DroneRenderData
2. Return array of render data

**Output Format:**
```
DroneRenderData {
  id: Integer
  x: Float
  y: Float
  heading: Float
  color: (r, g, b)
  selected: Boolean
}
```

---

### 6.5 Select Drone

**Operation:** `selectDrone(swarm: Swarm, droneId: Integer, multiSelect: Boolean): Swarm`

**Purpose:** Select a drone for waypoint assignment or inspection.

**Preconditions:**
- Drone with droneId exists in swarm

**Business Logic:**
1. If multiSelect is false:
   - Clear selected_drones set
2. If droneId is already in selected_drones:
   - Remove droneId from selected_drones (toggle off)
3. Else:
   - Add droneId to selected_drones
4. Return updated swarm

**Postconditions / Side Effects:**
- selected_drones updated
- Event SEL-001 emitted

---

### 6.6 Clear Selection

**Operation:** `clearSelection(swarm: Swarm): Swarm`

**Purpose:** Deselect all drones.

**Business Logic:**
1. Clear selected_drones set
2. Return updated swarm

---

### 6.7 Set Speed Multiplier

**Operation:** `setSpeedMultiplier(swarm: Swarm, multiplier: Float): Swarm`

**Purpose:** Adjust simulation speed.

**Preconditions:**
- multiplier >= 0.25 and multiplier <= 4.0

**Business Logic:**
1. Set swarm.speed_multiplier = multiplier
2. Return updated swarm

**Error Conditions:**
| Error Code | Condition | User Message |
|:-----------|:----------|:-------------|
| `INVALID_MULTIPLIER` | multiplier < 0.25 or > 4.0 | "Speed must be between 0.25x and 4x" |

---

### 6.8 Assign Waypoint to Selection

**Operation:** `assignWaypointToSelection(swarm: Swarm, waypoint: (x, y)): Swarm`

**Purpose:** Set waypoint for all selected drones.

**Preconditions:**
- selected_drones is not empty

**Business Logic:**
1. For each droneId in selected_drones:
   - Create ReachWaypoint objective with [waypoint]
   - Assign objective to drone
2. Return updated swarm

**Postconditions / Side Effects:**
- All selected drones now navigating to waypoint
- Event DRN-003 emitted for each drone

**Error Conditions:**
| Error Code | Condition | User Message |
|:-----------|:----------|:-------------|
| `NO_SELECTION` | selected_drones is empty | "No drones selected" |

---

### 6.9 Assign Path to Selection

**Operation:** `assignPathToSelection(swarm: Swarm, waypoints: [(x, y), ...]): Swarm`

**Purpose:** Set multi-waypoint path for all selected drones.

**Preconditions:**
- selected_drones is not empty
- waypoints is not empty

**Business Logic:**
1. For each droneId in selected_drones:
   - Create ReachWaypoint objective with waypoints array
   - Assign objective to drone
2. Return updated swarm

**Postconditions / Side Effects:**
- All selected drones now following path
- Event DRN-003 emitted for each drone

---

## 7. Calculations & Algorithms

### 7.1 Acceleration Toward Target

**Purpose:** Calculate acceleration vector to reach a target position smoothly.

**Formula:**
```
direction = normalize(target - position)
distance = |target - position|

if distance > stopping_distance:
    acceleration = direction × MAX_ACCELERATION
else:
    // Decelerate to stop at target
    required_decel = (speed²) / (2 × distance)
    acceleration = -velocity_direction × min(required_decel, MAX_ACCELERATION)
```

**Stopping Distance Calculation:**
```
stopping_distance = (current_speed²) / (2 × MAX_ACCELERATION)
```

**Examples:**
| Current Speed | Distance to Target | Acceleration | Result |
|:--------------|:-------------------|:-------------|:-------|
| 0 m/s | 100 units | 7.0 m/s² toward | Accelerating |
| 35 m/s | 100 units | 0 m/s² | Cruising (at max) |
| 35 m/s | 87.5 units | -7.0 m/s² | Decelerating (stopping_distance = 87.5) |
| 5 m/s | 2 units | -6.25 m/s² | Gentle deceleration |

---

### 7.2 Heading from Velocity

**Purpose:** Calculate heading angle from velocity vector.

**Formula:**
```
if speed > 0.1:  // Only update if moving
    heading = atan2(vy, vx)
// Otherwise keep previous heading
```

**Convention:**
- 0 radians = East (+X direction)
- π/2 radians = North (+Y direction)
- π radians = West (-X direction)
- -π/2 radians = South (-Y direction)

---

### 7.3 Waypoint Arrival Check

**Purpose:** Determine if drone has reached its target waypoint.

**Formula:**
```
distance = sqrt((drone.x - waypoint.x)² + (drone.y - waypoint.y)²)
arrived = distance < ARRIVAL_THRESHOLD  // 5.0 units
```

---

### 7.4 Toroidal Shortest Path

**Purpose:** Calculate direction to target in toroidal (wrap-around) space.

**Formula:**
```
// For each axis, find shortest distance considering wrap-around
dx = target.x - drone.x
dy = target.y - drone.y

// Check if wrapping is shorter for X
if abs(dx) > bounds.width / 2:
    if dx > 0:
        dx = dx - bounds.width  // Wrap left
    else:
        dx = dx + bounds.width  // Wrap right

// Check if wrapping is shorter for Y
if abs(dy) > bounds.height / 2:
    if dy > 0:
        dy = dy - bounds.height  // Wrap down
    else:
        dy = dy + bounds.height  // Wrap up

direction = normalize(dx, dy)
distance = sqrt(dx² + dy²)
```

**Example (1000x1000 bounds):**
| Drone Pos | Target Pos | Direct | Wrapped | Chosen Path |
|:----------|:-----------|:-------|:--------|:------------|
| (100, 500) | (900, 500) | 800 right | 200 left | Wrap left |
| (500, 100) | (500, 900) | 800 up | 200 down | Wrap down |
| (100, 100) | (900, 900) | 1131 diagonal | 283 wrap both | Wrap both |

---

## 8. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Start/Pause │  │   Reset     │  │  Assign Waypoints       │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SIMULATION ENGINE                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    SWARM STATE                            │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │   │
│  │  │ Drone 0 │ │ Drone 1 │ │ Drone 2 │ │ Drone N │  ...    │   │
│  │  │  state  │ │  state  │ │  state  │ │  state  │         │   │
│  │  │  obj    │ │  obj    │ │  obj    │ │  obj    │         │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              │ tick(dt)                          │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  PHYSICS ENGINE                           │   │
│  │  • Apply acceleration toward objectives                   │   │
│  │  • Clamp velocity and acceleration                        │   │
│  │  • Update positions                                       │   │
│  │  • Check waypoint arrivals                                │   │
│  │  • Enforce bounds                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │ getRenderState()
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RENDER LAYER                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    2D CANVAS                              │   │
│  │                                                           │   │
│  │     ●→         Drone: colored circle                      │   │
│  │                Arrow: heading indicator                   │   │
│  │         ●→                                                │   │
│  │                      ●→                                   │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Non-Functional Business Requirements

| Requirement | Description | Target |
|:------------|:------------|:-------|
| Frame Rate | Smooth visual updates | 60 FPS (16ms tick) |
| Drone Count | Concurrent drones supported | 100+ drones |
| Responsiveness | UI remains responsive during simulation | < 16ms per tick |
| Deployment | No backend server required | Static hosting (e.g., GitHub Pages) |
| Compatibility | Browser support | Modern browsers with WASM support |

---

## 10. Out of Scope (v1.0)

The following features are explicitly **not included** in the initial version:

- **3D visualization** - 2D canvas only
- **Collision avoidance** - Drones may overlap
- **Inter-drone communication** - No swarm coordination
- **Realistic flight dynamics** - Simplified point-mass physics
- **Battery/fuel simulation** - Infinite endurance
- **Obstacles/terrain** - Empty simulation space
- **Network multiplayer** - Single-user only
- **Persistent state** - No save/load functionality
- **Real-time external data** - No GPS, weather, etc.

---

## 11. Implementation Checklist

### Core Simulation (Rust/WASM)
- [ ] Implement SimulationConfig struct with spawn patterns
- [ ] Implement Swarm struct with drone collection and selection
- [ ] Implement Drone state management
- [ ] Implement Objective types and state machine
- [ ] Implement physics tick with velocity/acceleration clamping
- [ ] Implement toroidal boundary wrap-around
- [ ] Implement toroidal shortest-path navigation
- [ ] Implement waypoint navigation with smooth deceleration
- [ ] Implement speed multiplier support
- [ ] Implement getRenderState export for JS consumption
- [ ] Add WASM bindings for all public operations

### Web Frontend
- [ ] Set up canvas rendering pipeline
- [ ] Implement drone circle rendering with colors
- [ ] Implement heading arrow rendering
- [ ] Implement selection highlight rendering
- [ ] Wire up requestAnimationFrame loop
- [ ] Implement Start/Pause/Reset controls
- [ ] Implement speed multiplier slider (0.25x - 4x)
- [ ] Implement drone selection (click to select)
- [ ] Implement multi-select (Ctrl/Shift+click)
- [ ] Implement waypoint assignment UI (click on canvas)
- [ ] Implement config-based path loading
- [ ] Handle window resize with proper scaling

### Integration
- [ ] Connect WASM exports to Svelte component
- [ ] Implement state synchronization (Rust → JS)
- [ ] Load simulation config from JSON/config file
- [ ] Verify 60fps rendering with 100 drones
- [ ] Deploy to static hosting

---

## 12. Appendix: Default Configuration

### Physics Constants
| Parameter | Default Value | Description |
|:----------|:--------------|:------------|
| MAX_VELOCITY | 35.0 m/s | Maximum drone speed |
| MAX_ACCELERATION | 7.0 m/s² | Maximum acceleration/deceleration |
| ARRIVAL_THRESHOLD | 5.0 units | Distance to consider waypoint reached |

### Simulation Settings
| Parameter | Default Value | Description |
|:----------|:--------------|:------------|
| TICK_INTERVAL | 16 ms | Time between simulation updates |
| DEFAULT_DRONE_COUNT | 5 | Initial number of drones |
| SIMULATION_BOUNDS | (1000, 1000) | Default simulation space size |
| SPEED_MULTIPLIER | 1.0 | Simulation speed (0.25 - 4.0) |
| SPAWN_PATTERN | Grid | Default spawn arrangement |
| BOUNDARY_MODE | Toroidal | Wrap-around boundaries |

### Example Configuration (JSON)
```json
{
  "drone_count": 10,
  "spawn_pattern": "grid",
  "bounds": { "width": 1000, "height": 1000 },
  "speed_multiplier": 1.0,
  "predefined_paths": {
    "0": [[100, 100], [900, 100], [900, 900], [100, 900]],
    "1": [[500, 500]]
  }
}
```
