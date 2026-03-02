# Autonomy Stack Modular Breakout Specification

## Overview

This document proposes a MOSA-aligned refactoring of the drone-lib autonomy stack, decomposing the monolithic `FixedWing` implementation into three cohesive modules: **Platform**, **Behaviors**, and **Missions**.

## Current Architecture Analysis

### Existing Structure

```
drone-lib/
├── models/
│   ├── drone.rs          # Drone trait (interface)
│   └── fixed_wing.rs     # Monolithic 719-line implementation
├── behaviors/
│   ├── velocity_obstacle.rs
│   └── separation.rs
└── types/
    ├── state.rs          # State + Objective (mixed concerns)
    ├── physics.rs
    └── ...
```

### Current FixedWing Responsibilities (Tightly Coupled)

The `FixedWing` struct currently handles:

1. **Platform concerns**: Performance parameters (`DronePerfFeatures`), physics constraints (turn rate scaling, acceleration limits)
2. **Behavior concerns**: Collision avoidance orchestration, heading selection, movement execution
3. **Mission concerns**: Waypoint management, objective state machine, route following

### Key Pain Points

| Issue | Location | Impact |
|-------|----------|--------|
| Objective enum embedded in state | `types/state.rs` | Mission logic cannot be tested independently |
| Physics + behavior in one method | `move_to_heading()` | Cannot swap avoidance algorithms |
| Hardcoded behavior configs | `FixedWing` fields | Cannot compose different behavior sets |
| Spline computation in drone model | `compute_smoothed_heading()` | Path planning mixed with flight control |

---

## MOSA Principles Applied

Following [DoD MOSA guidelines](https://www.cto.mil/sea/mosa/), this refactoring targets:

1. **Loose coupling**: Modules communicate via well-defined interfaces
2. **High cohesion**: Each module has a single, clear responsibility
3. **Open interfaces**: Traits define contracts, not implementations
4. **Substitutability**: Modules can be replaced without affecting others

### Reference Architecture Alignment

The proposed structure aligns with the [Drone Reference Architecture](https://www.sciencedirect.com/science/article/abs/pii/S0141933122002356) layering:

| Layer | MOSA Module | Responsibility |
|-------|-------------|----------------|
| Flight Control | Platform | Physical dynamics, actuator limits |
| Flight Guidance | Behaviors | Reactive control, obstacle avoidance |
| Mission Management | Missions | Goal selection, task sequencing |

---

## Proposed Module Architecture

```
drone-lib/
├── platform/
│   ├── mod.rs
│   ├── traits.rs         # Platform trait definition
│   ├── fixed_wing.rs     # Fixed-wing kinematics
│   ├── multirotor.rs     # (future) Quadrotor kinematics
│   └── dynamics.rs       # Physics integration
│
├── behaviors/
│   ├── mod.rs
│   ├── traits.rs         # Behavior trait definition
│   ├── tree/             # Behavior Tree implementation
│   │   ├── mod.rs
│   │   ├── nodes.rs      # BT node types
│   │   ├── composite.rs  # Sequence, Selector, Parallel
│   │   └── decorator.rs  # Repeat, Invert, etc.
│   ├── actions/          # Leaf action nodes
│   │   ├── mod.rs
│   │   ├── seek.rs
│   │   ├── avoid.rs
│   │   └── hover.rs
│   └── conditions/       # Leaf condition nodes
│       ├── mod.rs
│       ├── at_waypoint.rs
│       └── collision_imminent.rs
│
├── missions/
│   ├── mod.rs
│   ├── traits.rs         # Mission trait definition
│   ├── task.rs           # Task enum + state machine
│   ├── waypoint.rs       # Waypoint/route management
│   └── planner.rs        # Path planning (splines, etc.)
│
├── agent/
│   ├── mod.rs
│   └── drone_agent.rs    # Composes Platform + Behaviors + Missions
│
└── types/                # (unchanged, shared types)
```

---

## Module Specifications

### 1. Platform Module

**Purpose**: Define physical characteristics and kinematic constraints of the aerial vehicle.

#### Platform Trait

```rust
pub trait Platform: Send + Sync {
    /// Unique identifier
    fn id(&self) -> usize;

    /// Current physical state
    fn state(&self) -> &State;
    fn state_mut(&mut self) -> &mut State;

    /// Performance envelope
    fn max_velocity(&self) -> f32;
    fn max_acceleration(&self) -> f32;
    fn max_turn_rate(&self) -> f32;

    /// Compute achievable turn rate at current velocity
    fn effective_turn_rate(&self) -> f32;

    /// Apply control input, respecting physical constraints
    fn apply_control(&mut self, desired_heading: Heading, desired_speed: f32, dt: f32);

    /// Get collision radius for avoidance calculations
    fn collision_radius(&self) -> f32;

    /// Create snapshot for swarm communication
    fn to_drone_info(&self) -> DroneInfo;
}
```

#### FixedWing Implementation

```rust
pub struct FixedWingPlatform {
    id: usize,
    state: State,
    perf: FixedWingPerformance,
    bounds: Bounds,
}

pub struct FixedWingPerformance {
    pub max_velocity: f32,        // 100.0 units/s
    pub min_velocity: f32,        // 20.0 units/s (stall speed)
    pub max_acceleration: f32,    // 21.0 units/s²
    pub max_turn_rate: f32,       // 2.0 rad/s
    pub collision_radius: f32,    // 15.0 units
}
```

**Key Design Decision**: Turn rate scales with velocity squared (existing behavior). This models fixed-wing aerodynamics where control authority depends on airspeed.

---

### 2. Behaviors Module

**Purpose**: Implement reactive behaviors using a Behavior Tree architecture.

#### Why Behavior Trees?

Per [Behavior Trees in Robotics and AI](https://arxiv.org/abs/1709.00084):

| Property | FSM (Current) | Behavior Tree |
|----------|---------------|---------------|
| Modularity | States tightly coupled | Subtrees are reusable |
| Readability | Transition spaghetti | Hierarchical, visual |
| Extensibility | Add state = update all transitions | Add subtree = plug in |
| Reactivity | Must encode in transitions | Built-in via tick propagation |

#### Behavior Trait

```rust
pub trait Behavior: Send + Sync {
    /// Execute one tick of the behavior tree
    /// Returns the recommended control output
    fn tick(&mut self, context: &BehaviorContext) -> BehaviorResult;

    /// Reset behavior state (e.g., when mission changes)
    fn reset(&mut self);
}

pub struct BehaviorContext<'a> {
    pub platform: &'a dyn Platform,
    pub swarm: &'a [DroneInfo],
    pub mission: &'a dyn Mission,
    pub dt: f32,
}

pub struct BehaviorResult {
    pub status: BehaviorStatus,
    pub control: Option<ControlOutput>,
}

pub enum BehaviorStatus {
    Running,
    Success,
    Failure,
}

pub struct ControlOutput {
    pub desired_heading: Heading,
    pub desired_speed: f32,
    pub urgency: f32,  // 0.0 = relaxed, 1.0 = emergency
}
```

#### Behavior Tree Node Types

```rust
pub enum BTNode {
    // Composites
    Sequence(Vec<BTNode>),      // Run children until one fails
    Selector(Vec<BTNode>),      // Run children until one succeeds
    Parallel(Vec<BTNode>),      // Run all children simultaneously

    // Decorators
    Repeat(Box<BTNode>, RepeatPolicy),
    Invert(Box<BTNode>),
    ForceSuccess(Box<BTNode>),

    // Leaves
    Action(Box<dyn ActionNode>),
    Condition(Box<dyn ConditionNode>),
}
```

#### Example: Default Flight Behavior Tree

```
Root (Selector)
├── Emergency Avoidance (Sequence)
│   ├── [Condition] CollisionImminent(threshold: 30.0)
│   └── [Action] EmergencyAvoid
│
├── Mission Execution (Sequence)
│   ├── [Condition] HasActiveTask
│   └── Navigate With Avoidance (Selector)
│       ├── Safe Navigation (Sequence)
│       │   ├── [Condition] PathClear(lookahead: 2.5s)
│       │   └── [Action] SeekWaypoint
│       └── [Action] AvoidAndSeek  // VO-based avoidance
│
└── [Action] Loiter  // Fallback: hover/circle
```

#### Migrating Existing Behaviors

| Current | New Location |
|---------|--------------|
| `calculate_velocity_obstacle()` | `actions/avoid.rs` → `AvoidAndSeek` action |
| `calculate_separation()` | `actions/separate.rs` → `Separate` action |
| `compute_smoothed_heading()` | `missions/planner.rs` (moved to Mission layer) |

---

### 3. Missions Module

**Purpose**: Manage high-level goals, task sequencing, and path planning.

#### Mission Trait

```rust
pub trait Mission: Send + Sync {
    /// Current task being executed
    fn current_task(&self) -> &Task;

    /// Get the immediate target (waypoint, position, etc.)
    fn current_target(&self) -> Option<Position>;

    /// Get planned path for visualization/prediction
    fn planned_path(&self) -> &[Position];

    /// Notify mission of progress (called by behaviors)
    fn on_waypoint_reached(&mut self, position: Position);

    /// Check if mission is complete
    fn is_complete(&self) -> bool;

    /// Assign new task
    fn assign(&mut self, task: Task);
}
```

#### Task Enum (Replaces Objective)

```rust
pub enum Task {
    /// Idle - no active goal
    Idle,

    /// Navigate to a single waypoint
    GoTo {
        target: Position,
        clearance: f32,
    },

    /// Follow a sequence of waypoints (one-shot)
    FollowPath {
        waypoints: VecDeque<Position>,
        clearance: f32,
    },

    /// Continuously loop through a route
    Patrol {
        route: Arc<[Position]>,
        current_index: usize,
        clearance: f32,
    },

    /// Track a moving target
    Track {
        target_id: usize,
        offset: Vec2,
    },

    /// Hold position (station-keeping)
    Loiter {
        center: Position,
        radius: f32,
    },
}
```

#### Path Planner

```rust
pub struct PathPlanner {
    smoothing_enabled: bool,
    lookahead_base: f32,
    lookahead_speed_factor: f32,
}

impl PathPlanner {
    /// Compute smoothed heading using Hermite splines + Stanley controller
    pub fn compute_guidance(
        &self,
        current: &State,
        waypoints: &[Position],
    ) -> GuidanceOutput;
}

pub struct GuidanceOutput {
    pub desired_heading: Heading,
    pub distance_to_waypoint: f32,
    pub cross_track_error: f32,
    pub spline_preview: Vec<Position>,  // For rendering
}
```

---

### 4. Agent Module (Composition Layer)

**Purpose**: Compose Platform, Behaviors, and Mission into a complete autonomous agent.

```rust
pub struct DroneAgent<P: Platform, B: Behavior, M: Mission> {
    platform: P,
    behavior: B,
    mission: M,
}

impl<P: Platform, B: Behavior, M: Mission> DroneAgent<P, B, M> {
    pub fn tick(&mut self, swarm: &[DroneInfo], dt: f32) {
        // 1. Build context
        let context = BehaviorContext {
            platform: &self.platform,
            swarm,
            mission: &self.mission,
            dt,
        };

        // 2. Run behavior tree
        let result = self.behavior.tick(&context);

        // 3. Apply control to platform
        if let Some(control) = result.control {
            self.platform.apply_control(
                control.desired_heading,
                control.desired_speed,
                dt,
            );
        }

        // 4. Update mission state
        if let Some(target) = self.mission.current_target() {
            let dist = self.platform.state().pos.distance_to(target);
            if dist < self.mission.current_task().clearance() {
                self.mission.on_waypoint_reached(target);
            }
        }
    }
}
```

#### Factory for Common Configurations

```rust
pub fn create_fixed_wing_agent(
    id: usize,
    initial_state: State,
    bounds: Bounds,
) -> DroneAgent<FixedWingPlatform, DefaultBehaviorTree, StandardMission> {
    DroneAgent {
        platform: FixedWingPlatform::new(id, initial_state, bounds),
        behavior: DefaultBehaviorTree::new(default_vo_config()),
        mission: StandardMission::new(),
    }
}
```

---

## Interface Contracts

### Cross-Module Communication

```
┌─────────────────────────────────────────────────────────────┐
│                        DroneAgent                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Mission   │───▶│  Behaviors  │───▶│  Platform   │     │
│  │             │    │             │    │             │     │
│  │ current_    │    │ tick()      │    │ apply_      │     │
│  │ target()    │    │   │         │    │ control()   │     │
│  │             │    │   ▼         │    │             │     │
│  │ planned_    │    │ Control     │    │ state()     │     │
│  │ path()      │    │ Output      │    │             │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         ▲                  │                  │             │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                    BehaviorContext                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Swarm (owns    │
                    │  Vec<DroneAgent>)│
                    └─────────────────┘
```

### Data Flow

1. **Mission → Behaviors**: Target position, planned path, task type
2. **Behaviors → Platform**: Desired heading, desired speed
3. **Platform → Behaviors**: Current state, constraints, drone info
4. **Swarm → Behaviors**: Other drone positions/velocities (via `&[DroneInfo]`)

---

## Migration Strategy

### Phase 1: Extract Platform (Low Risk)

1. Create `platform/traits.rs` with `Platform` trait
2. Move physics from `FixedWing` to `FixedWingPlatform`
3. `FixedWing` holds `FixedWingPlatform` internally (adapter pattern)
4. **Tests**: Existing tests pass unchanged

### Phase 2: Extract Missions (Medium Risk)

1. Create `missions/traits.rs` with `Mission` trait
2. Move `Objective` handling to `StandardMission`
3. Move path planning (`compute_smoothed_heading`) to `PathPlanner`
4. `FixedWing` holds `StandardMission` internally
5. **Tests**: Add mission-specific unit tests

### Phase 3: Introduce Behavior Trees (Higher Risk)

1. Implement BT infrastructure (`tree/`)
2. Port `calculate_velocity_obstacle` to `AvoidAndSeek` action
3. Port `calculate_separation` to `Separate` action
4. Build default behavior tree matching current logic
5. **Tests**: Behavior parity tests (same inputs → same outputs)

### Phase 4: Compose Agent (Final)

1. Create `DroneAgent<P, B, M>` composition
2. Update `Swarm` to use `DroneAgent`
3. Remove legacy `FixedWing` (or keep as alias)
4. **Tests**: Full integration tests

---

## Testing Strategy

### Unit Tests (Per Module)

```rust
// Platform tests
#[test]
fn fixed_wing_respects_turn_rate_at_low_speed() { ... }

// Behavior tests
#[test]
fn avoid_action_returns_evasive_heading_when_collision_imminent() { ... }

// Mission tests
#[test]
fn patrol_task_loops_to_start_after_last_waypoint() { ... }
```

### Integration Tests

```rust
#[test]
fn agent_avoids_collision_while_following_route() {
    let mut agent = create_fixed_wing_agent(...);
    let obstacle = DroneInfo { pos: ahead_of_agent, ... };

    for _ in 0..100 {
        agent.tick(&[obstacle], 0.016);
    }

    assert!(no_collision_occurred());
    assert!(agent_made_progress_toward_waypoint());
}
```

### Behavior Parity Tests

Ensure refactored code produces equivalent behavior:

```rust
#[test]
fn new_agent_matches_legacy_fixed_wing_behavior() {
    let legacy = FixedWing::new(...);
    let new_agent = create_fixed_wing_agent(...);

    for _ in 0..1000 {
        let swarm = [...];
        legacy.state_update(dt, &swarm);
        new_agent.tick(&swarm, dt);

        assert_states_approximately_equal(
            legacy.state(),
            new_agent.platform.state(),
        );
    }
}
```

---

## Design Decisions

### 1. Behavior Tree Execution Model

**Decision**: BT ticks once per frame (synchronous).

```rust
impl Behavior for BehaviorTree {
    fn tick(&mut self, context: &BehaviorContext) -> BehaviorResult {
        // Single traversal per frame
        self.root.tick(context)
    }
}
```

**Future consideration**: Migrate to async/coroutine model for long-running actions (e.g., search patterns, complex maneuvers). This will require:
- `async fn tick()` signature
- Cooperative yielding within action nodes
- Frame budget management

### 2. Mission Assignment via Command Queue

**Decision**: Missions are assigned through a command queue, not directly.

```rust
pub enum AgentCommand {
    AssignTask(Task),
    AbortTask,
    SetPriority(u8),
    UpdateBehaviorParam { key: String, value: f32 },
    ReplaceBehaviorTree(BehaviorTree),
}

pub struct CommandQueue {
    commands: VecDeque<AgentCommand>,
}

impl DroneAgent {
    pub fn enqueue(&mut self, cmd: AgentCommand) {
        self.command_queue.push_back(cmd);
    }

    pub fn tick(&mut self, swarm: &[DroneInfo], dt: f32) {
        // 1. Process pending commands
        while let Some(cmd) = self.command_queue.pop_front() {
            self.process_command(cmd);
        }

        // 2. Run behavior tree
        // ...
    }
}
```

**Benefits**:
- Decouples command source (UI, swarm coordinator, external API) from agent
- Enables command batching and prioritization
- Supports undo/replay for debugging

### 3. Swarm-Level Behaviors

**Decision**: Yes, implement `SwarmBehavior` trait for collective behaviors.

```rust
pub trait SwarmBehavior: Send + Sync {
    /// Execute swarm-level behavior, potentially issuing commands to agents
    fn tick(&mut self, context: &SwarmBehaviorContext) -> SwarmBehaviorResult;

    fn reset(&mut self);
}

pub struct SwarmBehaviorContext<'a> {
    pub agents: &'a [DroneAgent],
    pub swarm_state: &'a SwarmState,
    pub dt: f32,
}

pub struct SwarmBehaviorResult {
    pub status: BehaviorStatus,
    /// Commands to dispatch to individual agents
    pub agent_commands: Vec<(usize, AgentCommand)>,
}
```

#### Example Swarm Behaviors

```rust
pub struct FormationBehavior {
    formation: FormationType,
    leader_id: Option<usize>,
    spacing: f32,
}

pub enum FormationType {
    Line { heading: Heading },
    Wedge { angle: f32 },
    Circle { radius: f32 },
    Grid { rows: usize, cols: usize },
}

pub struct ConsensusRendezvous {
    target: Option<Position>,
    convergence_threshold: f32,
}

pub struct AreaCoverage {
    region: Bounds,
    cell_size: f32,
    assigned_cells: HashMap<usize, Vec<GridCell>>,
}
```

#### Swarm Tick Order

```rust
impl Swarm {
    pub fn tick(&mut self, dt: f32) {
        // 1. Run swarm-level behaviors (may enqueue agent commands)
        for swarm_behavior in &mut self.swarm_behaviors {
            let result = swarm_behavior.tick(&self.build_swarm_context(dt));
            for (agent_id, cmd) in result.agent_commands {
                self.agents[agent_id].enqueue(cmd);
            }
        }

        // 2. Build swarm info snapshot
        let swarm_info: Vec<DroneInfo> = self.agents
            .iter()
            .map(|a| a.platform.to_drone_info())
            .collect();

        // 3. Tick individual agents (processes commands, runs agent BT)
        for agent in &mut self.agents {
            agent.tick(&swarm_info, dt);
        }

        // 4. Collision detection
        self.detect_collisions();
    }
}
```

### 4. Runtime-Modifiable Behavior Trees

**Decision**: Behavior trees are fully modifiable at runtime.

```rust
pub struct BehaviorTree {
    root: BTNode,
    /// Registry of named subtrees for hot-swapping
    subtree_registry: HashMap<String, BTNode>,
}

impl BehaviorTree {
    /// Replace entire tree
    pub fn replace(&mut self, new_root: BTNode) {
        self.root = new_root;
    }

    /// Replace a named subtree
    pub fn replace_subtree(&mut self, name: &str, subtree: BTNode) -> Result<(), BTreeError> {
        self.find_and_replace(&mut self.root, name, subtree)
    }

    /// Update parameter on action/condition nodes
    pub fn set_param(&mut self, node_id: &str, key: &str, value: f32) -> Result<(), BTreeError> {
        self.find_node_mut(node_id)?.set_param(key, value)
    }
}
```

#### Serialization for External Editing

```rust
#[derive(Serialize, Deserialize)]
pub struct BTNodeSpec {
    pub node_type: String,
    pub id: Option<String>,
    pub params: HashMap<String, serde_json::Value>,
    pub children: Vec<BTNodeSpec>,
}

impl BehaviorTree {
    pub fn from_spec(spec: BTNodeSpec) -> Result<Self, BTreeError> { ... }
    pub fn to_spec(&self) -> BTNodeSpec { ... }
}
```

**Use cases**:
- Live tuning via UI sliders (avoidance weight, lookahead time)
- A/B testing different behavior strategies
- Loading behavior profiles from JSON/YAML
- Future: visual behavior tree editor in webapp

---

## Updated Module Structure

```
drone-lib/
├── platform/
│   ├── mod.rs
│   ├── traits.rs
│   ├── fixed_wing.rs
│   └── dynamics.rs
│
├── behaviors/
│   ├── mod.rs
│   ├── traits.rs
│   ├── tree/
│   │   ├── mod.rs
│   │   ├── node.rs           # BTNode enum + serialization
│   │   ├── composite.rs
│   │   ├── decorator.rs
│   │   └── registry.rs       # Subtree hot-swap support
│   ├── actions/
│   └── conditions/
│
├── missions/
│   ├── mod.rs
│   ├── traits.rs
│   ├── task.rs
│   ├── waypoint.rs
│   ├── planner.rs
│   └── command.rs            # AgentCommand + CommandQueue
│
├── swarm/
│   ├── mod.rs
│   ├── traits.rs             # SwarmBehavior trait
│   ├── formation.rs
│   ├── consensus.rs
│   └── coverage.rs
│
├── agent/
│   ├── mod.rs
│   └── drone_agent.rs
│
└── types/
```

---

## References

- [DoD MOSA Overview](https://www.cto.mil/sea/mosa/)
- [MOSA Implementation Guidebook (Feb 2025)](https://www.cto.mil/wp-content/uploads/2025/03/MOSA-Implementation-Guidebook-27Feb2025-Cleared.pdf)
- [Behavior Trees in Robotics and AI](https://arxiv.org/abs/1709.00084)
- [Drone Reference Architecture](https://www.sciencedirect.com/science/article/abs/pii/S0141933122002356)
- [NIST Autonomy Levels for Unmanned Systems](https://www.govinfo.gov/content/pkg/GOVPUB-C13-cbc9faa25f6d651e046c9df607d40d59/pdf/GOVPUB-C13-cbc9faa25f6d651e046c9df607d40d59.pdf)
- [A Survey of Behavior Trees in Robotics and AI](https://www.sciencedirect.com/science/article/pii/S0921889022000513)
