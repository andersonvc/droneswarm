# Implementation Plan
## DroneSwarm - Staged Implementation

**Created:** 2026-01-06
**Spec References:**
- [BUSINESS-LOGIC-SPEC.md](./BUSINESS-LOGIC-SPEC.md) - Domain model and rules
- [UI-SPEC.md](./UI-SPEC.md) - User interface specification
- [TECHNICAL-SPEC.md](./TECHNICAL-SPEC.md) - WASM integration details

---

## Overview

This plan breaks the DroneSwarm implementation into 4 stages with clear dependencies. Each stage produces a working, testable deliverable.

```
┌─────────────────┐
│  Stage 1:       │
│  WASM Foundation│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 2:       │
│  Core Simulation│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 3:       │
│  TypeScript     │
│  Bridge         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 4:       │
│  UI Components  │
└─────────────────┘
```

**Critical Path:** drone-lib refactor → wasm-lib core → WASM exports → TypeScript bridge → Svelte stores → UI components

---

## Stage 1: WASM Foundation

**Goal**: Refactor drone-lib for WASM compatibility and establish the build pipeline

**Status**: Not Started

### Why This First
The existing `drone-lib` uses `std::time::Instant` which is not available in WASM. Nothing else can proceed until this is resolved. This stage also validates the entire WASM toolchain works.

### Tasks

#### 1.1 Refactor drone-lib (MAJ-001 from ANALYSIS-TECHNICAL.md)

**File:** `drone-lib/src/models/quadcopter.rs`

| Line | Current | Change To |
|:-----|:--------|:----------|
| 5 | `use std::time::Instant` | Remove import |
| 12 | `clock_time: Instant` | Remove field |
| 71 | `timestamp: Instant` parameter | `dt: f32` parameter |
| 85 | `panic!("waypoint reached")` | State transition to `Sleep` |

**Changes required:**
```rust
// Before (quadcopter.rs)
pub fn state_update(&mut self, timestamp: Instant) {
    let dt = (timestamp - self.clock_time).as_secs_f32();
    self.clock_time = timestamp;
    // ... physics
}

// After
pub fn state_update(&mut self, dt: f32) {
    // dt passed directly from caller
    // ... physics (unchanged)
}
```

**Also update:**
- Remove `clock_time` from `QuadCopter::new()`
- Update any tests in `drone-lib/src/models/quadcopter.rs` or test files
- Change waypoint reached behavior from panic to state transition

#### 1.2 Create wasm-lib Foundation

**File:** `wasm-lib/Cargo.toml` - Add dependencies:
```toml
[dependencies]
wasm-bindgen = "0.2"
serde = { version = "1.0", features = ["derive"] }
serde-wasm-bindgen = "0.6"
console_error_panic_hook = "0.1"
drone-lib = { path = "../drone-lib" }

[lib]
crate-type = ["cdylib", "rlib"]
```

**File:** `wasm-lib/src/lib.rs` - Minimal implementation:
```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct SwarmHandle {
    // Will hold Swarm in Stage 2
}

#[wasm_bindgen]
pub fn init_swarm(_config: JsValue) -> SwarmHandle {
    SwarmHandle {}
}
```

#### 1.3 Set Up Build Pipeline

**File:** `wasm-lib/build.sh` (or npm script):
```bash
wasm-pack build --target web --out-dir ../webapp/src/lib/wasm/pkg
```

**File:** `webapp/vite.config.ts` - Ensure WASM support:
```typescript
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  optimizeDeps: {
    exclude: ['wasm-lib']
  }
});
```

#### 1.4 Verify WASM Loads

**File:** `webapp/src/routes/+page.svelte` - Minimal test:
```svelte
<script lang="ts">
  import { onMount } from 'svelte';

  onMount(async () => {
    const wasm = await import('$lib/wasm/pkg');
    console.log('WASM loaded successfully');
  });
</script>

<p>Check console for WASM load status</p>
```

### Success Criteria

- [ ] `drone-lib` compiles without `std::time::Instant`
- [ ] `drone-lib` unit tests pass with `dt: f32` parameter
- [ ] `wasm-lib` compiles to WASM with `wasm-pack build`
- [ ] WASM module loads in browser without errors
- [ ] Console shows "WASM loaded successfully"

### Tests

| Test | Type | Description |
|:-----|:-----|:------------|
| QuadCopter dt parameter | Unit | `QuadCopter::state_update(0.016)` works |
| Waypoint reached | Unit | No panic, transitions to Sleep state |
| WASM compilation | Build | `wasm-pack build` succeeds |
| Browser load | Manual | Console shows success message |

### Files Modified/Created

| File | Action |
|:-----|:-------|
| `drone-lib/src/models/quadcopter.rs` | Modify |
| `wasm-lib/Cargo.toml` | Modify |
| `wasm-lib/src/lib.rs` | Rewrite |
| `webapp/src/lib/wasm/pkg/` | Generated |
| `webapp/src/routes/+page.svelte` | Modify |

---

## Stage 2: Core Simulation

**Goal**: Implement full simulation logic in wasm-lib with all physics, spawning, and state management

**Status**: Not Started

**Depends On**: Stage 1 complete

### Tasks

#### 2.1 Define Core Types

**Reference:** TECHNICAL-SPEC.md §3.1 Type Definitions

```rust
// wasm-lib/src/lib.rs

use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DroneRenderData {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub heading: f32,
    pub color: Color,
    pub selected: bool,
    pub objective_type: String,
    pub target: Option<Point>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SwarmStatus {
    pub simulation_time: f32,
    pub drone_count: u32,
    pub selected_count: u32,
    pub speed_multiplier: f32,
    pub is_valid: bool,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationConfig {
    pub drone_count: u32,
    pub spawn_pattern: SpawnPattern,
    pub bounds: Bounds,
    pub speed_multiplier: Option<f32>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpawnPattern {
    Grid,
    Random,
    Cluster { center: Point, radius: f32 },
    Custom { positions: Vec<Point> },
}

#[derive(Clone, Deserialize)]
pub struct Bounds {
    pub width: f32,
    pub height: f32,
}
```

#### 2.2 Implement Swarm Struct

**Reference:** BUSINESS-LOGIC-SPEC.md §3.4 Swarm Entity

```rust
pub struct Swarm {
    drones: Vec<DroneState>,
    bounds: Bounds,
    simulation_time: f32,
    speed_multiplier: f32,
    selected_ids: HashSet<u32>,
}

struct DroneState {
    id: u32,
    quadcopter: QuadCopter,  // from drone-lib
    color: Color,
    selected: bool,
}

impl Swarm {
    pub fn new(config: SimulationConfig) -> Self {
        let positions = Self::generate_spawn_positions(&config);
        let drones = positions.into_iter().enumerate().map(|(i, pos)| {
            DroneState {
                id: i as u32,
                quadcopter: QuadCopter::new(pos.x, pos.y),
                color: Self::generate_color(i, config.drone_count as usize),
                selected: false,
            }
        }).collect();

        Swarm {
            drones,
            bounds: config.bounds,
            simulation_time: 0.0,
            speed_multiplier: config.speed_multiplier.unwrap_or(1.0),
            selected_ids: HashSet::new(),
        }
    }
}
```

#### 2.3 Implement Spawn Patterns

**Reference:** BUSINESS-LOGIC-SPEC.md §7.1 initializeSwarm

```rust
impl Swarm {
    fn generate_spawn_positions(config: &SimulationConfig) -> Vec<Point> {
        match &config.spawn_pattern {
            SpawnPattern::Grid => Self::spawn_grid(config.drone_count, &config.bounds),
            SpawnPattern::Random => Self::spawn_random(config.drone_count, &config.bounds),
            SpawnPattern::Cluster { center, radius } => {
                Self::spawn_cluster(config.drone_count, center, *radius)
            }
            SpawnPattern::Custom { positions } => positions.clone(),
        }
    }

    fn spawn_grid(count: u32, bounds: &Bounds) -> Vec<Point> {
        let cols = (count as f32).sqrt().ceil() as u32;
        let rows = (count + cols - 1) / cols;
        let spacing_x = bounds.width / (cols + 1) as f32;
        let spacing_y = bounds.height / (rows + 1) as f32;

        (0..count).map(|i| {
            let col = i % cols;
            let row = i / cols;
            Point {
                x: spacing_x * (col + 1) as f32,
                y: spacing_y * (row + 1) as f32,
            }
        }).collect()
    }

    fn spawn_random(count: u32, bounds: &Bounds) -> Vec<Point> {
        // Use simple PRNG for deterministic results
        // ...
    }

    fn spawn_cluster(count: u32, center: &Point, radius: f32) -> Vec<Point> {
        // Distribute around center point
        // ...
    }
}
```

#### 2.4 Implement Color Generation

**Reference:** BUSINESS-LOGIC-SPEC.md Rule VIS-003

```rust
impl Swarm {
    fn generate_color(index: usize, total: usize) -> Color {
        // HSL to RGB: hue = index / total * 360, saturation = 70%, lightness = 50%
        let hue = (index as f32 / total as f32) * 360.0;
        let (r, g, b) = hsl_to_rgb(hue, 0.7, 0.5);
        Color { r, g, b }
    }
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}
```

#### 2.5 Implement Physics Tick

**Reference:** BUSINESS-LOGIC-SPEC.md §7.2 tick, Rules PHY-001 to PHY-007

```rust
impl Swarm {
    pub fn tick(&mut self, dt: f32) {
        let effective_dt = dt * self.speed_multiplier;

        for drone in &mut self.drones {
            // Update physics via drone-lib
            drone.quadcopter.state_update(effective_dt);

            // Apply toroidal wrapping (SIM-004)
            let state = drone.quadcopter.state_mut();
            state.pos.x = Self::wrap(state.pos.x, self.bounds.width);
            state.pos.y = Self::wrap(state.pos.y, self.bounds.height);
        }

        self.simulation_time += effective_dt;
    }

    fn wrap(value: f32, max: f32) -> f32 {
        if value < 0.0 {
            value + max
        } else if value >= max {
            value - max
        } else {
            value
        }
    }
}
```

#### 2.6 Implement State Queries

```rust
impl Swarm {
    pub fn get_render_state(&self) -> Vec<DroneRenderData> {
        self.drones.iter().map(|d| {
            let state = d.quadcopter.state();
            let objective = d.quadcopter.objective();

            DroneRenderData {
                id: d.id,
                x: state.pos.x,
                y: state.pos.y,
                heading: state.hdg,
                color: d.color.clone(),
                selected: self.selected_ids.contains(&d.id),
                objective_type: objective.objective_type().to_string(),
                target: objective.target_position(),
            }
        }).collect()
    }

    pub fn get_status(&self) -> SwarmStatus {
        SwarmStatus {
            simulation_time: self.simulation_time,
            drone_count: self.drones.len() as u32,
            selected_count: self.selected_ids.len() as u32,
            speed_multiplier: self.speed_multiplier,
            is_valid: true,
        }
    }

    pub fn get_drone_at(&self, x: f32, y: f32, hit_radius: f32) -> Option<u32> {
        for drone in &self.drones {
            let state = drone.quadcopter.state();
            let dx = state.pos.x - x;
            let dy = state.pos.y - y;
            if dx * dx + dy * dy <= hit_radius * hit_radius {
                return Some(drone.id);
            }
        }
        None
    }
}
```

#### 2.7 Implement Selection

**Reference:** BUSINESS-LOGIC-SPEC.md Rules SEL-001 to SEL-005

```rust
impl Swarm {
    pub fn select_drone(&mut self, id: u32, multi_select: bool) {
        if !multi_select {
            self.selected_ids.clear();
        }

        if self.drones.iter().any(|d| d.id == id) {
            if self.selected_ids.contains(&id) && multi_select {
                self.selected_ids.remove(&id);  // Toggle off
            } else {
                self.selected_ids.insert(id);
            }
        }
    }

    pub fn clear_selection(&mut self) {
        self.selected_ids.clear();
    }
}
```

#### 2.8 Implement Waypoint Assignment

**Reference:** BUSINESS-LOGIC-SPEC.md §7.6 assignWaypointToSelection

```rust
impl Swarm {
    pub fn assign_waypoint(&mut self, x: f32, y: f32) {
        for drone in &mut self.drones {
            if self.selected_ids.contains(&drone.id) {
                drone.quadcopter.set_objective(Objective::ReachWaypoint(Point { x, y }));
            }
        }
    }

    pub fn assign_path(&mut self, waypoints: Vec<Point>) {
        for drone in &mut self.drones {
            if self.selected_ids.contains(&drone.id) {
                drone.quadcopter.set_path(waypoints.clone());
            }
        }
    }

    pub fn set_speed(&mut self, multiplier: f32) {
        self.speed_multiplier = multiplier.clamp(0.25, 4.0);
    }
}
```

### Success Criteria

- [ ] Swarm initializes with correct drone count
- [ ] Grid spawn produces evenly distributed positions
- [ ] Random spawn stays within bounds
- [ ] Cluster spawn centers around specified point
- [ ] tick() advances positions correctly
- [ ] Toroidal wrapping works at all boundaries
- [ ] Color generation produces distinct colors
- [ ] get_render_state() returns all drones
- [ ] get_drone_at() finds drones within hit radius
- [ ] Selection add/remove/toggle works
- [ ] Waypoint assignment affects only selected drones
- [ ] Speed multiplier clamps to valid range

### Tests

| Test | Type | Description |
|:-----|:-----|:------------|
| Grid spawn 9 drones | Unit | 3x3 grid, correct positions |
| Grid spawn 10 drones | Unit | 4x3 grid, correct positions |
| Random bounds | Unit | All positions within bounds |
| Tick physics | Unit | Position changes with velocity |
| Wrap left edge | Unit | x < 0 wraps to width |
| Wrap right edge | Unit | x >= width wraps to 0 |
| Wrap top/bottom | Unit | Same for y axis |
| Color distinct | Unit | No duplicate colors for N drones |
| Hit test center | Unit | Click on drone returns id |
| Hit test edge | Unit | Click within radius returns id |
| Hit test miss | Unit | Click outside all drones returns None |
| Select single | Unit | One drone selected, others cleared |
| Select multi | Unit | Multiple drones in selection set |
| Waypoint to selected | Unit | Only selected drones get waypoint |
| Speed clamp low | Unit | 0.1 clamps to 0.25 |
| Speed clamp high | Unit | 5.0 clamps to 4.0 |

### Files Modified/Created

| File | Action |
|:-----|:-------|
| `wasm-lib/src/lib.rs` | Expand with full implementation |
| `wasm-lib/src/types.rs` | Create (optional, can stay in lib.rs) |

---

## Stage 3: WASM Exports & TypeScript Bridge

**Goal**: Complete WASM API exports and create TypeScript integration layer

**Status**: Not Started

**Depends On**: Stage 2 complete

### Tasks

#### 3.1 Implement WASM Exports

**Reference:** TECHNICAL-SPEC.md §3 WASM API Contract

```rust
// wasm-lib/src/lib.rs

use wasm_bindgen::prelude::*;
use serde_wasm_bindgen;

#[wasm_bindgen]
pub struct SwarmHandle {
    swarm: Swarm,
}

#[wasm_bindgen]
impl SwarmHandle {
    pub fn free(self) {
        // Rust ownership handles cleanup
    }
}

#[wasm_bindgen]
pub fn init_swarm(config: JsValue) -> Result<SwarmHandle, JsValue> {
    let config: SimulationConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(SwarmHandle {
        swarm: Swarm::new(config),
    })
}

#[wasm_bindgen]
pub fn tick(handle: &mut SwarmHandle, dt: f32) {
    handle.swarm.tick(dt);
}

#[wasm_bindgen]
pub fn get_render_state(handle: &SwarmHandle) -> JsValue {
    let state = handle.swarm.get_render_state();
    serde_wasm_bindgen::to_value(&state).unwrap()
}

#[wasm_bindgen]
pub fn get_status(handle: &SwarmHandle) -> JsValue {
    let status = handle.swarm.get_status();
    serde_wasm_bindgen::to_value(&status).unwrap()
}

#[wasm_bindgen]
pub fn select_drone(handle: &mut SwarmHandle, id: u32, multi_select: bool) {
    handle.swarm.select_drone(id, multi_select);
}

#[wasm_bindgen]
pub fn clear_selection(handle: &mut SwarmHandle) {
    handle.swarm.clear_selection();
}

#[wasm_bindgen]
pub fn set_speed(handle: &mut SwarmHandle, multiplier: f32) {
    handle.swarm.set_speed(multiplier);
}

#[wasm_bindgen]
pub fn assign_waypoint(handle: &mut SwarmHandle, x: f32, y: f32) {
    handle.swarm.assign_waypoint(x, y);
}

#[wasm_bindgen]
pub fn assign_path(handle: &mut SwarmHandle, waypoints: JsValue) -> Result<(), JsValue> {
    let waypoints: Vec<Point> = serde_wasm_bindgen::from_value(waypoints)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    handle.swarm.assign_path(waypoints);
    Ok(())
}

#[wasm_bindgen]
pub fn get_drone_at(handle: &SwarmHandle, x: f32, y: f32, hit_radius: f32) -> Option<u32> {
    handle.swarm.get_drone_at(x, y, hit_radius)
}
```

#### 3.2 Create TypeScript Bridge

**Reference:** TECHNICAL-SPEC.md §5.4 Swarm Manager Class

**File:** `webapp/src/lib/wasm/bridge.ts`

```typescript
import init, {
  init_swarm,
  tick,
  get_render_state,
  get_status,
  select_drone,
  clear_selection,
  set_speed,
  assign_waypoint,
  assign_path,
  get_drone_at,
  type SwarmHandle,
} from './pkg';

// Re-export types
export interface Point {
  x: number;
  y: number;
}

export interface DroneRenderData {
  id: number;
  x: number;
  y: number;
  heading: number;
  color: { r: number; g: number; b: number };
  selected: boolean;
  objectiveType: string;
  target?: Point;
}

export interface SwarmStatus {
  simulationTime: number;
  droneCount: number;
  selectedCount: number;
  speedMultiplier: number;
  isValid: boolean;
}

export interface SimulationConfig {
  droneCount: number;
  spawnPattern: 'grid' | 'random' | { cluster: { center: Point; radius: number } } | { custom: { positions: Point[] } };
  bounds: { width: number; height: number };
  speedMultiplier?: number;
}

let wasmInitialized = false;

export async function initWasm(): Promise<void> {
  if (!wasmInitialized) {
    await init();
    wasmInitialized = true;
  }
}

export class SwarmManager {
  private handle: SwarmHandle | null = null;

  async initialize(config: SimulationConfig): Promise<void> {
    await initWasm();
    this.handle = init_swarm(config);
  }

  tick(dt: number): void {
    if (!this.handle) return;
    tick(this.handle, dt);
  }

  getRenderState(): DroneRenderData[] {
    if (!this.handle) return [];
    return get_render_state(this.handle) as DroneRenderData[];
  }

  getStatus(): SwarmStatus {
    if (!this.handle) {
      return {
        simulationTime: 0,
        droneCount: 0,
        selectedCount: 0,
        speedMultiplier: 1.0,
        isValid: false,
      };
    }
    return get_status(this.handle) as SwarmStatus;
  }

  selectDrone(id: number, multi: boolean): void {
    if (!this.handle) return;
    select_drone(this.handle, id, multi);
  }

  clearSelection(): void {
    if (!this.handle) return;
    clear_selection(this.handle);
  }

  setSpeed(speed: number): void {
    if (!this.handle) return;
    set_speed(this.handle, speed);
  }

  assignWaypoint(x: number, y: number): void {
    if (!this.handle) return;
    assign_waypoint(this.handle, x, y);
  }

  assignPath(waypoints: Point[]): void {
    if (!this.handle) return;
    assign_path(this.handle, waypoints);
  }

  getDroneAt(x: number, y: number, hitRadius: number = 20): number | undefined {
    if (!this.handle) return undefined;
    return get_drone_at(this.handle, x, y, hitRadius) ?? undefined;
  }

  destroy(): void {
    if (this.handle) {
      this.handle.free();
      this.handle = null;
    }
  }
}
```

#### 3.3 Create Config Loading

**Reference:** TECHNICAL-SPEC.md §5.2 Configuration Loading

**File:** `webapp/src/lib/wasm/config.ts`

```typescript
import type { SimulationConfig } from './bridge';

const DEFAULT_CONFIG: SimulationConfig = {
  droneCount: 5,
  spawnPattern: 'grid',
  bounds: { width: 1000, height: 1000 },
  speedMultiplier: 1.0,
};

export async function loadConfig(): Promise<SimulationConfig> {
  // 1. Try URL parameters first
  const urlConfig = parseUrlConfig();
  if (urlConfig) {
    console.log('Config loaded from URL parameters');
    return { ...DEFAULT_CONFIG, ...urlConfig };
  }

  // 2. Try loading config.json
  try {
    const response = await fetch('/config.json');
    if (response.ok) {
      const fileConfig = await response.json();
      console.log('Config loaded from config.json');
      return { ...DEFAULT_CONFIG, ...fileConfig };
    }
  } catch (e) {
    console.log('No config.json found');
  }

  // 3. Fall back to defaults
  console.log('Using default configuration');
  return DEFAULT_CONFIG;
}

function parseUrlConfig(): Partial<SimulationConfig> | null {
  if (typeof window === 'undefined') return null;

  const params = new URLSearchParams(window.location.search);
  const config: Partial<SimulationConfig> = {};

  if (params.has('drones')) {
    config.droneCount = parseInt(params.get('drones')!, 10);
  }
  if (params.has('pattern')) {
    config.spawnPattern = params.get('pattern') as 'grid' | 'random';
  }
  if (params.has('speed')) {
    config.speedMultiplier = parseFloat(params.get('speed')!);
  }

  return Object.keys(config).length > 0 ? config : null;
}
```

#### 3.4 Create Svelte Stores

**Reference:** TECHNICAL-SPEC.md §5.5 Svelte Store Integration

**File:** `webapp/src/lib/stores/simulation.ts`

```typescript
import { writable, derived, get } from 'svelte/store';
import { SwarmManager, type DroneRenderData, type SwarmStatus, type SimulationConfig } from '$lib/wasm/bridge';
import { loadConfig } from '$lib/wasm/config';

// Singleton manager
let manager: SwarmManager | null = null;

// Stores
export const isInitialized = writable(false);
export const isRunning = writable(false);
export const renderState = writable<DroneRenderData[]>([]);
export const status = writable<SwarmStatus>({
  simulationTime: 0,
  droneCount: 0,
  selectedCount: 0,
  speedMultiplier: 1.0,
  isValid: false,
});

// Path mode stores
export const pathMode = writable(false);
export const currentPath = writable<Array<{ x: number; y: number }>>([]);

// Hover store for tooltips
export const hoveredDroneId = writable<number | null>(null);

// Derived
export const selectedCount = derived(status, ($status) => $status.selectedCount);

// Actions
export async function initSimulation(config?: SimulationConfig): Promise<void> {
  const finalConfig = config ?? await loadConfig();
  manager = new SwarmManager();
  await manager.initialize(finalConfig);
  isInitialized.set(true);
  updateRenderState();
}

export function updateRenderState(): void {
  if (!manager) return;
  renderState.set(manager.getRenderState());
  status.set(manager.getStatus());
}

export function tickSimulation(dt: number): void {
  if (!manager) return;
  manager.tick(dt);
  updateRenderState();
}

export function selectDrone(id: number, multi: boolean): void {
  manager?.selectDrone(id, multi);
  updateRenderState();
}

export function clearSelection(): void {
  manager?.clearSelection();
  updateRenderState();
}

export function setSpeed(speed: number): void {
  manager?.setSpeed(speed);
  updateRenderState();
}

export function assignWaypoint(x: number, y: number): void {
  manager?.assignWaypoint(x, y);
  updateRenderState();
}

export function assignPath(waypoints: Array<{ x: number; y: number }>): void {
  manager?.assignPath(waypoints);
  currentPath.set([]);
  pathMode.set(false);
  updateRenderState();
}

export function getDroneAt(x: number, y: number): number | undefined {
  return manager?.getDroneAt(x, y, 20);
}

export function resetSimulation(): void {
  if (manager) {
    manager.destroy();
    manager = null;
  }
  isInitialized.set(false);
  isRunning.set(false);
  pathMode.set(false);
  currentPath.set([]);
}

export function destroySimulation(): void {
  resetSimulation();
}
```

### Success Criteria

- [ ] All WASM exports callable from TypeScript
- [ ] SwarmManager initializes without errors
- [ ] SwarmManager.destroy() frees WASM memory
- [ ] Config loads from URL parameters
- [ ] Config loads from config.json when present
- [ ] Config falls back to defaults
- [ ] Stores update when simulation changes
- [ ] tickSimulation advances simulation time

### Tests

| Test | Type | Description |
|:-----|:-----|:------------|
| WASM init | Integration | `initWasm()` completes |
| Manager lifecycle | Integration | Create, use, destroy without errors |
| Config URL parsing | Unit | `?drones=10` produces correct config |
| Config file loading | Integration | Fetches and parses config.json |
| Store reactivity | Integration | Store subscribers receive updates |

### Files Modified/Created

| File | Action |
|:-----|:-------|
| `wasm-lib/src/lib.rs` | Add wasm_bindgen exports |
| `webapp/src/lib/wasm/bridge.ts` | Create |
| `webapp/src/lib/wasm/config.ts` | Create |
| `webapp/src/lib/stores/simulation.ts` | Create |

---

## Stage 4: UI Implementation

**Goal**: Build complete interactive UI with canvas rendering and controls

**Status**: Not Started

**Depends On**: Stage 3 complete

### Tasks

#### 4.1 Create SimulationCanvas Component

**Reference:** UI-SPEC.md §4.1 Main Simulation View, §5.1 SimulationCanvas

**File:** `webapp/src/lib/components/SimulationCanvas.svelte`

Key features:
- Canvas element that fills container
- Renders grid overlay (100-unit spacing, 15% opacity)
- Renders each drone as colored circle with heading arrow
- Renders selection ring around selected drones
- Renders waypoint markers for selected drones with targets
- Renders path mode waypoints during path creation
- Handles mouse events for selection and waypoint assignment
- Responsive sizing

```svelte
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    renderState,
    pathMode,
    currentPath,
    hoveredDroneId,
    selectDrone,
    clearSelection,
    assignWaypoint,
    assignPath,
    getDroneAt
  } from '$lib/stores/simulation';

  export let width = 1000;
  export let height = 1000;

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;

  // Render loop, event handlers, etc.
</script>

<canvas
  bind:this={canvas}
  {width}
  {height}
  on:click={handleClick}
  on:contextmenu={handleRightClick}
  on:mousemove={handleMouseMove}
/>
```

**Rendering logic:**
```typescript
function render() {
  ctx.clearRect(0, 0, width, height);

  // Draw grid
  drawGrid();

  // Draw waypoint markers for selected drones
  drawWaypointMarkers();

  // Draw path mode waypoints
  if ($pathMode) {
    drawPathWaypoints();
  }

  // Draw drones
  for (const drone of $renderState) {
    drawDrone(drone);
  }
}

function drawDrone(drone: DroneRenderData) {
  const { x, y, heading, color, selected } = drone;

  // Selection ring
  if (selected) {
    ctx.strokeStyle = '#00ffff';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(x, y, 25, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Drone body
  ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
  ctx.beginPath();
  ctx.arc(x, y, 20, 0, Math.PI * 2);
  ctx.fill();

  // Heading arrow
  const arrowLength = 30;
  const endX = x + Math.cos(heading) * arrowLength;
  const endY = y + Math.sin(heading) * arrowLength;

  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(endX, endY);
  ctx.stroke();

  // Arrow head
  // ...
}
```

#### 4.2 Create ControlPanel Component

**Reference:** UI-SPEC.md §5.3 ControlPanel

**File:** `webapp/src/lib/components/ControlPanel.svelte`

```svelte
<script lang="ts">
  import { isRunning, pathMode, status } from '$lib/stores/simulation';
  import Button from './Button.svelte';
  import SpeedSlider from './SpeedSlider.svelte';

  export let onStart: () => void;
  export let onPause: () => void;
  export let onReset: () => void;
</script>

<div class="control-panel">
  <div class="playback-controls">
    {#if $isRunning}
      <Button variant="secondary" on:click={onPause}>Pause</Button>
    {:else}
      <Button variant="primary" on:click={onStart}>Start</Button>
    {/if}
    <Button variant="secondary" on:click={onReset}>Reset</Button>
  </div>

  <SpeedSlider />

  {#if $pathMode}
    <div class="path-mode-indicator">
      Path Mode: Click to add waypoints, Right-click to confirm
    </div>
  {/if}
</div>
```

#### 4.3 Create StatusBar Component

**Reference:** UI-SPEC.md §5.4 StatusBar

**File:** `webapp/src/lib/components/StatusBar.svelte`

```svelte
<script lang="ts">
  import { status } from '$lib/stores/simulation';

  $: formattedTime = $status.simulationTime.toFixed(1);
</script>

<div class="status-bar">
  <span>Drones: {$status.droneCount}</span>
  <span>Selected: {$status.selectedCount}</span>
  <span>Time: {formattedTime}s</span>
  <span>Speed: {$status.speedMultiplier}x</span>
</div>
```

#### 4.4 Create Button Component

**Reference:** UI-SPEC.md §5.2 Component Inventory

**File:** `webapp/src/lib/components/Button.svelte`

```svelte
<script lang="ts">
  export let variant: 'primary' | 'secondary' | 'danger' | 'ghost' = 'primary';
  export let disabled = false;
</script>

<button class="btn btn-{variant}" {disabled} on:click>
  <slot />
</button>

<style>
  .btn {
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: 500;
    cursor: pointer;
    border: none;
  }

  .btn-primary {
    background: #3b82f6;
    color: white;
  }

  .btn-secondary {
    background: #6b7280;
    color: white;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
```

#### 4.5 Create SpeedSlider Component

**Reference:** UI-SPEC.md §5.2 Component Inventory

**File:** `webapp/src/lib/components/SpeedSlider.svelte`

```svelte
<script lang="ts">
  import { status, setSpeed } from '$lib/stores/simulation';

  const speeds = [0.25, 0.5, 1, 2, 4];

  function handleChange(e: Event) {
    const value = parseFloat((e.target as HTMLInputElement).value);
    setSpeed(speeds[value]);
  }

  $: currentIndex = speeds.indexOf($status.speedMultiplier);
</script>

<div class="speed-slider">
  <label for="speed">Speed: {$status.speedMultiplier}x</label>
  <input
    id="speed"
    type="range"
    min="0"
    max="4"
    step="1"
    value={currentIndex}
    on:input={handleChange}
  />
</div>
```

#### 4.6 Create DroneTooltip Component

**Reference:** UI-SPEC.md §5.6 DroneTooltip

**File:** `webapp/src/lib/components/DroneTooltip.svelte`

```svelte
<script lang="ts">
  import { hoveredDroneId, renderState } from '$lib/stores/simulation';

  export let x: number;
  export let y: number;

  $: drone = $hoveredDroneId !== null
    ? $renderState.find(d => d.id === $hoveredDroneId)
    : null;
</script>

{#if drone}
  <div class="tooltip" style="left: {x + 20}px; top: {y - 10}px;">
    <div class="tooltip-header">Drone #{drone.id}</div>
    <div>Status: {drone.objectiveType}</div>
    <div>Position: ({drone.x.toFixed(0)}, {drone.y.toFixed(0)})</div>
    {#if drone.target}
      <div>Target: ({drone.target.x.toFixed(0)}, {drone.target.y.toFixed(0)})</div>
    {/if}
  </div>
{/if}

<style>
  .tooltip {
    position: absolute;
    background: rgba(0, 0, 0, 0.85);
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 12px;
    pointer-events: none;
    z-index: 100;
  }
</style>
```

#### 4.7 Create PathModeIndicator Component

**Reference:** UI-SPEC.md §5.7 PathModeIndicator

**File:** `webapp/src/lib/components/PathModeIndicator.svelte`

```svelte
<script lang="ts">
  import { pathMode, currentPath } from '$lib/stores/simulation';
</script>

{#if $pathMode}
  <div class="path-indicator">
    <span class="badge">PATH MODE</span>
    <span>Waypoints: {$currentPath.length}</span>
    <span class="hint">Click to add • Right-click to confirm • Escape to cancel</span>
  </div>
{/if}

<style>
  .path-indicator {
    position: fixed;
    top: 16px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 8px 16px;
    border-radius: 4px;
    display: flex;
    gap: 16px;
    align-items: center;
  }

  .badge {
    background: #f59e0b;
    color: black;
    padding: 2px 8px;
    border-radius: 2px;
    font-weight: bold;
  }

  .hint {
    color: #9ca3af;
    font-size: 12px;
  }
</style>
```

#### 4.8 Integrate Main Page

**File:** `webapp/src/routes/+page.svelte`

```svelte
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    isInitialized,
    isRunning,
    pathMode,
    currentPath,
    initSimulation,
    tickSimulation,
    resetSimulation,
    assignPath,
    clearSelection
  } from '$lib/stores/simulation';
  import SimulationCanvas from '$lib/components/SimulationCanvas.svelte';
  import ControlPanel from '$lib/components/ControlPanel.svelte';
  import StatusBar from '$lib/components/StatusBar.svelte';
  import DroneTooltip from '$lib/components/DroneTooltip.svelte';
  import PathModeIndicator from '$lib/components/PathModeIndicator.svelte';

  let animationFrame: number;
  let lastTime = 0;
  let tooltipX = 0;
  let tooltipY = 0;

  onMount(async () => {
    await initSimulation();
  });

  onDestroy(() => {
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
    }
  });

  function gameLoop(timestamp: number) {
    if (!$isRunning) return;

    const dt = lastTime ? (timestamp - lastTime) / 1000 : 0.016;
    lastTime = timestamp;

    tickSimulation(dt);
    animationFrame = requestAnimationFrame(gameLoop);
  }

  function handleStart() {
    isRunning.set(true);
    lastTime = 0;
    animationFrame = requestAnimationFrame(gameLoop);
  }

  function handlePause() {
    isRunning.set(false);
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
    }
  }

  async function handleReset() {
    handlePause();
    resetSimulation();
    await initSimulation();
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'p' || e.key === 'P') {
      pathMode.update(v => !v);
      if (!$pathMode) currentPath.set([]);
    } else if (e.key === 'Escape') {
      if ($pathMode) {
        pathMode.set(false);
        currentPath.set([]);
      } else {
        clearSelection();
      }
    } else if (e.key === 'Enter' && $pathMode && $currentPath.length > 0) {
      assignPath($currentPath);
    }
  }

  function handleMouseMove(e: MouseEvent) {
    tooltipX = e.clientX;
    tooltipY = e.clientY;
  }
</script>

<svelte:window on:keydown={handleKeydown} />

<main on:mousemove={handleMouseMove}>
  {#if $isInitialized}
    <SimulationCanvas />
    <ControlPanel
      onStart={handleStart}
      onPause={handlePause}
      onReset={handleReset}
    />
    <StatusBar />
    <DroneTooltip x={tooltipX} y={tooltipY} />
    <PathModeIndicator />
  {:else}
    <div class="loading">Loading simulation...</div>
  {/if}
</main>

<style>
  main {
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    background: #1a1a2e;
  }

  .loading {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: white;
    font-size: 24px;
  }
</style>
```

#### 4.9 Implement Interaction Handlers

In SimulationCanvas.svelte:

```typescript
function handleClick(e: MouseEvent) {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if ($pathMode) {
    // Add waypoint to path
    currentPath.update(path => [...path, { x, y }]);
    return;
  }

  const multi = e.ctrlKey || e.metaKey || e.shiftKey;
  const droneId = getDroneAt(x, y);

  if (droneId !== undefined) {
    selectDrone(droneId, multi);
  } else if (!multi) {
    clearSelection();
  }
}

function handleRightClick(e: MouseEvent) {
  e.preventDefault();

  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if ($pathMode) {
    // Confirm path
    if ($currentPath.length > 0) {
      assignPath($currentPath);
    }
    return;
  }

  // Assign waypoint
  if ($status.selectedCount > 0) {
    assignWaypoint(x, y);
  }
}

function handleMouseMove(e: MouseEvent) {
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  const droneId = getDroneAt(x, y);
  hoveredDroneId.set(droneId ?? null);
}
```

### Success Criteria

- [ ] Canvas renders all drones with correct colors
- [ ] Heading arrows point in correct direction
- [ ] Grid overlay visible at 15% opacity
- [ ] Selection ring appears around selected drones
- [ ] Click selects drone
- [ ] Ctrl/Shift+click adds to selection
- [ ] Click empty space clears selection
- [ ] Right-click assigns waypoint to selection
- [ ] Waypoint markers visible for drones with targets
- [ ] P key toggles path mode
- [ ] Path mode: clicks add waypoints
- [ ] Path mode: right-click confirms path
- [ ] Escape cancels path mode or clears selection
- [ ] Start button begins animation
- [ ] Pause button stops animation
- [ ] Reset button reinitializes simulation
- [ ] Speed slider changes simulation speed
- [ ] Status bar shows correct statistics
- [ ] Tooltip appears on drone hover after 300ms

### Tests

| Test | Type | Description |
|:-----|:-----|:------------|
| Drone rendering | Visual | 5 drones render with distinct colors |
| Selection visual | Visual | Cyan ring appears on click |
| Waypoint visual | Visual | X marker at waypoint location |
| Path mode flow | Manual | P → click × 3 → Right-click confirms |
| Keyboard shortcuts | Manual | P, Escape, Enter all work |
| Animation smooth | Manual | 60fps without stuttering |
| Responsive | Manual | Canvas resizes with window |

### Files Modified/Created

| File | Action |
|:-----|:-------|
| `webapp/src/lib/components/SimulationCanvas.svelte` | Create |
| `webapp/src/lib/components/ControlPanel.svelte` | Create |
| `webapp/src/lib/components/StatusBar.svelte` | Create |
| `webapp/src/lib/components/Button.svelte` | Create |
| `webapp/src/lib/components/SpeedSlider.svelte` | Create |
| `webapp/src/lib/components/DroneTooltip.svelte` | Create |
| `webapp/src/lib/components/PathModeIndicator.svelte` | Create |
| `webapp/src/routes/+page.svelte` | Rewrite |

---

## Summary

| Stage | Goal | Key Deliverable | Est. Files |
|:------|:-----|:----------------|:-----------|
| 1 | WASM Foundation | WASM loads in browser | 5 |
| 2 | Core Simulation | Full physics working | 1-2 |
| 3 | TypeScript Bridge | Stores drive UI | 3 |
| 4 | UI Implementation | Interactive app | 8 |

**Total estimated files to create/modify:** ~17

**Dependencies:**
```
Stage 1 ─┬─► Stage 2 ─┬─► Stage 3 ─┬─► Stage 4
         │            │            │
    (drone-lib)  (wasm-lib)   (bridge.ts)
```

Each stage produces a testable milestone. Stages can be committed independently.

---

## Appendix: Quick Reference

### Key Constants (from BUSINESS-LOGIC-SPEC.md)

| Constant | Value | Location |
|:---------|:------|:---------|
| MAX_VELOCITY | 35.0 m/s | drone-lib |
| MAX_ACCELERATION | 7.0 m/s² | drone-lib |
| ARRIVAL_THRESHOLD | 5.0 m | drone-lib |
| HIT_RADIUS | 20 px | wasm-lib |
| GRID_OPACITY | 15% | UI |
| GRID_SPACING | 100 units | UI |
| TOOLTIP_DELAY | 300ms | UI |
| SPEED_MIN | 0.25x | wasm-lib |
| SPEED_MAX | 4.0x | wasm-lib |

### File Structure After Implementation

```
droneswarm/
├── drone-lib/
│   └── src/
│       └── models/
│           └── quadcopter.rs    # Modified: remove Instant
├── wasm-lib/
│   ├── Cargo.toml               # Modified: add deps
│   └── src/
│       └── lib.rs               # Rewritten: full impl
└── webapp/
    └── src/
        ├── lib/
        │   ├── wasm/
        │   │   ├── pkg/         # Generated by wasm-pack
        │   │   ├── bridge.ts    # SwarmManager class
        │   │   └── config.ts    # Config loading
        │   ├── stores/
        │   │   └── simulation.ts
        │   └── components/
        │       ├── SimulationCanvas.svelte
        │       ├── ControlPanel.svelte
        │       ├── StatusBar.svelte
        │       ├── Button.svelte
        │       ├── SpeedSlider.svelte
        │       ├── DroneTooltip.svelte
        │       └── PathModeIndicator.svelte
        └── routes/
            └── +page.svelte     # Main page
```
