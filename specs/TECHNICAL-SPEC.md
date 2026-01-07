# DroneSwarm Technical Specification
## WASM Integration & Static Deployment

**Version:** 1.0
**Date:** 2026-01-06
**Business Spec Reference:** [BUSINESS-LOGIC-SPEC.md](./BUSINESS-LOGIC-SPEC.md)
**UI Spec Reference:** [UI-SPEC.md](./UI-SPEC.md)

---

## 1. Overview

### Architecture Summary

DroneSwarm is a **client-side only** application with no backend server. All simulation logic runs in WebAssembly (compiled from Rust), and the frontend is a static SvelteKit application.

```
┌─────────────────────────────────────────────────────────────────┐
│                         BROWSER                                  │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   SvelteKit Frontend                     │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │    │
│  │  │   Svelte    │  │   Stores    │  │  Canvas Render  │  │    │
│  │  │ Components  │  │   (State)   │  │     Logic       │  │    │
│  │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │    │
│  │         │                │                   │           │    │
│  │         └────────────────┼───────────────────┘           │    │
│  │                          │                               │    │
│  │                          ▼                               │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │              WASM Bridge Layer                   │    │    │
│  │  │         (wasm-bindgen bindings)                  │    │    │
│  │  └─────────────────────┬───────────────────────────┘    │    │
│  └────────────────────────┼────────────────────────────────┘    │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 WebAssembly Module                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │    │
│  │  │  Swarm      │  │   Physics   │  │   Drone State   │  │    │
│  │  │  Manager    │  │   Engine    │  │   Management    │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │    │
│  │                                                          │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │              drone-lib (Pure Rust)               │    │    │
│  │  │  QuadCopter, State, Objective, Physics Logic     │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

                              │
                              │ Static Files
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Static Hosting                                │
│              (GitHub Pages / Vercel / Netlify)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| Frontend Framework | SvelteKit 2.0 | UI components and routing |
| Build Tool | Vite 5.0 | Development server and bundling |
| WASM Compilation | wasm-pack | Rust → WASM compilation |
| WASM Bindings | wasm-bindgen 0.2.x | Rust ↔ JS interop |
| Serialization | serde + serde-wasm-bindgen | Data transfer across boundary |
| Simulation Core | drone-lib (Rust) | Physics and state management |
| Hosting | Static (GitHub Pages/Vercel) | No server required |

### Key Design Decisions

| Decision | Choice | Rationale |
|:---------|:-------|:----------|
| No Backend | Client-side only | Simpler deployment, no server costs |
| Rust for Simulation | WASM compilation | Performance for physics calculations |
| State in WASM | Rust owns simulation state | Avoids serialization on every tick |
| Render Data Only | JS receives minimal data | Reduces boundary crossing overhead |
| drone-lib Refactoring | Remove `Instant`, use `dt` param | WASM compatibility; makes lib reusable |
| Config Loading | URL → config.json → default | Maximum flexibility for users |
| WASM Loading | Eager (on page load) | Better UX, no delay when starting |

### Required drone-lib Refactoring

The existing `drone-lib` uses `std::time::Instant` which is **not available in WASM**. The following changes are required:

| File | Current | Change To |
|:-----|:--------|:----------|
| `quadcopter.rs:5` | `use std::time::Instant;` | Remove |
| `quadcopter.rs:12` | `clock_time: Instant` | Remove field |
| `quadcopter.rs:35` | `clock_time: Instant::now()` | Remove initialization |
| `quadcopter.rs:71` | `fn state_update(&mut self, timestamp: Instant)` | `fn state_update(&mut self, dt: f32)` |
| `quadcopter.rs:72` | `let dt = timestamp.duration_since(...)` | Use `dt` parameter directly |

**New Drone trait signature:**
```rust
pub trait Drone {
    fn state_update(&mut self, dt: f32);  // dt in seconds
    fn task_update(&mut self, objective: Option<Box<Objective>>);
    fn action(&mut self);
    fn broadcast_state(&self) -> &State;
}
```

---

## 2. WASM Module Structure

### 2.1 Crate Organization

```
wasm-lib/
├── Cargo.toml
├── src/
│   ├── lib.rs              # WASM entry point and exports
│   ├── swarm.rs            # Swarm management (WASM-specific)
│   ├── types.rs            # WASM-bindgen compatible types
│   └── bridge.rs           # JS interop helpers
└── pkg/                    # Generated by wasm-pack
    ├── wasm_lib.js         # JS wrapper
    ├── wasm_lib.d.ts       # TypeScript definitions
    ├── wasm_lib_bg.wasm    # WASM binary
    └── package.json
```

### 2.2 Dependencies

```toml
# wasm-lib/Cargo.toml
[package]
name = "wasm-lib"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2.90"
js-sys = "0.3.67"
serde = { version = "1.0", features = ["derive"] }
serde-wasm-bindgen = "0.6.3"
drone-lib = { path = "../drone-lib" }
getrandom = { version = "0.2", features = ["js"] }  # For random spawn

[dependencies.web-sys]
version = "0.3"
features = ["console"]

[profile.release]
opt-level = "s"      # Optimize for size
lto = true           # Link-time optimization
```

---

## 3. Data Types & Serialization

### 3.1 WASM-Exported Types

These types are exposed to JavaScript via wasm-bindgen:

#### SimulationConfig (JS → Rust)

```rust
// wasm-lib/src/types.rs
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationConfig {
    pub drone_count: u32,
    pub spawn_pattern: SpawnPattern,
    pub bounds: Bounds,
    pub speed_multiplier: Option<f32>,
    pub predefined_paths: Option<Vec<DronePath>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpawnPattern {
    Grid,
    Random,
    Cluster { center: Point, radius: f32 },
    Custom { positions: Vec<Point> },
}

#[derive(Deserialize, Serialize, Clone, Copy)]
pub struct Bounds {
    pub width: f32,
    pub height: f32,
}

#[derive(Deserialize, Serialize, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DronePath {
    pub drone_id: u32,
    pub waypoints: Vec<Point>,
}
```

#### DroneRenderData (Rust → JS)

```rust
#[derive(Serialize, Clone)]
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

#[derive(Serialize, Clone, Copy)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}
```

#### SwarmState (Rust → JS, for status bar)

```rust
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SwarmStatus {
    pub simulation_time: f32,
    pub drone_count: u32,
    pub selected_count: u32,
    pub speed_multiplier: f32,
    pub is_valid: bool,
}
```

### 3.2 Serialization Strategy

| Direction | Method | Rationale |
|:----------|:-------|:----------|
| JS → Rust (config) | `serde_wasm_bindgen::from_value()` | Parse complex nested JSON |
| Rust → JS (render data) | `serde_wasm_bindgen::to_value()` | Efficient array serialization |
| Rust → JS (simple) | Direct `wasm_bindgen` types | Avoid serialization overhead |

**Example:**
```rust
#[wasm_bindgen]
pub fn get_render_state(handle: &SwarmHandle) -> JsValue {
    let render_data: Vec<DroneRenderData> = handle.swarm.get_render_data();
    serde_wasm_bindgen::to_value(&render_data).unwrap()
}
```

---

## 4. WASM API Contract

### 4.1 Exported Functions

| Function | Input | Output | Purpose |
|:---------|:------|:-------|:--------|
| `init_swarm` | `JsValue` (config) | `SwarmHandle` | Create simulation |
| `tick` | `&mut SwarmHandle`, `f32` (dt) | `void` | Advance physics |
| `get_render_state` | `&SwarmHandle` | `JsValue` (array) | Get drone positions |
| `get_status` | `&SwarmHandle` | `JsValue` (status) | Get simulation status |
| `select_drone` | `&mut SwarmHandle`, `u32`, `bool` | `void` | Select/deselect drone |
| `clear_selection` | `&mut SwarmHandle` | `void` | Deselect all |
| `set_speed` | `&mut SwarmHandle`, `f32` | `Result<(), JsValue>` | Set speed multiplier |
| `assign_waypoint` | `&mut SwarmHandle`, `f32`, `f32` | `Result<(), JsValue>` | Set waypoint for selection |
| `assign_path` | `&mut SwarmHandle`, `JsValue` | `Result<(), JsValue>` | Set multi-waypoint path |
| `get_drone_at` | `&SwarmHandle`, `f32`, `f32`, `f32` | `Option<u32>` | Hit test for drone |

### 4.2 SwarmHandle (Opaque Handle Pattern)

The `SwarmHandle` is an opaque reference to the simulation state that JavaScript holds but cannot inspect directly:

```rust
#[wasm_bindgen]
pub struct SwarmHandle {
    swarm: Swarm,
}

#[wasm_bindgen]
impl SwarmHandle {
    // Private constructor - only created via init_swarm
    fn new(swarm: Swarm) -> SwarmHandle {
        SwarmHandle { swarm }
    }
}
```

**Rationale**: Keeping state in Rust avoids costly serialization on every frame. JS only receives render data when explicitly requested.

### 4.3 Function Signatures (Rust)

```rust
// wasm-lib/src/lib.rs
use wasm_bindgen::prelude::*;
use crate::types::*;

/// Initialize a new swarm from configuration
#[wasm_bindgen]
pub fn init_swarm(config: JsValue) -> Result<SwarmHandle, JsValue> {
    let config: SimulationConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let swarm = Swarm::new(config)?;
    Ok(SwarmHandle::new(swarm))
}

/// Advance simulation by dt seconds
#[wasm_bindgen]
pub fn tick(handle: &mut SwarmHandle, dt: f32) {
    handle.swarm.tick(dt);
}

/// Get render data for all drones
#[wasm_bindgen]
pub fn get_render_state(handle: &SwarmHandle) -> JsValue {
    let data = handle.swarm.get_render_data();
    serde_wasm_bindgen::to_value(&data).unwrap_or(JsValue::NULL)
}

/// Get simulation status
#[wasm_bindgen]
pub fn get_status(handle: &SwarmHandle) -> JsValue {
    let status = handle.swarm.get_status();
    serde_wasm_bindgen::to_value(&status).unwrap_or(JsValue::NULL)
}

/// Select or toggle a drone
#[wasm_bindgen]
pub fn select_drone(handle: &mut SwarmHandle, drone_id: u32, multi_select: bool) {
    handle.swarm.select_drone(drone_id, multi_select);
}

/// Clear all selections
#[wasm_bindgen]
pub fn clear_selection(handle: &mut SwarmHandle) {
    handle.swarm.clear_selection();
}

/// Set simulation speed multiplier
#[wasm_bindgen]
pub fn set_speed(handle: &mut SwarmHandle, speed: f32) -> Result<(), JsValue> {
    if speed < 0.25 || speed > 4.0 {
        return Err(JsValue::from_str("Speed must be between 0.25 and 4.0"));
    }
    handle.swarm.set_speed_multiplier(speed);
    Ok(())
}

/// Assign waypoint to all selected drones
#[wasm_bindgen]
pub fn assign_waypoint(handle: &mut SwarmHandle, x: f32, y: f32) -> Result<(), JsValue> {
    if handle.swarm.selected_count() == 0 {
        return Err(JsValue::from_str("No drones selected"));
    }
    handle.swarm.assign_waypoint_to_selection(x, y);
    Ok(())
}

/// Assign multi-waypoint path to all selected drones
#[wasm_bindgen]
pub fn assign_path(handle: &mut SwarmHandle, waypoints: JsValue) -> Result<(), JsValue> {
    let points: Vec<Point> = serde_wasm_bindgen::from_value(waypoints)
        .map_err(|e| JsValue::from_str(&format!("Invalid waypoints: {}", e)))?;

    if points.is_empty() {
        return Err(JsValue::from_str("Path must have at least one waypoint"));
    }
    if handle.swarm.selected_count() == 0 {
        return Err(JsValue::from_str("No drones selected"));
    }
    handle.swarm.assign_path_to_selection(points);
    Ok(())
}

/// Hit test: find drone at screen coordinates
#[wasm_bindgen]
pub fn get_drone_at(handle: &SwarmHandle, x: f32, y: f32, hit_radius: f32) -> Option<u32> {
    handle.swarm.get_drone_at(x, y, hit_radius)
}
```

### 4.4 TypeScript Definitions (Generated)

```typescript
// wasm-lib/pkg/wasm_lib.d.ts (auto-generated, shown for reference)
export class SwarmHandle {
  free(): void;
}

export function init_swarm(config: any): SwarmHandle;
export function tick(handle: SwarmHandle, dt: number): void;
export function get_render_state(handle: SwarmHandle): any;
export function get_status(handle: SwarmHandle): any;
export function select_drone(handle: SwarmHandle, drone_id: number, multi_select: boolean): void;
export function clear_selection(handle: SwarmHandle): void;
export function set_speed(handle: SwarmHandle, speed: number): void;
export function assign_waypoint(handle: SwarmHandle, x: number, y: number): void;
export function assign_path(handle: SwarmHandle, waypoints: any): void;
export function get_drone_at(handle: SwarmHandle, x: number, y: number, hit_radius: number): number | undefined;
```

---

## 5. JavaScript Bridge Layer

### 5.1 WASM Initialization

```typescript
// webapp/src/lib/wasm/bridge.ts
import init, {
  SwarmHandle,
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
} from 'wasm-lib';

let wasmInitialized = false;
let swarmHandle: SwarmHandle | null = null;

export async function initializeWasm(): Promise<void> {
  if (wasmInitialized) return;
  await init();
  wasmInitialized = true;
}

export function isWasmReady(): boolean {
  return wasmInitialized;
}
```

### 5.2 Configuration Loading

Configuration is loaded using a fallback chain: **URL parameters → config.json → defaults**.

```typescript
// webapp/src/lib/config/loader.ts

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
    console.log('No config.json found, using defaults');
  }

  // 3. Fall back to defaults
  console.log('Using default configuration');
  return DEFAULT_CONFIG;
}

function parseUrlConfig(): Partial<SimulationConfig> | null {
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

**URL Parameter Examples:**
```
# 10 drones in random pattern at 2x speed
?drones=10&pattern=random&speed=2

# Just change drone count
?drones=20
```

### 5.3 Type Definitions

```typescript
// webapp/src/lib/wasm/bridge.ts (continued)

export interface SimulationConfig {
  droneCount: number;
  spawnPattern: 'grid' | 'random' | { cluster: { center: Point; radius: number } } | { custom: { positions: Point[] } };
  bounds: { width: number; height: number };
  speedMultiplier?: number;
  predefinedPaths?: Array<{ droneId: number; waypoints: Point[] }>;
}

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
```

### 5.4 Swarm Manager Class

```typescript
export class SwarmManager {
  private handle: SwarmHandle | null = null;

  async initialize(config: SimulationConfig): Promise<void> {
    await initializeWasm();
    this.handle = init_swarm(config);
  }

  tick(dt: number): void {
    if (!this.handle) throw new Error('Swarm not initialized');
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

  selectDrone(droneId: number, multiSelect: boolean = false): void {
    if (!this.handle) return;
    select_drone(this.handle, droneId, multiSelect);
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

### 5.5 Svelte Store Integration

```typescript
// webapp/src/lib/stores/simulation.ts
import { writable, derived, get } from 'svelte/store';
import { SwarmManager, type DroneRenderData, type SwarmStatus, type SimulationConfig } from '$lib/wasm/bridge';

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

// Derived
export const selectedCount = derived(status, ($status) => $status.selectedCount);

// Actions
export async function initSimulation(config: SimulationConfig): Promise<void> {
  manager = new SwarmManager();
  await manager.initialize(config);
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
  return manager?.getDroneAt(x, y);
}

export function destroySimulation(): void {
  manager?.destroy();
  manager = null;
  isInitialized.set(false);
}
```

---

## 6. Build Configuration

### 6.1 WASM Build (wasm-pack)

```bash
# Build command
wasm-pack build --target web wasm-lib

# Output location: wasm-lib/pkg/
```

**Build targets:**
| Target | Use Case |
|:-------|:---------|
| `web` | Native ES modules (recommended for Vite) |
| `bundler` | For webpack/bundler integration |
| `nodejs` | For server-side use |

### 6.2 Vite Configuration

```typescript
// webapp/vite.config.ts
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  plugins: [
    sveltekit(),
    wasm(),
    topLevelAwait(),
  ],
  optimizeDeps: {
    exclude: ['wasm-lib'],
  },
  build: {
    target: 'esnext',
  },
});
```

### 6.3 Package.json Scripts

```json
{
  "scripts": {
    "dev": "vite dev",
    "build": "npm run build:wasm && vite build",
    "build:wasm": "wasm-pack build --target web ../wasm-lib --out-dir ../webapp/src/lib/wasm/pkg",
    "preview": "vite preview"
  }
}
```

### 6.4 SvelteKit Adapter (Static)

```javascript
// webapp/svelte.config.js
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

export default {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      pages: 'build',
      assets: 'build',
      fallback: 'index.html',
      precompress: false,
      strict: true,
    }),
  },
};
```

---

## 7. Deployment

### 7.1 Static Hosting Options

| Provider | Free Tier | Custom Domain | Build Command |
|:---------|:----------|:--------------|:--------------|
| GitHub Pages | Yes | Yes (CNAME) | `npm run build` |
| Vercel | Yes | Yes | Auto-detected |
| Netlify | Yes | Yes | `npm run build` |
| Cloudflare Pages | Yes | Yes | `npm run build` |

### 7.2 GitHub Pages Deployment

```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Install wasm-pack
        run: cargo install wasm-pack

      - name: Build WASM
        run: wasm-pack build --target web wasm-lib

      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: webapp/package-lock.json

      - name: Install dependencies
        run: npm ci
        working-directory: webapp

      - name: Build
        run: npm run build
        working-directory: webapp

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: webapp/build

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### 7.3 Build Output Structure

```
webapp/build/
├── index.html
├── _app/
│   ├── immutable/
│   │   ├── chunks/
│   │   ├── entry/
│   │   └── nodes/
│   └── version.json
├── wasm_lib_bg.wasm      # WASM binary (~100-500KB)
└── favicon.png
```

---

## 8. Error Handling

### 8.1 WASM Error Codes

| Error Code | Message | When Thrown |
|:-----------|:--------|:------------|
| `INIT_FAILED` | "Failed to initialize WASM module" | WASM load failure |
| `INVALID_CONFIG` | "Invalid simulation configuration" | Bad config JSON |
| `NO_SELECTION` | "No drones selected" | Waypoint without selection |
| `INVALID_SPEED` | "Speed must be between 0.25 and 4.0" | Out of range speed |
| `INVALID_PATH` | "Path must have at least one waypoint" | Empty path array |

### 8.2 Error Handling in Bridge

```typescript
// webapp/src/lib/wasm/bridge.ts
export class WasmError extends Error {
  constructor(
    message: string,
    public code: string,
  ) {
    super(message);
    this.name = 'WasmError';
  }
}

export function handleWasmError(error: unknown): never {
  if (error instanceof Error) {
    throw new WasmError(error.message, 'WASM_ERROR');
  }
  if (typeof error === 'string') {
    throw new WasmError(error, 'WASM_ERROR');
  }
  throw new WasmError('Unknown WASM error', 'UNKNOWN');
}
```

---

## 9. Performance Considerations

### 9.1 Optimization Strategies

| Strategy | Implementation | Impact |
|:---------|:---------------|:-------|
| Minimize boundary crossings | Only call `get_render_state` once per frame | High |
| Keep state in Rust | Use opaque handle pattern | High |
| Release build optimizations | `opt-level = "s"`, LTO | Medium |
| Avoid Vec allocations | Reuse buffers where possible | Low |

### 9.2 Expected Performance

| Metric | Target | Notes |
|:-------|:-------|:------|
| WASM load time | < 500ms | Depends on network |
| Tick latency (100 drones) | < 1ms | Physics update |
| Render state extraction | < 1ms | Serialization |
| Frame budget (60fps) | 16.67ms | Total per frame |

### 9.3 Memory Management

- **SwarmHandle.free()**: Must be called when simulation is destroyed
- **Automatic cleanup**: Rust's ownership handles internal memory
- **JS garbage collection**: Render data arrays are collected normally

---

## 10. Testing Strategy

### 10.1 Rust Unit Tests

```rust
// wasm-lib/src/lib.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_grid() {
        let config = SimulationConfig {
            drone_count: 4,
            spawn_pattern: SpawnPattern::Grid,
            bounds: Bounds { width: 100.0, height: 100.0 },
            speed_multiplier: None,
            predefined_paths: None,
        };
        let swarm = Swarm::new(config).unwrap();
        assert_eq!(swarm.drone_count(), 4);
    }

    #[test]
    fn test_selection() {
        // ... test selection logic
    }
}
```

### 10.2 Integration Tests (Browser)

```typescript
// webapp/tests/wasm.test.ts
import { describe, it, expect, beforeAll } from 'vitest';
import { SwarmManager } from '$lib/wasm/bridge';

describe('WASM Integration', () => {
  let manager: SwarmManager;

  beforeAll(async () => {
    manager = new SwarmManager();
    await manager.initialize({
      droneCount: 5,
      spawnPattern: 'grid',
      bounds: { width: 1000, height: 1000 },
    });
  });

  it('initializes swarm with correct drone count', () => {
    const status = manager.getStatus();
    expect(status.droneCount).toBe(5);
  });

  it('returns render state for all drones', () => {
    const state = manager.getRenderState();
    expect(state.length).toBe(5);
  });

  it('handles selection correctly', () => {
    manager.selectDrone(0, false);
    const status = manager.getStatus();
    expect(status.selectedCount).toBe(1);
  });
});
```

---

## 11. Implementation Checklist

### Phase 1: WASM Foundation
- [ ] Update `wasm-lib/Cargo.toml` with all dependencies
- [ ] Create `types.rs` with all data structures
- [ ] Implement `Swarm` struct with drone management
- [ ] Implement `SwarmHandle` with wasm-bindgen
- [ ] Export all required functions
- [ ] Add `getrandom` feature for random spawn

### Phase 2: Core Operations
- [ ] Implement `init_swarm` with config parsing
- [ ] Implement `tick` with physics integration
- [ ] Implement `get_render_state` serialization
- [ ] Implement `get_status` serialization
- [ ] Implement drone selection (single and multi)
- [ ] Implement `clear_selection`

### Phase 3: Waypoint Operations
- [ ] Implement `set_speed` with validation
- [ ] Implement `assign_waypoint` to selection
- [ ] Implement `assign_path` for multi-waypoint
- [ ] Implement `get_drone_at` hit testing
- [ ] Implement toroidal boundary wrapping

### Phase 4: JS Integration
- [ ] Create `bridge.ts` with initialization
- [ ] Create `SwarmManager` class
- [ ] Create Svelte stores integration
- [ ] Update Vite config for WASM
- [ ] Test WASM loading in browser

### Phase 5: Build & Deploy
- [ ] Configure wasm-pack build script
- [ ] Configure SvelteKit static adapter
- [ ] Create GitHub Actions workflow
- [ ] Test production build locally
- [ ] Deploy to static hosting

---

## 12. Appendix: Configuration Examples

### 12.1 Default Configuration

```json
{
  "droneCount": 5,
  "spawnPattern": "grid",
  "bounds": { "width": 1000, "height": 1000 },
  "speedMultiplier": 1.0
}
```

### 12.2 Custom Spawn Positions

```json
{
  "droneCount": 3,
  "spawnPattern": {
    "custom": {
      "positions": [
        { "x": 100, "y": 100 },
        { "x": 500, "y": 500 },
        { "x": 900, "y": 100 }
      ]
    }
  },
  "bounds": { "width": 1000, "height": 1000 }
}
```

### 12.3 Predefined Paths

```json
{
  "droneCount": 2,
  "spawnPattern": "grid",
  "bounds": { "width": 1000, "height": 1000 },
  "predefinedPaths": [
    {
      "droneId": 0,
      "waypoints": [
        { "x": 100, "y": 100 },
        { "x": 900, "y": 100 },
        { "x": 900, "y": 900 },
        { "x": 100, "y": 900 }
      ]
    }
  ]
}
```
