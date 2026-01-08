# Drone Swarm Simulation

A drone swarm simulation with collision avoidance using Rust/WASM and SvelteKit.

## Quick Start

### Development
```bash
./dev.sh
```
This will:
- Kill any old dev servers
- Clear caches
- Rebuild WASM
- Start the dev server at `localhost:5173`

### Production Build
```bash
./build.sh
```
This creates an optimized production build in `webapp/build/`.

## Manual Build

If you need to build manually:

```bash
# Build WASM
cd wasm-lib
wasm-pack build --target web

# Start dev server
cd ../webapp
npm run dev
```

**Important:** The WASM must be built to `wasm-lib/pkg/` (the default location), not `webapp/src/lib/wasm/pkg/`, as Vite loads it from the source directory.

## Project Structure

```
droneswarm/
├── drone-lib/          # Core Rust library
├── wasm-lib/          # WASM bindings
│   └── pkg/           # Built WASM (Vite loads from here)
└── webapp/            # SvelteKit frontend
```

## Features

- **Unbounded motion**: Drones can move beyond visible screen bounds
- **Collision avoidance**: Velocity obstacle algorithm
- **Path smoothing**: Hermite spline interpolation
- **Stanley controller**: Advanced path following
