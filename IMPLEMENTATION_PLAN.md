# Per-Drone Offset Routes for Formation Following

## Stage 1: Add compute_offset_route() Utility
**Goal**: Pure function that computes offset waypoints for a given slot
**Status**: Complete

## Stage 2: Remove Arc-Length Sync Machinery
**Goal**: Remove ParameterizedPath/PathProgress sync system
**Status**: Complete

## Stage 3: Distribute Offset Routes in assign_route_all()
**Goal**: Per-drone offset routes when assigning a route in formation mode
**Status**: Complete

## Stage 4: Add Speed-Based Formation Distance Control
**Goal**: Followers adjust speed based on along-track distance to leader
**Status**: Complete

## Stage 5: Integration Testing & Cleanup
**Goal**: End-to-end verification and final cleanup
**Status**: In Progress — all tests pass, wasm built, pending visual verification
