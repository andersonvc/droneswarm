import { writable, derived, get } from 'svelte/store';
import {
    SwarmManager,
    type DroneRenderData,
    type SwarmStatus,
    type SimulationConfig,
    type Point,
} from '$lib/wasm/bridge';
import { loadConfig } from '$lib/wasm/config';

// Singleton manager
let manager: SwarmManager | null = null;

// Core stores
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
export const currentPath = writable<Point[]>([]);

// Active route (persists across resets when in route mode)
export const activeRoute = writable<Point[]>([]);

// Coordination and objective modes
export type CoordinationMode = 'individual' | 'swarm';
export type ObjectiveMode = 'waypoint' | 'route';
export const coordinationMode = writable<CoordinationMode>('swarm');
export const objectiveMode = writable<ObjectiveMode>('waypoint');

// Swarm size configuration
export type SwarmSize = 'small' | 'medium' | 'large';
export const swarmSize = writable<SwarmSize>('small');

export const SWARM_SIZE_COUNTS: Record<SwarmSize, number> = {
    small: 6,
    medium: 20,
    large: 50,
};

// Hover store for tooltips
export const hoveredDroneId = writable<number | null>(null);

// Flight parameters store
export const flightParams = writable({
    maxVelocity: 120,
    maxAcceleration: 21,
    maxTurnRate: 4
});

// Velocity Obstacle config store
export interface VoConfig {
    lookaheadTime: number;
    timeSamples: number;
    safeDistance: number;
    detectionRange: number;
    avoidanceWeight: number;
}

export const voConfig = writable<VoConfig>({
    lookaheadTime: 1.0,
    timeSamples: 5,
    safeDistance: 50.0,
    detectionRange: 120.0,
    avoidanceWeight: 0.85,
});

// Waypoint clearance - how close to consider "arrived"
export const waypointClearance = writable(10.0);

// Legacy - kept for compatibility with AvoidanceSlider
export const avoidanceLookahead = writable(1.0);

// Derived stores
export const selectedCount = derived(status, ($status) => $status.selectedCount);

// Actions
export async function initSimulation(config?: SimulationConfig): Promise<void> {
    const finalConfig = config ?? (await loadConfig());
    manager = new SwarmManager();
    await manager.initialize(finalConfig);
    isInitialized.set(true);
    updateRenderState();
}

export async function reinitializeWithSize(size: SwarmSize): Promise<void> {
    // Stop simulation if running
    isRunning.set(false);

    // Destroy existing manager
    if (manager) {
        manager.destroy();
        manager = null;
    }

    // Reset state (preserve route/path)
    isInitialized.set(false);
    pathMode.set(false);

    // Update swarm size store
    swarmSize.set(size);

    // Get base config and override drone count
    const baseConfig = await loadConfig();
    const newConfig = {
        ...baseConfig,
        droneCount: SWARM_SIZE_COUNTS[size],
    };

    // Reinitialize and restore route
    await initSimulation(newConfig);
    restoreActiveRoute();
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

export function selectDronesInRect(minX: number, minY: number, maxX: number, maxY: number, multi: boolean): void {
    if (!manager) return;

    // Get current render state to find drones in the rectangle
    const state = manager.getRenderState();

    // If not multi-select, clear first
    if (!multi) {
        manager.clearSelection();
    }

    // Select all drones within the rectangle
    for (const drone of state) {
        if (drone.x >= minX && drone.x <= maxX && drone.y >= minY && drone.y <= maxY) {
            manager.selectDrone(drone.id, true); // Always multi=true to add to selection
        }
    }

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

export function assignWaypointAll(x: number, y: number): void {
    manager?.assignWaypointAll(x, y);
    updateRenderState();
}

export function assignPath(waypoints: Point[]): void {
    manager?.assignPath(waypoints);
    // Only clear path in waypoint mode; keep it visible in route mode
    if (get(objectiveMode) !== 'route') {
        currentPath.set([]);
    }
    pathMode.set(false);
    updateRenderState();
}

export function assignRouteAll(waypoints: Point[]): void {
    manager?.assignRouteAll(waypoints);
    // Save active route so it persists across resets
    activeRoute.set(waypoints);
    // Keep path visible in route mode (don't clear)
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
    // Don't clear currentPath/activeRoute here - they persist across resets in route mode
}

export function destroySimulation(): void {
    resetSimulation();
}

export function restoreActiveRoute(): void {
    const route = get(activeRoute);
    if (route.length > 0 && get(objectiveMode) === 'route') {
        manager?.assignRouteAll(route);
        updateRenderState();
    }
}

export function setFlightParams(maxVelocity: number, maxAcceleration: number, maxTurnRate: number): void {
    manager?.setFlightParams(maxVelocity, maxAcceleration, maxTurnRate);
    flightParams.set({ maxVelocity, maxAcceleration, maxTurnRate });
}

export function setAvoidanceLookahead(lookahead: number): void {
    manager?.setAvoidanceLookahead(lookahead);
    avoidanceLookahead.set(lookahead);
}

export function setVoConfig(config: VoConfig): void {
    manager?.setVoConfig(
        config.lookaheadTime,
        config.timeSamples,
        config.safeDistance,
        config.detectionRange,
        config.avoidanceWeight
    );
    voConfig.set(config);
}

export function setWaypointClearance(clearance: number): void {
    manager?.setWaypointClearance(clearance);
    waypointClearance.set(clearance);
}
