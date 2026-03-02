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
    speedMultiplier: 8.0,
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
export const objectiveMode = writable<ObjectiveMode>('route');

// Swarm size configuration
export type SwarmSize = 'small' | 'medium' | 'large';
export const swarmSize = writable<SwarmSize>('large');

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

// ORCA collision avoidance config store
export interface OrcaConfig {
    timeHorizon: number;
    agentRadius: number;
    neighborDist: number;
}

export const orcaConfig = writable<OrcaConfig>({
    timeHorizon: 2.0,         // Seconds to look ahead for collision
    agentRadius: 20.0,        // Collision radius per agent
    neighborDist: 80.0,       // How far to look for neighbors
});

// Waypoint clearance - how close to consider "arrived"
export const waypointClearance = writable(150.0);

// Consensus protocol for collision avoidance priority
export type ConsensusProtocol = 'priority_by_id' | 'priority_by_waypoint_dist';
export const consensusProtocol = writable<ConsensusProtocol>('priority_by_id');

// Formation configuration
export type FormationType = 'none' | 'line' | 'vee' | 'chevron' | 'diamond' | 'circle' | 'grid';
export interface FormationConfig {
    type: FormationType;
    spacing: number;
    leaderId?: number;
}
export const formationConfig = writable<FormationConfig>({
    type: 'chevron',
    spacing: 40,
    leaderId: undefined,
});

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

    // Set default patrol route (hourglass/bowtie — two crossing triangles)
    const defaultRoute = [
        { x: 200, y: 100 },  // 1: top-left
        { x: 550, y: 500 },  // 2: center
        { x: 900, y:  50 },  // 3: top-right
        { x: 850, y: 900 },  // 4: bottom-right
        { x: 550, y: 650 },  // 5: center-lower
        { x: 250, y: 850 },  // 6: bottom-left
    ];
    activeRoute.set(defaultRoute);
    currentPath.set(defaultRoute);
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

export function setOrcaConfig(config: OrcaConfig): void {
    manager?.setOrcaConfig(
        config.timeHorizon,
        config.agentRadius,
        config.neighborDist
    );
    orcaConfig.set(config);
}

export function setWaypointClearance(clearance: number): void {
    manager?.setWaypointClearance(clearance);
    waypointClearance.set(clearance);
}

export function setConsensusProtocol(protocol: ConsensusProtocol): void {
    manager?.setConsensusProtocol(protocol);
    consensusProtocol.set(protocol);
}

export function setFormation(type: FormationType, spacing: number, leaderId?: number): void {
    if (type === 'none') {
        manager?.clearFormation();
    } else {
        manager?.setFormation(type, spacing, leaderId);
    }
    formationConfig.set({ type, spacing, leaderId });
    updateRenderState();
}

export function clearFormation(): void {
    manager?.clearFormation();
    formationConfig.update(config => ({ ...config, type: 'none' }));
    updateRenderState();
}

export function formationCommand(command: 'hold' | 'advance' | 'disperse' | 'contract' | 'expand'): void {
    manager?.formationCommand(command);
    if (command === 'disperse') {
        formationConfig.update(config => ({ ...config, type: 'none' }));
    }
    updateRenderState();
}

export function updateFormation(): void {
    manager?.updateFormation();
}
