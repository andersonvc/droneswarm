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
    assign_waypoint_all,
    assign_route_all,
    get_drone_at,
    set_flight_params,
    set_avoidance_lookahead,
    set_vo_config,
    set_waypoint_clearance,
    set_consensus_protocol,
    type SwarmHandle,
} from 'wasm-lib';

// Type definitions
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
    splinePath: Point[];
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
    spawnPattern:
        | 'grid'
        | 'random'
        | { cluster: { center: Point; radius: number } }
        | { custom: { positions: Point[] } };
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

    assignWaypointAll(x: number, y: number): void {
        if (!this.handle) return;
        assign_waypoint_all(this.handle, x, y);
    }

    assignRouteAll(waypoints: Point[]): void {
        if (!this.handle) return;
        assign_route_all(this.handle, waypoints);
    }

    getDroneAt(x: number, y: number, hitRadius: number = 20): number | undefined {
        if (!this.handle) return undefined;
        return get_drone_at(this.handle, x, y, hitRadius) ?? undefined;
    }

    setFlightParams(maxVelocity: number, maxAcceleration: number, maxTurnRate: number): void {
        if (!this.handle) return;
        set_flight_params(this.handle, maxVelocity, maxAcceleration, maxTurnRate);
    }

    setAvoidanceLookahead(lookaheadTime: number): void {
        if (!this.handle) return;
        set_avoidance_lookahead(this.handle, lookaheadTime);
    }

    setVoConfig(
        lookaheadTime: number,
        timeSamples: number,
        safeDistance: number,
        detectionRange: number,
        avoidanceWeight: number
    ): void {
        if (!this.handle) return;
        set_vo_config(
            this.handle,
            lookaheadTime,
            timeSamples,
            safeDistance,
            detectionRange,
            avoidanceWeight
        );
    }

    setWaypointClearance(clearance: number): void {
        if (!this.handle) return;
        set_waypoint_clearance(this.handle, clearance);
    }

    setConsensusProtocol(protocol: string): void {
        if (!this.handle) return;
        set_consensus_protocol(this.handle, protocol);
    }

    destroy(): void {
        if (this.handle) {
            this.handle.free();
            this.handle = null;
        }
    }
}
