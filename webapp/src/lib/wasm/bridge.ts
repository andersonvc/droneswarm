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
    attack_target,
    detonate_drone,
    detonate_selected,
    assign_route_range,
    set_formation_for_range,
    get_drone_at,
    set_flight_params,
    set_avoidance_lookahead,
    set_orca_config,
    set_waypoint_clearance,
    intercept_drone,
    return_to_formation,
    set_group_split,
    set_protected_zones,
    set_consensus_protocol,
    set_formation,
    clear_formation,
    formation_command,
    update_formation,
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
    routePath: Point[];
    planningPath: Point[];
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
    /** World width in meters. Default: 500 */
    worldWidthMeters?: number;
    /** World height in meters. Default: 500 */
    worldHeightMeters?: number;
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
                speedMultiplier: 8.0,
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

    attackTarget(droneId: number, targetX: number, targetY: number): void {
        if (!this.handle) return;
        attack_target(this.handle, droneId, targetX, targetY);
    }

    detonateDrone(droneId: number): void {
        if (!this.handle) return;
        detonate_drone(this.handle, droneId);
    }

    detonateSelected(): void {
        if (!this.handle) return;
        detonate_selected(this.handle);
    }

    interceptDrone(attackerId: number, targetDroneId: number): void {
        if (!this.handle) return;
        intercept_drone(this.handle, attackerId, targetDroneId);
    }

    returnToFormation(droneId: number, groupStart: number, groupEnd: number): void {
        if (!this.handle) return;
        return_to_formation(this.handle, droneId, groupStart, groupEnd);
    }

    setGroupSplit(splitId: number): void {
        if (!this.handle) return;
        set_group_split(this.handle, splitId);
    }

    setProtectedZones(group: number, positions: Point[]): void {
        if (!this.handle) return;
        set_protected_zones(this.handle, group, positions);
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

    assignRouteRange(waypoints: Point[], startId: number, endId: number): void {
        if (!this.handle) return;
        assign_route_range(this.handle, waypoints, startId, endId);
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

    setOrcaConfig(
        timeHorizon: number,
        agentRadius: number,
        neighborDist: number
    ): void {
        if (!this.handle) return;
        set_orca_config(
            this.handle,
            timeHorizon,
            agentRadius,
            neighborDist
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

    setFormation(formationType: string, spacing: number, leaderId?: number): void {
        if (!this.handle) return;
        set_formation(this.handle, formationType, spacing, leaderId ?? null);
    }

    setFormationForRange(formationType: string, spacing: number, startId: number, endId: number): void {
        if (!this.handle) return;
        set_formation_for_range(this.handle, formationType, spacing, startId, endId);
    }

    clearFormation(): void {
        if (!this.handle) return;
        clear_formation(this.handle);
    }

    formationCommand(command: string): void {
        if (!this.handle) return;
        formation_command(this.handle, command);
    }

    updateFormation(): void {
        if (!this.handle) return;
        update_formation(this.handle);
    }

    destroy(): void {
        if (this.handle) {
            this.handle.free();
            this.handle = null;
        }
    }
}
