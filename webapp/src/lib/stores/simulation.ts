import { writable, derived, get } from 'svelte/store';
import {
    SwarmManager,
    type DroneRenderData,
    type SwarmStatus,
    type SimulationConfig,
    type Point,
} from '$lib/wasm/bridge';
import { loadConfig, generateTwoClusterPositions, CLUSTER_A, CLUSTER_B } from '$lib/wasm/config';

// Re-export from selection.ts
export { hoveredDroneId, selectDrone, selectDronesInRect, clearSelection, getDroneAt } from './selection';
// Re-export from game-state.ts
export {
    type Explosion,
    explosions,
    addExplosion,
    type Target,
    targets,
    targetCounts,
    checkTargetDestruction,
    type GameResult,
    gameResult,
    generateTargets,
    checkWinCondition,
    buildPatrolRoute,
    updateProtectedZones,
} from './game-state';

// Import from game-state for internal use
import {
    addExplosion,
    targets,
    gameResult,
    generateTargets,
    checkWinCondition,
    buildPatrolRoute,
    updateProtectedZones,
} from './game-state';

// Singleton manager
let manager: SwarmManager | null = null;

// Internal accessors for sibling modules
export function getManager(): SwarmManager | null {
    return manager;
}

// Track drones in attack mode (drone_id -> target position) for explosion visuals
const attackingDrones = new Map<number, Point>();
// Track drones in intercept mode (attacker_id -> target_drone_id)
const interceptingDrones = new Map<number, number>();

export function getAttackingDrones(): Map<number, Point> {
    return attackingDrones;
}

export function getInterceptingDrones(): Map<number, number> {
    return interceptingDrones;
}

/** Tracks the initial group split point (set during init) */
let groupSplitId = 25;

export function getGroupSplitId(): number {
    return groupSplitId;
}

// Core stores
export const isInitialized = writable(false);
export const isRunning = writable(false);
export const renderState = writable<DroneRenderData[]>([]);
export const droneCounts = derived(renderState, ($rs) => ({
    a: $rs.filter(d => d.id < groupSplitId).length,
    b: $rs.filter(d => d.id >= groupSplitId).length,
}));
export const status = writable<SwarmStatus>({
    simulationTime: 0,
    droneCount: 0,
    selectedCount: 0,
    speedMultiplier: 16.0,
    isValid: false,
});

// Path mode stores
export const pathMode = writable(false);
export const currentPath = writable<Point[]>([]);

// Active route (persists across resets when in route mode)
export const activeRoute = writable<Point[]>([]);

// Two-group route definitions
export const ROUTE_A: Point[] = [
    { x: 560, y: 480 },
    { x: 840, y: 800 },
    { x: 1120, y: 440 },
    { x: 1080, y: 1120 },
    { x: 840, y: 920 },
    { x: 600, y: 1080 },
];

export const ROUTE_B: Point[] = [
    { x: 2960, y: 2880 },
    { x: 3240, y: 3200 },
    { x: 3440, y: 2840 },
    { x: 3400, y: 3520 },
    { x: 3240, y: 3320 },
    { x: 3000, y: 3480 },
];

export const activeRoutes = writable<{ a: Point[]; b: Point[] }>({ a: ROUTE_A, b: ROUTE_B });

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

// Strategy configuration per group
export type StrategyType = 'none' | 'defend_area' | 'attack_zone' | 'patrol_perimeter' | 'doctrine_aggressive' | 'doctrine_defensive' | 'doctrine_rl';
export const groupAStrategy = writable<StrategyType>('doctrine_rl');
export const groupBStrategy = writable<StrategyType>('doctrine_defensive');

// Legacy - kept for compatibility with AvoidanceSlider
export const avoidanceLookahead = writable(1.0);

// Derived stores
export const selectedCount = derived(status, ($status) => $status.selectedCount);

// ========================================================================
// Combat Air Patrol (CAP) Strategy
// ========================================================================

/** Whether CAP mode is active */
export const capActive = writable(false);

// ---- CAP tuning knobs ----
/** Detection radius (world px) -- enemies inside this trigger intercepts */
const CAP_DETECT_RADIUS = 800;
/** Minimum fraction of group that must remain patrolling */
const CAP_MIN_PATROL_RATIO = 0.4;
/** Max fraction of group allowed on attack runs */
const CAP_MAX_ATTACK_RATIO = 0.3;
/** Ticks between attack-wave decisions */
const CAP_ATTACK_INTERVAL = 120;
/** Ticks between recalling stale interceptors whose targets fled the zone */
const CAP_RECALL_INTERVAL = 60;

let capTickCounter = 0;

/** How many drones to assign per enemy target */
const DRONES_PER_TARGET = 2;
/** Fraction of group to reserve for defense (patrol) */
const DEFENSE_RESERVE = 0.4;

/** Cached RL model JSON (fetched once, reused). */
let rlModelJsonCache: string | null = null;
/** Cached normalizer JSON (fetched once, reused). */
let rlNormalizerJsonCache: string | null = null;

async function loadRlModelJson(): Promise<string> {
    if (rlModelJsonCache) return rlModelJsonCache;
    const resp = await fetch('/models/best_model.json');
    if (!resp.ok) throw new Error(`Failed to load RL model: ${resp.statusText}`);
    rlModelJsonCache = await resp.text();
    return rlModelJsonCache;
}

async function loadRlNormalizerJson(): Promise<string | null> {
    if (rlNormalizerJsonCache !== null) return rlNormalizerJsonCache;
    try {
        const resp = await fetch('/models/best_model_normalizers.json');
        if (!resp.ok) return null;
        rlNormalizerJsonCache = await resp.text();
        return rlNormalizerJsonCache;
    } catch {
        return null;
    }
}

/** Track last known target destruction count to detect changes */
let lastTargetDestroyedCount = 0;

// ========================================================================
// Core Actions
// ========================================================================

export function updateRenderState(): void {
    if (!manager) return;
    renderState.set(manager.getRenderState());
    status.set(manager.getStatus());
}

export async function initSimulation(config?: SimulationConfig): Promise<void> {
    const finalConfig = config ?? (await loadConfig());
    manager = new SwarmManager();
    await manager.initialize(finalConfig);
    isInitialized.set(true);
    updateRenderState();

    // Split drones into two groups with independent formations and routes
    const halfCount = Math.ceil(finalConfig.droneCount / 2);
    groupSplitId = halfCount;
    attackingDrones.clear();
    interceptingDrones.clear();
    capTickCounter = 0;
    lastTargetDestroyedCount = 0;
    gameResult.set(null);

    // Tell WASM the group split so intercept can identify friendly zones
    manager.setGroupSplit(halfCount);

    // Set protected zones from current targets
    updateProtectedZones();

    // Set up chevron formation for each group (before assigning routes)
    const spacing = 40; // pixels
    manager.setFormationForRange('chevron', spacing, 0, halfCount);
    manager.setFormationForRange('chevron', spacing, halfCount, finalConfig.droneCount);

    // Build patrol routes around each group's friendly targets
    const currentTargets = get(targets);
    const friendlyA = currentTargets.filter(t => t.group === 'a' && !t.destroyed);
    const friendlyB = currentTargets.filter(t => t.group === 'b' && !t.destroyed);
    const routeA = friendlyA.length > 0 ? buildPatrolRoute(friendlyA) : ROUTE_A;
    const routeB = friendlyB.length > 0 ? buildPatrolRoute(friendlyB) : ROUTE_B;

    // Assign separate routes -- formation-aware (only leaders get routes)
    manager.assignRouteRange(routeA, 0, halfCount);
    manager.assignRouteRange(routeB, halfCount, finalConfig.droneCount);

    activeRoute.set(routeA);
    activeRoutes.set({ a: routeA, b: routeB });
    currentPath.set(routeA);

    // Apply default strategies
    const defaultA = get(groupAStrategy);
    const defaultB = get(groupBStrategy);
    if (defaultA !== 'none') applyStrategy(0, defaultA);
    if (defaultB !== 'none') applyStrategy(1, defaultB);

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
    targets.set(generateTargets());

    // Update swarm size store
    swarmSize.set(size);

    // Get base config and override drone count with fresh two-cluster positions
    const baseConfig = await loadConfig();
    const count = SWARM_SIZE_COUNTS[size];
    const newConfig = {
        ...baseConfig,
        droneCount: count,
        spawnPattern: { custom: { positions: generateTwoClusterPositions(count, CLUSTER_A, CLUSTER_B) } },
    };

    // Reinitialize and restore route
    await initSimulation(newConfig);
    restoreActiveRoute();
}

export function tickSimulation(dt: number): void {
    if (!manager) return;

    // Snapshot ALL drone positions before tick so we can show explosions for any that die
    const dronesBefore = new Map<number, Point>();
    for (const d of manager.getRenderState()) {
        dronesBefore.set(d.id, { x: d.x, y: d.y });
    }

    manager.tick(dt);
    updateRenderState();

    const currentDrones = manager.getRenderState();
    const currentIds = new Set(currentDrones.map(d => d.id));

    // Any drone that existed before the tick but not after -> explosion at last position
    for (const [id, pos] of dronesBefore) {
        if (!currentIds.has(id)) {
            addExplosion(pos.x, pos.y);
            attackingDrones.delete(id);
            interceptingDrones.delete(id);
        }
    }

    // Retarget: if an attacker's target has been destroyed, redirect to another enemy target
    if (attackingDrones.size > 0) {
        const currentTargets = get(targets);
        for (const [droneId, targetPos] of attackingDrones) {
            if (!currentIds.has(droneId)) continue; // drone is dead
            // Check if the target at this position was destroyed
            const targetHit = currentTargets.find(
                t => t.destroyed && Math.abs(t.x - targetPos.x) < 5 && Math.abs(t.y - targetPos.y) < 5
            );
            if (targetHit) {
                // Find another surviving enemy target
                const droneGroup = droneId < groupSplitId ? 'a' : 'b';
                const enemyGroup = droneGroup === 'a' ? 'b' : 'a';
                const surviving = currentTargets.filter(t => t.group === enemyGroup && !t.destroyed);
                if (surviving.length > 0) {
                    // Pick the closest surviving target
                    const drone = currentDrones.find(d => d.id === droneId);
                    if (drone) {
                        surviving.sort((a, b) => {
                            const da = (a.x - drone.x) ** 2 + (a.y - drone.y) ** 2;
                            const db = (b.x - drone.x) ** 2 + (b.y - drone.y) ** 2;
                            return da - db;
                        });
                        attackTarget(droneId, surviving[0].x, surviving[0].y);
                    }
                } else {
                    attackingDrones.delete(droneId);
                }
            }
        }
    }

    // Check if targets changed -- update protected zones + CAP routes
    {
        const destroyedNow = get(targets).filter(t => t.destroyed).length;
        if (destroyedNow !== lastTargetDestroyedCount) {
            lastTargetDestroyedCount = destroyedNow;
            updateProtectedZones();
            if (get(capActive)) {
                updateCAPRoutes(currentDrones);
            }
        }
    }

    // Check win condition
    checkWinCondition(currentDrones);

    // Run CAP strategy tick
    tickCAP();
}

// ========================================================================
// Combat Actions
// ========================================================================

export function detonateDrone(droneId: number): void {
    if (!manager) return;
    // Capture position before detonation
    const drones = manager.getRenderState();
    const drone = drones.find(d => d.id === droneId);
    if (drone) addExplosion(drone.x, drone.y);
    manager.detonateDrone(droneId);
    updateRenderState();
}

export function detonateSelected(): void {
    if (!manager) return;
    const drones = manager.getRenderState();
    const selected = drones.filter(d => d.selected);
    for (const drone of selected) addExplosion(drone.x, drone.y);
    manager.detonateSelected();
    updateRenderState();
}

export function detonateRandomDrone(): void {
    if (!manager) return;
    const drones = manager.getRenderState();
    if (drones.length === 0) return;
    const target = drones[Math.floor(Math.random() * drones.length)];
    addExplosion(target.x, target.y);
    manager.detonateDrone(target.id);
    updateRenderState();
}

export function attackTarget(droneId: number, targetX: number, targetY: number): void {
    if (!manager) return;
    manager.attackTarget(droneId, targetX, targetY);
    attackingDrones.set(droneId, { x: targetX, y: targetY });
    interceptingDrones.delete(droneId); // clear intercept if reassigned
    updateRenderState();
}

export function interceptDrone(attackerId: number, targetDroneId: number): void {
    if (!manager) return;
    manager.interceptDrone(attackerId, targetDroneId);
    interceptingDrones.set(attackerId, targetDroneId);
    attackingDrones.delete(attackerId); // clear attack if reassigned
    updateRenderState();
}

export function returnToFormation(droneId: number): void {
    if (!manager) return;
    const isGroupA = droneId < groupSplitId;
    const groupStart = isGroupA ? 0 : groupSplitId;
    const groupEnd = isGroupA ? groupSplitId : groupSplitId * 2;
    manager.returnToFormation(droneId, groupStart, groupEnd);
    attackingDrones.delete(droneId);
    interceptingDrones.delete(droneId);
    updateRenderState();
}

// ========================================================================
// Group Strategy
// ========================================================================

/**
 * Assign multiple drones from a group to attack all surviving enemy targets.
 * Distributes drones evenly across targets, respecting a defense reserve.
 */
function assignGroupAttackers(
    groupDrones: { id: number; x: number; y: number }[],
    enemyTargets: { id: number; x: number; y: number; group: 'a' | 'b'; destroyed: boolean }[],
): void {
    if (enemyTargets.length === 0 || groupDrones.length === 0) return;

    // Filter out drones already on attack runs
    const available = groupDrones.filter(d => !attackingDrones.has(d.id));
    if (available.length === 0) return;

    // Reserve some drones for defense
    const maxAttackers = Math.floor(available.length * (1 - DEFENSE_RESERVE));
    if (maxAttackers === 0) return;

    // Distribute attackers across targets
    const perTarget = Math.max(1, Math.min(DRONES_PER_TARGET, Math.floor(maxAttackers / enemyTargets.length)));
    let sent = 0;

    for (const tgt of enemyTargets) {
        // Pick the closest available drones to this target
        const remaining = available.filter(d => !attackingDrones.has(d.id));
        remaining.sort((a, b) => {
            const da = (a.x - tgt.x) ** 2 + (a.y - tgt.y) ** 2;
            const db = (b.x - tgt.x) ** 2 + (b.y - tgt.y) ** 2;
            return da - db;
        });

        for (let i = 0; i < perTarget && i < remaining.length && sent < maxAttackers; i++) {
            attackTarget(remaining[i].id, tgt.x, tgt.y);
            sent++;
        }
    }
}

/** Launch a coordinated attack wave -- each group sends drones to enemy targets. */
export function launchAttack(): void {
    if (!manager) return;
    const drones = manager.getRenderState();
    const currentTargets = get(targets);

    // Group A (IDs < groupSplitId) attacks blue targets
    const groupA = drones.filter(d => d.id < groupSplitId);
    const blueTargets = currentTargets.filter(t => t.group === 'b' && !t.destroyed);
    assignGroupAttackers(groupA, blueTargets);

    // Group B (IDs >= groupSplitId) attacks red targets
    const groupB = drones.filter(d => d.id >= groupSplitId);
    const redTargets = currentTargets.filter(t => t.group === 'a' && !t.destroyed);
    assignGroupAttackers(groupB, redTargets);
}

// ========================================================================
// CAP Helpers
// ========================================================================

interface GroupState {
    drones: { id: number; x: number; y: number }[];
    patrolling: { id: number; x: number; y: number }[];
    attacking: { id: number; x: number; y: number }[];
    intercepting: { id: number; x: number; y: number }[];
    friendlyTargets: { id: number; x: number; y: number; group: 'a' | 'b'; destroyed: boolean }[];
    enemyTargets: { id: number; x: number; y: number; group: 'a' | 'b'; destroyed: boolean }[];
    center: Point;
    groupStart: number;
    groupEnd: number;
    friendlyGroup: 'a' | 'b';
}

function buildGroupState(
    drones: { id: number; x: number; y: number }[],
    currentTargets: { id: number; x: number; y: number; group: 'a' | 'b'; destroyed: boolean }[],
    friendlyGroup: 'a' | 'b',
    idStart: number,
    idEnd: number,
): GroupState {
    const groupDrones = drones.filter(d => d.id >= idStart && d.id < idEnd);
    const friendlyTargets = currentTargets.filter(t => t.group === friendlyGroup && !t.destroyed);
    const enemyGroup = friendlyGroup === 'a' ? 'b' : 'a';
    const enemyTargets = currentTargets.filter(t => t.group === enemyGroup && !t.destroyed);

    const patrolling: typeof groupDrones = [];
    const attacking: typeof groupDrones = [];
    const intercepting: typeof groupDrones = [];
    for (const d of groupDrones) {
        if (attackingDrones.has(d.id)) attacking.push(d);
        else if (interceptingDrones.has(d.id)) intercepting.push(d);
        else patrolling.push(d);
    }

    const center = friendlyTargets.length > 0
        ? { x: friendlyTargets.reduce((s, t) => s + t.x, 0) / friendlyTargets.length,
            y: friendlyTargets.reduce((s, t) => s + t.y, 0) / friendlyTargets.length }
        : friendlyGroup === 'a' ? { x: 840, y: 800 } : { x: 3240, y: 3200 };

    return { drones: groupDrones, patrolling, attacking, intercepting, friendlyTargets, enemyTargets, center, groupStart: idStart, groupEnd: idEnd, friendlyGroup };
}

function capDecideGroup(
    group: GroupState,
    allDrones: { id: number; x: number; y: number }[],
): void {
    if (!manager) return;
    const total = group.drones.length;
    if (total === 0) return;

    const minPatrol = Math.max(1, Math.ceil(total * CAP_MIN_PATROL_RATIO));
    const maxAttack = Math.floor(total * CAP_MAX_ATTACK_RATIO);

    // ---- 1. RECALL: return interceptors whose targets left the zone ----
    if (capTickCounter % CAP_RECALL_INTERVAL === 0) {
        for (const d of group.intercepting) {
            const targetId = interceptingDrones.get(d.id);
            if (targetId === undefined) { returnToFormation(d.id); continue; }
            const targetDrone = allDrones.find(dr => dr.id === targetId);
            if (!targetDrone) { returnToFormation(d.id); continue; }
            // If the target has fled far from our patrol zone, recall
            const dx = targetDrone.x - group.center.x;
            const dy = targetDrone.y - group.center.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist > CAP_DETECT_RADIUS * 1.5) {
                returnToFormation(d.id);
            }
        }
    }

    // ---- 2. ENFORCE PATROL MINIMUM: recall farthest attacker if too few patrol ----
    // Recount after recalls
    const currentPatrol = group.drones.filter(d =>
        !attackingDrones.has(d.id) && !interceptingDrones.has(d.id)
    );
    if (currentPatrol.length < minPatrol) {
        // Recall the attacker farthest from our center
        const currentAttackers = group.drones.filter(d => attackingDrones.has(d.id));
        if (currentAttackers.length > 0) {
            currentAttackers.sort((a, b) => {
                const da = (a.x - group.center.x) ** 2 + (a.y - group.center.y) ** 2;
                const db = (b.x - group.center.x) ** 2 + (b.y - group.center.y) ** 2;
                return db - da; // farthest first
            });
            returnToFormation(currentAttackers[0].id);
        }
    }

    // ---- 3. INTERCEPT: send drones after enemies near friendly targets ----
    const enemyGroup = group.friendlyGroup === 'a' ? 'b' : 'a';
    const enemyDrones = allDrones.filter(d =>
        (enemyGroup === 'a' ? d.id < groupSplitId : d.id >= groupSplitId)
    );
    const enemiesNear = enemyDrones.filter(d => {
        const dx = d.x - group.center.x;
        const dy = d.y - group.center.y;
        return Math.sqrt(dx * dx + dy * dy) < CAP_DETECT_RADIUS;
    });

    for (const enemy of enemiesNear) {
        // Skip if already being intercepted
        if ([...interceptingDrones.values()].includes(enemy.id)) continue;

        // Only intercept if we have patrol budget
        const availableForIntercept = group.drones.filter(d =>
            !attackingDrones.has(d.id) && !interceptingDrones.has(d.id)
        );
        if (availableForIntercept.length <= minPatrol) break; // keep minimum patrol

        availableForIntercept.sort((a, b) => {
            const da = (a.x - enemy.x) ** 2 + (a.y - enemy.y) ** 2;
            const db = (b.x - enemy.x) ** 2 + (b.y - enemy.y) ** 2;
            return da - db;
        });
        interceptDrone(availableForIntercept[0].id, enemy.id);
    }

    // ---- 4. ATTACK: periodically send drones to enemy targets ----
    if (capTickCounter % CAP_ATTACK_INTERVAL === 0 && group.enemyTargets.length > 0) {
        const currentAttackCount = group.drones.filter(d => attackingDrones.has(d.id)).length;
        if (currentAttackCount < maxAttack) {
            const available = group.drones.filter(d =>
                !attackingDrones.has(d.id) && !interceptingDrones.has(d.id)
            );
            // Only send if above patrol minimum after sending
            if (available.length > minPatrol) {
                const target = group.enemyTargets[Math.floor(Math.random() * group.enemyTargets.length)];
                available.sort((a, b) => {
                    const da = (a.x - target.x) ** 2 + (a.y - target.y) ** 2;
                    const db = (b.x - target.x) ** 2 + (b.y - target.y) ** 2;
                    return da - db;
                });
                attackTarget(available[0].id, target.x, target.y);
            }
        }
    }
}

function updateCAPRoutes(currentDrones: { id: number }[]): void {
    if (!manager) return;
    const currentTargets = get(targets);

    // Rebuild patrol routes around surviving friendly targets
    const friendlyA = currentTargets.filter(t => t.group === 'a' && !t.destroyed);
    const friendlyB = currentTargets.filter(t => t.group === 'b' && !t.destroyed);

    const patrolA = friendlyA.length > 0 ? buildPatrolRoute(friendlyA) : ROUTE_A;
    const patrolB = friendlyB.length > 0 ? buildPatrolRoute(friendlyB) : ROUTE_B;

    // Use original group split; assignRouteRange only affects drones that still exist in range
    const totalDrones = get(status).droneCount;
    manager.assignRouteRange(patrolA, 0, groupSplitId);
    manager.assignRouteRange(patrolB, groupSplitId, groupSplitId + totalDrones);
    activeRoutes.set({ a: patrolA, b: patrolB });
    updateRenderState();
}

function tickCAP(): void {
    if (!manager || !get(capActive)) return;
    capTickCounter++;

    const drones = manager.getRenderState();
    const currentTargets = get(targets);
    const currentIds = new Set(drones.map(d => d.id));

    // Clean up tracking maps for dead drones
    for (const [id] of attackingDrones) { if (!currentIds.has(id)) attackingDrones.delete(id); }
    for (const [id] of interceptingDrones) { if (!currentIds.has(id)) interceptingDrones.delete(id); }

    // Clean up aborted interceptors (Rust set them to Sleep because they were near a friendly target)
    for (const [id] of interceptingDrones) {
        const drone = drones.find(d => d.id === id);
        if (drone && drone.objectiveType === 'Sleep') {
            interceptingDrones.delete(id);
            returnToFormation(id);
        }
    }

    // Build state for each group and run decisions
    const groupA = buildGroupState(drones, currentTargets, 'a', 0, groupSplitId);
    const groupB = buildGroupState(drones, currentTargets, 'b', groupSplitId, groupSplitId * 2);

    capDecideGroup(groupA, drones);
    capDecideGroup(groupB, drones);
}

/** Toggle Combat Air Patrol mode */
export function toggleCAP(): void {
    const wasActive = get(capActive);
    capActive.set(!wasActive);
    capTickCounter = 0;

    if (!manager) return;
    // Always use the original group split, not current drone count
    const endId = groupSplitId * 2;

    if (!wasActive) {
        // Activating CAP -- build patrol routes around friendly targets
        const currentTargets = get(targets);
        const friendlyA = currentTargets.filter(t => t.group === 'a' && !t.destroyed);
        const friendlyB = currentTargets.filter(t => t.group === 'b' && !t.destroyed);

        const patrolA = friendlyA.length > 0 ? buildPatrolRoute(friendlyA) : ROUTE_A;
        const patrolB = friendlyB.length > 0 ? buildPatrolRoute(friendlyB) : ROUTE_B;

        // Red drones (0..groupSplitId) patrol red targets
        // Blue drones (groupSplitId..end) patrol blue targets
        manager.assignRouteRange(patrolA, 0, groupSplitId);
        manager.assignRouteRange(patrolB, groupSplitId, endId);
        activeRoutes.set({ a: patrolA, b: patrolB });
        updateRenderState();
    } else {
        // Deactivating CAP -- restore original patrol routes
        manager.assignRouteRange(ROUTE_A, 0, groupSplitId);
        manager.assignRouteRange(ROUTE_B, groupSplitId, endId);
        activeRoutes.set({ a: ROUTE_A, b: ROUTE_B });
        updateRenderState();
    }
}

// ========================================================================
// Navigation & Config Actions
// ========================================================================

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

export function resetSimulation(): void {
    if (manager) {
        manager.destroy();
        manager = null;
    }
    isInitialized.set(false);
    isRunning.set(false);
    pathMode.set(false);
    targets.set(generateTargets());
    // Don't clear currentPath/activeRoute here - they persist across resets in route mode
}

export function destroySimulation(): void {
    resetSimulation();
}

export function restoreActiveRoute(): void {
    const routes = get(activeRoutes);
    if (routes.a.length > 0 && get(objectiveMode) === 'route') {
        const count = get(status).droneCount;
        const half = Math.ceil(count / 2);
        manager?.assignRouteRange(routes.a, 0, half);
        manager?.assignRouteRange(routes.b, half, count);
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

// ========================================================================
// Rust-side Strategy Management
// ========================================================================

export function setGroupStrategy(group: 0 | 1, strategy: StrategyType): void {
    if (!manager) return;

    // Remove RL agents for both groups (strategies are all cleared and re-applied)
    manager.removeRlAgent(0);
    manager.removeRlAgent(1);

    // First clear existing strategies (they'll be re-applied)
    manager.clearStrategies();

    // Update stores
    if (group === 0) groupAStrategy.set(strategy);
    else groupBStrategy.set(strategy);

    // Read both group strategies
    const stratA = group === 0 ? strategy : get(groupAStrategy);
    const stratB = group === 1 ? strategy : get(groupBStrategy);

    // Apply both strategies (clear_strategies removes all, so we re-apply both)
    applyStrategy(0, stratA);
    applyStrategy(1, stratB);

    updateRenderState();
}

function applyStrategy(group: 0 | 1, strategy: StrategyType): void {
    if (!manager || strategy === 'none') return;

    const drones = manager.getRenderState();
    const currentTargets = get(targets);

    // Get drone IDs for this group
    const groupDrones = group === 0
        ? drones.filter(d => d.id < groupSplitId)
        : drones.filter(d => d.id >= groupSplitId);
    const droneIds = groupDrones.map(d => d.id);
    if (droneIds.length === 0) return;

    // Determine center and radius from friendly targets
    const friendlyGroup = group === 0 ? 'a' : 'b';
    const friendlyTargets = currentTargets.filter(t => t.group === friendlyGroup && !t.destroyed);
    const enemyGroup = group === 0 ? 'b' : 'a';
    const enemyTargets = currentTargets.filter(t => t.group === enemyGroup && !t.destroyed);

    // Center defaults to cluster center if no targets
    const center = friendlyTargets.length > 0
        ? {
            x: friendlyTargets.reduce((s, t) => s + t.x, 0) / friendlyTargets.length,
            y: friendlyTargets.reduce((s, t) => s + t.y, 0) / friendlyTargets.length,
        }
        : group === 0 ? { x: 840, y: 800 } : { x: 3240, y: 3200 };

    switch (strategy) {
        case 'defend_area':
            manager.setStrategyDefendArea(droneIds, center.x, center.y, 600);
            break;
        case 'attack_zone': {
            // Attack enemy targets -- pass their positions directly
            const targetPositions = enemyTargets.map(t => ({ x: t.x, y: t.y }));
            if (targetPositions.length === 0) break;
            manager.setStrategyAttackZone(droneIds, targetPositions);
            break;
        }
        case 'patrol_perimeter': {
            // Build patrol perimeter around the group's friendly targets
            const patrolWaypoints = friendlyTargets.length > 0
                ? buildPatrolRoute(friendlyTargets)
                : (group === 0 ? ROUTE_A : ROUTE_B);
            manager.setStrategyPatrolPerimeter(droneIds, patrolWaypoints, 0.0);
            break;
        }
        case 'doctrine_aggressive':
        case 'doctrine_defensive': {
            // Autonomous force allocation: doctrine manages defense + offense
            const patrolWaypoints = friendlyTargets.length > 0
                ? buildPatrolRoute(friendlyTargets)
                : (group === 0 ? ROUTE_A : ROUTE_B);
            const friendlyPositions = friendlyTargets.map(t => ({ x: t.x, y: t.y }));
            const enemyPositions = enemyTargets.map(t => ({ x: t.x, y: t.y }));
            const mode = strategy === 'doctrine_aggressive' ? 'aggressive' : 'defensive';
            manager.setDoctrine(droneIds, friendlyPositions, enemyPositions, patrolWaypoints, mode);
            break;
        }
        case 'doctrine_rl': {
            // Set up doctrine with defensive as base mode, then load RL model on top
            const patrolWaypoints = friendlyTargets.length > 0
                ? buildPatrolRoute(friendlyTargets)
                : (group === 0 ? ROUTE_A : ROUTE_B);
            const friendlyPositions = friendlyTargets.map(t => ({ x: t.x, y: t.y }));
            const enemyPositions = enemyTargets.map(t => ({ x: t.x, y: t.y }));
            manager.setDoctrine(droneIds, friendlyPositions, enemyPositions, patrolWaypoints, 'defensive');

            // Load RL model asynchronously
            const initialOwnDrones = droneIds.length;
            const otherDrones = drones.filter(d =>
                group === 0 ? d.id >= groupSplitId : d.id < groupSplitId
            );
            const initialEnemyDrones = otherDrones.length;

            Promise.all([loadRlModelJson(), loadRlNormalizerJson()]).then(([modelJson, normJson]) => {
                if (!manager) return;
                manager.loadRlModelMulti(
                    group,
                    modelJson,
                    initialOwnDrones,
                    initialEnemyDrones,
                    friendlyTargets.length,
                    enemyTargets.length,
                );
                if (normJson) {
                    manager.loadRlNormalizers(group, normJson);
                }
            }).catch(err => {
                console.error('Failed to load RL model:', err);
            });
            break;
        }
    }
}
