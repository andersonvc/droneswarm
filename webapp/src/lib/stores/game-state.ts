import { writable, derived, get } from 'svelte/store';
import type { Point } from '$lib/wasm/bridge';
import { getManager, getGroupSplitId } from './simulation';

// Explosion visual effects
export interface Explosion {
    x: number;
    y: number;
    radius: number; // in world pixel coords
    time: number;    // timestamp when created
}
export const explosions = writable<Explosion[]>([]);

// Blast radius: 187.5m * (4000px / 10000m) = 75px in world coords.
const EXPLOSION_VISUAL_RADIUS = 75;
const EXPLOSION_DURATION_MS = 800;

export function addExplosion(x: number, y: number): void {
    explosions.update(list => [...list, { x, y, radius: EXPLOSION_VISUAL_RADIUS, time: Date.now() }]);
    checkTargetDestruction(x, y, EXPLOSION_VISUAL_RADIUS);
    setTimeout(() => {
        explosions.update(list => list.filter(e => Date.now() - e.time < EXPLOSION_DURATION_MS));
    }, EXPLOSION_DURATION_MS);
}

// Targets
export interface Target {
    id: number;
    x: number;
    y: number;
    group: 'a' | 'b'; // 'a' = red (for group 1), 'b' = blue (for group 2)
    destroyed: boolean;
}

const TARGET_SIZE = 30; // visual size in world px (half-width for hit detection)

export function generateTargets(): Target[] {
    const targets: Target[] = [];
    // Red targets ('a') near Group A's territory (top-left quadrant)
    for (let i = 0; i < 6; i++) {
        targets.push({
            id: i,
            x: 900 + Math.random() * 700,
            y: 900 + Math.random() * 700,
            group: 'a',
            destroyed: false,
        });
    }
    // Blue targets ('b') near Group B's territory (bottom-right quadrant)
    for (let i = 0; i < 6; i++) {
        targets.push({
            id: 6 + i,
            x: 2400 + Math.random() * 700,
            y: 2400 + Math.random() * 700,
            group: 'b',
            destroyed: false,
        });
    }
    return targets;
}

export const targets = writable<Target[]>(generateTargets());

export const targetCounts = derived(targets, ($targets) => ({
    a: $targets.filter(t => t.group === 'a' && !t.destroyed).length,
    b: $targets.filter(t => t.group === 'b' && !t.destroyed).length,
}));

export function checkTargetDestruction(explosionX: number, explosionY: number, blastRadius: number): void {
    targets.update(list =>
        list.map(t => {
            if (t.destroyed) return t;
            const dx = t.x - explosionX;
            const dy = t.y - explosionY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            // Target is hit if blast radius reaches any part of the square
            if (dist <= blastRadius + TARGET_SIZE / 2) {
                return { ...t, destroyed: true };
            }
            return t;
        })
    );
}

// Win condition
export type GameResult = null | 'a_wins' | 'b_wins' | 'draw';
export const gameResult = writable<GameResult>(null);

export function checkWinCondition(currentDrones: { id: number }[]): void {
    if (get(gameResult) !== null) return; // already decided

    const groupSplitId = getGroupSplitId();
    const currentTargets = get(targets);
    const redTargetsAlive = currentTargets.filter(t => t.group === 'a' && !t.destroyed).length;
    const blueTargetsAlive = currentTargets.filter(t => t.group === 'b' && !t.destroyed).length;

    const groupAAlive = currentDrones.filter(d => d.id < groupSplitId).length;
    const groupBAlive = currentDrones.filter(d => d.id >= groupSplitId).length;

    // Game continues if both sides have drones and targets.
    if (redTargetsAlive > 0 && blueTargetsAlive > 0 && groupAAlive > 0 && groupBAlive > 0) return;

    // When a side loses all drones, remaining enemy drones
    // effectively destroy that side's targets (1 drone = 1 target).
    const effA = groupAAlive === 0 ? Math.max(0, redTargetsAlive - groupBAlive) : redTargetsAlive;
    const effB = groupBAlive === 0 ? Math.max(0, blueTargetsAlive - groupAAlive) : blueTargetsAlive;

    const aEliminated = effA === 0;
    const bEliminated = effB === 0;

    if (aEliminated && bEliminated) {
        gameResult.set('draw');
    } else if (bEliminated && !aEliminated) {
        gameResult.set('a_wins');
    } else if (aEliminated && !bEliminated) {
        gameResult.set('b_wins');
    } else {
        // Both have effective targets. Compare who has more.
        if (effA > effB) {
            gameResult.set('a_wins');
        } else if (effB > effA) {
            gameResult.set('b_wins');
        } else {
            gameResult.set('draw');
        }
    }
}

/**
 * Compute the convex hull of a set of points (Andrew's monotone chain).
 * Returns points in counter-clockwise order.
 */
export function convexHull(points: Point[]): Point[] {
    if (points.length <= 1) return [...points];

    const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
    const n = sorted.length;

    // Cross product of vectors OA and OB where O is origin
    const cross = (o: Point, a: Point, b: Point) =>
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

    // Lower hull
    const lower: Point[] = [];
    for (const p of sorted) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
            lower.pop();
        lower.push(p);
    }

    // Upper hull
    const upper: Point[] = [];
    for (let i = n - 1; i >= 0; i--) {
        const p = sorted[i];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
            upper.pop();
        upper.push(p);
    }

    // Remove last point of each half because it's repeated
    lower.pop();
    upper.pop();
    return [...lower, ...upper];
}

/** Standoff distance (px) for patrol waypoints around friendly targets */
const CAP_PATROL_STANDOFF = 200;

/**
 * Build a convex patrol route around a set of targets.
 * Computes convex hull of target positions, then pushes each vertex
 * outward by CAP_PATROL_STANDOFF from the centroid.
 */
export function buildPatrolRoute(friendlyTargets: Target[]): Point[] {
    if (friendlyTargets.length === 0) return [];

    const targetPoints = friendlyTargets.map(t => ({ x: t.x, y: t.y }));

    // For 1-2 targets, create a simple polygon around them
    if (targetPoints.length <= 2) {
        const cx = targetPoints.reduce((s, p) => s + p.x, 0) / targetPoints.length;
        const cy = targetPoints.reduce((s, p) => s + p.y, 0) / targetPoints.length;
        // Generate a square/hexagon around the centroid
        const count = 4;
        const route: Point[] = [];
        for (let i = 0; i < count; i++) {
            const angle = (i / count) * Math.PI * 2;
            route.push({
                x: cx + CAP_PATROL_STANDOFF * Math.cos(angle),
                y: cy + CAP_PATROL_STANDOFF * Math.sin(angle),
            });
        }
        return route;
    }

    const hull = convexHull(targetPoints);
    const cx = hull.reduce((s, p) => s + p.x, 0) / hull.length;
    const cy = hull.reduce((s, p) => s + p.y, 0) / hull.length;

    return hull.map(p => {
        const dx = p.x - cx;
        const dy = p.y - cy;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        return {
            x: p.x + (dx / dist) * CAP_PATROL_STANDOFF,
            y: p.y + (dy / dist) * CAP_PATROL_STANDOFF,
        };
    });
}

/** Sync protected zones to WASM from current target state */
export function updateProtectedZones(): void {
    const manager = getManager();
    if (!manager) return;
    const currentTargets = get(targets);
    const zoneA = currentTargets
        .filter(t => t.group === 'a' && !t.destroyed)
        .map(t => ({ x: t.x, y: t.y }));
    const zoneB = currentTargets
        .filter(t => t.group === 'b' && !t.destroyed)
        .map(t => ({ x: t.x, y: t.y }));
    manager.setProtectedZones(0, zoneA); // group 0 defends 'a' targets
    manager.setProtectedZones(1, zoneB); // group 1 defends 'b' targets

    // Update doctrine strategies with current target state
    manager.updateDoctrineTargets(0, zoneA, zoneB); // group 0: friendly=A, enemy=B
    manager.updateDoctrineTargets(1, zoneB, zoneA); // group 1: friendly=B, enemy=A

    // Sync RL agent target counts
    manager.updateRlTargets(0, zoneA.length, zoneB.length);
    manager.updateRlTargets(1, zoneB.length, zoneA.length);
}
