import { base } from '$app/paths';
import type { SimulationConfig, Point } from './bridge';

export const CLUSTER_A = { center: { x: 760, y: 760 }, radius: 240 };
export const CLUSTER_B = { center: { x: 3240, y: 3240 }, radius: 240 };

export function generateTwoClusterPositions(
    droneCount: number,
    clusterA: { center: Point; radius: number },
    clusterB: { center: Point; radius: number },
): Point[] {
    const halfA = Math.ceil(droneCount / 2);
    const halfB = droneCount - halfA;
    const positions: Point[] = [];

    for (let i = 0; i < halfA; i++) {
        const angle = Math.random() * Math.PI * 2;
        const r = Math.sqrt(Math.random()) * clusterA.radius;
        positions.push({
            x: clusterA.center.x + r * Math.cos(angle),
            y: clusterA.center.y + r * Math.sin(angle),
        });
    }
    for (let i = 0; i < halfB; i++) {
        const angle = Math.random() * Math.PI * 2;
        const r = Math.sqrt(Math.random()) * clusterB.radius;
        positions.push({
            x: clusterB.center.x + r * Math.cos(angle),
            y: clusterB.center.y + r * Math.sin(angle),
        });
    }
    return positions;
}

const DEFAULT_DRONE_COUNT = 50;

const DEFAULT_CONFIG: SimulationConfig = {
    droneCount: DEFAULT_DRONE_COUNT,
    spawnPattern: { custom: { positions: generateTwoClusterPositions(DEFAULT_DRONE_COUNT, CLUSTER_A, CLUSTER_B) } },
    bounds: { width: 4000, height: 4000 },
    speedMultiplier: 16.0,
    worldWidthMeters: 10000,
    worldHeightMeters: 10000,
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
        const response = await fetch(`${base}/config.json`);
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
        const pattern = params.get('pattern')!;
        if (pattern === 'grid' || pattern === 'random') {
            config.spawnPattern = pattern;
        }
    }
    if (params.has('speed')) {
        config.speedMultiplier = parseFloat(params.get('speed')!);
    }

    return Object.keys(config).length > 0 ? config : null;
}
