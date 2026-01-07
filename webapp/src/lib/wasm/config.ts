import type { SimulationConfig } from './bridge';

const DEFAULT_CONFIG: SimulationConfig = {
    droneCount: 6, // Matches 'small' swarm size
    spawnPattern: 'random',
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
