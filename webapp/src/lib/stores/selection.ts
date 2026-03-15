import { writable } from 'svelte/store';
import { getManager, updateRenderState } from './simulation';

// Hover store for tooltips
export const hoveredDroneId = writable<number | null>(null);

// Selection actions
export function selectDrone(id: number, multi: boolean): void {
    getManager()?.selectDrone(id, multi);
    updateRenderState();
}

export function selectDronesInRect(minX: number, minY: number, maxX: number, maxY: number, multi: boolean): void {
    const manager = getManager();
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
    getManager()?.clearSelection();
    updateRenderState();
}

export function getDroneAt(x: number, y: number): number | undefined {
    return getManager()?.getDroneAt(x, y, 20);
}
