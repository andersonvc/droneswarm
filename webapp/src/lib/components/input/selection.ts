import { screenToWorld } from './camera';
import {
    selectDrone,
    clearSelection,
    selectDronesInRect,
    getDroneAt,
} from '$lib/stores/simulation';

export const DRAG_THRESHOLD = 5; // Minimum pixels to consider a drag

export interface SelectionState {
    isDragging: boolean;
    dragStart: { x: number; y: number };
    dragEnd: { x: number; y: number };
}

/** Handle the end of a drag (box selection) -- returns true if handled as drag */
export function handleBoxSelection(
    state: SelectionState,
    panX: number,
    panY: number,
    zoom: number,
    multi: boolean,
): boolean {
    if (!state.isDragging) return false;

    const startWorld = screenToWorld(state.dragStart.x, state.dragStart.y, panX, panY, zoom);
    const endWorld = screenToWorld(state.dragEnd.x, state.dragEnd.y, panX, panY, zoom);
    const minX = Math.min(startWorld.x, endWorld.x);
    const maxX = Math.max(startWorld.x, endWorld.x);
    const minY = Math.min(startWorld.y, endWorld.y);
    const maxY = Math.max(startWorld.y, endWorld.y);

    selectDronesInRect(minX, minY, maxX, maxY, multi);
    return true;
}

/** Handle a single click for drone selection */
export function handleClickSelection(
    worldX: number,
    worldY: number,
    multi: boolean,
): void {
    const droneId = getDroneAt(worldX, worldY);

    if (droneId !== undefined) {
        selectDrone(droneId, multi);
    } else if (!multi) {
        clearSelection();
    }
}
