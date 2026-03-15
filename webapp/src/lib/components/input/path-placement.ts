import type { Point } from '$lib/wasm/bridge';
import { screenToWorld } from './camera';

/** Handle a click in path mode -- returns the new world-space point */
export function handlePathClick(
    screenX: number,
    screenY: number,
    panX: number,
    panY: number,
    zoom: number,
): { x: number; y: number } {
    return screenToWorld(screenX, screenY, panX, panY, zoom);
}
