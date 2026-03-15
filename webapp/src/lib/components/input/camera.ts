export const MIN_ZOOM = 0.1;
export const MAX_ZOOM = 4.0;
export const ZOOM_SENSITIVITY = 0.001;

/** Convert screen coordinates to world coordinates */
export function screenToWorld(
    screenX: number,
    screenY: number,
    panX: number,
    panY: number,
    zoom: number,
): { x: number; y: number } {
    return {
        x: (screenX - panX) / zoom,
        y: (screenY - panY) / zoom,
    };
}

/** Convert world coordinates to screen coordinates */
export function worldToScreen(
    worldX: number,
    worldY: number,
    panX: number,
    panY: number,
    zoom: number,
): { x: number; y: number } {
    return {
        x: worldX * zoom + panX,
        y: worldY * zoom + panY,
    };
}

export interface WheelZoomResult {
    zoom: number;
    panX: number;
    panY: number;
}

/** Calculate new zoom and pan from a wheel event */
export function computeWheelZoom(
    deltaY: number,
    mouseX: number,
    mouseY: number,
    currentZoom: number,
    currentPanX: number,
    currentPanY: number,
): WheelZoomResult {
    const zoomDelta = -deltaY * ZOOM_SENSITIVITY;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, currentZoom * (1 + zoomDelta)));

    // Zoom toward mouse position
    const worldX = (mouseX - currentPanX) / currentZoom;
    const worldY = (mouseY - currentPanY) / currentZoom;

    // Adjust pan to keep mouse position fixed
    return {
        zoom: newZoom,
        panX: mouseX - worldX * newZoom,
        panY: mouseY - worldY * newZoom,
    };
}
