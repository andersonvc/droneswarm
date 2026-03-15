const DRONE_HEIGHT = 20;
const DRONE_WIDTH = 7;
const SELECTION_RADIUS = 20;
const COLLISION_RADIUS = 1; // Half of COLLISION_DISTANCE (30) from wasm-lib
const LIME_ACCENT = '#9DFF20';
const WAYPOINT_RADIUS = 8;

export interface DroneData {
    id: number;
    x: number;
    y: number;
    heading: number;
    color: { r: number; g: number; b: number };
    selected: boolean;
    target?: { x: number; y: number } | null;
    routePath?: { x: number; y: number }[] | null;
    splinePath?: { x: number; y: number }[] | null;
    planningPath?: { x: number; y: number }[] | null;
    approachMode: string;
    objectiveType: string;
}

export function drawDrone(
    ctx: CanvasRenderingContext2D,
    drone: DroneData,
): void {
    const { x, y, heading, color, selected } = drone;

    const cosH = Math.cos(heading);
    const sinH = Math.sin(heading);

    const tipX = x + cosH * (DRONE_HEIGHT / 2);
    const tipY = y + sinH * (DRONE_HEIGHT / 2);

    const backLeftX = x - cosH * (DRONE_HEIGHT / 2) - sinH * (DRONE_WIDTH / 2);
    const backLeftY = y - sinH * (DRONE_HEIGHT / 2) + cosH * (DRONE_WIDTH / 2);

    const backRightX = x - cosH * (DRONE_HEIGHT / 2) + sinH * (DRONE_WIDTH / 2);
    const backRightY = y - sinH * (DRONE_HEIGHT / 2) - cosH * (DRONE_WIDTH / 2);

    // Draw collision boundary (dashed circle, white 10% opacity)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(x, y, COLLISION_RADIUS, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    if (selected) {
        // Lime green selection ring with glow
        ctx.shadowColor = LIME_ACCENT;
        ctx.shadowBlur = 12;
        ctx.strokeStyle = LIME_ACCENT;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, SELECTION_RADIUS, 0, Math.PI * 2);
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(backLeftX, backLeftY);
    ctx.lineTo(backRightX, backRightY);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
}

export function drawWaypointMarkers(
    ctx: CanvasRenderingContext2D,
    drones: DroneData[],
): void {
    const LIME_DIM = 'rgba(157, 255, 32, 0.5)';

    // Collect unique waypoints and their associated drones
    const waypointMap = new Map<string, { x: number; y: number; drones: DroneData[] }>();

    for (const drone of drones) {
        if (drone.target) {
            const key = `${drone.target.x.toFixed(0)},${drone.target.y.toFixed(0)}`;
            if (!waypointMap.has(key)) {
                waypointMap.set(key, { x: drone.target.x, y: drone.target.y, drones: [] });
            }
            waypointMap.get(key)!.drones.push(drone);
        }
    }

    // Draw lines from each drone to its waypoint
    for (const drone of drones) {
        if (drone.target) {
            ctx.strokeStyle = drone.selected ? 'rgba(157, 255, 32, 0.6)' : 'rgba(157, 255, 32, 0.25)';
            ctx.lineWidth = drone.selected ? 2 : 1;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(drone.x, drone.y);
            ctx.lineTo(drone.target.x, drone.target.y);
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    // Draw waypoint markers
    for (const [, waypoint] of waypointMap) {
        const { x, y, drones: wpDrones } = waypoint;
        const hasSelectedDrone = wpDrones.some(d => d.selected);

        // Outer ring
        ctx.strokeStyle = hasSelectedDrone ? LIME_ACCENT : LIME_DIM;
        ctx.lineWidth = hasSelectedDrone ? 3 : 2;
        ctx.beginPath();
        ctx.arc(x, y, WAYPOINT_RADIUS + 4, 0, Math.PI * 2);
        ctx.stroke();

        // Inner filled circle
        ctx.fillStyle = hasSelectedDrone ? LIME_ACCENT : LIME_DIM;
        ctx.beginPath();
        ctx.arc(x, y, WAYPOINT_RADIUS, 0, Math.PI * 2);
        ctx.fill();

        // Drone count badge if multiple drones
        if (wpDrones.length > 1) {
            ctx.fillStyle = '#0a0a0f';
            ctx.font = 'bold 10px "IBM Plex Mono", monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(wpDrones.length), x, y);
        }
    }
}
