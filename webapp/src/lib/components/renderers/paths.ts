const LIME_ACCENT = '#9DFF20';
const OLIVE = '#345C00';

export function drawRoutePath(
    ctx: CanvasRenderingContext2D,
    points: { x: number; y: number }[],
    color: { r: number; g: number; b: number },
): void {
    if (points.length < 2) return;

    ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.4)`;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);

    // Points are already densely sampled Hermite spline from WASM
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }
    // Close the loop
    ctx.lineTo(points[0].x, points[0].y);

    ctx.stroke();
    ctx.setLineDash([]);
}

export function drawSplinePath(
    ctx: CanvasRenderingContext2D,
    points: { x: number; y: number }[],
): void {
    if (points.length < 2) return;

    ctx.strokeStyle = 'rgba(255, 0, 0, 0.7)'; // 70% translucent red
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 4]); // Dashed line

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }

    ctx.stroke();
    ctx.setLineDash([]); // Reset dash
}

export function drawPlanningPath(
    ctx: CanvasRenderingContext2D,
    points: { x: number; y: number }[],
    approachMode: string,
): void {
    if (points.length < 2) return;

    // Color based on approach mode:
    // - "approach": Yellow (gated spline approach)
    // - "pursuit": Cyan/Blue (lead pursuit)
    // - "correction": Green (close correction)
    // - "station_keeping": No path (but if shown, also green)
    // - "none": Skip (leader or not in formation)
    let color: string;
    switch (approachMode) {
        case 'approach':
            color = 'rgba(255, 220, 0, 0.7)'; // Yellow
            break;
        case 'pursuit':
            color = 'rgba(0, 180, 255, 0.7)'; // Cyan/Blue
            break;
        case 'correction':
        case 'station_keeping':
            color = 'rgba(0, 255, 100, 0.7)'; // Green
            break;
        default:
            return; // Don't draw for "none" or unknown modes
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]); // Dashed line

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }

    ctx.stroke();
    ctx.setLineDash([]); // Reset dash
}

export function drawPathWaypoints(
    ctx: CanvasRenderingContext2D,
    currentPath: { x: number; y: number }[],
    objectiveMode: string,
    pathMode: boolean,
): void {
    const isRoute = objectiveMode === 'route';
    const isPlacing = pathMode;

    // Use lime green for paths/routes
    const pointColor = isRoute && !isPlacing ? 'rgba(157, 255, 32, 0.5)' : LIME_ACCENT;
    const lineColor = isRoute && !isPlacing ? OLIVE : LIME_ACCENT;

    ctx.lineWidth = 2;

    // Connect with lines first (behind dots)
    if (currentPath.length > 1) {
        ctx.strokeStyle = lineColor;
        ctx.setLineDash(isRoute ? [5, 5] : []);
        ctx.beginPath();
        ctx.moveTo(currentPath[0].x, currentPath[0].y);
        for (let i = 1; i < currentPath.length; i++) {
            ctx.lineTo(currentPath[i].x, currentPath[i].y);
        }
        // If route mode, connect back to start
        if (isRoute && currentPath.length > 2) {
            ctx.lineTo(currentPath[0].x, currentPath[0].y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw waypoint dots with numbers
    for (let i = 0; i < currentPath.length; i++) {
        const { x, y } = currentPath[i];

        // Draw circle
        ctx.fillStyle = pointColor;
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fill();

        // Number label
        ctx.fillStyle = isRoute && !isPlacing ? 'rgba(10, 10, 15, 0.6)' : '#0a0a0f';
        ctx.font = 'bold 12px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(i + 1), x, y);
    }
}

export function drawGroupRoute(
    ctx: CanvasRenderingContext2D,
    waypoints: { x: number; y: number }[],
    pointColor: string,
    lineColor: string,
): void {
    if (waypoints.length === 0) return;

    ctx.lineWidth = 2;

    // Connect with dashed lines
    if (waypoints.length > 1) {
        ctx.strokeStyle = lineColor;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(waypoints[0].x, waypoints[0].y);
        for (let i = 1; i < waypoints.length; i++) {
            ctx.lineTo(waypoints[i].x, waypoints[i].y);
        }
        if (waypoints.length > 2) {
            ctx.lineTo(waypoints[0].x, waypoints[0].y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw waypoint dots with numbers
    for (let i = 0; i < waypoints.length; i++) {
        const { x, y } = waypoints[i];
        ctx.fillStyle = pointColor;
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = 'rgba(10, 10, 15, 0.6)';
        ctx.font = 'bold 12px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(i + 1), x, y);
    }
}
