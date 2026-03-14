<script lang="ts">
    import { onMount } from 'svelte';
    import {
        renderState,
        status,
        pathMode,
        currentPath,
        coordinationMode,
        objectiveMode,
        hoveredDroneId,
        selectDrone,
        clearSelection,
        assignWaypoint,
        assignWaypointAll,
        assignPath,
        assignRouteAll,
        getDroneAt,
        selectDronesInRect,
        activeRoutes,
        explosions,
        targets,
        targetCounts,
        gameResult,
    } from '$lib/stores/simulation';

    let { width = 1000, height = 1000, worldWidth = 4000, worldHeight = 4000 } = $props();

    let canvas: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D | null = $state(null);

    // Box selection state
    let isDragging = $state(false);
    let dragStart = $state({ x: 0, y: 0 });
    let dragEnd = $state({ x: 0, y: 0 });
    const DRAG_THRESHOLD = 5; // Minimum pixels to consider a drag

    // Zoom and pan state
    let zoom = $state(0.25);
    let panX = $state(0);
    let panY = $state(0);
    let isPanning = $state(false);
    let panStart = $state({ x: 0, y: 0 });
    const MIN_ZOOM = 0.1;
    const MAX_ZOOM = 4.0;
    const ZOOM_SENSITIVITY = 0.001;

    // Convert screen coordinates to world coordinates
    function screenToWorld(screenX: number, screenY: number): { x: number; y: number } {
        return {
            x: (screenX - panX) / zoom,
            y: (screenY - panY) / zoom
        };
    }

    // Convert world coordinates to screen coordinates
    function worldToScreen(worldX: number, worldY: number): { x: number; y: number } {
        return {
            x: worldX * zoom + panX,
            y: worldY * zoom + panY
        };
    }

    const GRID_SPACING = 100;
    const GRID_OPACITY = 0.08;
    const DRONE_HEIGHT = 15;
    const DRONE_WIDTH = 5;
    const SELECTION_RADIUS = 20;
    const WAYPOINT_RADIUS = 8;
    const COLLISION_RADIUS = 1; // Half of COLLISION_DISTANCE (30) from wasm-lib

    // Shield AI color palette
    const LIME_ACCENT = '#9DFF20';
    const LIME_GLOW = 'rgba(157, 255, 32, 0.3)';
    const LIME_DIM = 'rgba(157, 255, 32, 0.5)';
    const OLIVE = '#345C00';
    const CANVAS_BG = '#0a0a0f';

    onMount(() => {
        ctx = canvas.getContext('2d')!;
        render();
    });

    // Re-render when state changes
    $effect(() => {
        if (ctx && $renderState) {
            render();
        }
    });

    // Also re-render when path or drag changes
    $effect(() => {
        if (ctx && ($currentPath || isDragging || $activeRoutes)) {
            render();
        }
    });

    // Animate explosions with continuous re-renders until they fade out
    $effect(() => {
        if (!ctx || $explosions.length === 0) return;
        render();
        let frame: number;
        function loop() {
            render();
            if ($explosions.length > 0) {
                frame = requestAnimationFrame(loop);
            }
        }
        frame = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(frame);
    });

    function render() {
        if (!ctx) return;
        ctx.clearRect(0, 0, width, height);

        // Apply zoom and pan transform
        ctx.save();
        ctx.translate(panX, panY);
        ctx.scale(zoom, zoom);

        // Draw grid
        drawGrid();

        // Draw targets
        for (const target of $targets) {
            if (!target.destroyed) drawTarget(target);
        }

        // Draw waypoint markers for ALL drones with targets
        drawWaypointMarkers();

        // Draw path/route waypoints (show in path mode OR when route mode has waypoints)
        if ($pathMode && $currentPath.length > 0) {
            drawPathWaypoints();
        } else if ($objectiveMode === 'route') {
            // Draw both group routes with distinct colors
            const routes = $activeRoutes;
            if (routes.a.length > 0) drawGroupRoute(routes.a, 'rgba(157, 255, 32, 0.5)', OLIVE);
            if (routes.b.length > 0) drawGroupRoute(routes.b, 'rgba(0, 180, 255, 0.5)', 'rgba(0, 90, 128, 1.0)');
        }

        // Draw full route paths for all drones in route mode
        for (const drone of $renderState) {
            if (drone.routePath && drone.routePath.length > 1) {
                drawRoutePath(drone.routePath, drone.color);
            }
        }

        // Draw spline paths for drones in route mode
        for (const drone of $renderState) {
            if (drone.splinePath && drone.splinePath.length > 1) {
                drawSplinePath(drone.splinePath);
            }
        }

        // Draw planning paths (NLGL trajectory) for formation followers
        for (const drone of $renderState) {
            if (drone.planningPath && drone.planningPath.length > 1) {
                drawPlanningPath(drone.planningPath, drone.approachMode);
            }
        }

        // Draw drones
        for (const drone of $renderState) {
            drawDrone(drone);
        }

        // Draw explosion effects (in world space, before restore)
        for (const explosion of $explosions) {
            drawExplosion(explosion);
        }

        // Restore transform before drawing UI elements (selection box is in screen space)
        ctx.restore();

        // Draw selection box if dragging (in screen space)
        if (isDragging) {
            drawSelectionBox();
        }

        // Draw zoom indicator
        if (zoom !== 1.0) {
            drawZoomIndicator();
        }

        // Draw target score
        drawTargetScore();

        // Draw win banner if game is over
        if ($gameResult) {
            drawWinBanner();
        }
    }

    function drawZoomIndicator() {
        if (!ctx) return;
        const zoomText = `${Math.round(zoom * 100)}%`;
        ctx.fillStyle = 'rgba(157, 255, 32, 0.8)';
        ctx.font = '12px "IBM Plex Mono", monospace';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'bottom';
        ctx.fillText(zoomText, width - 10, height - 10);
    }

    function drawTarget(target: { x: number; y: number; group: 'a' | 'b' }) {
        if (!ctx) return;
        const size = 30;
        const half = size / 2;
        const color = target.group === 'a' ? 'rgba(255, 60, 60, 0.8)' : 'rgba(60, 140, 255, 0.8)';
        const border = target.group === 'a' ? 'rgba(255, 120, 120, 1)' : 'rgba(120, 180, 255, 1)';

        ctx.fillStyle = color;
        ctx.fillRect(target.x - half, target.y - half, size, size);
        ctx.strokeStyle = border;
        ctx.lineWidth = 2;
        ctx.strokeRect(target.x - half, target.y - half, size, size);
    }

    function drawTargetScore() {
        if (!ctx) return;
        const counts = $targetCounts;
        ctx.font = 'bold 14px "IBM Plex Mono", monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';

        // Red targets remaining (group A)
        ctx.fillStyle = 'rgba(255, 60, 60, 0.9)';
        ctx.fillText(`RED targets: ${counts.a}/6`, 10, 10);

        // Blue targets remaining (group B)
        ctx.fillStyle = 'rgba(60, 140, 255, 0.9)';
        ctx.fillText(`BLU targets: ${counts.b}/6`, 10, 30);
    }

    function drawWinBanner() {
        if (!ctx) return;
        const result = $gameResult;
        let text = '';
        let color = '';
        if (result === 'a_wins') {
            text = 'RED TEAM WINS';
            color = 'rgba(255, 60, 60, 0.95)';
        } else if (result === 'b_wins') {
            text = 'BLUE TEAM WINS';
            color = 'rgba(60, 140, 255, 0.95)';
        } else if (result === 'draw') {
            text = 'DRAW';
            color = 'rgba(200, 200, 200, 0.95)';
        }

        // Darken background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(0, 0, width, height);

        // Banner
        const bannerH = 80;
        const bannerY = height / 2 - bannerH / 2;
        ctx.fillStyle = 'rgba(10, 10, 15, 0.9)';
        ctx.fillRect(0, bannerY, width, bannerH);

        // Text
        ctx.font = 'bold 36px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = color;
        ctx.fillText(text, width / 2, height / 2);
    }

    function drawGrid() {
        if (!ctx) return;
        ctx.strokeStyle = `rgba(157, 255, 32, ${GRID_OPACITY})`;
        ctx.lineWidth = 1;

        for (let x = GRID_SPACING; x < worldWidth; x += GRID_SPACING) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, worldHeight);
            ctx.stroke();
        }

        for (let y = GRID_SPACING; y < worldHeight; y += GRID_SPACING) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(worldWidth, y);
            ctx.stroke();
        }
    }

    function drawRoutePath(points: { x: number; y: number }[], color: { r: number; g: number; b: number }) {
        if (!ctx || points.length < 2) return;

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

    function drawSplinePath(points: { x: number; y: number }[]) {
        if (!ctx || points.length < 2) return;

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

    function drawPlanningPath(points: { x: number; y: number }[], approachMode: string) {
        if (!ctx || points.length < 2) return;

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

    function drawWaypointMarkers() {
        if (!ctx) return;

        // Collect unique waypoints and their associated drones
        const waypointMap = new Map<string, { x: number; y: number; drones: typeof $renderState }>();

        for (const drone of $renderState) {
            if (drone.target) {
                const key = `${drone.target.x.toFixed(0)},${drone.target.y.toFixed(0)}`;
                if (!waypointMap.has(key)) {
                    waypointMap.set(key, { x: drone.target.x, y: drone.target.y, drones: [] });
                }
                waypointMap.get(key)!.drones.push(drone);
            }
        }

        // Draw lines from each drone to its waypoint
        for (const drone of $renderState) {
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
            const { x, y, drones } = waypoint;
            const hasSelectedDrone = drones.some(d => d.selected);

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
            if (drones.length > 1) {
                ctx.fillStyle = '#0a0a0f';
                ctx.font = 'bold 10px "IBM Plex Mono", monospace';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(String(drones.length), x, y);
            }
        }
    }

    function drawPathWaypoints() {
        if (!ctx) return;
        const isRoute = $objectiveMode === 'route';
        const isPlacing = $pathMode;

        // Use lime green for paths/routes
        const pointColor = isRoute && !isPlacing ? 'rgba(157, 255, 32, 0.5)' : LIME_ACCENT;
        const lineColor = isRoute && !isPlacing ? OLIVE : LIME_ACCENT;

        ctx.lineWidth = 2;

        // Connect with lines first (behind dots)
        if ($currentPath.length > 1) {
            ctx.strokeStyle = lineColor;
            ctx.setLineDash(isRoute ? [5, 5] : []);
            ctx.beginPath();
            ctx.moveTo($currentPath[0].x, $currentPath[0].y);
            for (let i = 1; i < $currentPath.length; i++) {
                ctx.lineTo($currentPath[i].x, $currentPath[i].y);
            }
            // If route mode, connect back to start
            if (isRoute && $currentPath.length > 2) {
                ctx.lineTo($currentPath[0].x, $currentPath[0].y);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Draw waypoint dots with numbers
        for (let i = 0; i < $currentPath.length; i++) {
            const { x, y } = $currentPath[i];

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

    function drawGroupRoute(waypoints: { x: number; y: number }[], pointColor: string, lineColor: string) {
        if (!ctx || waypoints.length === 0) return;

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

    function drawExplosion(explosion: { x: number; y: number; radius: number; time: number }) {
        if (!ctx) return;
        const elapsed = Date.now() - explosion.time;
        const duration = 800;
        if (elapsed >= duration) return;

        // Fade out over duration
        const progress = elapsed / duration;
        const alpha = 1.0 - progress;

        // Expanding ring effect
        const currentRadius = explosion.radius * (0.5 + 0.5 * progress);

        ctx.strokeStyle = `rgba(255, 40, 40, ${alpha})`;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.arc(explosion.x, explosion.y, currentRadius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Inner glow fill
        ctx.fillStyle = `rgba(255, 60, 20, ${alpha * 0.15})`;
        ctx.beginPath();
        ctx.arc(explosion.x, explosion.y, currentRadius, 0, Math.PI * 2);
        ctx.fill();
    }

    function drawDrone(drone: typeof $renderState[0]) {
        if (!ctx) return;
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

    function drawSelectionBox() {
        if (!ctx) return;

        const x = Math.min(dragStart.x, dragEnd.x);
        const y = Math.min(dragStart.y, dragEnd.y);
        const w = Math.abs(dragEnd.x - dragStart.x);
        const h = Math.abs(dragEnd.y - dragStart.y);

        // Fill
        ctx.fillStyle = 'rgba(157, 255, 32, 0.1)';
        ctx.fillRect(x, y, w, h);

        // Border
        ctx.strokeStyle = LIME_ACCENT;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
    }

    function handleMouseDown(e: MouseEvent) {
        const rect = canvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;

        // Middle mouse button or Ctrl+Left for panning
        if (e.button === 1 || (e.button === 0 && e.ctrlKey)) {
            isPanning = true;
            panStart = { x: screenX - panX, y: screenY - panY };
            e.preventDefault();
            return;
        }

        if (e.button !== 0) return; // Left click only for other actions
        if ($pathMode) return; // No box selection in path mode

        dragStart = { x: screenX, y: screenY };
        dragEnd = { x: screenX, y: screenY };
    }

    function handleMouseMove(e: MouseEvent) {
        const rect = canvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;

        // Handle panning
        if (isPanning) {
            panX = screenX - panStart.x;
            panY = screenY - panStart.y;
            render();
            return;
        }

        // Update hover state (convert to world coordinates)
        const world = screenToWorld(screenX, screenY);
        const droneId = getDroneAt(world.x, world.y);
        hoveredDroneId.set(droneId ?? null);

        // Handle dragging for box selection
        if (e.buttons === 1 && !$pathMode && !e.ctrlKey) { // Left mouse button held, not panning
            const dx = Math.abs(screenX - dragStart.x);
            const dy = Math.abs(screenY - dragStart.y);

            if (dx > DRAG_THRESHOLD || dy > DRAG_THRESHOLD) {
                isDragging = true;
            }

            if (isDragging) {
                dragEnd = { x: screenX, y: screenY };
                render();
            }
        }
    }

    function handleMouseUp(e: MouseEvent) {
        // End panning
        if (isPanning) {
            isPanning = false;
            return;
        }

        if (e.button !== 0) return;

        const rect = canvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;
        const world = screenToWorld(screenX, screenY);

        if (isDragging) {
            // Box selection complete - convert to world coordinates
            const startWorld = screenToWorld(dragStart.x, dragStart.y);
            const endWorld = screenToWorld(dragEnd.x, dragEnd.y);
            const minX = Math.min(startWorld.x, endWorld.x);
            const maxX = Math.max(startWorld.x, endWorld.x);
            const minY = Math.min(startWorld.y, endWorld.y);
            const maxY = Math.max(startWorld.y, endWorld.y);

            const multi = e.metaKey || e.shiftKey; // Note: ctrlKey is used for panning
            selectDronesInRect(minX, minY, maxX, maxY, multi);

            isDragging = false;
            render();
        } else if (!$pathMode && !e.ctrlKey) {
            // Regular click
            const multi = e.metaKey || e.shiftKey;
            const droneId = getDroneAt(world.x, world.y);

            if (droneId !== undefined) {
                selectDrone(droneId, multi);
            } else if (!multi) {
                clearSelection();
            }
        }
    }

    function handleWheel(e: WheelEvent) {
        e.preventDefault();

        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Calculate zoom change
        const zoomDelta = -e.deltaY * ZOOM_SENSITIVITY;
        const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * (1 + zoomDelta)));

        // Zoom toward mouse position
        const worldX = (mouseX - panX) / zoom;
        const worldY = (mouseY - panY) / zoom;

        zoom = newZoom;

        // Adjust pan to keep mouse position fixed
        panX = mouseX - worldX * zoom;
        panY = mouseY - worldY * zoom;

        render();
    }

    function handleClick(e: MouseEvent) {
        // Only handle clicks in path mode (box selection handles other clicks via mouseup)
        if (!$pathMode) return;
        if (e.ctrlKey) return; // Don't add waypoints while panning

        const rect = canvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;
        const world = screenToWorld(screenX, screenY);

        currentPath.update((path) => [...path, { x: world.x, y: world.y }]);
    }

    function handleRightClick(e: MouseEvent) {
        e.preventDefault();

        const rect = canvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;
        const world = screenToWorld(screenX, screenY);

        if ($pathMode) {
            if ($currentPath.length > 0) {
                if ($coordinationMode === 'swarm' && $objectiveMode === 'route') {
                    assignRouteAll($currentPath);
                } else {
                    assignPath($currentPath);
                }
            }
            return;
        }

        // Handle waypoint assignment (use world coordinates)
        if ($coordinationMode === 'swarm') {
            assignWaypointAll(world.x, world.y);
        } else if ($status.selectedCount > 0) {
            assignWaypoint(world.x, world.y);
        }
    }

    function resetView() {
        zoom = 1.0;
        panX = 0;
        panY = 0;
        render();
    }
</script>

<div class="canvas-wrapper">
    <canvas
        bind:this={canvas}
        {width}
        {height}
        onmousedown={handleMouseDown}
        onmousemove={handleMouseMove}
        onmouseup={handleMouseUp}
        onclick={handleClick}
        oncontextmenu={handleRightClick}
        onwheel={handleWheel}
    ></canvas>
    {#if zoom !== 1.0}
        <button class="reset-view-btn" onclick={resetView}>Reset View</button>
    {/if}
</div>

<style>
    .canvas-wrapper {
        width: 100%;
        height: 100%;
        overflow: hidden;
        background: #0a0a0f;
        position: relative;
    }

    canvas {
        background: #0a0a0f;
        cursor: crosshair;
        display: block;
    }

    .reset-view-btn {
        position: absolute;
        bottom: 10px;
        left: 10px;
        padding: 6px 12px;
        background: rgba(52, 92, 0, 0.8);
        color: #9DFF20;
        border: 1px solid #9DFF20;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        cursor: pointer;
        transition: background 0.2s;
    }

    .reset-view-btn:hover {
        background: rgba(157, 255, 32, 0.3);
    }
</style>
