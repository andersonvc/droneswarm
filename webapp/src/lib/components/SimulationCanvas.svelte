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
    } from '$lib/stores/simulation';

    let { width = 1000, height = 1000 } = $props();

    let canvas: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D | null = $state(null);

    // Box selection state
    let isDragging = $state(false);
    let dragStart = $state({ x: 0, y: 0 });
    let dragEnd = $state({ x: 0, y: 0 });
    const DRAG_THRESHOLD = 5; // Minimum pixels to consider a drag

    const GRID_SPACING = 100;
    const GRID_OPACITY = 0.08;
    const DRONE_HEIGHT = 30;
    const DRONE_WIDTH = 10;
    const SELECTION_RADIUS = 20;
    const WAYPOINT_RADIUS = 8;

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
        if (ctx && ($currentPath || isDragging)) {
            render();
        }
    });

    function render() {
        if (!ctx) return;
        ctx.clearRect(0, 0, width, height);

        // Draw grid
        drawGrid();

        // Draw waypoint markers for ALL drones with targets
        drawWaypointMarkers();

        // Draw path/route waypoints (show in path mode OR when route mode has waypoints)
        if ($currentPath.length > 0 && ($pathMode || $objectiveMode === 'route')) {
            drawPathWaypoints();
        }

        // Draw drones
        for (const drone of $renderState) {
            drawDrone(drone);
        }

        // Draw selection box if dragging
        if (isDragging) {
            drawSelectionBox();
        }
    }

    function drawGrid() {
        if (!ctx) return;
        ctx.strokeStyle = `rgba(157, 255, 32, ${GRID_OPACITY})`;
        ctx.lineWidth = 1;

        for (let x = GRID_SPACING; x < width; x += GRID_SPACING) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }

        for (let y = GRID_SPACING; y < height; y += GRID_SPACING) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
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
        if (e.button !== 0) return; // Left click only
        if ($pathMode) return; // No box selection in path mode

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        dragStart = { x, y };
        dragEnd = { x, y };
    }

    function handleMouseMove(e: MouseEvent) {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Update hover state
        const droneId = getDroneAt(x, y);
        hoveredDroneId.set(droneId ?? null);

        // Handle dragging
        if (e.buttons === 1 && !$pathMode) { // Left mouse button held
            const dx = Math.abs(x - dragStart.x);
            const dy = Math.abs(y - dragStart.y);

            if (dx > DRAG_THRESHOLD || dy > DRAG_THRESHOLD) {
                isDragging = true;
            }

            if (isDragging) {
                dragEnd = { x, y };
                render();
            }
        }
    }

    function handleMouseUp(e: MouseEvent) {
        if (e.button !== 0) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (isDragging) {
            // Box selection complete
            const minX = Math.min(dragStart.x, dragEnd.x);
            const maxX = Math.max(dragStart.x, dragEnd.x);
            const minY = Math.min(dragStart.y, dragEnd.y);
            const maxY = Math.max(dragStart.y, dragEnd.y);

            const multi = e.ctrlKey || e.metaKey || e.shiftKey;
            selectDronesInRect(minX, minY, maxX, maxY, multi);

            isDragging = false;
            render();
        } else if (!$pathMode) {
            // Regular click
            const multi = e.ctrlKey || e.metaKey || e.shiftKey;
            const droneId = getDroneAt(x, y);

            if (droneId !== undefined) {
                selectDrone(droneId, multi);
            } else if (!multi) {
                clearSelection();
            }
        }
    }

    function handleClick(e: MouseEvent) {
        // Only handle clicks in path mode (box selection handles other clicks via mouseup)
        if (!$pathMode) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        currentPath.update((path) => [...path, { x, y }]);
    }

    function handleRightClick(e: MouseEvent) {
        e.preventDefault();

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

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

        // Handle waypoint assignment
        if ($coordinationMode === 'swarm') {
            assignWaypointAll(x, y);
        } else if ($status.selectedCount > 0) {
            assignWaypoint(x, y);
        }
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
    ></canvas>
</div>

<style>
    .canvas-wrapper {
        width: 100%;
        height: 100%;
        overflow: hidden;
        background: #0a0a0f;
    }

    canvas {
        background: #0a0a0f;
        cursor: crosshair;
        display: block;
    }
</style>
