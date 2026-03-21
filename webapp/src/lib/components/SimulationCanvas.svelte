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
        assignWaypoint,
        assignWaypointAll,
        assignPath,
        assignRouteAll,
        getDroneAt,
        activeRoutes,
        explosions,
        targets,
        targetCounts,
        droneCounts,
        gameResult,
    } from '$lib/stores/simulation';

    // Renderers
    import { drawGrid } from './renderers/grid';
    import { drawDrone, drawWaypointMarkers } from './renderers/drones';
    import { drawRoutePath, drawSplinePath, drawPlanningPath, drawPathWaypoints, drawGroupRoute } from './renderers/paths';
    import { drawTarget, drawTargetScore } from './renderers/targets';
    import { drawExplosion, drawSelectionBox } from './renderers/effects';
    import { drawZoomIndicator, drawWinBanner } from './renderers/hud';

    // Input helpers
    import { screenToWorld, computeWheelZoom } from './input/camera';
    import { DRAG_THRESHOLD, handleBoxSelection, handleClickSelection } from './input/selection';
    import { handlePathClick } from './input/path-placement';

    let { width = 1000, height = 1000, worldWidth = 4000, worldHeight = 4000, hideViewControls = false } = $props();

    let canvas: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D | null = $state(null);

    // Box selection state
    let isDragging = $state(false);
    let dragStart = $state({ x: 0, y: 0 });
    let dragEnd = $state({ x: 0, y: 0 });

    // Zoom and pan state
    let zoom = $state(0.25);
    let panX = $state(0);
    let panY = $state(0);
    let isPanning = $state(false);
    let panStart = $state({ x: 0, y: 0 });

    // Touch interaction state
    let lastTouchDist = $state(0);
    let touchStartTime = $state(0);
    let touchStartPos = $state({ x: 0, y: 0 });
    let longPressTimer: ReturnType<typeof setTimeout> | null = null;
    let isTouchPanning = $state(false);
    let touchPanStart = $state({ x: 0, y: 0 });

    function applyDpiScaling() {
        if (!canvas || !ctx) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
    }

    onMount(() => {
        ctx = canvas.getContext('2d')!;
        applyDpiScaling();
        // Defer fitToView so renderState is populated
        requestAnimationFrame(() => fitToView());
    });

    // Re-apply DPI scaling when canvas dimensions change
    $effect(() => {
        // Access width and height to track them
        void width;
        void height;
        if (ctx) {
            applyDpiScaling();
            render();
        }
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

        drawGrid(ctx, worldWidth, worldHeight);

        for (const target of $targets) {
            if (!target.destroyed) drawTarget(ctx, target);
        }

        for (const drone of $renderState) {
            if (drone.planningPath && drone.planningPath.length > 1) {
                drawPlanningPath(ctx, drone.planningPath, drone.approachMode);
            }
        }

        for (const drone of $renderState) {
            drawDrone(ctx, drone);
        }

        for (const explosion of $explosions) {
            drawExplosion(ctx, explosion);
        }

        ctx.restore();

        if (isDragging) {
            drawSelectionBox(ctx, dragStart, dragEnd);
        }

        if (zoom !== 1.0) {
            if (!hideViewControls) drawZoomIndicator(ctx, zoom, width, height);
        }

        drawTargetScore(ctx, $targetCounts, $droneCounts);

        if ($gameResult) {
            drawWinBanner(ctx, $gameResult, width, height);
        }
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

        if (e.button !== 0) return;
        if ($pathMode) return;

        dragStart = { x: screenX, y: screenY };
        dragEnd = { x: screenX, y: screenY };
    }

    function handleMouseMove(e: MouseEvent) {
        const rect = canvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;

        if (isPanning) {
            panX = screenX - panStart.x;
            panY = screenY - panStart.y;
            render();
            return;
        }

        const world = screenToWorld(screenX, screenY, panX, panY, zoom);
        const droneId = getDroneAt(world.x, world.y);
        hoveredDroneId.set(droneId ?? null);

        if (e.buttons === 1 && !$pathMode && !e.ctrlKey) {
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
        if (isPanning) {
            isPanning = false;
            return;
        }

        if (e.button !== 0) return;

        const rect = canvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;
        const world = screenToWorld(screenX, screenY, panX, panY, zoom);

        if (isDragging) {
            const multi = e.metaKey || e.shiftKey;
            handleBoxSelection({ isDragging, dragStart, dragEnd }, panX, panY, zoom, multi);
            isDragging = false;
            render();
        } else if (!$pathMode && !e.ctrlKey) {
            const multi = e.metaKey || e.shiftKey;
            handleClickSelection(world.x, world.y, multi);
        }
    }

    function handleWheel(e: WheelEvent) {
        e.preventDefault();

        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const result = computeWheelZoom(e.deltaY, mouseX, mouseY, zoom, panX, panY);
        zoom = result.zoom;
        panX = result.panX;
        panY = result.panY;

        render();
    }

    function handleClick(e: MouseEvent) {
        if (!$pathMode) return;
        if (e.ctrlKey) return;

        const rect = canvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;
        const world = handlePathClick(screenX, screenY, panX, panY, zoom);

        currentPath.update((path) => [...path, { x: world.x, y: world.y }]);
    }

    function handleRightClick(e: MouseEvent) {
        e.preventDefault();

        const rect = canvas.getBoundingClientRect();
        const screenX = e.clientX - rect.left;
        const screenY = e.clientY - rect.top;
        const world = screenToWorld(screenX, screenY, panX, panY, zoom);

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

        if ($coordinationMode === 'swarm') {
            assignWaypointAll(world.x, world.y);
        } else if ($status.selectedCount > 0) {
            assignWaypoint(world.x, world.y);
        }
    }

    function getTouchDistance(t1: Touch, t2: Touch): number {
        const dx = t1.clientX - t2.clientX;
        const dy = t1.clientY - t2.clientY;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function handleTouchStart(e: TouchEvent) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();

        if (e.touches.length === 1) {
            const touch = e.touches[0];
            const screenX = touch.clientX - rect.left;
            const screenY = touch.clientY - rect.top;

            touchStartTime = Date.now();
            touchStartPos = { x: screenX, y: screenY };
            isTouchPanning = false;
            touchPanStart = { x: screenX - panX, y: screenY - panY };

            // Start long-press timer (right-click equivalent)
            if (longPressTimer) clearTimeout(longPressTimer);
            longPressTimer = setTimeout(() => {
                const world = screenToWorld(screenX, screenY, panX, panY, zoom);
                if ($coordinationMode === 'swarm') {
                    assignWaypointAll(world.x, world.y);
                } else if ($status.selectedCount > 0) {
                    assignWaypoint(world.x, world.y);
                }
                longPressTimer = null;
            }, 500);
        } else if (e.touches.length === 2) {
            // Cancel long press and single-finger actions on two-finger gesture
            if (longPressTimer) {
                clearTimeout(longPressTimer);
                longPressTimer = null;
            }
            isTouchPanning = false;
            lastTouchDist = getTouchDistance(e.touches[0], e.touches[1]);
        }
    }

    function handleTouchMove(e: TouchEvent) {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();

        if (e.touches.length === 1) {
            const touch = e.touches[0];
            const screenX = touch.clientX - rect.left;
            const screenY = touch.clientY - rect.top;

            const dx = Math.abs(screenX - touchStartPos.x);
            const dy = Math.abs(screenY - touchStartPos.y);

            if (dx > 10 || dy > 10) {
                // Movement exceeds threshold - this is a pan, cancel long press
                if (longPressTimer) {
                    clearTimeout(longPressTimer);
                    longPressTimer = null;
                }
                isTouchPanning = true;
            }

            if (isTouchPanning) {
                panX = screenX - touchPanStart.x;
                panY = screenY - touchPanStart.y;
                render();
            }
        } else if (e.touches.length === 2) {
            // Pinch to zoom
            const newDist = getTouchDistance(e.touches[0], e.touches[1]);
            if (lastTouchDist > 0) {
                const scale = newDist / lastTouchDist;

                // Calculate midpoint for zoom center
                const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left;
                const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2 - rect.top;

                const worldX = (midX - panX) / zoom;
                const worldY = (midY - panY) / zoom;

                const newZoom = Math.max(0.1, Math.min(4.0, zoom * scale));
                panX = midX - worldX * newZoom;
                panY = midY - worldY * newZoom;
                zoom = newZoom;

                render();
            }
            lastTouchDist = newDist;
        }
    }

    function handleTouchEnd(e: TouchEvent) {
        e.preventDefault();

        if (longPressTimer) {
            clearTimeout(longPressTimer);
            longPressTimer = null;
        }

        // Detect tap: short duration, minimal movement
        if (e.changedTouches.length === 1 && e.touches.length === 0) {
            const touch = e.changedTouches[0];
            const rect = canvas.getBoundingClientRect();
            const screenX = touch.clientX - rect.left;
            const screenY = touch.clientY - rect.top;

            const dx = Math.abs(screenX - touchStartPos.x);
            const dy = Math.abs(screenY - touchStartPos.y);
            const elapsed = Date.now() - touchStartTime;

            if (dx < 10 && dy < 10 && elapsed < 300) {
                // Tap - select drone at location
                const world = screenToWorld(screenX, screenY, panX, panY, zoom);
                handleClickSelection(world.x, world.y, false);
            }
        }

        // Reset pinch state
        lastTouchDist = 0;
        isTouchPanning = false;
    }

    function resetView() {
        zoom = 1.0;
        panX = 0;
        panY = 0;
        render();
    }

    /** Auto-fit zoom/pan so all targets and drones are visible with padding. */
    function fitToView() {
        const points: { x: number; y: number }[] = [];

        for (const t of $targets) {
            if (!t.destroyed) points.push(t);
        }
        for (const d of $renderState) {
            points.push({ x: d.x, y: d.y });
        }

        if (points.length === 0) return;

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const p of points) {
            if (p.x < minX) minX = p.x;
            if (p.y < minY) minY = p.y;
            if (p.x > maxX) maxX = p.x;
            if (p.y > maxY) maxY = p.y;
        }

        // Add 10% padding
        const padX = Math.max((maxX - minX) * 0.1, 100);
        const padY = Math.max((maxY - minY) * 0.1, 100);
        minX -= padX;
        minY -= padY;
        maxX += padX;
        maxY += padY;

        const boxW = maxX - minX;
        const boxH = maxY - minY;

        if (boxW <= 0 || boxH <= 0) return;

        // Fit to canvas, using the smaller scale so everything fits
        const scaleX = width / boxW;
        const scaleY = height / boxH;
        zoom = Math.min(scaleX, scaleY);

        // Center the bounding box in the canvas
        panX = (width - boxW * zoom) / 2 - minX * zoom;
        panY = (height - boxH * zoom) / 2 - minY * zoom;

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
        ontouchstart={handleTouchStart}
        ontouchmove={handleTouchMove}
        ontouchend={handleTouchEnd}
    ></canvas>
    {#if !hideViewControls}
        <div class="view-buttons">
            <button class="view-btn" onclick={fitToView}>Fit View</button>
            {#if zoom !== 1.0}
                <button class="view-btn" onclick={resetView}>Reset View</button>
            {/if}
        </div>
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

    .view-buttons {
        position: absolute;
        bottom: 10px;
        left: 10px;
        display: flex;
        gap: 6px;
    }

    .view-btn {
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

    .view-btn:hover {
        background: rgba(157, 255, 32, 0.3);
    }
</style>
