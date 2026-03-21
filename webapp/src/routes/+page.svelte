<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { page } from '$app/stores';
    import {
        isInitialized,
        isRunning,
        pathMode,
        currentPath,
        initSimulation,
        tickSimulation,
        resetSimulation,
        restoreActiveRoute,
        assignPath,
        clearSelection,
        detonateRandomDrone,
        launchAttack,
        gameResult,
        groupAStrategy,
        groupBStrategy,
        setGroupStrategy,
    } from '$lib/stores/simulation';
    import SimulationCanvas from '$lib/components/SimulationCanvas.svelte';
    import ControlPanel from '$lib/components/ControlPanel.svelte';
    import StatusBar from '$lib/components/StatusBar.svelte';
    import DroneTooltip from '$lib/components/DroneTooltip.svelte';
    import PathModeIndicator from '$lib/components/PathModeIndicator.svelte';
    import ConfigModal from '$lib/components/ConfigModal.svelte';
    import ConfigSidePanel from '$lib/components/ConfigSidePanel.svelte';
    import StrategySelector from '$lib/components/StrategySelector.svelte';

    // Demo mode: ?demo in URL hides controls, shows only canvas + score + strategies
    let demoMode = $derived($page.url.searchParams.has('demo'));

    let animationFrame: number;
    let lastTime = 0;
    let tooltipX = $state(0);
    let tooltipY = $state(0);
    let configPanelOpen = $state(false);
    let canvasWidth = $state(1000);
    let canvasHeight = $state(1000);
    let isMobile = $state(false);
    let mobileMenuOpen = $state(false);
    let restartTimer: ReturnType<typeof setTimeout> | null = null;

    function updateCanvasSize() {
        if (typeof window === 'undefined') return;

        isMobile = window.innerWidth < 768;
        const sidebarWidth = (demoMode || isMobile) ? 0 : 220;
        canvasWidth = window.innerWidth - sidebarWidth;
        canvasHeight = window.innerHeight;
    }

    let unsubGameResult: (() => void) | null = null;

    onMount(async () => {
        updateCanvasSize();
        window.addEventListener('resize', updateCanvasSize);

        await initSimulation();
        handleStart();

        // Auto-restart 3s after a win, preserving strategies
        unsubGameResult = gameResult.subscribe((result) => {
            if (result !== null && restartTimer === null) {
                restartTimer = setTimeout(async () => {
                    const savedA = $groupAStrategy;
                    const savedB = $groupBStrategy;
                    await handleReset();
                    if (savedA !== 'none') setGroupStrategy(0, savedA);
                    if (savedB !== 'none') setGroupStrategy(1, savedB);
                    restartTimer = null;
                }, 3000);
            }
        });
    });

    onDestroy(() => {
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
        }
        if (restartTimer) {
            clearTimeout(restartTimer);
        }
        unsubGameResult?.();
        if (typeof window !== 'undefined') {
            window.removeEventListener('resize', updateCanvasSize);
        }
    });

    function gameLoop(timestamp: number) {
        if (!$isRunning) return;

        const dt = lastTime ? (timestamp - lastTime) / 1000 : 0.016;
        lastTime = timestamp;

        tickSimulation(dt);
        animationFrame = requestAnimationFrame(gameLoop);
    }

    function handleStart() {
        isRunning.set(true);
        lastTime = 0;
        animationFrame = requestAnimationFrame(gameLoop);
    }

    function handlePause() {
        isRunning.set(false);
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
        }
    }

    async function handleReset() {
        handlePause();
        resetSimulation();
        await initSimulation();
        restoreActiveRoute();
        handleStart();
    }

    function handleKeydown(e: KeyboardEvent) {
        // Don't handle if typing in an input
        if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
            return;
        }

        if (e.key === 'p' || e.key === 'P') {
            pathMode.update((v) => !v);
            if (!$pathMode) currentPath.set([]);
        } else if (e.key === 'Escape') {
            if ($pathMode) {
                pathMode.set(false);
                currentPath.set([]);
            } else {
                clearSelection();
            }
        } else if (e.key === 'Enter' && $pathMode && $currentPath.length > 0) {
            assignPath($currentPath);
        } else if (e.key === 'd' || e.key === 'D') {
            detonateRandomDrone();
        } else if (e.key === 'a' || e.key === 'A') {
            launchAttack();
        } else if (e.key === ' ') {
            e.preventDefault();
            if ($isRunning) {
                handlePause();
            } else {
                handleStart();
            }
        }
    }

    function handleMouseMove(e: MouseEvent) {
        tooltipX = e.clientX;
        tooltipY = e.clientY;
    }
</script>

<svelte:window onkeydown={handleKeydown} />

<main onmousemove={handleMouseMove}>
    {#if $isInitialized}
        {#if demoMode}
            <div class="simulation-container">
                <SimulationCanvas width={canvasWidth} height={canvasHeight} worldWidth={4000} worldHeight={4000} hideViewControls={true} />
            </div>
            <div class="demo-overlay">
                <div class="demo-strategies">
                    <StrategySelector />
                </div>
            </div>
        {:else if isMobile}
            <div class="simulation-container">
                <SimulationCanvas width={canvasWidth} height={canvasHeight} worldWidth={4000} worldHeight={4000} />
            </div>
            <button class="mobile-menu-btn" onclick={() => mobileMenuOpen = !mobileMenuOpen}>
                &#9776;
            </button>
            {#if mobileMenuOpen}
                <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
                <div class="mobile-overlay" role="presentation" onclick={() => mobileMenuOpen = false}></div>
                <aside class="mobile-sidebar">
                    <ControlPanel
                        onStart={handleStart}
                        onPause={handlePause}
                        onReset={handleReset}
                    />
                    <button class="config-btn" onclick={() => configPanelOpen = true}>
                        Edit Configs
                    </button>
                    <div class="strategy-section">
                        <StrategySelector />
                    </div>
                    <div class="sidebar-spacer"></div>
                    <StatusBar />
                </aside>
            {/if}
            <PathModeIndicator />
            <ConfigModal />
            <ConfigSidePanel isOpen={configPanelOpen} onClose={() => configPanelOpen = false} />
        {:else}
            <aside class="controls-sidebar">
                <ControlPanel
                    onStart={handleStart}
                    onPause={handlePause}
                    onReset={handleReset}
                />
                <button class="config-btn" onclick={() => configPanelOpen = true}>
                    Edit Configs
                </button>
                <div class="strategy-section">
                    <StrategySelector />
                </div>
                <ConfigModal />
                <div class="sidebar-spacer"></div>
                <StatusBar />
            </aside>
            <div class="simulation-container">
                <SimulationCanvas width={canvasWidth} height={canvasHeight} worldWidth={4000} worldHeight={4000} />
            </div>
            <DroneTooltip x={tooltipX} y={tooltipY} />
            <PathModeIndicator />
            <ConfigSidePanel isOpen={configPanelOpen} onClose={() => configPanelOpen = false} />
        {/if}
    {:else}
        <div class="loading">
            <div class="loading-text">Initializing Swarm</div>
            <div class="loading-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    {/if}
</main>

<style>
    :global(body) {
        margin: 0;
        padding: 0;
        overflow: hidden;
    }

    main {
        width: 100vw;
        height: 100vh;
        display: flex;
        flex-direction: row;
        background: #0a0a0f;
    }

    .simulation-container {
        flex: 1;
        display: flex;
        align-items: stretch;
        justify-content: stretch;
        overflow: hidden;
    }

    .controls-sidebar {
        width: 220px;
        height: 100%;
        display: flex;
        flex-direction: column;
        background: #101014;
        border-right: 1px solid #1f1f25;
        padding: 16px;
        box-sizing: border-box;
    }

    .sidebar-spacer {
        flex: 1;
    }

    .loading {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        gap: 16px;
    }

    .loading-text {
        color: #9DFF20;
        font-size: 20px;
        font-family: 'DM Sans', system-ui, sans-serif;
        font-weight: 500;
        letter-spacing: 0.05em;
    }

    .loading-dots {
        display: flex;
        gap: 8px;
    }

    .loading-dots span {
        width: 8px;
        height: 8px;
        background: #9DFF20;
        border-radius: 50%;
        animation: pulse 1.4s ease-in-out infinite;
    }

    .loading-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }

    .loading-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }

    @keyframes pulse {
        0%, 80%, 100% {
            opacity: 0.3;
            transform: scale(0.8);
        }
        40% {
            opacity: 1;
            transform: scale(1);
        }
    }

    .config-btn {
        background: #1a1a1f;
        border: 1px solid #2a2a30;
        color: #9ca3af;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 13px;
        font-family: 'DM Sans', system-ui, sans-serif;
        cursor: pointer;
        transition: all 0.15s ease;
        margin-top: 12px;
        width: 100%;
    }

    .config-btn:hover {
        background: #252530;
        color: #fff;
        border-color: #9DFF20;
    }

    .strategy-section {
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid #1f1f25;
    }

    .demo-overlay {
        position: fixed;
        top: 0;
        right: 0;
        pointer-events: none;
        z-index: 10;
        padding: 16px;
    }

    .demo-strategies {
        pointer-events: auto;
        background: rgba(10, 10, 15, 0.8);
        border: 1px solid #1f1f25;
        border-radius: 8px;
        padding: 12px 16px;
    }

    .mobile-menu-btn {
        position: fixed;
        top: 12px;
        right: 12px;
        z-index: 50;
        width: 44px;
        height: 44px;
        background: rgba(10, 10, 15, 0.85);
        border: 1px solid #2a2a30;
        border-radius: 8px;
        color: #9DFF20;
        font-size: 22px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .mobile-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.5);
        z-index: 90;
    }

    .mobile-sidebar {
        position: fixed;
        top: 0;
        right: 0;
        width: 260px;
        height: 100%;
        background: #101014;
        border-left: 1px solid #1f1f25;
        padding: 16px;
        box-sizing: border-box;
        z-index: 100;
        display: flex;
        flex-direction: column;
        overflow-y: auto;
    }
</style>
