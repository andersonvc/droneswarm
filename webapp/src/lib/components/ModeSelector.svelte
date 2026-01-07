<script lang="ts">
    import {
        coordinationMode,
        objectiveMode,
        currentPath,
        pathMode,
        type CoordinationMode,
        type ObjectiveMode
    } from '$lib/stores/simulation';
    import RouteCreatorModal from './RouteCreatorModal.svelte';

    let routeModalOpen = $state(false);

    function setCoordination(mode: CoordinationMode) {
        coordinationMode.set(mode);
    }

    function setObjective(mode: ObjectiveMode) {
        objectiveMode.set(mode);
        if (mode === 'route') {
            routeModalOpen = true;
        } else if (mode === 'waypoint') {
            // Clear route when switching to waypoint mode
            currentPath.set([]);
            pathMode.set(false);
        }
    }
</script>

<div class="mode-selector">
    <div class="mode-group">
        <span class="mode-label">Coordination:</span>
        <div class="toggle-group">
            <button
                class="toggle-btn"
                class:active={$coordinationMode === 'individual'}
                onclick={() => setCoordination('individual')}
            >
                Individual
            </button>
            <button
                class="toggle-btn"
                class:active={$coordinationMode === 'swarm'}
                onclick={() => setCoordination('swarm')}
            >
                Swarm
            </button>
        </div>
    </div>

    <div class="mode-group">
        <span class="mode-label">Objective:</span>
        <div class="toggle-group">
            <button
                class="toggle-btn"
                class:active={$objectiveMode === 'waypoint'}
                onclick={() => setObjective('waypoint')}
            >
                Waypoint
            </button>
            <button
                class="toggle-btn"
                class:active={$objectiveMode === 'route'}
                onclick={() => setObjective('route')}
            >
                Route
            </button>
        </div>
    </div>
</div>

<RouteCreatorModal bind:open={routeModalOpen} />

<style>
    .mode-selector {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .mode-group {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .mode-label {
        color: #6b7280;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .toggle-group {
        display: flex;
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid #2a2a30;
    }

    .toggle-btn {
        flex: 1;
        padding: 6px 8px;
        border: none;
        background: #1a1a1f;
        color: #6b7280;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s;
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    .toggle-btn:hover:not(.active) {
        background: #252530;
        color: #9ca3af;
    }

    .toggle-btn.active {
        background: #345C00;
        color: #9DFF20;
    }

    .toggle-btn:not(:last-child) {
        border-right: 1px solid #2a2a30;
    }
</style>
