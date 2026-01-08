<script lang="ts">
    import { pathMode, currentPath, coordinationMode, objectiveMode } from '$lib/stores/simulation';

    let modeLabel = $derived(
        $objectiveMode === 'route' ? 'ROUTE MODE' : 'PATH MODE'
    );

    let hint = $derived(
        $coordinationMode === 'swarm' && $objectiveMode === 'route'
            ? 'Click to add nodes | Right-click to assign looping route to ALL | Escape to cancel'
            : 'Click to add | Right-click to confirm | Escape to cancel'
    );

    function handleClearRoute() {
        currentPath.set([]);
    }
</script>

{#if $pathMode}
    <div class="path-indicator">
        <span class="badge">{modeLabel}</span>
        <span class="waypoints">Waypoints: <span class="count">{$currentPath.length}</span></span>
        <span class="hint">{hint}</span>
        {#if $objectiveMode === 'route'}
            <button class="clear-btn" onclick={handleClearRoute}>
                Clear Route
            </button>
        {/if}
    </div>
{/if}

<style>
    .path-indicator {
        position: fixed;
        top: 16px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(10, 10, 15, 0.95);
        color: white;
        padding: 10px 20px;
        border-radius: 4px;
        display: flex;
        gap: 20px;
        align-items: center;
        z-index: 50;
        border: 1px solid #345C00;
        box-shadow: 0 0 30px rgba(157, 255, 32, 0.2);
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    .badge {
        background: #9DFF20;
        color: #0a0a0f;
        padding: 4px 10px;
        border-radius: 2px;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.1em;
        font-family: 'IBM Plex Mono', monospace;
    }

    .waypoints {
        color: #9ca3af;
        font-size: 14px;
    }

    .count {
        color: #9DFF20;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
    }

    .hint {
        color: #6b7280;
        font-size: 12px;
    }

    .clear-btn {
        background: #1a1a1f;
        border: 1px solid #2a2a30;
        color: #9ca3af;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-family: 'DM Sans', system-ui, sans-serif;
        cursor: pointer;
        transition: all 0.15s ease;
    }

    .clear-btn:hover {
        background: #252530;
        color: #fff;
        border-color: #9DFF20;
    }
</style>
