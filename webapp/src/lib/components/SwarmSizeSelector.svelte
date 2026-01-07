<script lang="ts">
    import {
        swarmSize,
        reinitializeWithSize,
        SWARM_SIZE_COUNTS,
        type SwarmSize
    } from '$lib/stores/simulation';

    async function setSize(size: SwarmSize) {
        if ($swarmSize === size) return;
        await reinitializeWithSize(size);
    }
</script>

<div class="size-selector">
    <span class="size-label">Swarm Size:</span>
    <div class="toggle-group">
        <button
            class="toggle-btn"
            class:active={$swarmSize === 'small'}
            onclick={() => setSize('small')}
            title="{SWARM_SIZE_COUNTS.small} drones"
        >
            Small
        </button>
        <button
            class="toggle-btn"
            class:active={$swarmSize === 'medium'}
            onclick={() => setSize('medium')}
            title="{SWARM_SIZE_COUNTS.medium} drones"
        >
            Medium
        </button>
        <button
            class="toggle-btn"
            class:active={$swarmSize === 'large'}
            onclick={() => setSize('large')}
            title="{SWARM_SIZE_COUNTS.large} drones"
        >
            Large
        </button>
    </div>
</div>

<style>
    .size-selector {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .size-label {
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
