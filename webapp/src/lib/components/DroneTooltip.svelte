<script lang="ts">
    import { hoveredDroneId, renderState } from '$lib/stores/simulation';

    interface Props {
        x: number;
        y: number;
    }

    let { x, y }: Props = $props();

    let drone = $derived(
        $hoveredDroneId !== null
            ? $renderState.find((d) => d.id === $hoveredDroneId)
            : null
    );
</script>

{#if drone}
    <div class="tooltip" style="left: {x + 20}px; top: {y - 10}px;">
        <div class="tooltip-header">Drone #{drone.id}</div>
        <div class="tooltip-row">
            <span class="label">Status</span>
            <span class="value">{drone.objectiveType}</span>
        </div>
        <div class="tooltip-row">
            <span class="label">Position</span>
            <span class="value">({drone.x.toFixed(0)}, {drone.y.toFixed(0)})</span>
        </div>
        {#if drone.target}
            <div class="tooltip-row">
                <span class="label">Target</span>
                <span class="value">({drone.target.x.toFixed(0)}, {drone.target.y.toFixed(0)})</span>
            </div>
        {/if}
    </div>
{/if}

<style>
    .tooltip {
        position: fixed;
        background: rgba(10, 10, 15, 0.95);
        color: #e5e7eb;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
        pointer-events: none;
        z-index: 100;
        border: 1px solid #345C00;
        box-shadow: 0 0 20px rgba(157, 255, 32, 0.15);
    }

    .tooltip-header {
        font-weight: 600;
        margin-bottom: 8px;
        color: #9DFF20;
        font-family: 'DM Sans', system-ui, sans-serif;
        font-size: 13px;
    }

    .tooltip-row {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 4px;
    }

    .label {
        color: #6b7280;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 0.05em;
    }

    .value {
        color: #e5e7eb;
    }
</style>
