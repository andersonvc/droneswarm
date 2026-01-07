<script lang="ts">
    import { status } from '$lib/stores/simulation';

    let formattedTime = $derived($status.simulationTime.toFixed(1));
</script>

<div class="status-bar">
    <span class="stat"><span class="label">Drones</span> <span class="value">{$status.droneCount}</span></span>
    <span class="stat"><span class="label">Selected</span> <span class="value">{$status.selectedCount}</span></span>
    <span class="stat"><span class="label">Time</span> <span class="value">{formattedTime}s</span></span>
    <span class="stat"><span class="label">Speed</span> <span class="value">{$status.speedMultiplier}x</span></span>
</div>

<style>
    .status-bar {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 12px 0;
        border-top: 1px solid #1f1f25;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
    }

    .stat {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .label {
        color: #6b7280;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 0.05em;
    }

    .value {
        color: #9DFF20;
        font-weight: 500;
    }
</style>
