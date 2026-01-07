<script lang="ts">
    import { isRunning, pathMode, objectiveMode } from '$lib/stores/simulation';
    import Button from './Button.svelte';
    import SpeedSlider from './SpeedSlider.svelte';
    import ModeSelector from './ModeSelector.svelte';
    import SwarmSizeSelector from './SwarmSizeSelector.svelte';

    interface Props {
        onStart: () => void;
        onPause: () => void;
        onReset: () => void;
    }

    let { onStart, onPause, onReset }: Props = $props();
</script>

<div class="control-panel">
    <div class="playback-controls">
        {#if $isRunning}
            <Button variant="secondary" onclick={onPause}>Pause</Button>
        {:else}
            <Button variant="primary" onclick={onStart}>Start</Button>
        {/if}
        <Button variant="secondary" onclick={onReset}>Reset</Button>
    </div>

    <SwarmSizeSelector />

    <ModeSelector />

    <SpeedSlider />

    {#if $pathMode}
        <div class="path-hint">
            {#if $objectiveMode === 'route'}
                Click to add route nodes, Right-click to confirm (loops)
            {:else}
                Click to add waypoints, Right-click to confirm
            {/if}
        </div>
    {/if}
</div>

<style>
    .control-panel {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }

    .playback-controls {
        display: flex;
        gap: 8px;
    }

    .playback-controls :global(button) {
        flex: 1;
    }

    .path-hint {
        color: #9DFF20;
        font-size: 12px;
        font-family: 'IBM Plex Mono', monospace;
        line-height: 1.4;
    }
</style>
