<script lang="ts">
    import { isRunning } from '$lib/stores/simulation';
    import Button from './Button.svelte';
    import SpeedSlider from './SpeedSlider.svelte';

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

    <SpeedSlider />
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
</style>
