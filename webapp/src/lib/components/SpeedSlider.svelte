<script lang="ts">
    import { status, setSpeed } from '$lib/stores/simulation';

    const speeds = [4, 8, 16, 24];

    function handleChange(e: Event) {
        const index = parseInt((e.target as HTMLInputElement).value, 10);
        setSpeed(speeds[index]);
    }

    let currentIndex = $derived(speeds.findIndex((s) => s === $status.speedMultiplier));
</script>

<div class="speed-slider">
    <label for="speed">Speed: <span class="speed-value">{$status.speedMultiplier}x</span></label>
    <input
        id="speed"
        type="range"
        min="0"
        max="3"
        step="1"
        value={currentIndex >= 0 ? currentIndex : 2}
        oninput={handleChange}
    />
</div>

<style>
    .speed-slider {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    label {
        display: flex;
        justify-content: space-between;
        color: #6b7280;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    .speed-value {
        color: #9DFF20;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
    }

    input[type='range'] {
        width: 100%;
        cursor: pointer;
        -webkit-appearance: none;
        appearance: none;
        background: #1a1a1f;
        height: 6px;
        border-radius: 3px;
    }

    input[type='range']::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #9DFF20;
        cursor: pointer;
        box-shadow: 0 0 10px rgba(157, 255, 32, 0.4);
    }

    input[type='range']::-moz-range-thumb {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #9DFF20;
        cursor: pointer;
        border: none;
        box-shadow: 0 0 10px rgba(157, 255, 32, 0.4);
    }

    input[type='range']::-webkit-slider-runnable-track {
        background: linear-gradient(to right, #345C00, #1a1a1f);
        border-radius: 3px;
    }
</style>
