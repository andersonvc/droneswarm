<script lang="ts">
    import { avoidanceLookahead, setAvoidanceLookahead } from '$lib/stores/simulation';

    function handleChange(e: Event) {
        const value = parseFloat((e.target as HTMLInputElement).value);
        setAvoidanceLookahead(value);
    }
</script>

<div class="avoidance-slider">
    <label for="lookahead">
        Lookahead: <span class="lookahead-value">{$avoidanceLookahead.toFixed(2)}s</span>
    </label>
    <input
        id="lookahead"
        type="range"
        min="0.1"
        max="2"
        step="0.05"
        value={$avoidanceLookahead}
        oninput={handleChange}
    />
</div>

<style>
    .avoidance-slider {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    label {
        color: #6b7280;
        font-size: 13px;
        min-width: 120px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    .lookahead-value {
        color: #9DFF20;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
    }

    input[type='range'] {
        width: 100px;
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
