<script lang="ts">
    import {
        voConfig,
        setVoConfig,
        waypointClearance,
        setWaypointClearance,
        consensusProtocol,
        setConsensusProtocol,
        type VoConfig,
        type ConsensusProtocol
    } from '$lib/stores/simulation';

    interface Props {
        isOpen: boolean;
        onClose: () => void;
    }

    let { isOpen, onClose }: Props = $props();

    // Local state for form inputs
    let localConfig = $state<VoConfig>({
        lookaheadTime: 1.0,
        timeSamples: 5,
        safeDistance: 50.0,
        detectionRange: 120.0,
        avoidanceWeight: 0.85,
    });

    let localWaypointClearance = $state(10.0);
    let localConsensusProtocol = $state<ConsensusProtocol>('priority_by_id');

    // Sync with store when opened
    $effect(() => {
        if (isOpen) {
            localConfig = { ...$voConfig };
            localWaypointClearance = $waypointClearance;
            localConsensusProtocol = $consensusProtocol;
        }
    });

    function handleVoChange() {
        setVoConfig(localConfig);
    }

    function handleWaypointClearanceChange() {
        setWaypointClearance(localWaypointClearance);
    }

    function handleConsensusProtocolChange(protocol: ConsensusProtocol) {
        localConsensusProtocol = protocol;
        setConsensusProtocol(protocol);
    }

    function handleClickOutside(e: MouseEvent) {
        if ((e.target as HTMLElement).classList.contains('overlay')) {
            onClose();
        }
    }
</script>

{#if isOpen}
    <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
    <div class="overlay" onclick={handleClickOutside}>
        <aside class="side-panel">
            <header>
                <h2>Configuration</h2>
                <button class="close-btn" onclick={onClose}>&times;</button>
            </header>

            <section>
                <h3>Velocity Obstacle</h3>

                <div class="config-row">
                    <label for="lookahead">
                        Lookahead Time
                        <span class="value">{localConfig.lookaheadTime.toFixed(2)}s</span>
                    </label>
                    <input
                        id="lookahead"
                        type="range"
                        min="0.1"
                        max="3"
                        step="0.1"
                        bind:value={localConfig.lookaheadTime}
                        oninput={handleVoChange}
                    />
                </div>

                <div class="config-row">
                    <label for="timeSamples">
                        Trajectory Samples
                        <span class="value">{localConfig.timeSamples}</span>
                    </label>
                    <input
                        id="timeSamples"
                        type="range"
                        min="5"
                        max="30"
                        step="1"
                        bind:value={localConfig.timeSamples}
                        oninput={handleVoChange}
                    />
                </div>

                <div class="config-row">
                    <label for="safeDistance">
                        Safe Distance
                        <span class="value">{localConfig.safeDistance.toFixed(0)}</span>
                    </label>
                    <input
                        id="safeDistance"
                        type="range"
                        min="30"
                        max="150"
                        step="30"
                        bind:value={localConfig.safeDistance}
                        oninput={handleVoChange}
                    />
                    <span class="unit-hint">multiples of collision diameter (30)</span>
                </div>

                <div class="config-row">
                    <label for="detectionRange">
                        Detection Range
                        <span class="value">{localConfig.detectionRange.toFixed(0)}</span>
                    </label>
                    <input
                        id="detectionRange"
                        type="range"
                        min="50"
                        max="300"
                        step="10"
                        bind:value={localConfig.detectionRange}
                        oninput={handleVoChange}
                    />
                </div>

                <div class="config-row">
                    <label for="avoidanceWeight">
                        Avoidance Weight
                        <span class="value">{localConfig.avoidanceWeight.toFixed(2)}</span>
                    </label>
                    <input
                        id="avoidanceWeight"
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        bind:value={localConfig.avoidanceWeight}
                        oninput={handleVoChange}
                    />
                </div>
            </section>

            <section>
                <h3>Navigation</h3>

                <div class="config-row">
                    <label for="waypointClearance">
                        Waypoint Clearance
                        <span class="value">{localWaypointClearance.toFixed(0)}</span>
                    </label>
                    <input
                        id="waypointClearance"
                        type="range"
                        min="1"
                        max="200"
                        step="1"
                        bind:value={localWaypointClearance}
                        oninput={handleWaypointClearanceChange}
                    />
                </div>
                <p class="description">
                    How close drone must be to waypoint to consider it reached.
                </p>
            </section>

            <section>
                <h3>Consensus Protocol</h3>
                <div class="protocol-options">
                    <button
                        class="protocol-btn"
                        class:active={localConsensusProtocol === 'priority_by_id'}
                        onclick={() => handleConsensusProtocolChange('priority_by_id')}
                    >
                        <span class="protocol-name">Priority by ID</span>
                        <span class="protocol-desc">Lower ID = higher priority</span>
                    </button>
                    <button
                        class="protocol-btn"
                        class:active={localConsensusProtocol === 'priority_by_waypoint_dist'}
                        onclick={() => handleConsensusProtocolChange('priority_by_waypoint_dist')}
                    >
                        <span class="protocol-name">Priority by Waypoint Dist</span>
                        <span class="protocol-desc">Closer to waypoint = higher priority</span>
                    </button>
                </div>
                <p class="description">
                    {#if localConsensusProtocol === 'priority_by_id'}
                        Lower ID drones have higher priority and plan first.
                        Higher ID drones yield to lower ID drones.
                    {:else}
                        Drones closest to their waypoint have priority.
                        Others yield to let them through.
                    {/if}
                </p>
            </section>
        </aside>
    </div>
{/if}

<style>
    .overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5);
        z-index: 100;
    }

    .side-panel {
        position: fixed;
        top: 0;
        left: 0;
        width: 320px;
        height: 100%;
        background: #101014;
        border-right: 1px solid #1f1f25;
        padding: 20px;
        overflow-y: auto;
        animation: slideIn 0.2s ease-out;
    }

    @keyframes slideIn {
        from {
            transform: translateX(-100%);
        }
        to {
            transform: translateX(0);
        }
    }

    header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid #1f1f25;
    }

    h2 {
        margin: 0;
        color: #fff;
        font-size: 18px;
        font-weight: 600;
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    h3 {
        margin: 0 0 16px 0;
        color: #9DFF20;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    .close-btn {
        background: none;
        border: none;
        color: #6b7280;
        font-size: 24px;
        cursor: pointer;
        padding: 0;
        line-height: 1;
    }

    .close-btn:hover {
        color: #fff;
    }

    section {
        margin-bottom: 28px;
    }

    .config-row {
        margin-bottom: 16px;
    }

    label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #9ca3af;
        font-size: 13px;
        margin-bottom: 8px;
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    .value {
        color: #9DFF20;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 500;
    }

    .unit-hint {
        display: block;
        color: #4b5563;
        font-size: 10px;
        margin-top: 4px;
        font-family: 'DM Sans', system-ui, sans-serif;
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

    .config-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .config-info p {
        margin: 0;
        color: #fff;
        font-size: 14px;
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    .badge {
        background: #345C00;
        color: #9DFF20;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
    }

    .description {
        color: #6b7280;
        font-size: 12px;
        line-height: 1.5;
        margin: 0;
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    .protocol-options {
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 12px;
    }

    .protocol-btn {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        padding: 12px;
        background: #1a1a1f;
        border: 1px solid #2a2a30;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.15s ease;
        text-align: left;
    }

    .protocol-btn:hover:not(.active) {
        background: #252530;
        border-color: #3a3a40;
    }

    .protocol-btn.active {
        background: #1a2a10;
        border-color: #9DFF20;
    }

    .protocol-name {
        color: #fff;
        font-size: 13px;
        font-weight: 500;
        font-family: 'DM Sans', system-ui, sans-serif;
        margin-bottom: 4px;
    }

    .protocol-btn.active .protocol-name {
        color: #9DFF20;
    }

    .protocol-desc {
        color: #6b7280;
        font-size: 11px;
        font-family: 'DM Sans', system-ui, sans-serif;
    }
</style>
