<script lang="ts">
    import {
        orcaConfig,
        setOrcaConfig,
        waypointClearance,
        setWaypointClearance,
        consensusProtocol,
        setConsensusProtocol,
        formationConfig,
        setFormation,
        formationCommand,
        type OrcaConfig,
        type ConsensusProtocol,
        type FormationType
    } from '$lib/stores/simulation';

    interface Props {
        isOpen: boolean;
        onClose: () => void;
    }

    let { isOpen, onClose }: Props = $props();

    // Local state for form inputs
    let localConfig = $state<OrcaConfig>({
        timeHorizon: 2.0,
        agentRadius: 20.0,
        neighborDist: 150.0,
    });

    let localWaypointClearance = $state(10.0);
    let localConsensusProtocol = $state<ConsensusProtocol>('priority_by_id');
    let localFormationType = $state<FormationType>('chevron');
    let localFormationSpacing = $state(40);

    // Sync with store when opened
    $effect(() => {
        if (isOpen) {
            localConfig = { ...$orcaConfig };
            localWaypointClearance = $waypointClearance;
            localConsensusProtocol = $consensusProtocol;
            localFormationType = $formationConfig.type;
            localFormationSpacing = $formationConfig.spacing;
        }
    });

    function handleOrcaChange() {
        setOrcaConfig(localConfig);
    }

    function handleWaypointClearanceChange() {
        setWaypointClearance(localWaypointClearance);
    }

    function handleConsensusProtocolChange(protocol: ConsensusProtocol) {
        localConsensusProtocol = protocol;
        setConsensusProtocol(protocol);
    }

    function handleFormationTypeChange(type: FormationType) {
        localFormationType = type;
        setFormation(type, localFormationSpacing);
    }

    function handleFormationSpacingChange() {
        if (localFormationType !== 'none') {
            setFormation(localFormationType, localFormationSpacing);
        }
    }

    function handleFormationCommand(cmd: 'contract' | 'expand' | 'disperse') {
        formationCommand(cmd);
        if (cmd === 'disperse') {
            localFormationType = 'none';
        }
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
                <h3>Collision Avoidance (ORCA)</h3>

                <div class="config-row">
                    <label for="timeHorizon">
                        Time Horizon
                        <span class="value">{localConfig.timeHorizon.toFixed(1)}s</span>
                    </label>
                    <input
                        id="timeHorizon"
                        type="range"
                        min="0.5"
                        max="5"
                        step="0.5"
                        bind:value={localConfig.timeHorizon}
                        oninput={handleOrcaChange}
                    />
                </div>

                <div class="config-row">
                    <label for="agentRadius">
                        Agent Radius
                        <span class="value">{localConfig.agentRadius.toFixed(0)}</span>
                    </label>
                    <input
                        id="agentRadius"
                        type="range"
                        min="10"
                        max="50"
                        step="5"
                        bind:value={localConfig.agentRadius}
                        oninput={handleOrcaChange}
                    />
                </div>

                <div class="config-row">
                    <label for="neighborDist">
                        Neighbor Distance
                        <span class="value">{localConfig.neighborDist.toFixed(0)}</span>
                    </label>
                    <input
                        id="neighborDist"
                        type="range"
                        min="10"
                        max="300"
                        step="10"
                        bind:value={localConfig.neighborDist}
                        oninput={handleOrcaChange}
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

            <section>
                <h3>Formation</h3>
                <div class="formation-types">
                    <button
                        class="formation-btn"
                        class:active={localFormationType === 'none'}
                        onclick={() => handleFormationTypeChange('none')}
                    >
                        None
                    </button>
                    <button
                        class="formation-btn"
                        class:active={localFormationType === 'line'}
                        onclick={() => handleFormationTypeChange('line')}
                    >
                        Line
                    </button>
                    <button
                        class="formation-btn"
                        class:active={localFormationType === 'vee'}
                        onclick={() => handleFormationTypeChange('vee')}
                    >
                        Vee
                    </button>
                    <button
                        class="formation-btn"
                        class:active={localFormationType === 'chevron'}
                        onclick={() => handleFormationTypeChange('chevron')}
                    >
                        Chevron
                    </button>
                    <button
                        class="formation-btn"
                        class:active={localFormationType === 'diamond'}
                        onclick={() => handleFormationTypeChange('diamond')}
                    >
                        Diamond
                    </button>
                    <button
                        class="formation-btn"
                        class:active={localFormationType === 'circle'}
                        onclick={() => handleFormationTypeChange('circle')}
                    >
                        Circle
                    </button>
                    <button
                        class="formation-btn"
                        class:active={localFormationType === 'grid'}
                        onclick={() => handleFormationTypeChange('grid')}
                    >
                        Grid
                    </button>
                </div>

                {#if localFormationType !== 'none'}
                    <div class="config-row">
                        <label for="formationSpacing">
                            Spacing
                            <span class="value">{localFormationSpacing}</span>
                        </label>
                        <input
                            id="formationSpacing"
                            type="range"
                            min="30"
                            max="150"
                            step="10"
                            bind:value={localFormationSpacing}
                            oninput={handleFormationSpacingChange}
                        />
                    </div>

                    <div class="formation-commands">
                        <button
                            class="cmd-btn"
                            onclick={() => handleFormationCommand('contract')}
                        >
                            Contract
                        </button>
                        <button
                            class="cmd-btn"
                            onclick={() => handleFormationCommand('expand')}
                        >
                            Expand
                        </button>
                        <button
                            class="cmd-btn cmd-danger"
                            onclick={() => handleFormationCommand('disperse')}
                        >
                            Disperse
                        </button>
                    </div>
                {/if}

                <p class="description">
                    {#if localFormationType === 'none'}
                        No formation active. Select a formation type to organize drones.
                    {:else if localFormationType === 'line'}
                        Drones arranged in a horizontal line.
                    {:else if localFormationType === 'vee'}
                        V-shaped wedge formation with leader at front.
                    {:else if localFormationType === 'chevron'}
                        Nested V layers, each row adds one more drone per side.
                    {:else if localFormationType === 'diamond'}
                        Diamond/rhombus formation with leader at front and tail at back.
                    {:else if localFormationType === 'circle'}
                        Drones arranged in a circle.
                    {:else if localFormationType === 'grid'}
                        Drones arranged in a grid pattern.
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

    .formation-types {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 16px;
    }

    .formation-btn {
        padding: 8px 12px;
        background: #1a1a1f;
        border: 1px solid #2a2a30;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.15s ease;
        color: #9ca3af;
        font-size: 12px;
        font-family: 'DM Sans', system-ui, sans-serif;
        font-weight: 500;
    }

    .formation-btn:hover:not(.active) {
        background: #252530;
        border-color: #3a3a40;
        color: #fff;
    }

    .formation-btn.active {
        background: #1a2a10;
        border-color: #9DFF20;
        color: #9DFF20;
    }

    .formation-commands {
        display: flex;
        gap: 8px;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .cmd-btn {
        flex: 1;
        padding: 8px 12px;
        background: #1a1a1f;
        border: 1px solid #2a2a30;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.15s ease;
        color: #9ca3af;
        font-size: 11px;
        font-family: 'DM Sans', system-ui, sans-serif;
        font-weight: 500;
    }

    .cmd-btn:hover {
        background: #252530;
        border-color: #3a3a40;
        color: #fff;
    }

    .cmd-btn.cmd-danger:hover {
        background: #2a1a1a;
        border-color: #ff4444;
        color: #ff4444;
    }
</style>
