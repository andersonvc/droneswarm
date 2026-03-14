<script lang="ts">
    import {
        groupAStrategy,
        groupBStrategy,
        setGroupStrategy,
        type StrategyType,
    } from '$lib/stores/simulation';

    function handleChange(group: 0 | 1, event: Event) {
        const value = (event.target as HTMLSelectElement).value as StrategyType;
        setGroupStrategy(group, value);
    }
</script>

<div class="strategy-selector">
    <div class="strategy-group">
        <label class="strategy-label" for="strat-a">Group A Strategy</label>
        <select
            id="strat-a"
            class="strategy-select"
            value={$groupAStrategy}
            onchange={(e) => handleChange(0, e)}
        >
            <option value="none">None (Route)</option>
            <option value="defend_area">Defend Area</option>
            <option value="attack_zone">Attack Zone</option>
            <option value="patrol_perimeter">Patrol Perimeter</option>
            <option value="doctrine_aggressive">Doctrine: Aggressive</option>
            <option value="doctrine_defensive">Doctrine: Defensive</option>
            <option value="doctrine_rl">Doctrine: RL Agent</option>
        </select>
    </div>

    <div class="strategy-group">
        <label class="strategy-label" for="strat-b">Group B Strategy</label>
        <select
            id="strat-b"
            class="strategy-select"
            value={$groupBStrategy}
            onchange={(e) => handleChange(1, e)}
        >
            <option value="none">None (Route)</option>
            <option value="defend_area">Defend Area</option>
            <option value="attack_zone">Attack Zone</option>
            <option value="patrol_perimeter">Patrol Perimeter</option>
            <option value="doctrine_aggressive">Doctrine: Aggressive</option>
            <option value="doctrine_defensive">Doctrine: Defensive</option>
            <option value="doctrine_rl">Doctrine: RL Agent</option>
        </select>
    </div>
</div>

<style>
    .strategy-selector {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .strategy-group {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .strategy-label {
        color: #6b7280;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    .strategy-select {
        background: #1a1a1f;
        border: 1px solid #2a2a30;
        color: #9ca3af;
        padding: 8px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-family: 'DM Sans', system-ui, sans-serif;
        cursor: pointer;
        transition: all 0.15s ease;
        width: 100%;
        appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b7280' d='M3 5l3 3 3-3'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: right 8px center;
        padding-right: 28px;
    }

    .strategy-select:hover {
        background-color: #252530;
        color: #fff;
        border-color: #9DFF20;
    }

    .strategy-select:focus {
        outline: none;
        border-color: #9DFF20;
        color: #fff;
    }
</style>
