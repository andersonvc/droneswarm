<script lang="ts">
    import type { Snippet } from 'svelte';
    import type { HTMLButtonAttributes } from 'svelte/elements';

    interface Props extends HTMLButtonAttributes {
        variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
        disabled?: boolean;
        children?: Snippet;
    }

    let { variant = 'primary', disabled = false, children, ...restProps }: Props = $props();
</script>

<button class="btn btn-{variant}" {disabled} {...restProps}>
    {#if children}
        {@render children()}
    {/if}
</button>

<style>
    .btn {
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: 500;
        cursor: pointer;
        border: none;
        transition: all 0.2s;
        font-size: 14px;
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    .btn-primary {
        background: #9DFF20;
        color: #0a0a0f;
    }
    .btn-primary:hover:not(:disabled) {
        background: #b4ff4d;
        box-shadow: 0 0 20px rgba(157, 255, 32, 0.3);
    }

    .btn-secondary {
        background: #1a1a1f;
        color: #9ca3af;
        border: 1px solid #2a2a30;
    }
    .btn-secondary:hover:not(:disabled) {
        background: #252530;
        color: #9DFF20;
        border-color: #345C00;
    }

    .btn-danger {
        background: #dc2626;
        color: white;
    }
    .btn-danger:hover:not(:disabled) {
        background: #ef4444;
    }

    .btn-ghost {
        background: transparent;
        color: #6b7280;
    }
    .btn-ghost:hover:not(:disabled) {
        background: rgba(157, 255, 32, 0.1);
        color: #9DFF20;
    }

    .btn:disabled {
        opacity: 0.4;
        cursor: not-allowed;
    }
</style>
