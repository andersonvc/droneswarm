<script lang="ts">
	import { cn } from '$lib/utils/cn';
	import type { HTMLInputAttributes } from 'svelte/elements';

	interface Props extends Omit<HTMLInputAttributes, 'value'> {
		class?: string;
		value?: number[];
		min?: number;
		max?: number;
		step?: number;
	}

	let { class: className = '', value = $bindable([0]), min = 0, max = 100, step = 1, ...restProps }: Props = $props();

	function handleInput(e: Event) {
		const target = e.target as HTMLInputElement;
		value = [parseFloat(target.value)];
	}
</script>

<div class={cn('relative flex w-full touch-none select-none items-center', className)}>
	<input
		type="range"
		{min}
		{max}
		{step}
		value={value[0]}
		oninput={handleInput}
		class="w-full h-2 bg-secondary rounded-full appearance-none cursor-pointer accent-primary
			[&::-webkit-slider-thumb]:appearance-none
			[&::-webkit-slider-thumb]:w-5
			[&::-webkit-slider-thumb]:h-5
			[&::-webkit-slider-thumb]:rounded-full
			[&::-webkit-slider-thumb]:bg-background
			[&::-webkit-slider-thumb]:border-2
			[&::-webkit-slider-thumb]:border-primary
			[&::-webkit-slider-thumb]:cursor-pointer
			[&::-moz-range-thumb]:w-5
			[&::-moz-range-thumb]:h-5
			[&::-moz-range-thumb]:rounded-full
			[&::-moz-range-thumb]:bg-background
			[&::-moz-range-thumb]:border-2
			[&::-moz-range-thumb]:border-primary
			[&::-moz-range-thumb]:cursor-pointer"
		{...restProps}
	/>
</div>
