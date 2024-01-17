<h1>Welcome to SvelteKit</h1>
<p>Visit <a href="https://kit.svelte.dev">kit.svelte.dev</a> to read the documentation</p>
<p>{my_text}</p>

<script>
        import init, {hello} from 'wasm-lib';
        import { onMount, afterUpdate, onDestroy, setContext } from 'svelte';

        let my_text = 0;

    onMount(
        async() => {
            await init();
            my_text = Number(hello());
            console.log(my_text);
        }
    );

</script>
