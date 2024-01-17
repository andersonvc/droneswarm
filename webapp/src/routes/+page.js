
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
