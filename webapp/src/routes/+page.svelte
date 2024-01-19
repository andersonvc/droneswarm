<h1>Drone Visualization Map</h1>
<button on:click={handleClick} >Reset</button>
<canvas id="my-canvas" bind:this={canvasElement}></canvas>

<script lang="ts">
    import init, {hello} from 'wasm-lib';
    import run_plot from 'wasm-lib';
    import MyDummy from 'wasm-lib';
    import DronePos from 'wasm-lib';
    import { MyDummy, DronePos } from 'wasm-lib';
    import DroneTelemetry from 'wasm-lib';
    import { onMount, afterUpdate, onDestroy, setContext } from 'svelte';
	import { get } from 'svelte/store';

    let canvasElement: HTMLCanvasElement;
    let res;
    let my_text: number;
    let circle_offset: { x_offset: number; y_offset: number; radius_offset: number; };
    

    const drawCircle = (x: number, y: number) => {
        let ctx = canvasElement.getContext('2d')!;
        ctx.beginPath();
        ctx.arc(circle_offset.x_offset*x, circle_offset.y_offset*(1-y), circle_offset.radius_offset, 0, 2 * Math.PI);
        ctx.stroke();
    };

    const resizeCanvas = () => {
        let offset_pct = 0.02;
        let new_width = (window.innerWidth-canvasElement.getBoundingClientRect().left) * (1-2*offset_pct);
        let width_offset = window.innerWidth * offset_pct;
        canvasElement.style.left = width_offset.toString() + "px";
        canvasElement.style.width = new_width.toString() + "px";

        let new_height = (window.innerHeight-canvasElement.getBoundingClientRect().top) * (1-2*offset_pct);
        let height_offset = window.innerHeight * offset_pct;
        canvasElement.style.top = height_offset.toString() + "px";
        canvasElement.style.height = new_height.toString() + "px";

        var scale = window.devicePixelRatio; // Change to 1 on retina screens to see blurry canvas.
        canvasElement.width = new_width * scale;
        canvasElement.height = new_height * scale;

        let ctx = canvasElement.getContext('2d')!;
        ctx.scale(scale, scale);
        circle_offset = {
            x_offset: new_width,
            y_offset: new_height,
            radius_offset: Math.min(new_width, new_height)*0.01,
        };

        drawCircle(0.5,0.5);
        drawCircle(0.25,0.25);
    };

    /*
    function draw() {
        if (typeof document === 'undefined') return; // Check if running in a non-browser environment
  
        if (!ctx) {
            canvasElement = document.createElement('canvas');
            ctx = canvasElement.getContext('2d')!;
            document.querySelector('.canvas-container')?.appendChild(canvas);
        }
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height); // Clear the canvas before drawing
    }
    */

    onMount(
        // canvasElement = document.querySelector('canvas')!,
        
        async() => {
            await init();
            my_text = Number(hello());
            console.log(my_text);
            let ctx = canvasElement.getContext('2d')!;
            window.addEventListener('resize', resizeCanvas);
            resizeCanvas();

            var bbb = await run_plot();
            console.log(bbb);
            alert(bbb.drone[0].x);
            
        },
    );
    
    const handleClick = () => {
        console.log('start');

    };

</script>


<style>  

    canvas {
        border: 2px solid black;
        z-index: 0;
        position: relative;

    }

</style>

