<script lang="ts">
    import * as Dialog from '$lib/components/ui/dialog';
    import { Button } from '$lib/components/ui/button';
    import { pathMode, currentPath } from '$lib/stores/simulation';

    let { open = $bindable(false) }: { open: boolean } = $props();

    function handleStartPlacing() {
        pathMode.set(true);
        open = false;
    }

    function handleReset() {
        currentPath.set([]);
        pathMode.set(false);
    }

    function handleClose() {
        open = false;
    }
</script>

<Dialog.Root bind:open>
    <Dialog.Content class="sm:max-w-[350px]">
        <Dialog.Header>
            <Dialog.Title>Route Creator</Dialog.Title>
            <Dialog.Description>
                Create a looping route for drones to follow continuously.
            </Dialog.Description>
        </Dialog.Header>
        <div class="grid gap-3 py-4">
            <Button onclick={handleStartPlacing} class="w-full">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mr-2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="16"/>
                    <line x1="8" y1="12" x2="16" y2="12"/>
                </svg>
                Start Placing Waypoints
            </Button>
            <Button variant="outline" onclick={handleReset} class="w-full">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mr-2">
                    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                    <path d="M3 3v5h5"/>
                </svg>
                Reset Route
            </Button>
        </div>
        <Dialog.Footer>
            <Button variant="ghost" onclick={handleClose}>Close</Button>
        </Dialog.Footer>
    </Dialog.Content>
</Dialog.Root>
