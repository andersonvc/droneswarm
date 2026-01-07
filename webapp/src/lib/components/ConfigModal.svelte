<script lang="ts">
    import * as Dialog from '$lib/components/ui/dialog';
    import { Button } from '$lib/components/ui/button';
    import { Slider } from '$lib/components/ui/slider';
    import { Label } from '$lib/components/ui/label';
    import { flightParams, setFlightParams } from '$lib/stores/simulation';

    let open = $state(false);

    let maxVelocityArr = $state([$flightParams.maxVelocity]);
    let maxAccelerationArr = $state([$flightParams.maxAcceleration]);
    let maxTurnRateArr = $state([$flightParams.maxTurnRate]);

    // Sync local values when modal opens
    $effect(() => {
        if (open) {
            maxVelocityArr = [$flightParams.maxVelocity];
            maxAccelerationArr = [$flightParams.maxAcceleration];
            maxTurnRateArr = [$flightParams.maxTurnRate];
        }
    });

    function handleApply() {
        setFlightParams(maxVelocityArr[0], maxAccelerationArr[0], maxTurnRateArr[0]);
        open = false;
    }

    function handleReset() {
        maxVelocityArr = [120];
        maxAccelerationArr = [21];
        maxTurnRateArr = [4];
    }

    function openModal() {
        open = true;
    }
</script>

<Button variant="ghost" size="icon" class="text-muted-foreground hover:text-foreground" onclick={openModal}>
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>
        <circle cx="12" cy="12" r="3"/>
    </svg>
</Button>

<Dialog.Root bind:open>
    <Dialog.Content class="sm:max-w-[425px]">
        <Dialog.Header>
            <Dialog.Title>Flight Parameters</Dialog.Title>
            <Dialog.Description>
                Adjust drone flight characteristics. Changes apply to all drones immediately.
            </Dialog.Description>
        </Dialog.Header>
        <div class="grid gap-6 py-4">
            <div class="grid gap-2">
                <div class="flex items-center justify-between">
                    <Label>Max Velocity</Label>
                    <span class="text-sm text-muted-foreground">{maxVelocityArr[0].toFixed(0)} m/s</span>
                </div>
                <Slider
                    bind:value={maxVelocityArr}
                    min={10}
                    max={300}
                    step={5}
                />
            </div>
            <div class="grid gap-2">
                <div class="flex items-center justify-between">
                    <Label>Max Acceleration</Label>
                    <span class="text-sm text-muted-foreground">{maxAccelerationArr[0].toFixed(0)} m/s²</span>
                </div>
                <Slider
                    bind:value={maxAccelerationArr}
                    min={1}
                    max={50}
                    step={1}
                />
            </div>
            <div class="grid gap-2">
                <div class="flex items-center justify-between">
                    <Label>Max Turn Rate</Label>
                    <span class="text-sm text-muted-foreground">{maxTurnRateArr[0].toFixed(1)} rad/s</span>
                </div>
                <Slider
                    bind:value={maxTurnRateArr}
                    min={0.5}
                    max={10}
                    step={0.5}
                />
            </div>
        </div>
        <Dialog.Footer>
            <Button variant="outline" onclick={handleReset}>Reset</Button>
            <Button onclick={handleApply}>Apply</Button>
        </Dialog.Footer>
    </Dialog.Content>
</Dialog.Root>
