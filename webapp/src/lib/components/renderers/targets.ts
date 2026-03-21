export function drawTarget(
    ctx: CanvasRenderingContext2D,
    target: { x: number; y: number; group: 'a' | 'b' },
): void {
    const size = 40;
    const half = size / 2;
    const color = target.group === 'a' ? 'rgba(255, 60, 60, 0.8)' : 'rgba(60, 140, 255, 0.8)';
    const border = target.group === 'a' ? 'rgba(255, 120, 120, 1)' : 'rgba(120, 180, 255, 1)';

    ctx.fillStyle = color;
    ctx.fillRect(target.x - half, target.y - half, size, size);
    ctx.strokeStyle = border;
    ctx.lineWidth = 2;
    ctx.strokeRect(target.x - half, target.y - half, size, size);
}

export function drawTargetScore(
    ctx: CanvasRenderingContext2D,
    targetCounts: { a: number; b: number },
    droneCounts?: { a: number; b: number },
): void {
    ctx.font = 'bold 16px "IBM Plex Mono", monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';

    // Red team
    ctx.fillStyle = 'rgba(255, 60, 60, 0.9)';
    let redText = `RED targets: ${targetCounts.a}`;
    if (droneCounts) redText += `  drones: ${droneCounts.a}`;
    ctx.fillText(redText, 10, 14);

    // Blue team
    ctx.fillStyle = 'rgba(60, 140, 255, 0.9)';
    let bluText = `BLU targets: ${targetCounts.b}`;
    if (droneCounts) bluText += `  drones: ${droneCounts.b}`;
    ctx.fillText(bluText, 10, 38);
}
