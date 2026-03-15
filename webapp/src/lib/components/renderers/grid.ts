const GRID_SPACING = 100;
const GRID_OPACITY = 0.08;

export function drawGrid(
    ctx: CanvasRenderingContext2D,
    worldWidth: number,
    worldHeight: number,
): void {
    ctx.strokeStyle = `rgba(157, 255, 32, ${GRID_OPACITY})`;
    ctx.lineWidth = 1;

    for (let x = GRID_SPACING; x < worldWidth; x += GRID_SPACING) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, worldHeight);
        ctx.stroke();
    }

    for (let y = GRID_SPACING; y < worldHeight; y += GRID_SPACING) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(worldWidth, y);
        ctx.stroke();
    }
}
