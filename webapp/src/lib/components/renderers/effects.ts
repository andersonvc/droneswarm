const LIME_ACCENT = '#9DFF20';

export function drawExplosion(
    ctx: CanvasRenderingContext2D,
    explosion: { x: number; y: number; radius: number; time: number },
): void {
    const elapsed = Date.now() - explosion.time;
    const duration = 800;
    if (elapsed >= duration) return;

    // Fade out over duration
    const progress = elapsed / duration;
    const alpha = 1.0 - progress;

    // Expanding ring effect
    const currentRadius = explosion.radius * (0.5 + 0.5 * progress);

    ctx.strokeStyle = `rgba(255, 40, 40, ${alpha})`;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.arc(explosion.x, explosion.y, currentRadius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Inner glow fill
    ctx.fillStyle = `rgba(255, 60, 20, ${alpha * 0.15})`;
    ctx.beginPath();
    ctx.arc(explosion.x, explosion.y, currentRadius, 0, Math.PI * 2);
    ctx.fill();
}

export function drawSelectionBox(
    ctx: CanvasRenderingContext2D,
    dragStart: { x: number; y: number },
    dragEnd: { x: number; y: number },
): void {
    const x = Math.min(dragStart.x, dragEnd.x);
    const y = Math.min(dragStart.y, dragEnd.y);
    const w = Math.abs(dragEnd.x - dragStart.x);
    const h = Math.abs(dragEnd.y - dragStart.y);

    // Fill
    ctx.fillStyle = 'rgba(157, 255, 32, 0.1)';
    ctx.fillRect(x, y, w, h);

    // Border
    ctx.strokeStyle = LIME_ACCENT;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
}
