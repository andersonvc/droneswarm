export function drawZoomIndicator(
    ctx: CanvasRenderingContext2D,
    zoom: number,
    width: number,
    height: number,
): void {
    const zoomText = `${Math.round(zoom * 100)}%`;
    ctx.fillStyle = 'rgba(157, 255, 32, 0.8)';
    ctx.font = '14px "IBM Plex Mono", monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(zoomText, width - 10, height - 10);
}

export function drawWinBanner(
    ctx: CanvasRenderingContext2D,
    result: string,
    width: number,
    height: number,
): void {
    let text = '';
    let color = '';
    if (result === 'a_wins') {
        text = 'RED TEAM WINS';
        color = 'rgba(255, 60, 60, 0.95)';
    } else if (result === 'b_wins') {
        text = 'BLUE TEAM WINS';
        color = 'rgba(60, 140, 255, 0.95)';
    } else if (result === 'draw') {
        text = 'DRAW';
        color = 'rgba(200, 200, 200, 0.95)';
    }

    // Darken background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(0, 0, width, height);

    // Banner
    const bannerH = Math.min(80, height * 0.12);
    const bannerY = height / 2 - bannerH / 2;
    ctx.fillStyle = 'rgba(10, 10, 15, 0.9)';
    ctx.fillRect(0, bannerY, width, bannerH);

    // Text
    const fontSize = Math.min(36, width * 0.06);
    ctx.font = `bold ${fontSize}px "IBM Plex Mono", monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    ctx.fillText(text, width / 2, height / 2);
}
