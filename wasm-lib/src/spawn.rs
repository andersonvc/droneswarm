use drone_lib::game::rng::lcg_next;

use crate::types::{Bounds, Color, Point, SimulationConfig, SpawnPattern};

/// Generate spawn positions (in pixels) from a simulation config.
pub(crate) fn generate_spawn_positions(config: &SimulationConfig) -> Vec<Point> {
    match &config.spawn_pattern {
        SpawnPattern::Grid => spawn_grid(config.drone_count, &config.bounds),
        SpawnPattern::Random => spawn_random(config.drone_count, &config.bounds),
        SpawnPattern::Cluster { center, radius } => {
            spawn_cluster(config.drone_count, center, *radius)
        }
        SpawnPattern::Custom { positions } => positions.clone(),
    }
}

/// Generate a random seed from JavaScript Math.random()
pub(crate) fn random_seed() -> u32 {
    (js_sys::Math::random() * u32::MAX as f64) as u32
}

fn spawn_grid(count: u32, bounds: &Bounds) -> Vec<Point> {
    // Grid positions with random jitter for variety
    let cols = (count as f32).sqrt().ceil() as u32;
    let rows = count.div_ceil(cols);
    let spacing_x = bounds.width / (cols + 1) as f32;
    let spacing_y = bounds.height / (rows + 1) as f32;
    let jitter = spacing_x.min(spacing_y) * 0.3; // 30% jitter

    let mut seed = random_seed();

    (0..count)
        .map(|i| {
            let col = i % cols;
            let row = i / cols;
            // Add random jitter
            lcg_next(&mut seed);
            let jx = ((seed as f32 / u32::MAX as f32) - 0.5) * jitter;
            lcg_next(&mut seed);
            let jy = ((seed as f32 / u32::MAX as f32) - 0.5) * jitter;
            Point {
                x: spacing_x * (col + 1) as f32 + jx,
                y: spacing_y * (row + 1) as f32 + jy,
            }
        })
        .collect()
}

fn spawn_random(count: u32, bounds: &Bounds) -> Vec<Point> {
    // Use random seed from JavaScript for true randomness on each init
    let mut seed = random_seed();
    (0..count)
        .map(|_| {
            lcg_next(&mut seed);
            let x = (seed as f32 / u32::MAX as f32) * bounds.width;
            lcg_next(&mut seed);
            let y = (seed as f32 / u32::MAX as f32) * bounds.height;
            Point { x, y }
        })
        .collect()
}

fn spawn_cluster(count: u32, center: &Point, radius: f32) -> Vec<Point> {
    // Use random seed from JavaScript for true randomness on each init
    let mut seed = random_seed();
    (0..count)
        .map(|_| {
            lcg_next(&mut seed);
            let angle = (seed as f32 / u32::MAX as f32) * std::f32::consts::TAU;
            lcg_next(&mut seed);
            let r = (seed as f32 / u32::MAX as f32).sqrt() * radius;
            Point {
                x: center.x + r * angle.cos(),
                y: center.y + r * angle.sin(),
            }
        })
        .collect()
}

/// Generate a color for a drone based on its index and the total count.
pub(crate) fn generate_color(index: usize, total: usize) -> Color {
    let half = (total + 1) / 2;
    if index < half {
        // Group A — red
        Color { r: 220, g: 60, b: 60 }
    } else {
        // Group B — blue
        Color { r: 60, g: 120, b: 220 }
    }
}
