use crate::types::Position;

/// Convex hull using Andrew's monotone chain algorithm.
pub fn convex_hull(points: &[Position]) -> Vec<Position> {
    if points.len() <= 1 {
        return points.to_vec();
    }

    let mut sorted: Vec<Position> = points.to_vec();
    sorted.sort_by(|a, b| {
        a.x()
            .partial_cmp(&b.x())
            .unwrap()
            .then(a.y().partial_cmp(&b.y()).unwrap())
    });

    let cross = |o: Position, a: Position, b: Position| -> f32 {
        (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x())
    };

    let mut lower: Vec<Position> = Vec::new();
    for &p in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }

    let mut upper: Vec<Position> = Vec::new();
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Build patrol waypoints around targets: convex hull pushed outward by standoff.
pub fn build_patrol_route(targets: &[Position], standoff: f32) -> Vec<Position> {
    if targets.is_empty() {
        return Vec::new();
    }

    let cx = targets.iter().map(|p| p.x()).sum::<f32>() / targets.len() as f32;
    let cy = targets.iter().map(|p| p.y()).sum::<f32>() / targets.len() as f32;

    if targets.len() <= 2 {
        let n = 4;
        return (0..n)
            .map(|i| {
                let angle = (i as f32 / n as f32) * std::f32::consts::TAU;
                Position::new(cx + standoff * angle.cos(), cy + standoff * angle.sin())
            })
            .collect();
    }

    let hull = convex_hull(targets);
    hull.iter()
        .map(|p| {
            let dx = p.x() - cx;
            let dy = p.y() - cy;
            let dist = (dx * dx + dy * dy).sqrt().max(1.0);
            Position::new(p.x() + (dx / dist) * standoff, p.y() + (dy / dist) * standoff)
        })
        .collect()
}
