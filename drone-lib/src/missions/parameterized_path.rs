//! Arc-length parameterized path for synchronized formation following.

use std::sync::Arc;

use crate::types::{Bounds, Position, Vec2};

/// Arc-length parameterized path for synchronized formation following.
///
/// This struct provides a way for all drones in a formation to track progress
/// along a shared path using a common arc-length parameter `s`. This enables:
///
/// - Synchronized position tracking (all drones at same progress)
/// - Speed adjustments based on being ahead/behind the parameter
/// - Formation slot positions computed at the current `s`
#[derive(Debug, Clone)]
pub struct ParameterizedPath {
    /// Route waypoints.
    waypoints: Arc<[Position]>,
    /// Cumulative arc length at each waypoint.
    /// `arc_lengths[i]` = total distance from `waypoints[0]` to `waypoints[i]`
    arc_lengths: Vec<f32>,
    /// Total path length (arc_lengths.last()).
    total_length: f32,
    /// World bounds for toroidal distance calculations.
    bounds: Bounds,
}

impl ParameterizedPath {
    /// Create a looping parameterized path that includes a closing segment
    /// from the last waypoint back to the first.
    ///
    /// Used for `FollowRoute` formation following where the route repeats.
    /// The closing segment allows smooth tracking across the loop boundary.
    pub fn new_loop(waypoints: Arc<[Position]>, bounds: Bounds) -> Self {
        if waypoints.len() < 2 {
            return Self::new(waypoints, bounds);
        }

        // Append the first waypoint to close the loop
        let mut loop_wps: Vec<Position> = waypoints.iter().copied().collect();
        loop_wps.push(waypoints[0]);
        Self::new(Arc::from(loop_wps), bounds)
    }

    /// Create a new parameterized path from waypoints.
    ///
    /// Computes cumulative arc lengths along the path.
    pub fn new(waypoints: Arc<[Position]>, bounds: Bounds) -> Self {
        let mut arc_lengths = Vec::with_capacity(waypoints.len());
        let mut cumulative = 0.0;

        arc_lengths.push(0.0);
        for i in 1..waypoints.len() {
            let p0 = waypoints[i - 1].as_vec2();
            let p1 = waypoints[i].as_vec2();
            let delta = bounds.delta(p0, p1);
            cumulative += delta.magnitude();
            arc_lengths.push(cumulative);
        }

        let total_length = cumulative;

        ParameterizedPath {
            waypoints,
            arc_lengths,
            total_length,
            bounds,
        }
    }

    /// Get position at arc-length parameter `s` (0 to total_length).
    ///
    /// Linearly interpolates between waypoints.
    pub fn position_at(&self, s: f32) -> Position {
        if self.waypoints.is_empty() {
            return Position::new(0.0, 0.0);
        }

        if self.waypoints.len() == 1 {
            return self.waypoints[0];
        }

        let s_clamped = s.clamp(0.0, self.total_length);

        // Find segment containing s
        let segment_idx = self
            .arc_lengths
            .iter()
            .position(|&len| len > s_clamped)
            .unwrap_or(self.arc_lengths.len())
            .saturating_sub(1)
            .min(self.waypoints.len().saturating_sub(2));

        // Handle edge case at end of path
        if segment_idx >= self.waypoints.len().saturating_sub(1) {
            return self.waypoints[self.waypoints.len() - 1];
        }

        // Interpolate within segment
        let segment_start_s = self.arc_lengths[segment_idx];
        let segment_end_s = self.arc_lengths[segment_idx + 1];
        let segment_length = segment_end_s - segment_start_s;

        if segment_length < f32::EPSILON {
            return self.waypoints[segment_idx];
        }

        let t = (s_clamped - segment_start_s) / segment_length;
        let p0 = self.waypoints[segment_idx].as_vec2();
        let p1 = self.waypoints[segment_idx + 1].as_vec2();

        // Use toroidal interpolation
        let delta = self.bounds.delta(p0, p1);
        let pos = Vec2::new(p0.x + delta.x * t, p0.y + delta.y * t);

        Position::new(pos.x, pos.y)
    }

    /// Get heading at arc-length parameter `s`.
    ///
    /// Returns the heading along the path segment containing `s`.
    pub fn heading_at(&self, s: f32) -> f32 {
        if self.waypoints.len() < 2 {
            return 0.0;
        }

        let s_clamped = s.clamp(0.0, self.total_length);

        // Find segment containing s
        let segment_idx = self
            .arc_lengths
            .iter()
            .position(|&len| len > s_clamped)
            .unwrap_or(self.arc_lengths.len())
            .saturating_sub(1)
            .min(self.waypoints.len().saturating_sub(2));

        // At end, use last segment heading
        if segment_idx >= self.waypoints.len().saturating_sub(1) {
            let last = self.waypoints.len() - 1;
            let delta = self.bounds.delta(
                self.waypoints[last - 1].as_vec2(),
                self.waypoints[last].as_vec2(),
            );
            return delta.heading();
        }

        let delta = self.bounds.delta(
            self.waypoints[segment_idx].as_vec2(),
            self.waypoints[segment_idx + 1].as_vec2(),
        );
        delta.heading()
    }

    /// Find arc-length parameter for a given position (nearest point on path).
    ///
    /// Projects the position onto each segment and returns the `s` value
    /// of the closest point on the path.
    pub fn project_position(&self, pos: Position) -> f32 {
        if self.waypoints.is_empty() {
            return 0.0;
        }

        if self.waypoints.len() == 1 {
            return 0.0;
        }

        let pos_vec = pos.as_vec2();
        let mut best_s = 0.0;
        let mut best_dist_sq = f32::MAX;

        for i in 0..self.waypoints.len().saturating_sub(1) {
            let p0 = self.waypoints[i].as_vec2();
            let p1 = self.waypoints[i + 1].as_vec2();
            let segment = self.bounds.delta(p0, p1);
            let to_pos = self.bounds.delta(p0, pos_vec);

            let segment_len_sq = segment.magnitude_squared();
            if segment_len_sq < f32::EPSILON {
                continue;
            }

            // Project onto segment (t in [0, 1])
            let t = (to_pos.dot(segment) / segment_len_sq).clamp(0.0, 1.0);
            let projected = Vec2::new(p0.x + segment.x * t, p0.y + segment.y * t);
            let dist_sq = self.bounds.delta(pos_vec, projected).magnitude_squared();

            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                let segment_length = self.arc_lengths[i + 1] - self.arc_lengths[i];
                best_s = self.arc_lengths[i] + t * segment_length;
            }
        }

        best_s
    }

    /// Get the total path length.
    pub fn total_length(&self) -> f32 {
        self.total_length
    }

    /// Get the waypoints.
    pub fn waypoints(&self) -> &[Position] {
        &self.waypoints
    }

    /// Get the bounds.
    pub fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    /// Check if the path is empty.
    pub fn is_empty(&self) -> bool {
        self.waypoints.is_empty()
    }

    /// Get the number of waypoints.
    pub fn len(&self) -> usize {
        self.waypoints.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameterized_path_arc_lengths() {
        let bounds = Bounds::new(500.0, 500.0).unwrap();
        let waypoints: Arc<[Position]> = Arc::from(vec![
            Position::new(0.0, 0.0),
            Position::new(100.0, 0.0),   // 100m segment
            Position::new(100.0, 100.0), // 100m segment
        ]);

        let path = ParameterizedPath::new(waypoints, bounds);

        assert!((path.total_length() - 200.0).abs() < 0.1);
    }

    #[test]
    fn test_parameterized_path_position_at() {
        let bounds = Bounds::new(500.0, 500.0).unwrap();
        let waypoints: Arc<[Position]> = Arc::from(vec![
            Position::new(0.0, 0.0),
            Position::new(100.0, 0.0),
            Position::new(100.0, 100.0),
        ]);

        let path = ParameterizedPath::new(waypoints, bounds);

        // At start
        let p0 = path.position_at(0.0);
        assert!((p0.x() - 0.0).abs() < 0.1);
        assert!((p0.y() - 0.0).abs() < 0.1);

        // At midpoint of first segment
        let p50 = path.position_at(50.0);
        assert!((p50.x() - 50.0).abs() < 0.1);
        assert!((p50.y() - 0.0).abs() < 0.1);

        // At first waypoint (s = 100)
        let p100 = path.position_at(100.0);
        assert!((p100.x() - 100.0).abs() < 0.1);
        assert!((p100.y() - 0.0).abs() < 0.1);

        // At midpoint of second segment
        let p150 = path.position_at(150.0);
        assert!((p150.x() - 100.0).abs() < 0.1);
        assert!((p150.y() - 50.0).abs() < 0.1);

        // At end
        let p200 = path.position_at(200.0);
        assert!((p200.x() - 100.0).abs() < 0.1);
        assert!((p200.y() - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_parameterized_path_heading_at() {
        let bounds = Bounds::new(500.0, 500.0).unwrap();
        let waypoints: Arc<[Position]> = Arc::from(vec![
            Position::new(0.0, 0.0),
            Position::new(100.0, 0.0),   // Heading = 0 (east)
            Position::new(100.0, 100.0), // Heading = π/2 (north)
        ]);

        let path = ParameterizedPath::new(waypoints, bounds);

        // First segment (heading east = 0)
        let h0 = path.heading_at(50.0);
        assert!(h0.abs() < 0.1);

        // Second segment (heading north = π/2)
        let h150 = path.heading_at(150.0);
        assert!((h150 - std::f32::consts::FRAC_PI_2).abs() < 0.1);
    }

    #[test]
    fn test_parameterized_path_project_position() {
        let bounds = Bounds::new(500.0, 500.0).unwrap();
        let waypoints: Arc<[Position]> = Arc::from(vec![
            Position::new(0.0, 0.0),
            Position::new(100.0, 0.0),
            Position::new(100.0, 100.0),
        ]);

        let path = ParameterizedPath::new(waypoints, bounds);

        // Point on first segment
        let s1 = path.project_position(Position::new(50.0, 0.0));
        assert!((s1 - 50.0).abs() < 1.0);

        // Point slightly off first segment
        let s2 = path.project_position(Position::new(50.0, 5.0));
        assert!((s2 - 50.0).abs() < 1.0);

        // Point on second segment
        let s3 = path.project_position(Position::new(100.0, 50.0));
        assert!((s3 - 150.0).abs() < 1.0);
    }

    #[test]
    fn test_parameterized_path_empty() {
        let bounds = Bounds::new(500.0, 500.0).unwrap();
        let waypoints: Arc<[Position]> = Arc::from(vec![]);

        let path = ParameterizedPath::new(waypoints, bounds);

        assert!(path.is_empty());
        assert_eq!(path.len(), 0);
        assert_eq!(path.total_length(), 0.0);
    }

    #[test]
    fn test_parameterized_path_single_waypoint() {
        let bounds = Bounds::new(500.0, 500.0).unwrap();
        let waypoints: Arc<[Position]> = Arc::from(vec![Position::new(50.0, 50.0)]);

        let path = ParameterizedPath::new(waypoints, bounds);

        assert!(!path.is_empty());
        assert_eq!(path.len(), 1);
        assert_eq!(path.total_length(), 0.0);

        let pos = path.position_at(0.0);
        assert!((pos.x() - 50.0).abs() < 0.1);
    }
}
