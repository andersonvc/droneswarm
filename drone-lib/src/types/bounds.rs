use super::{Position, Vec2};
use crate::types::error::{DroneError, DroneResult};

/// World bounds for toroidal wrapping
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    width: f32,
    height: f32,
}

impl Bounds {
    /// Create new bounds with validation
    ///
    /// # Errors
    /// Returns `DroneError::InvalidBounds` if width or height is <= 0
    pub fn new(width: f32, height: f32) -> DroneResult<Bounds> {
        if width <= 0.0 || height <= 0.0 {
            return Err(DroneError::InvalidBounds { width, height });
        }
        Ok(Bounds { width, height })
    }

    /// Create bounds without validation (for internal use or when already validated)
    ///
    /// # Panics
    /// This function does not panic, but passing non-positive values will cause
    /// undefined behavior in wrap calculations (division by zero, infinite loops).
    #[inline]
    pub fn new_unchecked(width: f32, height: f32) -> Self {
        Bounds { width, height }
    }

    /// Get the width of the bounds
    #[inline]
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Get the height of the bounds
    #[inline]
    pub fn height(&self) -> f32 {
        self.height
    }

    /// Wrap a position to stay within bounds
    #[inline]
    pub fn wrap_position(&self, pos: Position) -> Position {
        Position::new(wrap(pos.x(), self.width), wrap(pos.y(), self.height))
    }

    /// Wrap a Vec2 to stay within bounds
    #[inline]
    pub fn wrap_vec2(&self, v: Vec2) -> Vec2 {
        Vec2::new(wrap(v.x, self.width), wrap(v.y, self.height))
    }

    /// Calculate toroidal (wrap-around) distance between two points
    #[inline]
    pub fn toroidal_distance(&self, from: Vec2, to: Vec2) -> f32 {
        self.toroidal_delta(from, to).magnitude()
    }

    /// Calculate the shortest delta vector between two points in toroidal space
    #[inline]
    pub fn toroidal_delta(&self, from: Vec2, to: Vec2) -> Vec2 {
        let dx = wrap_delta(to.x - from.x, self.width);
        let dy = wrap_delta(to.y - from.y, self.height);
        Vec2::new(dx, dy)
    }
}

/// Wrap a delta value to the range [-size/2, size/2) using branch-free arithmetic
#[inline]
fn wrap_delta(d: f32, size: f32) -> f32 {
    let half = size / 2.0;
    ((d + half).rem_euclid(size)) - half
}

impl TryFrom<(f32, f32)> for Bounds {
    type Error = DroneError;

    fn try_from(tuple: (f32, f32)) -> DroneResult<Self> {
        Bounds::new(tuple.0, tuple.1)
    }
}

/// Wrap value to stay within [0, max) using modulo arithmetic
#[inline]
fn wrap(value: f32, max: f32) -> f32 {
    ((value % max) + max) % max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_new_valid() {
        let bounds = Bounds::new(100.0, 200.0);
        assert!(bounds.is_ok());
        let b = bounds.unwrap();
        assert!((b.width() - 100.0).abs() < f32::EPSILON);
        assert!((b.height() - 200.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounds_new_invalid() {
        assert!(Bounds::new(0.0, 100.0).is_err());
        assert!(Bounds::new(100.0, 0.0).is_err());
        assert!(Bounds::new(-10.0, 100.0).is_err());
        assert!(Bounds::new(100.0, -10.0).is_err());
    }

    #[test]
    fn test_bounds_wrap() {
        let bounds = Bounds::new(100.0, 100.0).unwrap();

        let pos1 = Position::new(150.0, 50.0);
        let wrapped1 = bounds.wrap_position(pos1);
        assert!((wrapped1.x() - 50.0).abs() < 1e-6);

        let pos2 = Position::new(-10.0, 50.0);
        let wrapped2 = bounds.wrap_position(pos2);
        assert!((wrapped2.x() - 90.0).abs() < 1e-6);
    }

    #[test]
    fn test_wrap_extreme_values() {
        let bounds = Bounds::new(100.0, 100.0).unwrap();

        // Multiple wraps needed
        let pos = Position::new(350.0, -250.0);
        let wrapped = bounds.wrap_position(pos);
        assert!((wrapped.x() - 50.0).abs() < 1e-6);
        assert!((wrapped.y() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_toroidal_distance_no_wrap() {
        let bounds = Bounds::new(1000.0, 1000.0).unwrap();
        let from = Vec2::new(100.0, 100.0);
        let to = Vec2::new(200.0, 100.0);
        let dist = bounds.toroidal_distance(from, to);
        assert!((dist - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_toroidal_distance_wrap_x() {
        let bounds = Bounds::new(1000.0, 1000.0).unwrap();
        let from = Vec2::new(999.0, 500.0);
        let to = Vec2::new(1.0, 500.0);
        let dist = bounds.toroidal_distance(from, to);
        assert!((dist - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_toroidal_distance_wrap_y() {
        let bounds = Bounds::new(1000.0, 1000.0).unwrap();
        let from = Vec2::new(500.0, 999.0);
        let to = Vec2::new(500.0, 1.0);
        let dist = bounds.toroidal_distance(from, to);
        assert!((dist - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_toroidal_distance_wrap_both() {
        let bounds = Bounds::new(1000.0, 1000.0).unwrap();
        let from = Vec2::new(999.0, 999.0);
        let to = Vec2::new(1.0, 1.0);
        let dist = bounds.toroidal_distance(from, to);
        let expected = (8.0_f32).sqrt();
        assert!((dist - expected).abs() < 1e-6);
    }

    #[test]
    fn test_toroidal_delta_direction() {
        let bounds = Bounds::new(1000.0, 1000.0).unwrap();
        let from = Vec2::new(999.0, 500.0);
        let to = Vec2::new(1.0, 500.0);
        let delta = bounds.toroidal_delta(from, to);
        assert!((delta.x - 2.0).abs() < 1e-6);
        assert!((delta.y - 0.0).abs() < 1e-6);
    }
}
