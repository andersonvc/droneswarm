use std::f32::consts::{PI, TAU};
use std::ops::{Add, AddAssign, Sub};

use super::Vec2;

/// Normalize an angle to the range [-PI, PI] using modulo arithmetic
#[inline]
pub fn normalize_angle(angle: f32) -> f32 {
    let mut a = angle % TAU;
    if a > PI {
        a -= TAU;
    } else if a < -PI {
        a += TAU;
    }
    a
}

/// Angle in radians, always normalized to [-PI, PI]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Heading(f32);

impl Heading {
    /// Create a new heading, automatically normalizing to [-PI, PI]
    #[inline]
    pub fn new(radians: f32) -> Self {
        Heading(normalize_angle(radians))
    }

    /// Create heading from degrees
    #[inline]
    pub fn from_degrees(degrees: f32) -> Self {
        Heading::new(degrees.to_radians())
    }

    /// Create heading pointing from one position to another
    #[inline]
    pub fn from_direction(from: Vec2, to: Vec2) -> Self {
        Heading::new((to - from).heading())
    }

    /// Get the angle in radians
    #[inline]
    pub fn radians(self) -> f32 {
        self.0
    }

    /// Get the angle in degrees
    #[inline]
    pub fn degrees(self) -> f32 {
        self.0.to_degrees()
    }

    /// Convert to a unit direction vector
    #[inline]
    pub fn to_vec2(self) -> Vec2 {
        Vec2::from_angle(self.0)
    }

    /// Shortest angular difference to another heading (signed)
    #[inline]
    pub fn difference(self, other: Heading) -> f32 {
        normalize_angle(other.0 - self.0)
    }

    /// Absolute angular difference (always positive)
    #[inline]
    pub fn abs_difference(self, other: Heading) -> f32 {
        self.difference(other).abs()
    }

    /// Rotate by an angle (radians)
    #[inline]
    #[must_use]
    pub fn rotated(self, angle: f32) -> Heading {
        Heading::new(self.0 + angle)
    }

    /// Linear interpolation towards another heading (takes shortest path)
    #[inline]
    #[must_use]
    pub fn lerp(self, target: Heading, t: f32) -> Heading {
        let diff = self.difference(target);
        Heading::new(self.0 + diff * t)
    }

    /// Move towards target heading by at most max_delta radians
    #[inline]
    #[must_use]
    pub fn move_towards(self, target: Heading, max_delta: f32) -> Heading {
        let diff = self.difference(target);
        let clamped = diff.clamp(-max_delta, max_delta);
        Heading::new(self.0 + clamped)
    }
}

impl Default for Heading {
    fn default() -> Self {
        Heading(0.0)
    }
}

impl From<f32> for Heading {
    fn from(radians: f32) -> Self {
        Heading::new(radians)
    }
}

impl Add<f32> for Heading {
    type Output = Heading;
    #[inline]
    fn add(self, rhs: f32) -> Heading {
        Heading::new(self.0 + rhs)
    }
}

impl Sub<f32> for Heading {
    type Output = Heading;
    #[inline]
    fn sub(self, rhs: f32) -> Heading {
        Heading::new(self.0 - rhs)
    }
}

impl AddAssign<f32> for Heading {
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        *self = Heading::new(self.0 + rhs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_angle_positive() {
        let angle = 3.0 * PI;
        let normalized = normalize_angle(angle);
        assert!((normalized - PI).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_angle_negative() {
        let angle = -3.0 * PI;
        let normalized = normalize_angle(angle);
        assert!((normalized - (-PI)).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_angle_already_normalized() {
        let angle = PI / 4.0;
        let normalized = normalize_angle(angle);
        assert!((normalized - angle).abs() < 1e-6);
    }

    #[test]
    fn test_heading_normalization() {
        let h = Heading::new(3.0 * PI);
        assert!((h.radians() - PI).abs() < 1e-6);

        let h2 = Heading::new(-3.0 * PI);
        assert!((h2.radians() - (-PI)).abs() < 1e-6);
    }

    #[test]
    fn test_heading_difference() {
        let h1 = Heading::new(0.0);
        let h2 = Heading::new(PI / 2.0);
        assert!((h1.difference(h2) - PI / 2.0).abs() < 1e-6);

        // Test wrap-around
        let h3 = Heading::new(PI - 0.1);
        let h4 = Heading::new(-PI + 0.1);
        assert!((h3.difference(h4).abs() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_heading_to_vec2() {
        let h = Heading::new(0.0);
        let v = h.to_vec2();
        assert!((v.x - 1.0).abs() < 1e-6);
        assert!(v.y.abs() < 1e-6);

        let h2 = Heading::new(PI / 2.0);
        let v2 = h2.to_vec2();
        assert!(v2.x.abs() < 1e-6);
        assert!((v2.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_heading_move_towards() {
        let h1 = Heading::new(0.0);
        let h2 = Heading::new(1.0);
        let moved = h1.move_towards(h2, 0.5);
        assert!((moved.radians() - 0.5).abs() < 1e-6);
    }
}
