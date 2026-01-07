use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// 2D vector with named x/y components.
///
/// This is intentionally a transparent value type with public fields,
/// similar to how `f32` itself is used. Unlike semantic newtypes like
/// `Position` or `Velocity`, `Vec2` is a general-purpose math primitive
/// where direct field access is expected and encouraged.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };
    pub const UNIT_X: Vec2 = Vec2 { x: 1.0, y: 0.0 };
    pub const UNIT_Y: Vec2 = Vec2 { x: 0.0, y: 1.0 };

    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }

    #[inline]
    pub fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    pub fn splat(v: f32) -> Self {
        Vec2 { x: v, y: v }
    }

    /// Magnitude (length) of the vector
    #[inline]
    pub fn magnitude(self) -> f32 {
        self.magnitude_squared().sqrt()
    }

    /// Squared magnitude (avoids sqrt for comparisons)
    #[inline]
    pub fn magnitude_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Calculate heading angle from this vector (atan2)
    #[inline]
    pub fn heading(self) -> f32 {
        self.y.atan2(self.x)
    }

    /// Create a unit vector from an angle (radians)
    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        Vec2 {
            x: angle.cos(),
            y: angle.sin(),
        }
    }

    /// Dot product
    #[inline]
    pub fn dot(self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Cross product (returns scalar z-component for 2D)
    #[inline]
    pub fn cross(self, other: Vec2) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Normalize to unit vector (returns zero vector if magnitude is zero)
    #[inline]
    #[must_use]
    pub fn normalized(self) -> Self {
        let mag = self.magnitude();
        if mag > f32::EPSILON {
            self / mag
        } else {
            Self::ZERO
        }
    }

    /// Try to normalize, returning None if magnitude is too small
    #[inline]
    pub fn try_normalized(self) -> Option<Self> {
        let mag = self.magnitude();
        if mag > f32::EPSILON {
            Some(self / mag)
        } else {
            None
        }
    }

    /// Scale to specific length
    #[inline]
    #[must_use]
    pub fn with_magnitude(self, length: f32) -> Self {
        self.normalized() * length
    }

    /// Rotate by angle (radians)
    #[inline]
    #[must_use]
    pub fn rotated(self, angle: f32) -> Self {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        Vec2::new(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )
    }

    /// Perpendicular vector (90° counterclockwise)
    #[inline]
    #[must_use]
    pub fn perp(self) -> Self {
        Vec2::new(-self.y, self.x)
    }

    /// Perpendicular vector (90° clockwise)
    #[inline]
    #[must_use]
    pub fn perp_cw(self) -> Self {
        Vec2::new(self.y, -self.x)
    }

    /// Linear interpolation between self and other
    #[inline]
    #[must_use]
    pub fn lerp(self, other: Vec2, t: f32) -> Self {
        self + (other - self) * t
    }

    /// Distance to another point
    #[inline]
    pub fn distance(self, other: Vec2) -> f32 {
        (other - self).magnitude()
    }

    /// Squared distance to another point
    #[inline]
    pub fn distance_squared(self, other: Vec2) -> f32 {
        (other - self).magnitude_squared()
    }

    /// Angle between two vectors (radians)
    #[inline]
    pub fn angle_to(self, other: Vec2) -> f32 {
        let cross = self.cross(other);
        let dot = self.dot(other);
        cross.atan2(dot)
    }

    /// Project this vector onto another
    #[inline]
    #[must_use]
    pub fn project_onto(self, other: Vec2) -> Self {
        let other_mag_sq = other.magnitude_squared();
        if other_mag_sq > f32::EPSILON {
            other * (self.dot(other) / other_mag_sq)
        } else {
            Self::ZERO
        }
    }

    /// Reflect this vector about a normal
    #[inline]
    #[must_use]
    pub fn reflect(self, normal: Vec2) -> Self {
        self - normal * 2.0 * self.dot(normal)
    }

    /// Clamp magnitude to max value
    #[inline]
    #[must_use]
    pub fn clamp_magnitude(self, max: f32) -> Self {
        let mag = self.magnitude();
        if mag > max {
            self * (max / mag)
        } else {
            self
        }
    }

    /// Component-wise absolute value
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        Vec2::new(self.x.abs(), self.y.abs())
    }

    /// Component-wise minimum
    #[inline]
    #[must_use]
    pub fn min(self, other: Vec2) -> Self {
        Vec2::new(self.x.min(other.x), self.y.min(other.y))
    }

    /// Component-wise maximum
    #[inline]
    #[must_use]
    pub fn max(self, other: Vec2) -> Self {
        Vec2::new(self.x.max(other.x), self.y.max(other.y))
    }

    /// Approximate equality check with epsilon tolerance
    #[inline]
    pub fn approx_eq(self, other: Vec2, epsilon: f32) -> bool {
        (self.x - other.x).abs() < epsilon && (self.y - other.y).abs() < epsilon
    }
}

// Operator implementations
impl Add for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Vec2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl AddAssign for Vec2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Vec2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl SubAssign for Vec2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Vec2::new(self.x * scalar, self.y * scalar)
    }
}

impl Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline]
    fn mul(self, vec: Vec2) -> Vec2 {
        Vec2::new(self * vec.x, self * vec.y)
    }
}

impl MulAssign<f32> for Vec2 {
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        self.x *= scalar;
        self.y *= scalar;
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f32) -> Self {
        Vec2::new(self.x / scalar, self.y / scalar)
    }
}

impl DivAssign<f32> for Vec2 {
    #[inline]
    fn div_assign(&mut self, scalar: f32) {
        self.x /= scalar;
        self.y /= scalar;
    }
}

impl Neg for Vec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Vec2::new(-self.x, -self.y)
    }
}

impl From<(f32, f32)> for Vec2 {
    fn from(tuple: (f32, f32)) -> Self {
        Vec2 {
            x: tuple.0,
            y: tuple.1,
        }
    }
}

impl From<Vec2> for (f32, f32) {
    fn from(v: Vec2) -> Self {
        (v.x, v.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_vec2_magnitude() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.magnitude() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec2_from_tuple() {
        let v: Vec2 = (3.0, 4.0).into();
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 4.0);
    }

    #[test]
    fn test_vec2_operators() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);

        let sum = a + b;
        assert_eq!(sum, Vec2::new(4.0, 6.0));

        let diff = b - a;
        assert_eq!(diff, Vec2::new(2.0, 2.0));

        let scaled = a * 2.0;
        assert_eq!(scaled, Vec2::new(2.0, 4.0));

        let divided = b / 2.0;
        assert_eq!(divided, Vec2::new(1.5, 2.0));

        let negated = -a;
        assert_eq!(negated, Vec2::new(-1.0, -2.0));
    }

    #[test]
    fn test_vec2_dot() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!((a.dot(b)).abs() < 1e-6); // Perpendicular = 0

        let c = Vec2::new(1.0, 0.0);
        assert!((a.dot(c) - 1.0).abs() < 1e-6); // Parallel = 1
    }

    #[test]
    fn test_vec2_normalized() {
        let v = Vec2::new(3.0, 4.0);
        let n = v.normalized();
        assert!((n.magnitude() - 1.0).abs() < 1e-6);
        assert!((n.x - 0.6).abs() < 1e-6);
        assert!((n.y - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_vec2_rotated() {
        let v = Vec2::new(1.0, 0.0);
        let rotated = v.rotated(PI / 2.0);
        assert!(rotated.x.abs() < 1e-6);
        assert!((rotated.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec2_perp() {
        let v = Vec2::new(1.0, 0.0);
        let perp = v.perp();
        assert!(perp.x.abs() < 1e-6);
        assert!((perp.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec2_lerp() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 20.0);
        let mid = a.lerp(b, 0.5);
        assert_eq!(mid, Vec2::new(5.0, 10.0));
    }

    #[test]
    fn test_vec2_approx_eq() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(1.0000001, 2.0000001);
        assert!(a.approx_eq(b, 1e-6));
        assert!(!a.approx_eq(Vec2::new(1.1, 2.0), 1e-6));
    }
}
