use std::ops::{Add, AddAssign, Mul, Sub};

use super::{Bounds, Heading, Vec2};

/// Position in world space
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Position(Vec2);

impl Position {
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Position(Vec2::new(x, y))
    }

    #[inline]
    pub fn from_vec2(v: Vec2) -> Self {
        Position(v)
    }

    #[inline]
    pub fn x(self) -> f32 {
        self.0.x
    }

    #[inline]
    pub fn y(self) -> f32 {
        self.0.y
    }

    #[inline]
    pub fn as_vec2(self) -> Vec2 {
        self.0
    }

    #[inline]
    pub fn into_vec2(self) -> Vec2 {
        self.0
    }

    /// Distance to another position (Euclidean)
    #[inline]
    pub fn distance(self, other: Position) -> f32 {
        self.0.distance(other.0)
    }

    /// Toroidal distance to another position
    #[inline]
    pub fn toroidal_distance(self, other: Position, bounds: Bounds) -> f32 {
        bounds.toroidal_distance(self.0, other.0)
    }

    /// Direction heading to another position
    #[inline]
    pub fn heading_to(self, other: Position) -> Heading {
        Heading::from_direction(self.0, other.0)
    }

    /// Wrap position to stay within bounds
    #[inline]
    #[must_use]
    pub fn wrapped(self, bounds: Bounds) -> Position {
        bounds.wrap_position(self)
    }
}

impl Add<Velocity> for Position {
    type Output = Position;
    #[inline]
    fn add(self, rhs: Velocity) -> Position {
        Position(self.0 + rhs.as_vec2())
    }
}

impl AddAssign<Velocity> for Position {
    #[inline]
    fn add_assign(&mut self, rhs: Velocity) {
        self.0 += rhs.as_vec2();
    }
}

impl Sub for Position {
    type Output = Vec2;
    #[inline]
    fn sub(self, rhs: Position) -> Vec2 {
        self.0 - rhs.0
    }
}

impl From<Vec2> for Position {
    fn from(v: Vec2) -> Self {
        Position(v)
    }
}

impl From<Position> for Vec2 {
    fn from(p: Position) -> Self {
        p.0
    }
}

/// Velocity vector (units per second)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Velocity(Vec2);

impl Velocity {
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Velocity(Vec2::new(x, y))
    }

    #[inline]
    pub fn zero() -> Self {
        Velocity(Vec2::ZERO)
    }

    #[inline]
    pub fn from_vec2(v: Vec2) -> Self {
        Velocity(v)
    }

    #[inline]
    pub fn from_heading_and_speed(heading: Heading, speed: f32) -> Self {
        Velocity(heading.to_vec2() * speed)
    }

    #[inline]
    pub fn as_vec2(self) -> Vec2 {
        self.0
    }

    #[inline]
    pub fn into_vec2(self) -> Vec2 {
        self.0
    }

    /// Speed (magnitude of velocity)
    #[inline]
    pub fn speed(self) -> f32 {
        self.0.magnitude()
    }

    /// Squared speed (avoids sqrt)
    #[inline]
    pub fn speed_squared(self) -> f32 {
        self.0.magnitude_squared()
    }

    /// Direction of travel
    #[inline]
    pub fn heading(self) -> Heading {
        Heading::new(self.0.heading())
    }

    /// Scale velocity for time step (returns displacement)
    #[inline]
    #[must_use]
    pub fn scaled(self, dt: f32) -> Velocity {
        Velocity(self.0 * dt)
    }

    /// Clamp speed to maximum
    #[inline]
    #[must_use]
    pub fn clamp_speed(self, max_speed: f32) -> Velocity {
        Velocity(self.0.clamp_magnitude(max_speed))
    }
}

impl Add for Velocity {
    type Output = Velocity;
    #[inline]
    fn add(self, rhs: Velocity) -> Velocity {
        Velocity(self.0 + rhs.0)
    }
}

impl AddAssign for Velocity {
    #[inline]
    fn add_assign(&mut self, rhs: Velocity) {
        self.0 += rhs.0;
    }
}

impl Add<Acceleration> for Velocity {
    type Output = Velocity;
    #[inline]
    fn add(self, rhs: Acceleration) -> Velocity {
        Velocity(self.0 + rhs.as_vec2())
    }
}

impl AddAssign<Acceleration> for Velocity {
    #[inline]
    fn add_assign(&mut self, rhs: Acceleration) {
        self.0 += rhs.as_vec2();
    }
}

impl Sub for Velocity {
    type Output = Velocity;
    #[inline]
    fn sub(self, rhs: Velocity) -> Velocity {
        Velocity(self.0 - rhs.0)
    }
}

impl Mul<f32> for Velocity {
    type Output = Velocity;
    #[inline]
    fn mul(self, scalar: f32) -> Velocity {
        Velocity(self.0 * scalar)
    }
}

impl From<Vec2> for Velocity {
    fn from(v: Vec2) -> Self {
        Velocity(v)
    }
}

impl From<Velocity> for Vec2 {
    fn from(v: Velocity) -> Self {
        v.0
    }
}

/// Acceleration vector (units per second squared)
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Acceleration(Vec2);

impl Acceleration {
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Acceleration(Vec2::new(x, y))
    }

    #[inline]
    pub fn zero() -> Self {
        Acceleration(Vec2::ZERO)
    }

    #[inline]
    pub fn from_vec2(v: Vec2) -> Self {
        Acceleration(v)
    }

    #[inline]
    pub fn from_heading_and_magnitude(heading: Heading, magnitude: f32) -> Self {
        Acceleration(heading.to_vec2() * magnitude)
    }

    #[inline]
    pub fn as_vec2(self) -> Vec2 {
        self.0
    }

    #[inline]
    pub fn into_vec2(self) -> Vec2 {
        self.0
    }

    /// Magnitude of acceleration
    #[inline]
    pub fn magnitude(self) -> f32 {
        self.0.magnitude()
    }

    /// Scale acceleration for time step (returns velocity delta)
    #[inline]
    #[must_use]
    pub fn scaled(self, dt: f32) -> Velocity {
        Velocity::from_vec2(self.0 * dt)
    }
}

impl Add for Acceleration {
    type Output = Acceleration;
    #[inline]
    fn add(self, rhs: Acceleration) -> Acceleration {
        Acceleration(self.0 + rhs.0)
    }
}

impl Mul<f32> for Acceleration {
    type Output = Acceleration;
    #[inline]
    fn mul(self, scalar: f32) -> Acceleration {
        Acceleration(self.0 * scalar)
    }
}

impl From<Vec2> for Acceleration {
    fn from(v: Vec2) -> Self {
        Acceleration(v)
    }
}

impl From<Acceleration> for Vec2 {
    fn from(a: Acceleration) -> Self {
        a.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_add_velocity() {
        let pos = Position::new(10.0, 20.0);
        let vel = Velocity::new(1.0, 2.0);
        let new_pos = pos + vel;
        assert_eq!(new_pos, Position::new(11.0, 22.0));
    }

    #[test]
    fn test_velocity_add_acceleration() {
        let vel = Velocity::new(5.0, 5.0);
        let acc = Acceleration::new(1.0, 1.0);
        let new_vel = vel + acc;
        assert_eq!(new_vel, Velocity::new(6.0, 6.0));
    }

    #[test]
    fn test_velocity_from_heading() {
        let vel = Velocity::from_heading_and_speed(Heading::new(0.0), 10.0);
        assert!((vel.as_vec2().x - 10.0).abs() < 1e-6);
        assert!(vel.as_vec2().y.abs() < 1e-6);
    }

    #[test]
    fn test_acceleration_scaled() {
        let acc = Acceleration::new(10.0, 20.0);
        let vel_delta = acc.scaled(0.1);
        assert_eq!(vel_delta, Velocity::new(1.0, 2.0));
    }

    #[test]
    fn test_velocity_speed_squared() {
        let vel = Velocity::new(3.0, 4.0);
        assert!((vel.speed_squared() - 25.0).abs() < 1e-6);
        assert!((vel.speed() - 5.0).abs() < 1e-6);
    }
}
