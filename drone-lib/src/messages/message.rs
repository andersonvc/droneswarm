//! Message types for inter-drone communication.

use crate::types::Vec2;

/// Formation command variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormationCommand {
    /// Hold current formation positions.
    Hold,
    /// Advance formation in direction of travel.
    Advance,
    /// Disperse from current formation.
    Disperse,
    /// Tighten formation spacing.
    Contract,
    /// Loosen formation spacing.
    Expand,
}

/// Formation position assignment for a single drone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FormationSlot {
    /// Target position relative to formation center.
    pub offset: Vec2,
    /// Slot priority (lower = higher priority in update order).
    pub priority: u8,
}

impl FormationSlot {
    /// Create a new formation slot.
    pub fn new(offset: Vec2, priority: u8) -> Self {
        FormationSlot { offset, priority }
    }
}

/// Velocity consensus message from leader to followers.
///
/// Used for synchronized formation movement where all followers
/// aim to match the leader's velocity (with corrections for position).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityConsensus {
    /// Target formation velocity vector (m/s).
    pub target_velocity: Vec2,
    /// Target formation speed magnitude (m/s).
    pub target_speed: f32,
    /// Whether leader is actively moving (vs stopped at waypoint).
    pub is_moving: bool,
}

impl VelocityConsensus {
    /// Create a new velocity consensus from a velocity vector.
    pub fn new(velocity: Vec2, is_moving: bool) -> Self {
        VelocityConsensus {
            target_velocity: velocity,
            target_speed: velocity.magnitude(),
            is_moving,
        }
    }

    /// Create a stopped velocity consensus.
    pub fn stopped() -> Self {
        VelocityConsensus {
            target_velocity: Vec2::ZERO,
            target_speed: 0.0,
            is_moving: false,
        }
    }

    /// Scale the velocity consensus by a factor.
    pub fn scaled(&self, factor: f32) -> Self {
        VelocityConsensus {
            target_velocity: self.target_velocity * factor,
            target_speed: self.target_speed * factor,
            is_moving: self.is_moving,
        }
    }
}

impl Default for VelocityConsensus {
    fn default() -> Self {
        Self::stopped()
    }
}

/// Path progress state for synchronized formation following.
///
/// Tracks progress along a parameterized path using arc-length.
/// All drones in a formation share the same path progress to
/// maintain synchronized positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathProgress {
    /// Current arc-length parameter (meters along path).
    pub s: f32,
    /// Rate of change of s (m/s along path).
    pub s_dot: f32,
    /// Total path length for normalization.
    pub total_length: f32,
}

impl PathProgress {
    /// Create a new path progress.
    pub fn new(s: f32, s_dot: f32, total_length: f32) -> Self {
        PathProgress { s, s_dot, total_length }
    }

    /// Get normalized progress (0.0 to 1.0).
    pub fn normalized(&self) -> f32 {
        if self.total_length > f32::EPSILON {
            (self.s / self.total_length).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Check if at or past end of path.
    pub fn is_complete(&self) -> bool {
        self.s >= self.total_length - f32::EPSILON
    }
}

impl Default for PathProgress {
    fn default() -> Self {
        PathProgress {
            s: 0.0,
            s_dot: 0.0,
            total_length: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formation_slot_new() {
        let slot = FormationSlot::new(Vec2::new(10.0, -5.0), 2);
        assert_eq!(slot.offset.x, 10.0);
        assert_eq!(slot.offset.y, -5.0);
        assert_eq!(slot.priority, 2);
    }

    #[test]
    fn test_formation_command_equality() {
        assert_eq!(FormationCommand::Hold, FormationCommand::Hold);
        assert_ne!(FormationCommand::Hold, FormationCommand::Advance);
    }

    #[test]
    fn test_velocity_consensus_new() {
        let vel = Vec2::new(5.0, 0.0);
        let consensus = VelocityConsensus::new(vel, true);
        assert_eq!(consensus.target_velocity.x, 5.0);
        assert_eq!(consensus.target_velocity.y, 0.0);
        assert_eq!(consensus.target_speed, 5.0);
        assert!(consensus.is_moving);
    }

    #[test]
    fn test_velocity_consensus_stopped() {
        let consensus = VelocityConsensus::stopped();
        assert_eq!(consensus.target_speed, 0.0);
        assert!(!consensus.is_moving);
    }

    #[test]
    fn test_velocity_consensus_scaled() {
        let consensus = VelocityConsensus::new(Vec2::new(10.0, 0.0), true);
        let scaled = consensus.scaled(0.5);
        assert_eq!(scaled.target_speed, 5.0);
        assert_eq!(scaled.target_velocity.x, 5.0);
        assert!(scaled.is_moving); // is_moving preserved
    }

    #[test]
    fn test_path_progress_new() {
        let progress = PathProgress::new(50.0, 5.0, 100.0);
        assert_eq!(progress.s, 50.0);
        assert_eq!(progress.s_dot, 5.0);
        assert_eq!(progress.total_length, 100.0);
    }

    #[test]
    fn test_path_progress_normalized() {
        let progress = PathProgress::new(25.0, 5.0, 100.0);
        assert!((progress.normalized() - 0.25).abs() < 0.001);

        let at_end = PathProgress::new(100.0, 0.0, 100.0);
        assert!((at_end.normalized() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_path_progress_is_complete() {
        let in_progress = PathProgress::new(50.0, 5.0, 100.0);
        assert!(!in_progress.is_complete());

        let complete = PathProgress::new(100.0, 0.0, 100.0);
        assert!(complete.is_complete());
    }
}
