pub mod behaviors;
pub mod models {
    pub mod drone;
    pub mod fixed_wing;
}
pub mod types;

pub use behaviors::{
    calculate_separation, calculate_velocity_obstacle, get_recommended_heading, AvoidanceResult,
    SeparationConfig, VelocityObstacleConfig, COLLISION_RADIUS,
};
pub use models::drone::Drone;
pub use models::fixed_wing::FixedWing;
pub use types::{
    // Core types
    Acceleration, Bounds, Heading, Position, Vec2, Velocity,
    // State types
    DroneInfo, DronePerfFeatures, Objective, ObjectiveType, State,
    // Error types
    DroneError, DroneResult,
    // Utility functions
    normalize_angle,
};
