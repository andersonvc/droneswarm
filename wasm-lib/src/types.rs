use drone_lib::{Position, Vec2};
use serde::{Deserialize, Serialize};

// ============================================================================
// Consensus Protocol
// ============================================================================

/// Consensus protocol for determining drone priority during collision avoidance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsensusProtocol {
    /// Lower drone ID = higher priority (original behavior)
    #[default]
    PriorityById,
    /// Drone closest to its waypoint = higher priority
    PriorityByWaypointDist,
}

// ============================================================================
// Type Definitions
// ============================================================================

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl From<Point> for Position {
    fn from(p: Point) -> Self {
        Position::new(p.x, p.y)
    }
}

impl From<Position> for Point {
    fn from(p: Position) -> Self {
        Point { x: p.x(), y: p.y() }
    }
}

impl From<Point> for Vec2 {
    fn from(p: Point) -> Self {
        Vec2::new(p.x, p.y)
    }
}

#[derive(Clone, Copy, Serialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DroneRenderData {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub heading: f32,
    pub color: Color,
    pub selected: bool,
    pub objective_type: String,
    pub target: Option<Point>,
    pub spline_path: Vec<Point>,
    /// Full route waypoints for the drone (closed loop).
    pub route_path: Vec<Point>,
    /// NLGL planning path to formation slot (for followers only).
    pub planning_path: Vec<Point>,
    /// Formation approach mode for color-coded visualization.
    /// One of: "none", "station_keeping", "correction", "pursuit", "approach"
    pub approach_mode: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SwarmStatus {
    pub simulation_time: f32,
    pub drone_count: u32,
    pub selected_count: u32,
    pub speed_multiplier: f32,
    pub is_valid: bool,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationConfig {
    pub drone_count: u32,
    pub spawn_pattern: SpawnPattern,
    pub bounds: Bounds,
    pub speed_multiplier: Option<f32>,
    /// World width in meters. Defaults to 500.0 if not specified.
    pub world_width_meters: Option<f32>,
    /// World height in meters. Defaults to 500.0 if not specified.
    pub world_height_meters: Option<f32>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpawnPattern {
    Grid,
    Random,
    Cluster { center: Point, radius: f32 },
    Custom { positions: Vec<Point> },
}

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct Bounds {
    pub width: f32,
    pub height: f32,
}
