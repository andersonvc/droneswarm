/// Shared game constants used by both sim_runner and wasm-lib.
#[derive(Debug, Clone)]
pub struct GameConfig {
    /// Half-length of drone model in meters.
    pub drone_length_meters: f32,
    /// Blast radius for detonations (drone_length_meters * 5).
    pub detonation_radius: f32,
    /// Distance below which two drones collide.
    pub collision_distance: f32,
    /// Distance from blast epicenter within which a target is hit.
    pub target_hit_radius: f32,
    /// Standoff distance for patrol routes around target centroid.
    pub patrol_standoff: f32,
    /// Multiplier for threat detection radius (detonation_radius * this).
    pub threat_radius_multiplier: f32,
    /// Max nearby threats for observation normalization.
    pub max_nearby_threats: f32,
}

impl Default for GameConfig {
    fn default() -> Self {
        let drone_length_meters = 37.5;
        let detonation_radius = drone_length_meters * 5.0; // 187.5
        GameConfig {
            drone_length_meters,
            detonation_radius,
            collision_distance: 1.0,
            target_hit_radius: detonation_radius + drone_length_meters, // 225.0
            patrol_standoff: 200.0,
            threat_radius_multiplier: 5.0,
            max_nearby_threats: 20.0,
        }
    }
}
