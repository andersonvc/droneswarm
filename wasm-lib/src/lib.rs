use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;

use drone_lib::models::drone::Drone;
use drone_lib::models::fixed_wing::FixedWing;
use drone_lib::{
    Bounds as LibBounds, DroneInfo, DronePerfFeatures, Heading, Objective, ObjectiveType, Position,
    Vec2, VelocityObstacleConfig,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

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
// Panic Hook Setup
// ============================================================================

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
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

impl Bounds {
    fn to_lib_bounds(self) -> Result<LibBounds, String> {
        LibBounds::new(self.width, self.height)
            .map_err(|e| format!("Invalid bounds: {}", e))
    }
}

// ============================================================================
// Internal Structs
// ============================================================================

struct DroneState {
    id: u32,
    fixed_wing: FixedWing,
    color: Color,
}

pub struct Swarm {
    drones: Vec<DroneState>,
    lib_bounds: LibBounds,
    simulation_time: f32,
    speed_multiplier: f32,
    selected_ids: HashSet<u32>,
    consensus_protocol: ConsensusProtocol,
}

impl Swarm {
    pub fn new(config: SimulationConfig) -> Result<Self, String> {
        let lib_bounds = config.bounds.to_lib_bounds()?;
        let positions = Self::generate_spawn_positions(&config);
        let drone_count = positions.len();

        // Generate random headings using LCG
        let mut hdg_seed: u32 = 98765;
        let drones = positions
            .into_iter()
            .enumerate()
            .map(|(i, pos)| {
                hdg_seed = hdg_seed.wrapping_mul(1103515245).wrapping_add(12345);
                let hdg = (hdg_seed as f32 / u32::MAX as f32) * std::f32::consts::TAU
                    - std::f32::consts::PI;

                DroneState {
                    id: i as u32,
                    fixed_wing: FixedWing::new(
                        i,
                        Position::new(pos.x, pos.y),
                        Heading::new(hdg),
                        lib_bounds,
                    ),
                    color: Self::generate_color(i, drone_count),
                }
            })
            .collect();

        Ok(Swarm {
            drones,
            lib_bounds,
            simulation_time: 0.0,
            speed_multiplier: config.speed_multiplier.unwrap_or(1.0),
            selected_ids: HashSet::new(),
            consensus_protocol: ConsensusProtocol::default(),
        })
    }

    // ========================================================================
    // Spawn Pattern Generation
    // ========================================================================

    fn generate_spawn_positions(config: &SimulationConfig) -> Vec<Point> {
        match &config.spawn_pattern {
            SpawnPattern::Grid => Self::spawn_grid(config.drone_count, &config.bounds),
            SpawnPattern::Random => Self::spawn_random(config.drone_count, &config.bounds),
            SpawnPattern::Cluster { center, radius } => {
                Self::spawn_cluster(config.drone_count, center, *radius)
            }
            SpawnPattern::Custom { positions } => positions.clone(),
        }
    }

    fn spawn_grid(count: u32, bounds: &Bounds) -> Vec<Point> {
        let cols = (count as f32).sqrt().ceil() as u32;
        let rows = count.div_ceil(cols);
        let spacing_x = bounds.width / (cols + 1) as f32;
        let spacing_y = bounds.height / (rows + 1) as f32;

        (0..count)
            .map(|i| {
                let col = i % cols;
                let row = i / cols;
                Point {
                    x: spacing_x * (col + 1) as f32,
                    y: spacing_y * (row + 1) as f32,
                }
            })
            .collect()
    }

    fn spawn_random(count: u32, bounds: &Bounds) -> Vec<Point> {
        // Simple LCG for deterministic "random" positions
        let mut seed: u32 = 12345;
        (0..count)
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let x = (seed as f32 / u32::MAX as f32) * bounds.width;
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let y = (seed as f32 / u32::MAX as f32) * bounds.height;
                Point { x, y }
            })
            .collect()
    }

    fn spawn_cluster(count: u32, center: &Point, radius: f32) -> Vec<Point> {
        let mut seed: u32 = 54321;
        (0..count)
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let angle = (seed as f32 / u32::MAX as f32) * std::f32::consts::TAU;
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let r = (seed as f32 / u32::MAX as f32).sqrt() * radius;
                Point {
                    x: center.x + r * angle.cos(),
                    y: center.y + r * angle.sin(),
                }
            })
            .collect()
    }

    // ========================================================================
    // Color Generation
    // ========================================================================

    fn generate_color(index: usize, total: usize) -> Color {
        let hue = (index as f32 / total as f32) * 360.0;
        let (r, g, b) = hsl_to_rgb(hue, 0.7, 0.5);
        Color { r, g, b }
    }

    // ========================================================================
    // Physics Update
    // ========================================================================

    pub fn tick(&mut self, dt: f32) {
        const COLLISION_DISTANCE: f32 = 30.0; // 1 drone length

        let effective_dt = dt * self.speed_multiplier;

        // Priority-Based Consensus: higher priority drones plan first
        // Lower priority drones see their updated state

        // Sort drone indices based on consensus protocol
        let mut indices: Vec<usize> = (0..self.drones.len()).collect();

        match self.consensus_protocol {
            ConsensusProtocol::PriorityById => {
                // Lower ID = higher priority
                indices.sort_by_key(|&i| self.drones[i].id);
            }
            ConsensusProtocol::PriorityByWaypointDist => {
                // Closer to waypoint = higher priority (sort ascending by distance)
                indices.sort_by(|&a, &b| {
                    let dist_a = self.distance_to_waypoint(a);
                    let dist_b = self.distance_to_waypoint(b);
                    dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // Initialize swarm info with current state
        let mut swarm_info: Vec<DroneInfo> = self
            .drones
            .iter()
            .map(|d| d.fixed_wing.get_info())
            .collect();

        // Process drones in priority order
        for &idx in &indices {
            // Update this drone using current swarm info
            // (higher priority drones have already updated their entries)
            self.drones[idx].fixed_wing.state_update(effective_dt, &swarm_info);

            // Update swarm info with this drone's new state
            // so lower priority drones will see where we're going
            swarm_info[idx] = self.drones[idx].fixed_wing.get_info();
        }

        // Collision detection - find all colliding drone pairs
        let mut destroyed_ids: HashSet<u32> = HashSet::new();

        for i in 0..self.drones.len() {
            for j in (i + 1)..self.drones.len() {
                let pos_i = self.drones[i].fixed_wing.state().pos;
                let pos_j = self.drones[j].fixed_wing.state().pos;

                // Use distance from bounds
                let dist = self.lib_bounds.distance(pos_i.as_vec2(), pos_j.as_vec2());

                if dist < COLLISION_DISTANCE {
                    let id_i = self.drones[i].id;
                    let id_j = self.drones[j].id;

                    // Log collision
                    web_sys::console::info_1(
                        &format!("collision - Drone {} & Drone {}", id_i, id_j).into(),
                    );

                    destroyed_ids.insert(id_i);
                    destroyed_ids.insert(id_j);
                }
            }
        }

        // Remove destroyed drones
        if !destroyed_ids.is_empty() {
            self.drones.retain(|d| !destroyed_ids.contains(&d.id));
            // Also remove from selection
            self.selected_ids.retain(|id| !destroyed_ids.contains(id));
        }

        self.simulation_time += effective_dt;
    }

    // ========================================================================
    // State Queries
    // ========================================================================

    pub fn get_render_state(&self) -> Vec<DroneRenderData> {
        self.drones
            .iter()
            .map(|d| {
                let state = d.fixed_wing.state();
                let objective = d.fixed_wing.objective();

                // Get spline path (20 points for smooth visualization)
                let spline_path: Vec<Point> = d
                    .fixed_wing
                    .get_spline_path(20)
                    .into_iter()
                    .map(|v| Point { x: v.x, y: v.y })
                    .collect();

                DroneRenderData {
                    id: d.id,
                    x: state.pos.x(),
                    y: state.pos.y(),
                    heading: state.hdg.radians(),
                    color: d.color,
                    selected: self.selected_ids.contains(&d.id),
                    objective_type: format!("{:?}", objective.task),
                    target: objective.waypoints.front().map(|&p| p.into()),
                    spline_path,
                }
            })
            .collect()
    }

    pub fn get_status(&self) -> SwarmStatus {
        SwarmStatus {
            simulation_time: self.simulation_time,
            drone_count: self.drones.len() as u32,
            selected_count: self.selected_ids.len() as u32,
            speed_multiplier: self.speed_multiplier,
            is_valid: true,
        }
    }

    pub fn get_drone_at(&self, x: f32, y: f32, hit_radius: f32) -> Option<u32> {
        let hit_radius_sq = hit_radius * hit_radius;
        for drone in &self.drones {
            let state = drone.fixed_wing.state();
            let dx = state.pos.x() - x;
            let dy = state.pos.y() - y;
            if dx * dx + dy * dy <= hit_radius_sq {
                return Some(drone.id);
            }
        }
        None
    }

    // ========================================================================
    // Selection Management
    // ========================================================================

    pub fn select_drone(&mut self, id: u32, multi_select: bool) {
        if !multi_select {
            self.selected_ids.clear();
        }

        if self.drones.iter().any(|d| d.id == id) {
            if self.selected_ids.contains(&id) && multi_select {
                self.selected_ids.remove(&id); // Toggle off
            } else {
                self.selected_ids.insert(id);
            }
        }
    }

    pub fn clear_selection(&mut self) {
        self.selected_ids.clear();
    }

    // ========================================================================
    // Waypoint Assignment
    // ========================================================================

    pub fn assign_waypoint(&mut self, x: f32, y: f32) {
        let mut waypoints = VecDeque::new();
        waypoints.push_back(Position::new(x, y));

        for drone in &mut self.drones {
            if self.selected_ids.contains(&drone.id) {
                let objective = Objective {
                    task: ObjectiveType::ReachWaypoint,
                    waypoints: waypoints.clone(),
                    route: None,
                    targets: None,
                };
                drone.fixed_wing.set_objective(objective);
            }
        }
    }

    pub fn assign_waypoint_all(&mut self, x: f32, y: f32) {
        let mut waypoints = VecDeque::new();
        waypoints.push_back(Position::new(x, y));

        for drone in &mut self.drones {
            let objective = Objective {
                task: ObjectiveType::ReachWaypoint,
                waypoints: waypoints.clone(),
                route: None,
                targets: None,
            };
            drone.fixed_wing.set_objective(objective);
        }
    }

    pub fn assign_path(&mut self, waypoints: Vec<Point>) {
        let waypoint_deque: VecDeque<Position> =
            waypoints.iter().map(|p| Position::new(p.x, p.y)).collect();

        for drone in &mut self.drones {
            if self.selected_ids.contains(&drone.id) {
                let objective = Objective {
                    task: ObjectiveType::ReachWaypoint,
                    waypoints: waypoint_deque.clone(),
                    route: None,
                    targets: None,
                };
                drone.fixed_wing.set_objective(objective);
            }
        }
    }

    pub fn assign_route_all(&mut self, waypoints: Vec<Point>) {
        let waypoint_vec: Vec<Position> =
            waypoints.iter().map(|p| Position::new(p.x, p.y)).collect();
        let waypoint_deque: VecDeque<Position> = waypoint_vec.iter().copied().collect();
        let route: Arc<[Position]> = Arc::from(waypoint_vec);

        for drone in &mut self.drones {
            let objective = Objective {
                task: ObjectiveType::FollowRoute,
                waypoints: waypoint_deque.clone(),
                route: Some(route.clone()), // Shared route for all drones
                targets: None,
            };
            drone.fixed_wing.set_objective(objective);
        }
    }

    pub fn set_speed(&mut self, multiplier: f32) {
        self.speed_multiplier = multiplier.clamp(0.25, 4.0);
    }

    pub fn set_flight_params(
        &mut self,
        max_velocity: f32,
        max_acceleration: f32,
        max_turn_rate: f32,
    ) {
        let params = DronePerfFeatures::new_unchecked(max_velocity, max_acceleration, max_turn_rate);
        for drone in &mut self.drones {
            drone.fixed_wing.set_flight_params(params);
        }
    }

    pub fn set_avoidance_lookahead(&mut self, lookahead_time: f32) {
        let lookahead = lookahead_time.clamp(0.1, 2.0);
        for drone in &mut self.drones {
            let mut config = *drone.fixed_wing.velocity_obstacle_config();
            config.lookahead_time = lookahead;
            drone.fixed_wing.set_velocity_obstacle_config(config);
        }
    }

    pub fn set_vo_config(
        &mut self,
        lookahead_time: f32,
        time_samples: usize,
        safe_distance: f32,
        detection_range: f32,
        avoidance_weight: f32,
    ) {
        for drone in &mut self.drones {
            let mut config = *drone.fixed_wing.velocity_obstacle_config();
            config.lookahead_time = lookahead_time.clamp(0.1, 3.0);
            config.time_samples = time_samples.clamp(5, 30);
            config.safe_distance = safe_distance.clamp(30.0, 150.0);
            config.detection_range = detection_range.clamp(50.0, 300.0);
            config.avoidance_weight = avoidance_weight.clamp(0.0, 1.0);
            drone.fixed_wing.set_velocity_obstacle_config(config);
        }
    }

    pub fn set_waypoint_clearance(&mut self, clearance: f32) {
        let clearance = clearance.clamp(1.0, 200.0);
        for drone in &mut self.drones {
            drone.fixed_wing.set_waypoint_clearance(clearance);
        }
    }

    pub fn set_consensus_protocol(&mut self, protocol: ConsensusProtocol) {
        self.consensus_protocol = protocol;
    }

    /// Helper: get distance from drone to its next waypoint (or MAX if none)
    fn distance_to_waypoint(&self, drone_idx: usize) -> f32 {
        let drone = &self.drones[drone_idx];
        let objective = drone.fixed_wing.objective();

        if let Some(waypoint) = objective.waypoints.front() {
            let pos = drone.fixed_wing.state().pos;
            self.lib_bounds.distance(pos.as_vec2(), waypoint.as_vec2())
        } else {
            f32::MAX // No waypoint = lowest priority
        }
    }
}

// ============================================================================
// HSL to RGB Conversion
// ============================================================================

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let m = l - c / 2.0;

    let (r, g, b) = match h_prime as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

// ============================================================================
// WASM Exports
// ============================================================================

#[wasm_bindgen]
pub struct SwarmHandle {
    swarm: Swarm,
}

#[wasm_bindgen]
impl SwarmHandle {
    pub fn free(self) {
        // Rust ownership handles cleanup
    }
}

#[wasm_bindgen]
pub fn init_swarm(config: JsValue) -> Result<SwarmHandle, JsValue> {
    let config: SimulationConfig = serde_wasm_bindgen::from_value(config)
        .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

    let swarm = Swarm::new(config)
        .map_err(|e| JsValue::from_str(&e))?;

    Ok(SwarmHandle { swarm })
}

#[wasm_bindgen]
pub fn tick(handle: &mut SwarmHandle, dt: f32) {
    handle.swarm.tick(dt);
}

#[wasm_bindgen]
pub fn get_render_state(handle: &SwarmHandle) -> JsValue {
    let state = handle.swarm.get_render_state();
    serde_wasm_bindgen::to_value(&state).unwrap_or(JsValue::NULL)
}

#[wasm_bindgen]
pub fn get_status(handle: &SwarmHandle) -> JsValue {
    let status = handle.swarm.get_status();
    serde_wasm_bindgen::to_value(&status).unwrap_or(JsValue::NULL)
}

#[wasm_bindgen]
pub fn select_drone(handle: &mut SwarmHandle, id: u32, multi_select: bool) {
    handle.swarm.select_drone(id, multi_select);
}

#[wasm_bindgen]
pub fn clear_selection(handle: &mut SwarmHandle) {
    handle.swarm.clear_selection();
}

#[wasm_bindgen]
pub fn set_speed(handle: &mut SwarmHandle, multiplier: f32) {
    handle.swarm.set_speed(multiplier);
}

#[wasm_bindgen]
pub fn assign_waypoint(handle: &mut SwarmHandle, x: f32, y: f32) {
    handle.swarm.assign_waypoint(x, y);
}

#[wasm_bindgen]
pub fn assign_path(handle: &mut SwarmHandle, waypoints: JsValue) -> Result<(), JsValue> {
    let waypoints: Vec<Point> = serde_wasm_bindgen::from_value(waypoints)
        .map_err(|e| JsValue::from_str(&format!("Invalid waypoints: {}", e)))?;
    handle.swarm.assign_path(waypoints);
    Ok(())
}

#[wasm_bindgen]
pub fn assign_waypoint_all(handle: &mut SwarmHandle, x: f32, y: f32) {
    handle.swarm.assign_waypoint_all(x, y);
}

#[wasm_bindgen]
pub fn assign_route_all(handle: &mut SwarmHandle, waypoints: JsValue) -> Result<(), JsValue> {
    let waypoints: Vec<Point> = serde_wasm_bindgen::from_value(waypoints)
        .map_err(|e| JsValue::from_str(&format!("Invalid waypoints: {}", e)))?;
    handle.swarm.assign_route_all(waypoints);
    Ok(())
}

#[wasm_bindgen]
pub fn get_drone_at(handle: &SwarmHandle, x: f32, y: f32, hit_radius: f32) -> Option<u32> {
    handle.swarm.get_drone_at(x, y, hit_radius)
}

#[wasm_bindgen]
pub fn set_flight_params(
    handle: &mut SwarmHandle,
    max_velocity: f32,
    max_acceleration: f32,
    max_turn_rate: f32,
) {
    handle
        .swarm
        .set_flight_params(max_velocity, max_acceleration, max_turn_rate);
}

#[wasm_bindgen]
pub fn set_avoidance_lookahead(handle: &mut SwarmHandle, lookahead_time: f32) {
    handle.swarm.set_avoidance_lookahead(lookahead_time);
}

#[wasm_bindgen]
pub fn set_vo_config(
    handle: &mut SwarmHandle,
    lookahead_time: f32,
    time_samples: usize,
    safe_distance: f32,
    detection_range: f32,
    avoidance_weight: f32,
) {
    handle.swarm.set_vo_config(
        lookahead_time,
        time_samples,
        safe_distance,
        detection_range,
        avoidance_weight,
    );
}

#[wasm_bindgen]
pub fn set_waypoint_clearance(handle: &mut SwarmHandle, clearance: f32) {
    handle.swarm.set_waypoint_clearance(clearance);
}

#[wasm_bindgen]
pub fn set_consensus_protocol(handle: &mut SwarmHandle, protocol: &str) {
    let protocol = match protocol {
        "priority_by_id" => ConsensusProtocol::PriorityById,
        "priority_by_waypoint_dist" => ConsensusProtocol::PriorityByWaypointDist,
        _ => ConsensusProtocol::PriorityById,
    };
    handle.swarm.set_consensus_protocol(protocol);
}
