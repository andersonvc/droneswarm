use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;

use drone_lib::DroneAgent;
use drone_lib::{
    Bounds as LibBounds, DroneInfo, DronePerfFeatures, FormationApproachMode, FormationCommand,
    FormationSlot, FormationType, Heading, Objective, PathPlanner, Position, Vec2,
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


// ============================================================================
// World Scale - Coordinate Translation
// ============================================================================

/// World scale configuration for coordinate translation.
///
/// The frontend works in pixels (canvas coordinates), while drone-lib
/// works in meters (real-world SI units). This struct handles the
/// conversion between the two coordinate systems.
#[derive(Debug, Clone, Copy)]
pub struct WorldScale {
    /// Pixels per meter (e.g., 2.0 means 500m = 1000px)
    pub px_per_meter: f32,
    /// World width in meters
    pub world_width_meters: f32,
    /// World height in meters
    pub world_height_meters: f32,
    /// Canvas width in pixels
    pub canvas_width_px: f32,
    /// Canvas height in pixels
    pub canvas_height_px: f32,
}

impl WorldScale {
    /// Create from canvas size (pixels) and world size (meters).
    ///
    /// Uses the smaller ratio to ensure the world fits in the canvas.
    pub fn new(
        canvas_width_px: f32,
        canvas_height_px: f32,
        world_width_meters: f32,
        world_height_meters: f32,
    ) -> Self {
        // Calculate scale (use the smaller ratio to ensure world fits)
        let scale_x = canvas_width_px / world_width_meters;
        let scale_y = canvas_height_px / world_height_meters;
        let px_per_meter = scale_x.min(scale_y);

        WorldScale {
            px_per_meter,
            world_width_meters,
            world_height_meters,
            canvas_width_px,
            canvas_height_px,
        }
    }

    /// Default scale: 1000x1000 pixels = 2500x2500 meters (0.4 px/m)
    pub fn default_scale() -> Self {
        Self::new(1000.0, 1000.0, 2500.0, 2500.0)
    }

    /// Convert pixel distance to meters.
    #[inline]
    pub fn px_to_meters(&self, px: f32) -> f32 {
        px / self.px_per_meter
    }

    /// Convert meters to pixel distance.
    #[inline]
    pub fn meters_to_px(&self, meters: f32) -> f32 {
        meters * self.px_per_meter
    }

    /// Convert a Point from pixels to meters.
    pub fn point_px_to_meters(&self, p: Point) -> Point {
        Point {
            x: self.px_to_meters(p.x),
            y: self.px_to_meters(p.y),
        }
    }

    /// Convert a Point from meters to pixels.
    pub fn point_meters_to_px(&self, p: Point) -> Point {
        Point {
            x: self.meters_to_px(p.x),
            y: self.meters_to_px(p.y),
        }
    }

    /// Convert Position (meters) to Point (pixels).
    pub fn position_to_point_px(&self, p: Position) -> Point {
        Point {
            x: self.meters_to_px(p.x()),
            y: self.meters_to_px(p.y()),
        }
    }

    /// Convert Point (pixels) to Position (meters).
    pub fn point_px_to_position(&self, p: Point) -> Position {
        Position::new(self.px_to_meters(p.x), self.px_to_meters(p.y))
    }

    /// Convert Vec2 (meters) to Point (pixels).
    pub fn vec2_to_point_px(&self, v: Vec2) -> Point {
        Point {
            x: self.meters_to_px(v.x),
            y: self.meters_to_px(v.y),
        }
    }
}

// ============================================================================
// Internal Structs
// ============================================================================

struct DroneState {
    id: u32,
    agent: DroneAgent,
    color: Color,
}

/// Active formation state for the swarm.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct FormationState {
    formation_type: FormationType,
    leader_id: Option<usize>,
    slot_assignments: Vec<(usize, FormationSlot)>,
    center: Position,
    heading: f32,
    /// Smoothed heading for gradual formation rotation during turns
    smoothed_heading: f32,
    /// Leader's current target waypoint (to detect when it changes)
    leader_target: Option<Position>,
    /// Whether we're in route-following mode (vs position-tracking mode)
    route_mode: bool,
    /// Original leader route waypoints (for recomputation on leader succession)
    leader_route: Option<Arc<[Position]>>,
}

pub struct Swarm {
    drones: Vec<DroneState>,
    /// Bounds in meters (for drone-lib)
    lib_bounds: LibBounds,
    /// Coordinate translation between pixels and meters
    world_scale: WorldScale,
    simulation_time: f32,
    speed_multiplier: f32,
    selected_ids: HashSet<u32>,
    consensus_protocol: ConsensusProtocol,
    formation: Option<FormationState>,
}

impl Swarm {
    pub fn new(config: SimulationConfig) -> Result<Self, String> {
        // Get world dimensions in meters (default 500x500)
        let world_width_meters = config.world_width_meters.unwrap_or(2500.0);
        let world_height_meters = config.world_height_meters.unwrap_or(2500.0);

        // Create world scale for coordinate translation
        let world_scale = WorldScale::new(
            config.bounds.width,
            config.bounds.height,
            world_width_meters,
            world_height_meters,
        );

        // Create lib_bounds in METERS (not pixels)
        let lib_bounds = LibBounds::new(world_width_meters, world_height_meters)
            .map_err(|e| format!("Invalid world bounds: {}", e))?;

        // Generate spawn positions (in pixels from config)
        let positions_px = Self::generate_spawn_positions(&config);
        let drone_count = positions_px.len();

        // Generate random headings using random seed from JavaScript
        let mut hdg_seed = Self::random_seed();
        let drones = positions_px
            .into_iter()
            .enumerate()
            .map(|(i, pos_px)| {
                hdg_seed = hdg_seed.wrapping_mul(1103515245).wrapping_add(12345);
                let hdg = (hdg_seed as f32 / u32::MAX as f32) * std::f32::consts::TAU
                    - std::f32::consts::PI;

                // Convert spawn position from pixels to meters
                let pos_meters = world_scale.point_px_to_position(pos_px);

                DroneState {
                    id: i as u32,
                    agent: DroneAgent::new(
                        i,
                        pos_meters,
                        Heading::new(hdg),
                        lib_bounds,
                    ),
                    color: Self::generate_color(i, drone_count),
                }
            })
            .collect();

        let mut swarm = Swarm {
            drones,
            lib_bounds,
            world_scale,
            simulation_time: 0.0,
            speed_multiplier: config.speed_multiplier.unwrap_or(8.0),
            selected_ids: HashSet::new(),
            consensus_protocol: ConsensusProtocol::default(),
            formation: None,
        };

        // Enable formation by default - Chevron formation with spacing=40px
        // Leader (lowest ID) follows waypoints, others maintain formation
        let default_spacing_m = swarm.world_scale.px_to_meters(40.0);
        swarm.set_formation(FormationType::Chevron { spacing: default_spacing_m, angle: std::f32::consts::FRAC_PI_4 }, None);

        // Assign default patrol route (hourglass/bowtie pattern — two crossing triangles)
        let default_route = vec![
            Point { x: 200.0, y: 100.0 },  // 1: top-left
            Point { x: 550.0, y: 500.0 },  // 2: center
            Point { x: 900.0, y:  50.0 },  // 3: top-right
            Point { x: 850.0, y: 900.0 },  // 4: bottom-right
            Point { x: 550.0, y: 650.0 },  // 5: center-lower
            Point { x: 250.0, y: 850.0 },  // 6: bottom-left
        ];
        swarm.assign_route_all(default_route);

        Ok(swarm)
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

    /// Generate a random seed from JavaScript Math.random()
    fn random_seed() -> u32 {
        (js_sys::Math::random() * u32::MAX as f64) as u32
    }

    fn spawn_grid(count: u32, bounds: &Bounds) -> Vec<Point> {
        // Grid positions with random jitter for variety
        let cols = (count as f32).sqrt().ceil() as u32;
        let rows = count.div_ceil(cols);
        let spacing_x = bounds.width / (cols + 1) as f32;
        let spacing_y = bounds.height / (rows + 1) as f32;
        let jitter = spacing_x.min(spacing_y) * 0.3; // 30% jitter

        let mut seed = Self::random_seed();

        (0..count)
            .map(|i| {
                let col = i % cols;
                let row = i / cols;
                // Add random jitter
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let jx = ((seed as f32 / u32::MAX as f32) - 0.5) * jitter;
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let jy = ((seed as f32 / u32::MAX as f32) - 0.5) * jitter;
                Point {
                    x: spacing_x * (col + 1) as f32 + jx,
                    y: spacing_y * (row + 1) as f32 + jy,
                }
            })
            .collect()
    }

    fn spawn_random(count: u32, bounds: &Bounds) -> Vec<Point> {
        // Use random seed from JavaScript for true randomness on each init
        let mut seed = Self::random_seed();
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
        // Use random seed from JavaScript for true randomness on each init
        let mut seed = Self::random_seed();
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
        // Collision threshold in meters (approximately 1 drone diameter)
        const COLLISION_DISTANCE: f32 = 1.0;

        // Clamp dt to prevent physics explosions when tab is backgrounded.
        // requestAnimationFrame pauses in inactive tabs, causing huge dt on return.
        let clamped_dt = dt.min(0.05); // Max ~20 FPS equivalent
        let effective_dt = clamped_dt * self.speed_multiplier;

        // Update formation positions (followers track the leader)
        self.update_formation(effective_dt);

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
            .map(|d| d.agent.get_info())
            .collect();

        // Process drones in priority order
        for &idx in &indices {
            // Update this drone using current swarm info
            // (higher priority drones have already updated their entries)
            self.drones[idx].agent.state_update(effective_dt, &swarm_info);

            // Update swarm info with this drone's new state
            // so lower priority drones will see where we're going
            swarm_info[idx] = self.drones[idx].agent.get_info();
        }

        // Collision detection - find all colliding drone pairs
        let mut destroyed_ids: HashSet<u32> = HashSet::new();

        for i in 0..self.drones.len() {
            for j in (i + 1)..self.drones.len() {
                let pos_i = self.drones[i].agent.state().pos;
                let pos_j = self.drones[j].agent.state().pos;

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
            // Before removing, save leader's route if leader is being destroyed
            let leader_route = self.save_leader_route_if_destroyed(&destroyed_ids);

            self.drones.retain(|d| !destroyed_ids.contains(&d.id));
            // Also remove from selection
            self.selected_ids.retain(|id| !destroyed_ids.contains(id));
            // Check if leader was destroyed - promote successor if needed
            self.check_leader_succession_with_route(leader_route);
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
                let state = d.agent.state();
                let objective = d.agent.objective();

                // Hide all waypoint/path visualization for formation followers
                let is_follower = self.formation.as_ref()
                    .is_some_and(|f| f.leader_id != Some(d.id as usize));

                let spline_path: Vec<Point> = if is_follower {
                    Vec::new()
                } else {
                    d.agent
                        .get_spline_path(20)
                        .into_iter()
                        .map(|v| self.world_scale.vec2_to_point_px(v))
                        .collect()
                };

                let route_path: Vec<Point> = if is_follower {
                    Vec::new()
                } else {
                    match &objective {
                        Objective::FollowRoute { route, .. } if route.len() >= 2 => {
                            PathPlanner::get_full_route_spline(route, &self.lib_bounds, 10)
                                .into_iter()
                                .map(|v| self.world_scale.vec2_to_point_px(v))
                                .collect()
                        }
                        _ => Vec::new(),
                    }
                };

                let planning_path: Vec<Point> = if is_follower {
                    Vec::new()
                } else {
                    d.agent
                        .get_formation_planning_path(20)
                        .into_iter()
                        .map(|v| self.world_scale.vec2_to_point_px(v))
                        .collect()
                };

                let target = if is_follower {
                    None
                } else {
                    match &objective {
                        Objective::ReachWaypoint { waypoints } | Objective::FollowRoute { waypoints, .. } => {
                            waypoints.front().map(|&p| self.world_scale.position_to_point_px(p))
                        }
                        _ => None,
                    }
                };

                // Objective variant name for display
                let objective_type = match &objective {
                    Objective::Sleep => "Sleep",
                    Objective::ReachWaypoint { .. } => "ReachWaypoint",
                    Objective::FollowRoute { .. } => "FollowRoute",
                    Objective::FollowTarget { .. } => "FollowTarget",
                    Objective::Loiter { .. } => "Loiter",
                }.to_string();

                // Get formation approach mode for color-coded visualization
                let approach_mode = match d.agent.get_formation_approach_mode() {
                    FormationApproachMode::None => "none",
                    FormationApproachMode::StationKeeping => "station_keeping",
                    FormationApproachMode::Correction => "correction",
                    FormationApproachMode::Pursuit => "pursuit",
                    FormationApproachMode::Approach => "approach",
                }
                .to_string();

                DroneRenderData {
                    id: d.id,
                    // Convert position from meters to pixels
                    x: self.world_scale.meters_to_px(state.pos.x()),
                    y: self.world_scale.meters_to_px(state.pos.y()),
                    heading: state.hdg.radians(),
                    color: d.color,
                    selected: self.selected_ids.contains(&d.id),
                    objective_type,
                    target,
                    spline_path,
                    route_path,
                    planning_path,
                    approach_mode,
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
        // Convert input coordinates from pixels to meters
        let x_m = self.world_scale.px_to_meters(x);
        let y_m = self.world_scale.px_to_meters(y);
        let hit_radius_m = self.world_scale.px_to_meters(hit_radius);
        let hit_radius_sq = hit_radius_m * hit_radius_m;

        for drone in &self.drones {
            let state = drone.agent.state();
            let dx = state.pos.x() - x_m;
            let dy = state.pos.y() - y_m;
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
        // Convert pixel coordinates to meters
        let x_m = self.world_scale.px_to_meters(x);
        let y_m = self.world_scale.px_to_meters(y);

        let mut waypoints = VecDeque::new();
        waypoints.push_back(Position::new(x_m, y_m));

        for drone in &mut self.drones {
            if self.selected_ids.contains(&drone.id) {
                drone.agent.set_objective(Objective::ReachWaypoint {
                    waypoints: waypoints.clone(),
                });
            }
        }
    }

    pub fn assign_waypoint_all(&mut self, x: f32, y: f32) {
        // Convert pixel coordinates to meters
        let x_m = self.world_scale.px_to_meters(x);
        let y_m = self.world_scale.px_to_meters(y);

        let mut waypoints = VecDeque::new();
        waypoints.push_back(Position::new(x_m, y_m));

        for drone in &mut self.drones {
            drone.agent.set_objective(Objective::ReachWaypoint {
                waypoints: waypoints.clone(),
            });
        }
    }

    pub fn assign_path(&mut self, waypoints: Vec<Point>) {
        // Convert pixel coordinates to meters
        let waypoint_deque: VecDeque<Position> = waypoints
            .iter()
            .map(|p| self.world_scale.point_px_to_position(*p))
            .collect();

        for drone in &mut self.drones {
            if self.selected_ids.contains(&drone.id) {
                drone.agent.set_objective(Objective::ReachWaypoint {
                    waypoints: waypoint_deque.clone(),
                });
            }
        }
    }

    pub fn assign_route_all(&mut self, waypoints: Vec<Point>) {
        // Convert pixel coordinates to meters
        let waypoint_vec: Vec<Position> = waypoints
            .iter()
            .map(|p| self.world_scale.point_px_to_position(*p))
            .collect();
        let route: Arc<[Position]> = Arc::from(waypoint_vec);

        // If in formation mode, only the leader gets the route.
        // Followers stay in position-tracking mode — they dynamically track their
        // formation slot positions around the moving leader each tick.
        if let Some(ref mut formation) = self.formation {
            // Store the leader route for recomputation on succession
            formation.leader_route = Some(route.clone());

            // Find the leader's nearest waypoint so it joins the route at the closest point
            let leader_nearest_idx = if let Some(leader_id) = formation.leader_id {
                self.drones.iter()
                    .find(|d| d.id == leader_id as u32)
                    .map(|d| {
                        let leader_pos = d.agent.get_info().pos;
                        route.iter().enumerate()
                            .min_by(|(_, a), (_, b)| {
                                let da = self.lib_bounds.distance(leader_pos.as_vec2(), a.as_vec2());
                                let db = self.lib_bounds.distance(leader_pos.as_vec2(), b.as_vec2());
                                da.partial_cmp(&db).unwrap()
                            })
                            .map(|(i, _)| i)
                            .unwrap_or(0)
                    })
                    .unwrap_or(0)
            } else {
                0
            };

            // Leader gets the original route, starting from nearest waypoint
            if let Some(leader_id) = formation.leader_id {
                if let Some(leader_drone) = self.drones.iter_mut().find(|d| d.id == leader_id as u32) {
                    let mut leader_deque: VecDeque<Position> = VecDeque::with_capacity(route.len());
                    for i in 0..route.len() {
                        leader_deque.push_back(route[(leader_nearest_idx + i) % route.len()]);
                    }

                    leader_drone.agent.set_objective(Objective::FollowRoute {
                        waypoints: leader_deque,
                        route: route.clone(),
                    });
                }
            }

            // Followers do NOT get their own routes — they track formation slots
            // around the leader via update_formation_reference() each tick.
            formation.route_mode = false;
            formation.leader_target = None;
            return;
        }

        // No formation - all drones follow the same route, each starting from nearest waypoint
        for drone in &mut self.drones {
            let drone_pos = drone.agent.get_info().pos;
            let nearest_idx = route.iter().enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = self.lib_bounds.distance(drone_pos.as_vec2(), a.as_vec2());
                    let db = self.lib_bounds.distance(drone_pos.as_vec2(), b.as_vec2());
                    da.partial_cmp(&db).unwrap()
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            let mut deque: VecDeque<Position> = VecDeque::with_capacity(route.len());
            for i in 0..route.len() {
                deque.push_back(route[(nearest_idx + i) % route.len()]);
            }

            drone.agent.set_objective(Objective::FollowRoute {
                waypoints: deque,
                route: route.clone(),
            });
        }
    }

    pub fn set_speed(&mut self, multiplier: f32) {
        self.speed_multiplier = multiplier.clamp(0.25, 16.0);
    }

    pub fn set_flight_params(
        &mut self,
        max_velocity: f32,
        max_acceleration: f32,
        max_turn_rate: f32,
    ) {
        let params = DronePerfFeatures::new_unchecked(max_velocity, max_acceleration, max_turn_rate);
        for drone in &mut self.drones {
            drone.agent.set_flight_params(params);
        }
    }

    pub fn set_avoidance_lookahead(&mut self, lookahead_time: f32) {
        let time_horizon = lookahead_time.clamp(0.5, 5.0);
        for drone in &mut self.drones {
            let mut config = *drone.agent.orca_config();
            config.time_horizon = time_horizon;
            drone.agent.set_orca_config(config);
        }
    }

    pub fn set_orca_config(
        &mut self,
        time_horizon: f32,
        agent_radius: f32,
        neighbor_dist: f32,
    ) {
        for drone in &mut self.drones {
            let mut config = *drone.agent.orca_config();
            config.time_horizon = time_horizon.clamp(1.5, 10.0);
            config.agent_radius = agent_radius.clamp(5.0, 50.0);
            config.neighbor_dist = neighbor_dist.clamp(10.0, 300.0);
            drone.agent.set_orca_config(config);
        }
    }

    pub fn set_waypoint_clearance(&mut self, clearance: f32) {
        let clearance = clearance.clamp(1.0, 200.0);
        for drone in &mut self.drones {
            drone.agent.set_waypoint_clearance(clearance);
        }
    }

    pub fn set_consensus_protocol(&mut self, protocol: ConsensusProtocol) {
        self.consensus_protocol = protocol;
    }

    // ========================================================================
    // Formation Control
    // ========================================================================

    pub fn set_formation(
        &mut self,
        formation_type: FormationType,
        leader_id: Option<usize>,
    ) {
        if self.drones.is_empty() {
            return;
        }

        // Clear any existing formation state first
        for drone in &mut self.drones {
            drone.agent.set_formation_leader(false);
            drone.agent.clear_formation_slot();
        }

        // Determine leader
        let resolved_leader = leader_id.or_else(|| {
            self.drones.iter().map(|d| d.id as usize).min()
        });

        // Get center and heading from leader
        let (center, heading) = if let Some(lid) = resolved_leader {
            self.drones
                .iter()
                .find(|d| d.id == lid as u32)
                .map(|d| (d.agent.state().pos, d.agent.state().hdg.radians()))
                .unwrap_or_else(|| (self.calculate_centroid(), 0.0))
        } else {
            (self.calculate_centroid(), 0.0)
        };

        // Get drone IDs, with leader first
        let mut drone_ids: Vec<usize> = self.drones.iter().map(|d| d.id as usize).collect();
        if let Some(lid) = resolved_leader {
            if let Some(pos) = drone_ids.iter().position(|&id| id == lid) {
                drone_ids.remove(pos);
                drone_ids.insert(0, lid);
            }
        } else {
            drone_ids.sort();
        }

        // Compute slots
        let slots = formation_type.compute_slots(drone_ids.len());

        // Build assignments
        let slot_assignments: Vec<(usize, FormationSlot)> = drone_ids
            .into_iter()
            .zip(slots)
            .collect();

        // Set formation leader flag on the leader drone
        if let Some(lid) = resolved_leader {
            if let Some(leader_drone) = self.drones.iter_mut().find(|d| d.id == lid as u32) {
                leader_drone.agent.set_formation_leader(true);
            }
        }

        // Apply slots to drones (skip leader - it follows its waypoint)
        for (drone_id, slot) in &slot_assignments {
            // Skip the leader - it keeps its current waypoint/mission
            if Some(*drone_id) == resolved_leader {
                continue;
            }
            if let Some(drone) = self.drones.iter_mut().find(|d| d.id == *drone_id as u32) {
                drone.agent.set_formation_slot(*slot, center, heading);
            }
        }

        self.formation = Some(FormationState {
            formation_type,
            leader_id: resolved_leader,
            slot_assignments,
            center,
            heading,
            smoothed_heading: heading,
            leader_target: None,
            route_mode: false,
            leader_route: None,
        });
    }

    pub fn clear_formation(&mut self) {
        for drone in &mut self.drones {
            drone.agent.clear_formation_slot();
            drone.agent.set_formation_leader(false);
        }
        self.formation = None;
    }

    /// Save the leader's route before the leader is destroyed.
    ///
    /// Returns the route if the current leader is in the destroyed set and has a FollowRoute objective.
    fn save_leader_route_if_destroyed(&self, destroyed_ids: &HashSet<u32>) -> Option<Arc<[Position]>> {
        let state = self.formation.as_ref()?;
        let leader_id = state.leader_id?;

        // Check if leader is being destroyed
        if !destroyed_ids.contains(&(leader_id as u32)) {
            return None;
        }

        // Find the leader and get their route
        self.drones
            .iter()
            .find(|d| d.id == leader_id as u32)
            .and_then(|d| {
                match d.agent.objective() {
                    Objective::FollowRoute { route, .. } => Some(route),
                    _ => None,
                }
            })
    }

    /// Check if the current leader is still alive; if not, promote successor.
    ///
    /// When the leader crashes, the drone with the next lowest ID becomes
    /// the new leader and formation slots are reassigned. If the old leader
    /// had a route, it's transferred to the new leader.
    fn check_leader_succession_with_route(&mut self, leader_route: Option<Arc<[Position]>>) {
        let Some(ref state) = self.formation else { return };

        let current_leader = state.leader_id;
        let formation_type = state.formation_type;
        let saved_leader_route = state.leader_route.clone();
        let was_route_mode = state.route_mode;

        // Check if current leader still exists
        let leader_alive = current_leader
            .map(|lid| self.drones.iter().any(|d| d.id == lid as u32))
            .unwrap_or(false);

        if leader_alive {
            return; // Leader is fine, no succession needed
        }

        // Leader is gone - find the new leader (lowest ID among remaining drones)
        let new_leader_id = self.drones.iter().map(|d| d.id as usize).min();

        if new_leader_id.is_none() {
            // No drones left
            self.formation = None;
            return;
        }

        // Reassign formation with new leader
        self.set_formation(formation_type, new_leader_id);

        // Transfer route to new leader if the old leader had one
        let route = leader_route.or(saved_leader_route);
        if let Some(route) = route {
            // Use assign_route_all to recompute offset routes for all drones
            let waypoints_px: Vec<Point> = route.iter().map(|p| {
                self.world_scale.position_to_point_px(*p)
            }).collect();
            self.assign_route_all(waypoints_px);

            if let Some(ref mut formation) = self.formation {
                formation.route_mode = was_route_mode;
            }
        }
    }

    pub fn formation_command(&mut self, cmd: FormationCommand) {
        let Some(ref state) = self.formation else { return };

        // Extract values we need before potentially modifying self
        let formation_type = state.formation_type;
        let leader_id = state.leader_id;

        match cmd {
            FormationCommand::Disperse => {
                for drone in &mut self.drones {
                    drone.agent.handle_formation_command(cmd);
                }
                self.formation = None;
            }
            FormationCommand::Contract => {
                // Reduce spacing by 20%
                let new_type = Self::scale_formation_type(&formation_type, 0.8);
                self.set_formation(new_type, leader_id);
            }
            FormationCommand::Expand => {
                // Increase spacing by 20%
                let new_type = Self::scale_formation_type(&formation_type, 1.2);
                self.set_formation(new_type, leader_id);
            }
            FormationCommand::Hold | FormationCommand::Advance => {
                for drone in &mut self.drones {
                    drone.agent.handle_formation_command(cmd);
                }
            }
        }
    }

    pub fn update_formation(&mut self, _dt: f32) {
        let Some(ref mut state) = self.formation else { return };

        // Get leader info
        let leader_id = state.leader_id;
        let Some(lid) = leader_id else { return };

        let (leader_pos, leader_heading, leader_velocity) = {
            if let Some(leader) = self.drones.iter().find(|d| d.id == lid as u32) {
                let pos = leader.agent.state().pos;
                let hdg = leader.agent.state().hdg.radians();
                let vel = leader.agent.state().vel.as_vec2();
                (pos, hdg, vel)
            } else {
                return;
            }
        };

        state.center = leader_pos;

        // Use leader's heading directly for formation orientation
        let target_heading = leader_heading;
        state.heading = target_heading;

        // Smooth the formation heading to prevent jerky turns
        const HEADING_SMOOTHING: f32 = 0.08;

        let mut heading_diff = target_heading - state.smoothed_heading;
        while heading_diff > std::f32::consts::PI {
            heading_diff -= std::f32::consts::TAU;
        }
        while heading_diff < -std::f32::consts::PI {
            heading_diff += std::f32::consts::TAU;
        }

        state.smoothed_heading += heading_diff * HEADING_SMOOTHING;
        while state.smoothed_heading > std::f32::consts::PI {
            state.smoothed_heading -= std::f32::consts::TAU;
        }
        while state.smoothed_heading < -std::f32::consts::PI {
            state.smoothed_heading += std::f32::consts::TAU;
        }

        let smoothed_heading = state.smoothed_heading;
        let is_route_mode = state.route_mode;
        for drone in &mut self.drones {
            if leader_id == Some(drone.id as usize) {
                continue;
            }
            if drone.agent.in_formation() {
                if is_route_mode {
                    // In route mode, followers independently follow their own offset routes.
                    // Only update center/heading/velocity for reference (used by speed factor),
                    // but do NOT overwrite their mission waypoints.
                    drone.agent.update_formation_reference_no_waypoint(
                        state.center, smoothed_heading, leader_velocity,
                    );
                } else {
                    // Position-tracking mode: update waypoint to track leader's slot position
                    drone.agent.update_formation_reference(
                        state.center, smoothed_heading, leader_velocity,
                    );
                }
            }
        }
    }

    fn scale_formation_type(formation_type: &FormationType, scale: f32) -> FormationType {
        match *formation_type {
            FormationType::Line { spacing } => FormationType::Line { spacing: spacing * scale },
            FormationType::Vee { spacing, angle } => FormationType::Vee { spacing: spacing * scale, angle },
            FormationType::Diamond { spacing } => FormationType::Diamond { spacing: spacing * scale },
            FormationType::Circle { radius } => FormationType::Circle { radius: radius * scale },
            FormationType::Grid { spacing, cols } => FormationType::Grid { spacing: spacing * scale, cols },
            FormationType::Chevron { spacing, angle } => FormationType::Chevron { spacing: spacing * scale, angle },
        }
    }

    fn calculate_centroid(&self) -> Position {
        if self.drones.is_empty() {
            return Position::new(0.0, 0.0);
        }

        let sum_x: f32 = self.drones.iter().map(|d| d.agent.state().pos.x()).sum();
        let sum_y: f32 = self.drones.iter().map(|d| d.agent.state().pos.y()).sum();
        let count = self.drones.len() as f32;

        Position::new(sum_x / count, sum_y / count)
    }

    /// Helper: get distance from drone to its next waypoint (or MAX if none)
    fn distance_to_waypoint(&self, drone_idx: usize) -> f32 {
        let drone = &self.drones[drone_idx];
        let objective = drone.agent.objective();

        let waypoint = match &objective {
            Objective::ReachWaypoint { waypoints } | Objective::FollowRoute { waypoints, .. } => {
                waypoints.front().copied()
            }
            _ => None,
        };

        if let Some(waypoint) = waypoint {
            let pos = drone.agent.state().pos;
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
pub fn set_orca_config(
    handle: &mut SwarmHandle,
    time_horizon: f32,
    agent_radius: f32,
    neighbor_dist: f32,
) {
    handle.swarm.set_orca_config(
        time_horizon,
        agent_radius,
        neighbor_dist,
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

// ============================================================================
// Formation Control WASM Exports
// ============================================================================

/// Set the swarm formation.
///
/// # Arguments
/// * `formation_type` - One of: "line", "vee", "diamond", "circle", "grid"
/// * `spacing` - Distance between drones (pixels) - converted to meters internally
/// * `leader_id` - Optional drone ID to be the formation leader
///
/// # Examples (JavaScript)
/// ```js
/// set_formation(handle, "vee", 60.0, 0);  // V formation, drone 0 as leader
/// set_formation(handle, "diamond", 80.0, 0);  // Diamond formation
/// set_formation(handle, "line", 50.0, null);  // Line formation, no leader
/// set_formation(handle, "circle", 80.0, null);  // Circle formation
/// ```
#[wasm_bindgen]
pub fn set_formation(
    handle: &mut SwarmHandle,
    formation_type: &str,
    spacing: f32,
    leader_id: Option<u32>,
) {
    // Convert spacing from pixels to meters
    let spacing_m = handle.swarm.world_scale.px_to_meters(spacing);

    let formation = match formation_type {
        "line" => FormationType::Line { spacing: spacing_m },
        "vee" => FormationType::Vee {
            spacing: spacing_m,
            angle: std::f32::consts::FRAC_PI_4, // 45 degrees
        },
        "diamond" => FormationType::Diamond { spacing: spacing_m },
        "circle" => FormationType::Circle { radius: spacing_m },
        "grid" => {
            let drone_count = handle.swarm.drones.len();
            let cols = (drone_count as f32).sqrt().ceil() as usize;
            FormationType::Grid { spacing: spacing_m, cols }
        }
        "chevron" => FormationType::Chevron {
            spacing: spacing_m,
            angle: std::f32::consts::FRAC_PI_4,
        },
        _ => {
            web_sys::console::warn_1(
                &format!("Unknown formation type '{}', using line", formation_type).into(),
            );
            FormationType::Line { spacing: spacing_m }
        }
    };

    handle.swarm.set_formation(formation, leader_id.map(|id| id as usize));
}

/// Clear the current formation, returning drones to independent control.
#[wasm_bindgen]
pub fn clear_formation(handle: &mut SwarmHandle) {
    handle.swarm.clear_formation();
}

/// Issue a formation command to all drones.
///
/// # Arguments
/// * `command` - One of: "hold", "advance", "disperse", "contract", "expand"
///
/// # Examples (JavaScript)
/// ```js
/// formation_command(handle, "disperse");  // Drones scatter
/// formation_command(handle, "contract");  // Tighten formation
/// formation_command(handle, "expand");    // Loosen formation
/// ```
#[wasm_bindgen]
pub fn formation_command(handle: &mut SwarmHandle, command: &str) {
    let cmd = match command {
        "hold" => FormationCommand::Hold,
        "advance" => FormationCommand::Advance,
        "disperse" => FormationCommand::Disperse,
        "contract" => FormationCommand::Contract,
        "expand" => FormationCommand::Expand,
        _ => {
            web_sys::console::warn_1(
                &format!("Unknown formation command '{}', using hold", command).into(),
            );
            FormationCommand::Hold
        }
    };

    handle.swarm.formation_command(cmd);
}

/// Update formation center and heading based on leader position.
/// Call this each tick if you want the formation to follow the leader.
#[wasm_bindgen]
pub fn update_formation(handle: &mut SwarmHandle) {
    // Use a small dt for manual updates (formation already updated in tick loop)
    handle.swarm.update_formation(0.016);
}
