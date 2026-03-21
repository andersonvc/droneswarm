use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use drone_lib::DroneAgent;
use drone_lib::{
    APFConfig, Bounds as LibBounds, DroneInfo, DronePerfFeatures,
    FormationCommand, FormationSlot, FormationType, Heading, Objective, Position,
    TaskStatus, Vec2,
};
use drone_lib::doctrine::{DoctrineMode, SwarmDoctrine};
use drone_lib::inference::InferenceNetV2;
use drone_lib::strategies::attack_zone::AttackZoneStrategy;
use drone_lib::strategies::defend_area::DefendAreaStrategy;
use drone_lib::strategies::patrol_perimeter::PatrolPerimeterStrategy;
use drone_lib::strategies::{StrategyDroneState, SwarmStrategy, TaskAssignment};
use drone_lib::tasks::attack::AttackTask;
use drone_lib::tasks::defend::DefendTask;
use drone_lib::tasks::evade::EvadeTask;
use drone_lib::tasks::intercept::InterceptTask;
use drone_lib::tasks::intercept_group::InterceptGroupTask;
use drone_lib::tasks::loiter::LoiterTask;
use drone_lib::tasks::patrol::PatrolTask;
use wasm_bindgen::prelude::*;

mod types;
mod world_scale;
mod spawn;
mod render;
mod formation;

pub use types::*;
pub use world_scale::WorldScale;

// ============================================================================
// Panic Hook Setup
// ============================================================================

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

// ============================================================================
// Internal Structs
// ============================================================================

pub(crate) struct DroneState {
    pub(crate) id: u32,
    pub(crate) agent: DroneAgent,
    pub(crate) color: Color,
}

/// Active formation state for a group of drones.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct FormationState {
    pub(crate) formation_type: FormationType,
    pub(crate) leader_id: Option<usize>,
    pub(crate) slot_assignments: Vec<(usize, FormationSlot)>,
    /// Which drone IDs belong to this formation group
    pub(crate) drone_ids: HashSet<u32>,
    pub(crate) center: Position,
    pub(crate) heading: f32,
    /// Smoothed heading for gradual formation rotation during turns
    pub(crate) smoothed_heading: f32,
    /// Leader's current target waypoint (to detect when it changes)
    pub(crate) leader_target: Option<Position>,
    /// Whether we're in route-following mode (vs position-tracking mode)
    pub(crate) route_mode: bool,
    /// Original leader route waypoints (for recomputation on leader succession)
    pub(crate) leader_route: Option<Arc<[Position]>>,
}

/// Drone length in meters — matches the 15px visual size at 0.4 px/m scale.
const DRONE_LENGTH_METERS: f32 = 37.5;
/// Detonation blast radius = 5x drone length
const DETONATION_RADIUS: f32 = DRONE_LENGTH_METERS * 5.0;

/// RL agent state for a single group.
struct RlAgentState {
    /// Trained policy network.
    model: InferenceNetV2,
    /// Which group this agent controls (0 or 1).
    group: u32,
    /// Whether the agent is actively making decisions.
    enabled: bool,
    /// Tick counter for decision frequency.
    tick_counter: u32,
    /// Initial drone count for observation normalization.
    initial_own_drones: f32,
    /// Initial enemy drone count for observation normalization.
    initial_enemy_drones: f32,
    /// Initial friendly target count.
    initial_friendly_targets: f32,
    /// Initial enemy target count.
    initial_enemy_targets: f32,
    /// Current friendly target count (updated externally).
    current_friendly_targets: f32,
    /// Current enemy target count (updated externally).
    current_enemy_targets: f32,
    /// Multi-agent mode: per-drone inference instead of doctrine-level.
    multi_agent: bool,
}

/// How often the RL agent makes decisions (in ticks).
const RL_DECISION_INTERVAL: u32 = 60;
/// How often the multi-agent RL makes decisions (in ticks).
const RL_MULTI_DECISION_INTERVAL: u32 = 20;
/// Max simulation ticks for observation normalization.
const RL_MAX_TICKS: f32 = 10000.0;
/// Max nearby threats for observation normalization.
const RL_MAX_NEARBY_THREATS: f32 = 20.0;
/// Threat detection radius multiplier (relative to DETONATION_RADIUS).
const RL_THREAT_RADIUS_MULTIPLIER: f32 = 5.0;
/// Max drone velocity for per-drone obs normalization.
const RL_MAX_VELOCITY: f32 = 20.0;
/// Per-drone action dimensionality.
const RL_ACT_DIM: usize = 13;
/// Patrol standoff distance for RL patrol action.
const RL_PATROL_STANDOFF: f32 = 200.0;

pub struct Swarm {
    pub(crate) drones: Vec<DroneState>,
    /// Drone IDs queued for detonation (processed next tick)
    pub(crate) pending_detonations: HashSet<u32>,
    /// Drones in "attack target" mode: drone_id → target position (meters)
    pub(crate) attack_targets: HashMap<u32, Position>,
    /// Drones in "intercept" mode: attacker_id → target_drone_id
    pub(crate) intercept_targets: HashMap<u32, u32>,
    /// Protected positions (meters) per group: interceptors must not detonate near these.
    /// Key = group id (0 or 1), Value = positions to protect.
    pub(crate) protected_zones: HashMap<u32, Vec<Position>>,
    /// Drone ID that splits group 0 (< split) from group 1 (>= split).
    pub(crate) group_split_id: u32,
    /// Bounds in meters (for drone-lib)
    pub(crate) lib_bounds: LibBounds,
    /// Coordinate translation between pixels and meters
    pub(crate) world_scale: WorldScale,
    pub(crate) simulation_time: f32,
    pub(crate) speed_multiplier: f32,
    pub(crate) selected_ids: HashSet<u32>,
    pub(crate) consensus_protocol: ConsensusProtocol,
    pub(crate) formations: Vec<FormationState>,
    /// Active swarm-level strategies.
    pub(crate) strategies: Vec<Box<dyn SwarmStrategy>>,
    /// RL agents (one per group, if loaded).
    pub(crate) rl_agents: Vec<RlAgentState>,
    /// Last RL action per drone (for observation encoding).
    pub(crate) rl_last_actions: HashMap<u32, u32>,
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
        let positions_px = spawn::generate_spawn_positions(&config);
        let drone_count = positions_px.len();

        // Generate random headings using random seed from JavaScript
        let mut hdg_seed = spawn::random_seed();
        let drones = positions_px
            .into_iter()
            .enumerate()
            .map(|(i, pos_px)| {
                drone_lib::game::rng::lcg_next(&mut hdg_seed);
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
                    color: spawn::generate_color(i, drone_count),
                }
            })
            .collect();

        let swarm = Swarm {
            drones,
            pending_detonations: HashSet::new(),
            attack_targets: HashMap::new(),
            intercept_targets: HashMap::new(),
            protected_zones: HashMap::new(),
            group_split_id: 0,
            lib_bounds,
            world_scale,
            simulation_time: 0.0,
            speed_multiplier: config.speed_multiplier.unwrap_or(8.0),
            selected_ids: HashSet::new(),
            consensus_protocol: ConsensusProtocol::default(),
            formations: Vec::new(),
            strategies: Vec::new(),
            rl_agents: Vec::new(),
            rl_last_actions: HashMap::new(),
        };

        Ok(swarm)
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

        // Process task-based drones: check for detonation requests and task completion
        {
            let mut task_detonations: Vec<u32> = Vec::new();
            let mut task_completed: Vec<u32> = Vec::new();
            let mut task_failed: Vec<u32> = Vec::new();

            for drone in &self.drones {
                let id = drone.id;

                // Check if task requested detonation
                if drone.agent.should_detonate() {
                    // For intercept drones, check protected zone safety
                    if self.intercept_targets.contains_key(&id) {
                        let drone_pos = drone.agent.state().pos;
                        let group = if id < self.group_split_id { 0 } else { 1 };
                        let would_hit_friendly = self.protected_zones
                            .get(&group)
                            .map(|zones| {
                                zones.iter().any(|z| {
                                    self.lib_bounds.distance(drone_pos.as_vec2(), z.as_vec2())
                                        <= DETONATION_RADIUS
                                })
                            })
                            .unwrap_or(false);

                        if would_hit_friendly {
                            // Abort — too close to friendly
                            task_failed.push(id);
                            continue;
                        }
                    }
                    task_detonations.push(id);
                    continue;
                }

                // Check for task completion/failure without detonation
                if let Some(status) = drone.agent.task_status() {
                    match status {
                        TaskStatus::Complete => task_completed.push(id),
                        TaskStatus::Failed => task_failed.push(id),
                        TaskStatus::Active => {}
                    }
                }
            }

            for id in &task_detonations {
                self.intercept_targets.remove(id);
                self.attack_targets.remove(id);
                self.pending_detonations.insert(*id);
            }
            for id in &task_completed {
                self.intercept_targets.remove(id);
                self.attack_targets.remove(id);
                if let Some(drone) = self.drones.iter_mut().find(|d| d.id == *id) {
                    drone.agent.clear_task();
                    drone.agent.set_objective(Objective::Sleep);
                }
            }
            for id in &task_failed {
                self.intercept_targets.remove(id);
                self.attack_targets.remove(id);
                if let Some(drone) = self.drones.iter_mut().find(|d| d.id == *id) {
                    drone.agent.clear_task();
                    drone.agent.set_objective(Objective::Sleep);
                }
            }
        }

        // Clean up attack/intercept targets for drones that no longer exist
        self.attack_targets.retain(|id, _| self.drones.iter().any(|d| d.id == *id));
        self.intercept_targets.retain(|id, _| self.drones.iter().any(|d| d.id == *id));

        // Process detonations — the detonating drone and everything in blast radius
        let mut destroyed_ids: HashSet<u32> = HashSet::new();

        if !self.pending_detonations.is_empty() {
            let detonations: Vec<u32> = self.pending_detonations.drain().collect();
            for det_id in &detonations {
                // Find the detonating drone's position
                let det_pos = self.drones.iter()
                    .find(|d| d.id == *det_id)
                    .map(|d| d.agent.state().pos);

                let Some(det_pos) = det_pos else { continue };

                // The detonating drone is always destroyed
                destroyed_ids.insert(*det_id);

                web_sys::console::info_1(
                    &format!("DETONATION - Drone {} at ({:.0}, {:.0}), blast radius {:.1}m",
                        det_id, det_pos.x(), det_pos.y(), DETONATION_RADIUS).into(),
                );

                // Destroy all drones within blast radius
                for drone in &self.drones {
                    if drone.id == *det_id { continue; }
                    let dist = self.lib_bounds.distance(
                        det_pos.as_vec2(), drone.agent.state().pos.as_vec2()
                    );
                    if dist <= DETONATION_RADIUS {
                        destroyed_ids.insert(drone.id);
                        web_sys::console::info_1(
                            &format!("  blast killed Drone {} (dist {:.1}m)", drone.id, dist).into(),
                        );
                    }
                }
            }
        }

        // Collision detection - find all colliding drone pairs

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
            // Before removing, save leader routes if leaders are being destroyed
            let saved_routes = self.save_leader_routes_if_destroyed(&destroyed_ids);

            self.drones.retain(|d| !destroyed_ids.contains(&d.id));
            // Also remove from selection
            self.selected_ids.retain(|id| !destroyed_ids.contains(id));
            // Remove destroyed drone IDs from formation groups
            for formation in &mut self.formations {
                formation.drone_ids.retain(|id| !destroyed_ids.contains(id));
            }
            // Check if any leaders were destroyed - promote successors if needed
            self.check_leader_successions(saved_routes);
        }

        // Process swarm strategies — after all deaths and cleanup,
        // strategies see clean state and produce task assignments for next tick.
        if !self.strategies.is_empty() {
            let swarm_info: Vec<DroneInfo> = self.drones.iter()
                .map(|d| d.agent.get_info())
                .collect();
            self.process_strategies(&swarm_info, effective_dt);
        }

        // RL agent decisions (modifies doctrine mode).
        if !self.rl_agents.is_empty() {
            self.tick_rl_agents();
        }

        self.simulation_time += effective_dt;
    }

    /// Process active strategies: build drone states, tick each strategy,
    /// apply returned task assignments, and remove completed strategies.
    fn process_strategies(&mut self, swarm_info: &[DroneInfo], dt: f32) {
        let mut all_assignments: Vec<TaskAssignment> = Vec::new();
        let bounds = self.lib_bounds;
        let drones = &self.drones;

        for strategy in &mut self.strategies {
            let own_drones: Vec<StrategyDroneState> = strategy
                .drone_ids()
                .iter()
                .filter_map(|&id| {
                    drones.iter().find(|d| d.id == id as u32).map(|d| {
                        let available = match d.agent.task_status() {
                            None => true,
                            Some(TaskStatus::Active) => false,
                            Some(TaskStatus::Complete) | Some(TaskStatus::Failed) => true,
                        };
                        StrategyDroneState {
                            id,
                            pos: d.agent.state().pos,
                            vel: d.agent.state().vel,
                            available,
                        }
                    })
                })
                .collect();

            let assignments = strategy.tick(&own_drones, swarm_info, &bounds, dt);
            all_assignments.extend(assignments);
        }

        for assignment in all_assignments {
            self.apply_task_assignment(assignment);
        }

        self.strategies.retain(|s| !s.is_complete());
    }

    /// Apply a task assignment from a strategy to a drone.
    fn apply_task_assignment(&mut self, assignment: TaskAssignment) {
        // PatrolFormation is handled specially — it manages a group, not a single drone
        if let TaskAssignment::PatrolFormation {
            leader_id,
            follower_ids,
            waypoints,
            loiter_duration,
        } = assignment
        {
            self.apply_patrol_formation(leader_id, follower_ids, waypoints, loiter_duration);
            return;
        }

        // Extract drone_id for cleanup
        let drone_id_u32 = match &assignment {
            TaskAssignment::Intercept { drone_id, .. }
            | TaskAssignment::InterceptGroup { drone_id, .. }
            | TaskAssignment::Attack { drone_id, .. }
            | TaskAssignment::Defend { drone_id, .. }
            | TaskAssignment::Patrol { drone_id, .. }
            | TaskAssignment::Loiter { drone_id, .. } => *drone_id as u32,
            TaskAssignment::PatrolFormation { .. } => unreachable!(),
        };

        // Clean up old tracking
        self.attack_targets.remove(&drone_id_u32);
        self.intercept_targets.remove(&drone_id_u32);

        match assignment {
            TaskAssignment::Intercept {
                drone_id,
                target_id,
            } => {
                self.intercept_drone(drone_id as u32, target_id as u32);
            }
            TaskAssignment::InterceptGroup { drone_id } => {
                let id = drone_id as u32;
                self.remove_drone_from_formation(id);
                let group = if id < self.group_split_id { 0 } else { 1 };
                if let Some(drone) = self.drones.iter_mut().find(|d| d.id == id) {
                    let task = drone_lib::tasks::intercept_group::InterceptGroupTask::new(
                        group,
                        DETONATION_RADIUS,
                    );
                    drone.agent.set_task(Box::new(task));
                }
            }
            TaskAssignment::Attack { drone_id, target } => {
                let id = drone_id as u32;
                self.remove_drone_from_formation(id);
                if let Some(drone) = self.drones.iter_mut().find(|d| d.id == id) {
                    let task = AttackTask::new(target, DETONATION_RADIUS);
                    drone.agent.set_task(Box::new(task));
                }
                self.attack_targets.insert(id, target);
            }
            TaskAssignment::Defend {
                drone_id,
                center,
                orbit_radius,
                engage_radius,
            } => {
                let id = drone_id as u32;
                self.remove_drone_from_formation(id);
                let group = if id < self.group_split_id { 0 } else { 1 };
                if let Some(drone) = self.drones.iter_mut().find(|d| d.id == id) {
                    let task = DefendTask::new(
                        drone_id,
                        group,
                        center,
                        orbit_radius,
                        engage_radius,
                        DETONATION_RADIUS,
                    );
                    drone.agent.set_task(Box::new(task));
                }
            }
            TaskAssignment::Patrol {
                drone_id,
                waypoints,
                loiter_duration,
            } => {
                let id = drone_id as u32;
                self.remove_drone_from_formation(id);
                if let Some(drone) = self.drones.iter_mut().find(|d| d.id == id) {
                    let task = PatrolTask::new(waypoints, 50.0, loiter_duration);
                    drone.agent.set_task(Box::new(task));
                }
            }
            TaskAssignment::Loiter { drone_id, position } => {
                let id = drone_id as u32;
                self.remove_drone_from_formation(id);
                if let Some(drone) = self.drones.iter_mut().find(|d| d.id == id) {
                    let task = LoiterTask::new(position, 50.0);
                    drone.agent.set_task(Box::new(task));
                }
            }
            TaskAssignment::PatrolFormation { .. } => unreachable!("handled above"),
        }
    }

    /// Set up a patrol formation squad: create a chevron formation for the
    /// squad and assign the leader a patrol task. Followers automatically
    /// maintain formation around the leader.
    fn apply_patrol_formation(
        &mut self,
        leader_id: usize,
        follower_ids: Vec<usize>,
        waypoints: Vec<Position>,
        loiter_duration: f32,
    ) {
        let leader_u32 = leader_id as u32;
        let spacing_m = DETONATION_RADIUS * 1.2; // Keep drones > detonation radius apart

        // Build the set of all drone IDs in this squad
        let mut group_ids: HashSet<u32> = HashSet::new();
        group_ids.insert(leader_u32);
        for &fid in &follower_ids {
            group_ids.insert(fid as u32);
        }

        // Remove all squad drones from existing formations and tracking
        for &id in &group_ids {
            self.remove_drone_from_formation(id);
            self.attack_targets.remove(&id);
            self.intercept_targets.remove(&id);
        }

        // Clear any existing task on followers (they'll follow via formation)
        for &fid in &follower_ids {
            if let Some(drone) = self.drones.iter_mut().find(|d| d.id == fid as u32) {
                drone.agent.clear_task();
            }
        }

        // Give leader the patrol task
        if let Some(leader) = self.drones.iter_mut().find(|d| d.id == leader_u32) {
            let task = PatrolTask::new(waypoints, 50.0, loiter_duration);
            leader.agent.set_task(Box::new(task));
        }

        // Create chevron formation for the squad
        let formation_type = FormationType::Chevron {
            spacing: spacing_m,
            angle: std::f32::consts::FRAC_PI_4,
        };
        self.set_formation_for_group(formation_type, group_ids, Some(leader_id));
    }

    // ========================================================================
    // State Queries
    // ========================================================================

    /// Queue a drone for detonation. The drone and anything within
    /// DETONATION_RADIUS (3x drone length) will be destroyed next tick.
    pub fn detonate_drone(&mut self, drone_id: u32) {
        // Verify drone exists
        if self.drones.iter().any(|d| d.id == drone_id) {
            self.pending_detonations.insert(drone_id);
        }
    }

    /// Detonate selected drones.
    pub fn detonate_selected(&mut self) {
        let selected: Vec<u32> = self.selected_ids.iter().copied().collect();
        for id in selected {
            self.detonate_drone(id);
        }
    }

    /// Assign a drone to attack a target position (in pixels).
    /// Uses the AttackTask state machine — navigates at full speed and detonates on arrival.
    pub fn attack_target(&mut self, drone_id: u32, target_x: f32, target_y: f32) {
        let target_m = Position::new(
            self.world_scale.px_to_meters(target_x),
            self.world_scale.px_to_meters(target_y),
        );

        // Remove from formation (promotes new leader if needed)
        self.remove_drone_from_formation(drone_id);

        // Set up AttackTask on the drone agent
        if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
            let task = AttackTask::new(target_m, DETONATION_RADIUS);
            drone.agent.set_task(Box::new(task));
        }

        // Register for tracking (used for cleanup)
        self.attack_targets.insert(drone_id, target_m);
    }

    /// Assign a drone to intercept (chase) an enemy drone.
    /// Uses the InterceptTask state machine for lead pursuit with evade/re-engage phases.
    pub fn intercept_drone(&mut self, attacker_id: u32, target_drone_id: u32) {
        // Don't intercept self
        if attacker_id == target_drone_id { return; }
        // Verify both drones exist
        if !self.drones.iter().any(|d| d.id == attacker_id) { return; }
        if !self.drones.iter().any(|d| d.id == target_drone_id) { return; }

        // Remove from formation (promotes new leader if needed)
        self.remove_drone_from_formation(attacker_id);

        // Determine attacker's group
        let group = if attacker_id < self.group_split_id { 0 } else { 1 };

        // Set up InterceptTask on the drone agent
        if let Some(drone) = self.drones.iter_mut().find(|d| d.id == attacker_id) {
            let task = InterceptTask::new(
                attacker_id as usize,
                target_drone_id as usize,
                group,
                DETONATION_RADIUS,
            );
            drone.agent.set_task(Box::new(task));
        }

        // Remove from attack_targets if it was there
        self.attack_targets.remove(&attacker_id);
        // Register for intercept tracking (used for cleanup and protected zone checks)
        self.intercept_targets.insert(attacker_id, target_drone_id);
    }

    /// Assign a drone to loiter (hold position) at a target position (in pixels).
    pub fn loiter_at(&mut self, drone_id: u32, target_x: f32, target_y: f32) {
        let target_m = Position::new(
            self.world_scale.px_to_meters(target_x),
            self.world_scale.px_to_meters(target_y),
        );

        self.remove_drone_from_formation(drone_id);
        self.attack_targets.remove(&drone_id);
        self.intercept_targets.remove(&drone_id);

        if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
            let task = LoiterTask::new(target_m, 50.0);
            drone.agent.set_task(Box::new(task));
        }
    }

    /// Assign a drone to patrol a route of waypoints (in pixels).
    /// `loiter_duration` is how many seconds to hold at each waypoint.
    pub fn patrol_route(&mut self, drone_id: u32, waypoints_px: &[Point], loiter_duration: f32) {
        if waypoints_px.is_empty() { return; }

        self.remove_drone_from_formation(drone_id);
        self.attack_targets.remove(&drone_id);
        self.intercept_targets.remove(&drone_id);

        let waypoints: Vec<Position> = waypoints_px.iter()
            .map(|p| Position::new(
                self.world_scale.px_to_meters(p.x),
                self.world_scale.px_to_meters(p.y),
            ))
            .collect();

        if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
            let task = PatrolTask::new(waypoints, 50.0, loiter_duration);
            drone.agent.set_task(Box::new(task));
        }
    }

    /// Assign a drone to defend a position (in pixels).
    /// The drone orbits the position and engages enemies that enter the zone.
    pub fn defend_position(
        &mut self,
        drone_id: u32,
        center_x: f32,
        center_y: f32,
        orbit_radius_px: f32,
        engage_radius_px: f32,
    ) {
        let center_m = Position::new(
            self.world_scale.px_to_meters(center_x),
            self.world_scale.px_to_meters(center_y),
        );
        let orbit_radius = self.world_scale.px_to_meters(orbit_radius_px);
        let engage_radius = self.world_scale.px_to_meters(engage_radius_px);

        self.remove_drone_from_formation(drone_id);
        self.attack_targets.remove(&drone_id);
        self.intercept_targets.remove(&drone_id);

        let group = if drone_id < self.group_split_id { 0 } else { 1 };

        if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
            let task = DefendTask::new(
                drone_id as usize,
                group,
                center_m,
                orbit_radius,
                engage_radius,
                DETONATION_RADIUS,
            );
            drone.agent.set_task(Box::new(task));
        }
    }

    // ========================================================================
    // Strategy Commands
    // ========================================================================

    /// Set a defend area strategy for a group of drones.
    /// Positions in pixels, converted to meters internally.
    pub fn set_strategy_defend_area(
        &mut self,
        drone_ids: &[u32],
        center_x: f32,
        center_y: f32,
        radius: f32,
    ) {
        let center = Position::new(
            self.world_scale.px_to_meters(center_x),
            self.world_scale.px_to_meters(center_y),
        );
        let radius_m = self.world_scale.px_to_meters(radius);
        let group = drone_ids
            .first()
            .map(|&id| if id < self.group_split_id { 0 } else { 1 })
            .unwrap_or(0);
        let ids: Vec<usize> = drone_ids.iter().map(|&id| id as usize).collect();

        let strategy = DefendAreaStrategy::new(ids, group, center, radius_m, DETONATION_RADIUS);
        self.strategies.push(Box::new(strategy));
    }

    /// Set an attack zone strategy for a group of drones.
    /// Target positions in pixels, converted to meters internally.
    pub fn set_strategy_attack_zone(
        &mut self,
        drone_ids: &[u32],
        target_positions_px: &[Point],
    ) {
        let target_positions: Vec<Position> = target_positions_px
            .iter()
            .map(|p| {
                Position::new(
                    self.world_scale.px_to_meters(p.x),
                    self.world_scale.px_to_meters(p.y),
                )
            })
            .collect();
        let ids: Vec<usize> = drone_ids.iter().map(|&id| id as usize).collect();

        let strategy = AttackZoneStrategy::new(ids, target_positions, DETONATION_RADIUS);
        self.strategies.push(Box::new(strategy));
    }

    /// Set a patrol perimeter strategy for a group of drones.
    /// Waypoints in pixels, converted to meters internally.
    pub fn set_strategy_patrol_perimeter(
        &mut self,
        drone_ids: &[u32],
        waypoints_px: &[Point],
        loiter_duration: f32,
    ) {
        let waypoints: Vec<Position> = waypoints_px
            .iter()
            .map(|p| {
                Position::new(
                    self.world_scale.px_to_meters(p.x),
                    self.world_scale.px_to_meters(p.y),
                )
            })
            .collect();
        let group = drone_ids
            .first()
            .map(|&id| if id < self.group_split_id { 0 } else { 1 })
            .unwrap_or(0);
        let ids: Vec<usize> = drone_ids.iter().map(|&id| id as usize).collect();

        let strategy = PatrolPerimeterStrategy::new(ids, waypoints, loiter_duration, group, DETONATION_RADIUS);
        self.strategies.push(Box::new(strategy));
    }

    /// Remove all active strategies.
    pub fn clear_strategies(&mut self) {
        self.strategies.clear();
    }

    /// Set a doctrine (autonomous force allocation) for a group of drones.
    /// Positions in pixels, converted to meters internally.
    /// `mode`: "aggressive" or "defensive".
    pub fn set_doctrine(
        &mut self,
        drone_ids: &[u32],
        friendly_targets_px: &[Point],
        enemy_targets_px: &[Point],
        patrol_waypoints_px: &[Point],
        mode: DoctrineMode,
    ) {
        let group = drone_ids
            .first()
            .map(|&id| if id < self.group_split_id { 0 } else { 1 })
            .unwrap_or(0);
        let ids: Vec<usize> = drone_ids.iter().map(|&id| id as usize).collect();
        let friendly: Vec<Position> = friendly_targets_px
            .iter()
            .map(|p| self.world_scale.point_px_to_position(*p))
            .collect();
        let enemy: Vec<Position> = enemy_targets_px
            .iter()
            .map(|p| self.world_scale.point_px_to_position(*p))
            .collect();
        let waypoints: Vec<Position> = patrol_waypoints_px
            .iter()
            .map(|p| self.world_scale.point_px_to_position(*p))
            .collect();

        let doctrine = SwarmDoctrine::new(ids, group, friendly, enemy, waypoints, DETONATION_RADIUS, mode);
        self.strategies.push(Box::new(doctrine));
    }

    /// Update target positions for all doctrine strategies belonging to a group.
    /// Called when targets are destroyed/changed during gameplay.
    pub fn update_doctrine_targets(
        &mut self,
        group: u32,
        friendly_targets_px: &[Point],
        enemy_targets_px: &[Point],
    ) {
        let friendly: Vec<Position> = friendly_targets_px
            .iter()
            .map(|p| self.world_scale.point_px_to_position(*p))
            .collect();
        let enemy: Vec<Position> = enemy_targets_px
            .iter()
            .map(|p| self.world_scale.point_px_to_position(*p))
            .collect();

        for strategy in &mut self.strategies {
            if strategy.group() == Some(group) {
                strategy.update_targets(&friendly, &enemy);
            }
        }
    }

    /// Set the drone ID that splits group 0 from group 1.
    /// Also assigns group IDs to all drone agents and configures
    /// ORCA and APF enemy avoidance parameters.
    ///
    /// Two-layer avoidance:
    /// - ORCA: hard avoidance at blast radius (the real danger zone).
    ///   Long time horizon forecasts collision courses well in advance.
    /// - APF: soft repulsive field at 10x blast radius.
    ///   Gently pushes drones away from enemies at long range.
    pub fn set_group_split(&mut self, split_id: u32) {
        self.group_split_id = split_id;
        // ORCA: avoid the actual blast zone, forecast far ahead
        let enemy_orca_radius = DETONATION_RADIUS;
        // APF: soft repulsion at 10x blast radius
        let enemy_apf_distance = DETONATION_RADIUS * 10.0;
        for drone in &mut self.drones {
            let group = if drone.id < split_id { 0 } else { 1 };
            drone.agent.set_group(group);

            let mut orca_config = *drone.agent.orca_config();
            orca_config.enemy_avoidance_radius = enemy_orca_radius;
            drone.agent.set_orca_config(orca_config);

            drone.agent.set_apf_config(APFConfig {
                influence_distance: DETONATION_RADIUS * 1.2, // keep friendlies outside blast radius
                repulsion_strength: 10000000.0,
                min_distance: 35.0,
                max_force: 80.0,
                enemy_influence_distance: enemy_apf_distance,
            });
        }
    }

    /// Set the protected positions for a group. Interceptors belonging to this
    /// group will not auto-detonate if their blast would reach any of these positions.
    /// Positions are in pixels and converted to meters internally.
    pub fn set_protected_zones(&mut self, group: u32, positions_px: Vec<Point>) {
        let positions: Vec<Position> = positions_px
            .iter()
            .map(|p| self.world_scale.point_px_to_position(*p))
            .collect();
        self.protected_zones.insert(group, positions);
    }

    // ========================================================================
    // RL Agent Integration
    // ========================================================================

    /// Load a trained RL model for a group.
    pub fn load_rl_model(
        &mut self,
        group: u32,
        model_json: &str,
        initial_own_drones: f32,
        initial_enemy_drones: f32,
        initial_friendly_targets: f32,
        initial_enemy_targets: f32,
    ) -> Result<(), String> {
        self.load_rl_model_inner(
            group, model_json,
            initial_own_drones, initial_enemy_drones,
            initial_friendly_targets, initial_enemy_targets,
            false,
        )
    }

    /// Load a per-drone RL model (multi-agent mode) for a group.
    pub fn load_rl_model_multi(
        &mut self,
        group: u32,
        model_json: &str,
        initial_own_drones: f32,
        initial_enemy_drones: f32,
        initial_friendly_targets: f32,
        initial_enemy_targets: f32,
    ) -> Result<(), String> {
        self.load_rl_model_inner(
            group, model_json,
            initial_own_drones, initial_enemy_drones,
            initial_friendly_targets, initial_enemy_targets,
            true,
        )
    }

    fn load_rl_model_inner(
        &mut self,
        group: u32,
        model_json: &str,
        initial_own_drones: f32,
        initial_enemy_drones: f32,
        initial_friendly_targets: f32,
        initial_enemy_targets: f32,
        multi_agent: bool,
    ) -> Result<(), String> {
        let model = InferenceNetV2::from_json(model_json)?;

        // Remove any existing agent for this group.
        self.rl_agents.retain(|a| a.group != group);

        self.rl_agents.push(RlAgentState {
            model,
            group,
            enabled: true,
            tick_counter: 0,
            initial_own_drones,
            initial_enemy_drones,
            initial_friendly_targets,
            initial_enemy_targets,
            current_friendly_targets: initial_friendly_targets,
            current_enemy_targets: initial_enemy_targets,
            multi_agent,
        });

        Ok(())
    }

    /// Load normalizer stats for an RL agent's model.
    /// Should be called after load_rl_model/load_rl_model_multi.
    pub fn load_rl_normalizers(&mut self, group: u32, normalizer_json: &str) -> Result<(), String> {
        let agent = self.rl_agents.iter_mut().find(|a| a.group == group)
            .ok_or_else(|| format!("No RL agent for group {}", group))?;
        agent.model.load_normalizers(normalizer_json)
    }

    /// Enable or disable the RL agent for a group.
    pub fn set_rl_agent_enabled(&mut self, group: u32, enabled: bool) {
        for agent in &mut self.rl_agents {
            if agent.group == group {
                agent.enabled = enabled;
                if enabled {
                    agent.tick_counter = 0;
                }
            }
        }
    }

    /// Update the RL agent's target counts (called when targets are destroyed).
    pub fn update_rl_targets(
        &mut self,
        group: u32,
        friendly_targets: f32,
        enemy_targets: f32,
    ) {
        for agent in &mut self.rl_agents {
            if agent.group == group {
                agent.current_friendly_targets = friendly_targets;
                agent.current_enemy_targets = enemy_targets;
            }
        }
    }

    /// Remove the RL agent for a group.
    pub fn remove_rl_agent(&mut self, group: u32) {
        self.rl_agents.retain(|a| a.group != group);
    }

    /// Tick all enabled RL agents.
    fn tick_rl_agents(&mut self) {
        // Check if any agent uses multi-agent mode.
        let has_multi = self.rl_agents.iter().any(|a| a.enabled && a.multi_agent);
        let has_doctrine = self.rl_agents.iter().any(|a| a.enabled && !a.multi_agent);

        if has_multi {
            self.tick_rl_agents_multi();
        }
        if has_doctrine {
            self.tick_rl_agents_doctrine();
        }
    }

    /// Tick doctrine-level RL agents (8-dim obs, 3 actions — legacy mode).
    fn tick_rl_agents_doctrine(&mut self) {
        let mut decisions: Vec<(u32, DoctrineMode)> = Vec::new();

        for agent in &mut self.rl_agents {
            if !agent.enabled || agent.multi_agent {
                continue;
            }
            agent.tick_counter += 1;
            if agent.tick_counter % RL_DECISION_INTERVAL != 0 {
                continue;
            }

            let group = agent.group;
            let other_group = if group == 0 { 1 } else { 0 };

            let own_count = self.drones.iter()
                .filter(|d| {
                    if group == 0 { d.id < self.group_split_id }
                    else { d.id >= self.group_split_id }
                })
                .count() as f32;
            let enemy_count = self.drones.iter()
                .filter(|d| {
                    if other_group == 0 { d.id < self.group_split_id }
                    else { d.id >= self.group_split_id }
                })
                .count() as f32;

            let friendly_centroid = self.protected_zones
                .get(&group)
                .filter(|zones| !zones.is_empty())
                .map(|zones| {
                    let cx = zones.iter().map(|p| p.x()).sum::<f32>() / zones.len() as f32;
                    let cy = zones.iter().map(|p| p.y()).sum::<f32>() / zones.len() as f32;
                    (cx, cy)
                });

            let nearby_threats = if let Some((cx, cy)) = friendly_centroid {
                let centroid = Vec2::new(cx, cy);
                let threat_radius = DETONATION_RADIUS * RL_THREAT_RADIUS_MULTIPLIER;
                self.drones.iter()
                    .filter(|d| {
                        let is_enemy = if other_group == 0 { d.id < self.group_split_id }
                            else { d.id >= self.group_split_id };
                        is_enemy && self.lib_bounds.distance(
                            centroid,
                            d.agent.state().pos.as_vec2(),
                        ) <= threat_radius
                    })
                    .count() as f32
            } else {
                0.0
            };

            let (defend_frac, attack_frac) = self.strategies.iter()
                .find(|s| s.group() == Some(group))
                .and_then(|s| s.defend_attack_counts())
                .map(|(d, a)| {
                    let total = own_count;
                    if total > 0.0 {
                        (d as f32 / total, a as f32 / total)
                    } else {
                        (0.0, 0.0)
                    }
                })
                .unwrap_or((0.5, 0.5));

            let obs = [
                (own_count / agent.initial_own_drones).clamp(0.0, 1.0),
                (enemy_count / agent.initial_enemy_drones).clamp(0.0, 1.0),
                (agent.current_friendly_targets / agent.initial_friendly_targets).clamp(0.0, 1.0),
                (agent.current_enemy_targets / agent.initial_enemy_targets).clamp(0.0, 1.0),
                (nearby_threats / RL_MAX_NEARBY_THREATS).clamp(0.0, 1.0),
                (agent.tick_counter as f32 / RL_MAX_TICKS).clamp(0.0, 1.0),
                defend_frac.clamp(0.0, 1.0),
                attack_frac.clamp(0.0, 1.0),
            ];

            let action = agent.model.act(&obs, &[], 0);

            match action {
                0 => decisions.push((group, DoctrineMode::Aggressive)),
                1 => decisions.push((group, DoctrineMode::Defensive)),
                _ => {}
            }
        }

        for (group, mode) in decisions {
            for strategy in &mut self.strategies {
                if strategy.group() == Some(group) {
                    strategy.set_doctrine_mode(mode);
                }
            }
        }
    }

    /// Tick multi-agent RL: per-drone entity-based observation + action (V2).
    fn tick_rl_agents_multi(&mut self) {
        let mut drone_actions: Vec<(u32, u32)> = Vec::new();

        for agent in &mut self.rl_agents {
            if !agent.enabled || !agent.multi_agent { continue; }
            agent.tick_counter += 1;
            if agent.tick_counter % RL_MULTI_DECISION_INTERVAL != 0 { continue; }

            let group = agent.group;
            let other_group = if group == 0 { 1 } else { 0 };
            let w = self.lib_bounds.width();
            let diag = (w * w + w * w).sqrt();

            // Get target positions from protected_zones.
            let friendly_targets: Vec<Position> = self.protected_zones.get(&group).cloned().unwrap_or_default();
            let enemy_targets: Vec<Position> = self.protected_zones.get(&other_group).cloned().unwrap_or_default();

            // Global features.
            let own_count = self.drones.iter()
                .filter(|d| Self::drone_in_group(d.id, group, self.group_split_id))
                .count() as f32;
            let enemy_count = self.drones.iter()
                .filter(|d| Self::drone_in_group(d.id, other_group, self.group_split_id))
                .count() as f32;

            let friendly_centroid = if !friendly_targets.is_empty() {
                let cx = friendly_targets.iter().map(|p| p.x()).sum::<f32>() / friendly_targets.len() as f32;
                let cy = friendly_targets.iter().map(|p| p.y()).sum::<f32>() / friendly_targets.len() as f32;
                Some(Vec2::new(cx, cy))
            } else {
                None
            };

            let nearby_threats = if let Some(centroid) = friendly_centroid {
                let threat_radius = DETONATION_RADIUS * RL_THREAT_RADIUS_MULTIPLIER;
                self.drones.iter()
                    .filter(|d| Self::drone_in_group(d.id, other_group, self.group_split_id)
                        && self.lib_bounds.distance(centroid, d.agent.state().pos.as_vec2()) <= threat_radius)
                    .count() as f32
            } else { 0.0 };

            // Build sorted alive same-group IDs for relative_alive_index.
            let mut alive_same_group: Vec<u32> = self.drones.iter()
                .filter(|d| Self::drone_in_group(d.id, group, self.group_split_id))
                .map(|d| d.id)
                .collect();
            alive_same_group.sort_unstable();
            let alive_count = alive_same_group.len().max(1) as f32;

            let drone_indices: Vec<usize> = self.drones.iter().enumerate()
                .filter(|(_, d)| Self::drone_in_group(d.id, group, self.group_split_id))
                .map(|(i, _)| i)
                .collect();

            for &drone_idx in &drone_indices {
                let drone = &self.drones[drone_idx];
                let state = drone.agent.state();
                let my_pos = state.pos.as_vec2();
                let my_heading = state.hdg.radians();

                // === EGO (25-dim) ===
                let mut ego = [0.0f32; 25];
                ego[0] = state.pos.x() / w;
                ego[1] = state.pos.y() / w;
                ego[2] = state.vel.as_vec2().x / RL_MAX_VELOCITY;
                ego[3] = state.vel.as_vec2().y / RL_MAX_VELOCITY;
                ego[4] = my_heading / (2.0 * std::f32::consts::PI);
                let last_action = self.rl_last_actions.get(&drone.id).copied().unwrap_or(0) as f32;
                ego[5] = last_action / RL_ACT_DIM as f32;

                // Task type one-hot [6..15] and phase [15]
                if let Some((task_name, phase_name)) = drone.agent.task_info() {
                    let task_idx = match task_name {
                        "Attack" => 1, "AttackEvasive" => 2, "Defend" => 3,
                        "Intercept" => 4, "InterceptGroup" => 5, "Evade" => 6,
                        "Loiter" => 7, "Patrol" => 8, _ => 0,
                    };
                    if task_idx > 0 && task_idx < 9 { ego[6 + task_idx] = 1.0; }
                    ego[15] = match phase_name {
                        "navigate" | "approach" | "transit" | "orbit" => 0.0,
                        "engage" | "pursue" | "pursue_cluster" | "flee" | "hold" | "loiter" => 0.33,
                        "terminal" => 0.67,
                        "complete" | "done" => 1.0,
                        _ => 0.0,
                    };
                }

                // Relative alive index
                let my_rank = alive_same_group.iter().position(|&id| id == drone.id).unwrap_or(0) as f32;
                ego[16] = my_rank / alive_count;

                // Global features [17..25]
                ego[17] = own_count / agent.initial_own_drones;
                ego[18] = enemy_count / agent.initial_enemy_drones;
                ego[19] = (agent.current_friendly_targets / agent.initial_friendly_targets).clamp(0.0, 1.0);
                ego[20] = (agent.current_enemy_targets / agent.initial_enemy_targets).clamp(0.0, 1.0);
                ego[21] = (agent.tick_counter as f32 / RL_MAX_TICKS).clamp(0.0, 1.0);
                ego[22] = (nearby_threats / RL_MAX_NEARBY_THREATS).clamp(0.0, 1.0);

                let nearest_friendly_dist = self.drones.iter()
                    .filter(|d| Self::drone_in_group(d.id, group, self.group_split_id) && d.id != drone.id)
                    .map(|d| self.lib_bounds.distance(my_pos, d.agent.state().pos.as_vec2()))
                    .fold(f32::INFINITY, f32::min);
                ego[23] = if nearest_friendly_dist.is_finite() { nearest_friendly_dist / diag } else { 1.0 };

                let friendlies_in_blast = self.drones.iter()
                    .filter(|d| Self::drone_in_group(d.id, group, self.group_split_id) && d.id != drone.id)
                    .filter(|d| self.lib_bounds.distance(my_pos, d.agent.state().pos.as_vec2()) <= DETONATION_RADIUS)
                    .count() as f32;
                ego[24] = (friendlies_in_blast / 8.0).clamp(0.0, 1.0);

                // === ENTITY TOKENS (8-dim each) ===
                let mut entities: Vec<f32> = Vec::new();

                // Helper closure to normalize angle to [-PI, PI]
                let normalize_angle = |a: f32| -> f32 {
                    let mut x = a;
                    while x > std::f32::consts::PI { x -= 2.0 * std::f32::consts::PI; }
                    while x < -std::f32::consts::PI { x += 2.0 * std::f32::consts::PI; }
                    x
                };

                // Enemy drones (sorted by distance)
                let mut enemy_drones: Vec<(f32, usize)> = self.drones.iter().enumerate()
                    .filter(|(_, d)| Self::drone_in_group(d.id, other_group, self.group_split_id))
                    .map(|(idx, d)| (self.lib_bounds.distance(my_pos, d.agent.state().pos.as_vec2()), idx))
                    .collect();
                enemy_drones.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for &(_, idx) in &enemy_drones {
                    let d = &self.drones[idx];
                    let es = d.agent.state();
                    let epos = es.pos.as_vec2();
                    let dx = epos.x - my_pos.x;
                    let dy = epos.y - my_pos.y;
                    let dist = self.lib_bounds.distance(my_pos, epos);
                    let heading_rel = normalize_angle(dy.atan2(dx) - my_heading) / std::f32::consts::PI;
                    entities.extend_from_slice(&[
                        dx / w, dy / w, dist / diag,
                        es.vel.as_vec2().x / RL_MAX_VELOCITY,
                        es.vel.as_vec2().y / RL_MAX_VELOCITY,
                        heading_rel,
                        0.0, // ENEMY_DRONE type flag (entity_type::ENEMY_DRONE)
                        1.0, // alive
                        0.0, // assignment_count
                        0.0, // is_current_target
                    ]);
                }

                // Friendly drones (excl self, sorted by distance)
                let mut friendly_drones: Vec<(f32, usize)> = self.drones.iter().enumerate()
                    .filter(|(_, d)| Self::drone_in_group(d.id, group, self.group_split_id) && d.id != drone.id)
                    .map(|(idx, d)| (self.lib_bounds.distance(my_pos, d.agent.state().pos.as_vec2()), idx))
                    .collect();
                friendly_drones.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for &(_, idx) in &friendly_drones {
                    let d = &self.drones[idx];
                    let es = d.agent.state();
                    let epos = es.pos.as_vec2();
                    let dx = epos.x - my_pos.x;
                    let dy = epos.y - my_pos.y;
                    let dist = self.lib_bounds.distance(my_pos, epos);
                    let heading_rel = normalize_angle(dy.atan2(dx) - my_heading) / std::f32::consts::PI;
                    entities.extend_from_slice(&[
                        dx / w, dy / w, dist / diag,
                        es.vel.as_vec2().x / RL_MAX_VELOCITY,
                        es.vel.as_vec2().y / RL_MAX_VELOCITY,
                        heading_rel,
                        0.33, // FRIENDLY_DRONE type flag (entity_type::FRIENDLY_DRONE)
                        1.0,
                        0.0, // assignment_count
                        0.0, // is_current_target
                    ]);
                }

                // Enemy targets (sorted by distance)
                let mut sorted_et: Vec<(f32, &Position)> = enemy_targets.iter()
                    .map(|p| (self.lib_bounds.distance(my_pos, p.as_vec2()), p))
                    .collect();
                sorted_et.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for &(_, p) in &sorted_et {
                    let dx = p.x() - my_pos.x;
                    let dy = p.y() - my_pos.y;
                    let dist = self.lib_bounds.distance(my_pos, p.as_vec2());
                    let heading_rel = normalize_angle(dy.atan2(dx) - my_heading) / std::f32::consts::PI;
                    entities.extend_from_slice(&[
                        dx / w, dy / w, dist / diag,
                        0.0, 0.0, // targets have no velocity
                        heading_rel,
                        0.67, // ENEMY_TARGET type flag (entity_type::ENEMY_TARGET)
                        1.0,
                        0.0, // assignment_count
                        0.0, // is_current_target
                    ]);
                }

                // Friendly targets (sorted by distance)
                let mut sorted_ft: Vec<(f32, &Position)> = friendly_targets.iter()
                    .map(|p| (self.lib_bounds.distance(my_pos, p.as_vec2()), p))
                    .collect();
                sorted_ft.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for &(_, p) in &sorted_ft {
                    let dx = p.x() - my_pos.x;
                    let dy = p.y() - my_pos.y;
                    let dist = self.lib_bounds.distance(my_pos, p.as_vec2());
                    let heading_rel = normalize_angle(dy.atan2(dx) - my_heading) / std::f32::consts::PI;
                    entities.extend_from_slice(&[
                        dx / w, dy / w, dist / diag,
                        0.0, 0.0,
                        heading_rel,
                        1.0, // FRIENDLY_TARGET type flag (entity_type::FRIENDLY_TARGET)
                        1.0,
                        0.0, // assignment_count
                        0.0, // is_current_target
                    ]);
                }

                let n_entities = entities.len() / 10;

                // Run V2 policy.
                let action = agent.model.act(&ego, &entities, n_entities);
                drone_actions.push((drone.id, action));
            }
        }

        // Apply actions.
        for (drone_id, action) in drone_actions {
            self.apply_multi_rl_action(drone_id, action);
        }
    }

    /// Apply a per-drone RL action (0-12) to a specific drone.
    ///
    /// Action space (13 actions):
    ///   0  = Attack nearest enemy target (direct)
    ///   1  = Attack farthest enemy target (direct)
    ///   2  = Attack least-defended enemy target (direct)
    ///   3  = Attack nearest enemy target (evasive)
    ///   4  = Attack farthest enemy target (evasive)
    ///   5  = Attack least-defended enemy target (evasive)
    ///   6  = Intercept nearest enemy drone
    ///   7  = Intercept 2nd nearest enemy drone
    ///   8  = Intercept enemy cluster
    ///   9  = Defend nearest friendly target (tight: 100m/300m)
    ///   10 = Defend nearest friendly target (wide: 250m/600m)
    ///   11 = Patrol perimeter
    ///   12 = Evade nearest threat
    fn apply_multi_rl_action(&mut self, drone_id: u32, action: u32) {
        // Record action for observation encoding.
        self.rl_last_actions.insert(drone_id, action);

        let group = if drone_id < self.group_split_id { 0u32 } else { 1u32 };
        let other_group = if group == 0 { 1 } else { 0 };

        let drone_pos = match self.drones.iter().find(|d| d.id == drone_id) {
            Some(d) => d.agent.state().pos.as_vec2(),
            None => return,
        };

        match action {
            // --- Attack nearest enemy target (direct) ---
            0 => {
                let target_pos = self.nth_nearest_enemy_target(drone_pos, other_group, 0);
                if let Some(target) = target_pos {
                    self.clear_drone_tasks_wasm(drone_id);
                    if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(AttackTask::new(target, DETONATION_RADIUS)));
                    }
                    self.attack_targets.insert(drone_id, target);
                }
            }

            // --- Attack farthest enemy target (direct) ---
            1 => {
                let target_pos = self.farthest_enemy_target(drone_pos, other_group);
                if let Some(target) = target_pos {
                    self.clear_drone_tasks_wasm(drone_id);
                    if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(AttackTask::new(target, DETONATION_RADIUS)));
                    }
                    self.attack_targets.insert(drone_id, target);
                }
            }

            // --- Attack least-defended enemy target (direct) ---
            2 => {
                let target_pos = self.least_defended_enemy_target(drone_pos, other_group);
                if let Some(target) = target_pos {
                    self.clear_drone_tasks_wasm(drone_id);
                    if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(AttackTask::new(target, DETONATION_RADIUS)));
                    }
                    self.attack_targets.insert(drone_id, target);
                }
            }

            // --- Attack nearest enemy target (evasive) ---
            3 => {
                let target_pos = self.nth_nearest_enemy_target(drone_pos, other_group, 0);
                if let Some(target) = target_pos {
                    self.clear_drone_tasks_wasm(drone_id);
                    if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(AttackTask::new_evasive(target, DETONATION_RADIUS)));
                    }
                    self.attack_targets.insert(drone_id, target);
                }
            }

            // --- Attack farthest enemy target (evasive) ---
            4 => {
                let target_pos = self.farthest_enemy_target(drone_pos, other_group);
                if let Some(target) = target_pos {
                    self.clear_drone_tasks_wasm(drone_id);
                    if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(AttackTask::new_evasive(target, DETONATION_RADIUS)));
                    }
                    self.attack_targets.insert(drone_id, target);
                }
            }

            // --- Attack least-defended enemy target (evasive) ---
            5 => {
                let target_pos = self.least_defended_enemy_target(drone_pos, other_group);
                if let Some(target) = target_pos {
                    self.clear_drone_tasks_wasm(drone_id);
                    if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(AttackTask::new_evasive(target, DETONATION_RADIUS)));
                    }
                    self.attack_targets.insert(drone_id, target);
                }
            }

            // --- Intercept nearest enemy drone ---
            6 => {
                let enemy_id = self.nth_nearest_enemy_drone_wasm(drone_pos, other_group, 0);
                if let Some(eid) = enemy_id {
                    self.clear_drone_tasks_wasm(drone_id);
                    if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                        let task = InterceptTask::new(drone_id as usize, eid as usize, group, DETONATION_RADIUS);
                        drone.agent.set_task(Box::new(task));
                    }
                    self.intercept_targets.insert(drone_id, eid);
                }
            }

            // --- Intercept 2nd nearest enemy drone ---
            7 => {
                let enemy_id = self.nth_nearest_enemy_drone_wasm(drone_pos, other_group, 1);
                if let Some(eid) = enemy_id {
                    self.clear_drone_tasks_wasm(drone_id);
                    if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                        let task = InterceptTask::new(drone_id as usize, eid as usize, group, DETONATION_RADIUS);
                        drone.agent.set_task(Box::new(task));
                    }
                    self.intercept_targets.insert(drone_id, eid);
                }
            }

            // --- Intercept enemy cluster ---
            8 => {
                self.clear_drone_tasks_wasm(drone_id);
                if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                    drone.agent.set_task(Box::new(InterceptGroupTask::new(group, DETONATION_RADIUS)));
                }
            }

            // --- Defend nearest friendly target (tight: 100m/300m) ---
            9 => {
                let target_pos = self.nth_nearest_friendly_target(drone_pos, group, 0);
                if let Some(center) = target_pos {
                    self.clear_drone_tasks_wasm(drone_id);
                    if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(DefendTask::new(
                            drone_id as usize, group, center, 100.0, 300.0, DETONATION_RADIUS,
                        )));
                    }
                }
            }

            // --- Defend nearest friendly target (wide: 250m/600m) ---
            10 => {
                let target_pos = self.nth_nearest_friendly_target(drone_pos, group, 0);
                if let Some(center) = target_pos {
                    self.clear_drone_tasks_wasm(drone_id);
                    if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(DefendTask::new(
                            drone_id as usize, group, center, 250.0, 600.0, DETONATION_RADIUS,
                        )));
                    }
                }
            }

            // --- Patrol perimeter ---
            11 => {
                let friendly_positions: Vec<Position> = self.protected_zones
                    .get(&group)
                    .cloned()
                    .unwrap_or_default();
                if !friendly_positions.is_empty() {
                    let waypoints = Self::build_patrol_route_static(&friendly_positions, RL_PATROL_STANDOFF);
                    self.clear_drone_tasks_wasm(drone_id);
                    if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(PatrolTask::new(waypoints, 50.0, 2.0)));
                    }
                }
            }

            // --- Evade nearest threat ---
            12 => {
                self.clear_drone_tasks_wasm(drone_id);
                if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                    drone.agent.set_task(Box::new(EvadeTask::new(group)));
                }
            }

            _ => {} // Unknown action — ignore
        }
    }

    /// Get the farthest enemy target position (from protected_zones).
    fn farthest_enemy_target(&self, drone_pos: Vec2, enemy_group: u32) -> Option<Position> {
        let targets = self.protected_zones.get(&enemy_group)?;
        targets.iter()
            .max_by(|a, b| {
                let da = self.lib_bounds.distance(drone_pos, a.as_vec2());
                let db = self.lib_bounds.distance(drone_pos, b.as_vec2());
                da.partial_cmp(&db).unwrap()
            })
            .copied()
    }

    /// Get the least-defended enemy target (fewest attackers assigned within 50m).
    fn least_defended_enemy_target(&self, _drone_pos: Vec2, enemy_group: u32) -> Option<Position> {
        let targets = self.protected_zones.get(&enemy_group)?;
        if targets.is_empty() { return None; }
        targets.iter()
            .min_by_key(|t| {
                self.attack_targets.values()
                    .filter(|at_pos| self.lib_bounds.distance(t.as_vec2(), at_pos.as_vec2()) <= 50.0)
                    .count()
            })
            .copied()
    }


    /// Clear attack/intercept tracking for a drone.
    fn clear_drone_tasks_wasm(&mut self, drone_id: u32) {
        self.attack_targets.remove(&drone_id);
        self.intercept_targets.remove(&drone_id);
    }

    /// Get the Nth nearest enemy target position (from protected_zones).
    fn nth_nearest_enemy_target(&self, drone_pos: Vec2, enemy_group: u32, nth: usize) -> Option<Position> {
        let targets = self.protected_zones.get(&enemy_group)?;
        let mut sorted: Vec<(f32, Position)> = targets.iter()
            .map(|p| (self.lib_bounds.distance(drone_pos, p.as_vec2()), *p))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        sorted.get(nth).map(|(_, p)| *p)
    }

    /// Get the Nth nearest friendly target position (from protected_zones).
    fn nth_nearest_friendly_target(&self, drone_pos: Vec2, own_group: u32, nth: usize) -> Option<Position> {
        let targets = self.protected_zones.get(&own_group)?;
        let mut sorted: Vec<(f32, Position)> = targets.iter()
            .map(|p| (self.lib_bounds.distance(drone_pos, p.as_vec2()), *p))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        sorted.get(nth).map(|(_, p)| *p)
    }

    /// Get the ID of the Nth nearest enemy drone.
    fn nth_nearest_enemy_drone_wasm(&self, drone_pos: Vec2, enemy_group: u32, nth: usize) -> Option<u32> {
        let mut enemies: Vec<(f32, u32)> = self.drones.iter()
            .filter(|d| Self::drone_in_group(d.id, enemy_group, self.group_split_id))
            .map(|d| (self.lib_bounds.distance(drone_pos, d.agent.state().pos.as_vec2()), d.id))
            .collect();
        enemies.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        enemies.get(nth).map(|(_, id)| *id)
    }

    /// Check if a drone ID belongs to a given group.
    fn drone_in_group(drone_id: u32, group: u32, split_id: u32) -> bool {
        if group == 0 { drone_id < split_id } else { drone_id >= split_id }
    }

    /// Build patrol waypoints around target positions (static helper).
    fn build_patrol_route_static(targets: &[Position], standoff: f32) -> Vec<Position> {
        if targets.is_empty() {
            return Vec::new();
        }
        let cx = targets.iter().map(|p| p.x()).sum::<f32>() / targets.len() as f32;
        let cy = targets.iter().map(|p| p.y()).sum::<f32>() / targets.len() as f32;

        if targets.len() <= 2 {
            return (0..4)
                .map(|i| {
                    let angle = (i as f32 / 4.0) * std::f32::consts::TAU;
                    Position::new(cx + standoff * angle.cos(), cy + standoff * angle.sin())
                })
                .collect();
        }

        // Push hull points outward from centroid.
        targets.iter()
            .map(|p| {
                let dx = p.x() - cx;
                let dy = p.y() - cy;
                let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                Position::new(p.x() + (dx / dist) * standoff, p.y() + (dy / dist) * standoff)
            })
            .collect()
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

        // If formations exist, assign the route to each formation's leader.
        // Followers stay in position-tracking mode.
        if !self.formations.is_empty() {
            for formation in &mut self.formations {
                formation.leader_route = Some(route.clone());

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

                formation.route_mode = false;
                formation.leader_target = None;
            }
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

    /// Assign a route to a subset of drones by ID range [start_id, end_id).
    /// If these drones have a formation, only the leader gets the route.
    pub fn assign_route_range(&mut self, waypoints: Vec<Point>, start_id: u32, end_id: u32) {
        let waypoint_vec: Vec<Position> = waypoints
            .iter()
            .map(|p| self.world_scale.point_px_to_position(*p))
            .collect();
        let route: Arc<[Position]> = Arc::from(waypoint_vec);

        // Check if this range has an associated formation
        let formation_idx = self.formations.iter().position(|f| {
            f.drone_ids.iter().any(|id| *id >= start_id && *id < end_id)
        });

        if let Some(idx) = formation_idx {
            let formation = &mut self.formations[idx];
            formation.leader_route = Some(route.clone());

            if let Some(leader_id) = formation.leader_id {
                let leader_nearest_idx = self.drones.iter()
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
                    .unwrap_or(0);

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

            let formation = &mut self.formations[idx];
            formation.route_mode = false;
            formation.leader_target = None;
            return;
        }

        // No formation - all drones in range follow the route directly
        for drone in &mut self.drones {
            if drone.id >= start_id && drone.id < end_id {
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
pub fn attack_target(handle: &mut SwarmHandle, drone_id: u32, target_x: f32, target_y: f32) {
    handle.swarm.attack_target(drone_id, target_x, target_y);
}

#[wasm_bindgen]
pub fn intercept_drone(handle: &mut SwarmHandle, attacker_id: u32, target_drone_id: u32) {
    handle.swarm.intercept_drone(attacker_id, target_drone_id);
}

#[wasm_bindgen]
pub fn loiter_at(handle: &mut SwarmHandle, drone_id: u32, target_x: f32, target_y: f32) {
    handle.swarm.loiter_at(drone_id, target_x, target_y);
}

#[wasm_bindgen]
pub fn patrol_route(handle: &mut SwarmHandle, drone_id: u32, waypoints: JsValue, loiter_duration: f32) -> Result<(), JsValue> {
    let points: Vec<Point> = serde_wasm_bindgen::from_value(waypoints)
        .map_err(|e| JsValue::from_str(&format!("Invalid waypoints: {}", e)))?;
    handle.swarm.patrol_route(drone_id, &points, loiter_duration);
    Ok(())
}

#[wasm_bindgen]
pub fn defend_position(
    handle: &mut SwarmHandle,
    drone_id: u32,
    center_x: f32,
    center_y: f32,
    orbit_radius: f32,
    engage_radius: f32,
) {
    handle.swarm.defend_position(drone_id, center_x, center_y, orbit_radius, engage_radius);
}

#[wasm_bindgen]
pub fn return_to_formation(handle: &mut SwarmHandle, drone_id: u32, group_start: u32, group_end: u32) {
    handle.swarm.return_to_formation(drone_id, group_start, group_end);
}

#[wasm_bindgen]
pub fn set_strategy_defend_area(
    handle: &mut SwarmHandle,
    drone_ids: JsValue,
    center_x: f32,
    center_y: f32,
    radius: f32,
) -> Result<(), JsValue> {
    let ids: Vec<u32> = serde_wasm_bindgen::from_value(drone_ids)
        .map_err(|e| JsValue::from_str(&format!("Invalid drone IDs: {}", e)))?;
    handle.swarm.set_strategy_defend_area(&ids, center_x, center_y, radius);
    Ok(())
}

#[wasm_bindgen]
pub fn set_strategy_attack_zone(
    handle: &mut SwarmHandle,
    drone_ids: JsValue,
    target_positions: JsValue,
) -> Result<(), JsValue> {
    let ids: Vec<u32> = serde_wasm_bindgen::from_value(drone_ids)
        .map_err(|e| JsValue::from_str(&format!("Invalid drone IDs: {}", e)))?;
    let positions: Vec<Point> = serde_wasm_bindgen::from_value(target_positions)
        .map_err(|e| JsValue::from_str(&format!("Invalid target positions: {}", e)))?;
    handle.swarm.set_strategy_attack_zone(&ids, &positions);
    Ok(())
}

#[wasm_bindgen]
pub fn set_strategy_patrol_perimeter(
    handle: &mut SwarmHandle,
    drone_ids: JsValue,
    waypoints: JsValue,
    loiter_duration: f32,
) -> Result<(), JsValue> {
    let ids: Vec<u32> = serde_wasm_bindgen::from_value(drone_ids)
        .map_err(|e| JsValue::from_str(&format!("Invalid drone IDs: {}", e)))?;
    let points: Vec<Point> = serde_wasm_bindgen::from_value(waypoints)
        .map_err(|e| JsValue::from_str(&format!("Invalid waypoints: {}", e)))?;
    handle.swarm.set_strategy_patrol_perimeter(&ids, &points, loiter_duration);
    Ok(())
}

#[wasm_bindgen]
pub fn clear_strategies(handle: &mut SwarmHandle) {
    handle.swarm.clear_strategies();
}

#[wasm_bindgen]
pub fn set_doctrine(
    handle: &mut SwarmHandle,
    drone_ids: JsValue,
    friendly_targets: JsValue,
    enemy_targets: JsValue,
    patrol_waypoints: JsValue,
    mode: &str,
) -> Result<(), JsValue> {
    let ids: Vec<u32> = serde_wasm_bindgen::from_value(drone_ids)
        .map_err(|e| JsValue::from_str(&format!("Invalid drone IDs: {}", e)))?;
    let friendly: Vec<Point> = serde_wasm_bindgen::from_value(friendly_targets)
        .map_err(|e| JsValue::from_str(&format!("Invalid friendly targets: {}", e)))?;
    let enemy: Vec<Point> = serde_wasm_bindgen::from_value(enemy_targets)
        .map_err(|e| JsValue::from_str(&format!("Invalid enemy targets: {}", e)))?;
    let waypoints: Vec<Point> = serde_wasm_bindgen::from_value(patrol_waypoints)
        .map_err(|e| JsValue::from_str(&format!("Invalid patrol waypoints: {}", e)))?;
    let doctrine_mode = match mode {
        "defensive" => DoctrineMode::Defensive,
        _ => DoctrineMode::Aggressive,
    };
    handle.swarm.set_doctrine(&ids, &friendly, &enemy, &waypoints, doctrine_mode);
    Ok(())
}

#[wasm_bindgen]
pub fn update_doctrine_targets(
    handle: &mut SwarmHandle,
    group: u32,
    friendly_targets: JsValue,
    enemy_targets: JsValue,
) -> Result<(), JsValue> {
    let friendly: Vec<Point> = serde_wasm_bindgen::from_value(friendly_targets)
        .map_err(|e| JsValue::from_str(&format!("Invalid friendly targets: {}", e)))?;
    let enemy: Vec<Point> = serde_wasm_bindgen::from_value(enemy_targets)
        .map_err(|e| JsValue::from_str(&format!("Invalid enemy targets: {}", e)))?;
    handle.swarm.update_doctrine_targets(group, &friendly, &enemy);
    Ok(())
}

#[wasm_bindgen]
pub fn set_group_split(handle: &mut SwarmHandle, split_id: u32) {
    handle.swarm.set_group_split(split_id);
}

#[wasm_bindgen]
pub fn set_protected_zones(handle: &mut SwarmHandle, group: u32, positions: JsValue) -> Result<(), JsValue> {
    let positions: Vec<Point> = serde_wasm_bindgen::from_value(positions)
        .map_err(|e| JsValue::from_str(&format!("Invalid positions: {}", e)))?;
    handle.swarm.set_protected_zones(group, positions);
    Ok(())
}

// ========================================================================
// RL Agent WASM Bindings
// ========================================================================

#[wasm_bindgen]
pub fn load_rl_model(
    handle: &mut SwarmHandle,
    group: u32,
    model_json: &str,
    initial_own_drones: f32,
    initial_enemy_drones: f32,
    initial_friendly_targets: f32,
    initial_enemy_targets: f32,
) -> Result<(), JsValue> {
    handle.swarm.load_rl_model(
        group,
        model_json,
        initial_own_drones,
        initial_enemy_drones,
        initial_friendly_targets,
        initial_enemy_targets,
    ).map_err(|e| JsValue::from_str(&e))
}

#[wasm_bindgen]
pub fn load_rl_model_multi(
    handle: &mut SwarmHandle,
    group: u32,
    model_json: &str,
    initial_own_drones: f32,
    initial_enemy_drones: f32,
    initial_friendly_targets: f32,
    initial_enemy_targets: f32,
) -> Result<(), JsValue> {
    handle.swarm.load_rl_model_multi(
        group,
        model_json,
        initial_own_drones,
        initial_enemy_drones,
        initial_friendly_targets,
        initial_enemy_targets,
    ).map_err(|e| JsValue::from_str(&e))
}

#[wasm_bindgen]
pub fn load_rl_normalizers(
    handle: &mut SwarmHandle,
    group: u32,
    normalizer_json: &str,
) -> Result<(), JsValue> {
    handle.swarm.load_rl_normalizers(group, normalizer_json)
        .map_err(|e| JsValue::from_str(&e))
}

#[wasm_bindgen]
pub fn set_rl_agent_enabled(handle: &mut SwarmHandle, group: u32, enabled: bool) {
    handle.swarm.set_rl_agent_enabled(group, enabled);
}

#[wasm_bindgen]
pub fn update_rl_targets(handle: &mut SwarmHandle, group: u32, friendly_count: f32, enemy_count: f32) {
    handle.swarm.update_rl_targets(group, friendly_count, enemy_count);
}

#[wasm_bindgen]
pub fn remove_rl_agent(handle: &mut SwarmHandle, group: u32) {
    handle.swarm.remove_rl_agent(group);
}

#[wasm_bindgen]
pub fn detonate_drone(handle: &mut SwarmHandle, drone_id: u32) {
    handle.swarm.detonate_drone(drone_id);
}

#[wasm_bindgen]
pub fn detonate_selected(handle: &mut SwarmHandle) {
    handle.swarm.detonate_selected();
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
pub fn assign_route_range(
    handle: &mut SwarmHandle,
    waypoints: JsValue,
    start_id: u32,
    end_id: u32,
) -> Result<(), JsValue> {
    let waypoints: Vec<Point> = serde_wasm_bindgen::from_value(waypoints)
        .map_err(|e| JsValue::from_str(&format!("Invalid waypoints: {}", e)))?;
    handle.swarm.assign_route_range(waypoints, start_id, end_id);
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

/// Set formation for a specific range of drones [start_id, end_id).
#[wasm_bindgen]
pub fn set_formation_for_range(
    handle: &mut SwarmHandle,
    formation_type: &str,
    spacing: f32,
    start_id: u32,
    end_id: u32,
) {
    let spacing_m = handle.swarm.world_scale.px_to_meters(spacing);

    let group_ids: HashSet<u32> = handle.swarm.drones.iter()
        .filter(|d| d.id >= start_id && d.id < end_id)
        .map(|d| d.id)
        .collect();

    let drone_count = group_ids.len();
    let formation = match formation_type {
        "line" => FormationType::Line { spacing: spacing_m },
        "vee" => FormationType::Vee {
            spacing: spacing_m,
            angle: std::f32::consts::FRAC_PI_4,
        },
        "diamond" => FormationType::Diamond { spacing: spacing_m },
        "circle" => FormationType::Circle { radius: spacing_m },
        "grid" => {
            let cols = (drone_count as f32).sqrt().ceil() as usize;
            FormationType::Grid { spacing: spacing_m, cols }
        }
        "chevron" => FormationType::Chevron {
            spacing: spacing_m,
            angle: std::f32::consts::FRAC_PI_4,
        },
        _ => FormationType::Line { spacing: spacing_m },
    };

    let leader_id = group_ids.iter().map(|&id| id as usize).min();
    handle.swarm.set_formation_for_group(formation, group_ids, leader_id);
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
