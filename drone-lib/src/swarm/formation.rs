//! Formation behaviors for swarm coordination.
//!
//! Provides formation flying capabilities with direct drone control.

use super::traits::SwarmBehavior;
use crate::agent::DroneAgent;
use crate::messages::{FormationCommand, FormationSlot, VelocityConsensus};
use crate::types::{Bounds, DroneInfo, Position, Vec2};

/// Compute the world position of a formation slot given the formation center,
/// heading, and the slot's local offset.
pub fn compute_slot_world_position(center: Position, heading: f32, offset: Vec2) -> Vec2 {
    let (sin_h, cos_h) = heading.sin_cos();
    Vec2::new(
        center.x() + offset.x * cos_h - offset.y * sin_h,
        center.y() + offset.x * sin_h + offset.y * cos_h,
    )
}

/// Compute offset waypoints for a follower drone given the leader's route.
///
/// For each leader waypoint, computes the route heading (bisector of incoming
/// and outgoing segment headings), then applies the formation slot offset
/// rotated by that heading to produce the follower's offset waypoint.
///
/// For closed-loop routes (which wrap), the heading at waypoint\[0\] uses
/// the direction from waypoint\[last\] as the incoming segment.
pub fn compute_offset_route(
    waypoints: &[Position],
    offset: Vec2,
    bounds: &Bounds,
) -> Vec<Position> {
    let n = waypoints.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        let pos = compute_slot_world_position(waypoints[0], 0.0, offset);
        return vec![Position::new(pos.x, pos.y)];
    }

    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        // Compute incoming and outgoing segment headings.
        // For a closed loop: wrap indices.
        let prev = if i == 0 { n - 1 } else { i - 1 };
        let next = if i == n - 1 { 0 } else { i + 1 };

        let d_in = bounds.delta(waypoints[prev].as_vec2(), waypoints[i].as_vec2());
        let d_out = bounds.delta(waypoints[i].as_vec2(), waypoints[next].as_vec2());

        let hdg_in = d_in.y.atan2(d_in.x);
        let hdg_out = d_out.y.atan2(d_out.x);

        // Bisector heading: average of incoming and outgoing headings,
        // handling angle wrapping.
        let bisector = angle_bisector(hdg_in, hdg_out);

        let pos = compute_slot_world_position(waypoints[i], bisector, offset);
        result.push(Position::new(pos.x, pos.y));
    }

    result
}

/// Compute the bisector angle between two headings.
///
/// Returns the angle that bisects the smaller arc between the two headings.
fn angle_bisector(a: f32, b: f32) -> f32 {
    let mut diff = b - a;
    // Normalize to [-PI, PI]
    while diff > std::f32::consts::PI {
        diff -= std::f32::consts::TAU;
    }
    while diff < -std::f32::consts::PI {
        diff += std::f32::consts::TAU;
    }
    a + diff * 0.5
}

/// Formation control default constants.
pub mod defaults {
    /// Station-keeping threshold (meters).
    pub const STATION_KEEPING_DIST: f32 = 25.0;

    /// Start deceleration distance (meters).
    pub const DECEL_START: f32 = 60.0;

    /// "At position" threshold (meters).
    pub const AT_POSITION_DIST: f32 = 7.5;

    /// Default formation spacing (meters).
    pub const DEFAULT_SPACING: f32 = 50.0;

    // Stability Thresholds
    /// Nominal slot deviation threshold (meters).
    pub const NOMINAL_DEVIATION: f32 = 10.0;

    /// Degraded slot deviation threshold (meters).
    pub const DEGRADED_DEVIATION: f32 = 30.0;

    /// Critical slot deviation threshold (meters).
    pub const CRITICAL_DEVIATION: f32 = 75.0;

    /// Nominal path lag threshold (meters along path).
    pub const NOMINAL_LAG: f32 = 15.0;

    /// Degraded path lag threshold (meters along path).
    pub const DEGRADED_LAG: f32 = 40.0;

    /// Critical path lag threshold (meters along path).
    pub const CRITICAL_LAG: f32 = 100.0;

    // Automatic Response
    /// Fraction of drones that can be degraded before formation slows.
    pub const DEGRADED_FRACTION: f32 = 0.25;

    /// Fraction of drones that can be critical before emergency slowdown.
    pub const CRITICAL_FRACTION: f32 = 0.1;

    /// Speed reduction factor when formation is degraded.
    pub const DEGRADED_SPEED_FACTOR: f32 = 0.7;

    /// Speed reduction factor when formation is critical.
    pub const CRITICAL_SPEED_FACTOR: f32 = 0.3;

    // Approach Gate (Rear Entry)
    /// Radius of the "capture ball" around each formation slot (meters).
    pub const APPROACH_BALL_RADIUS: f32 = 25.0;

    /// Distance behind the slot where the approach gate is located (meters).
    pub const APPROACH_GATE_OFFSET: f32 = 35.0;

    /// Angle tolerance for rear approach (radians).
    pub const APPROACH_ANGLE_TOLERANCE: f32 = 1.3;
}

/// Formation types with spacing parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FormationType {
    /// Line formation perpendicular to direction of travel.
    Line { spacing: f32 },
    /// V-shaped formation (wedge).
    Vee { spacing: f32, angle: f32 },
    /// Diamond formation (rhombus shape).
    Diamond { spacing: f32 },
    /// Circular formation around a center point.
    Circle { radius: f32 },
    /// Grid formation.
    Grid { spacing: f32, cols: usize },
    /// Chevron formation — nested V layers, each layer adds one more drone per side.
    Chevron { spacing: f32, angle: f32 },
}

impl FormationType {
    /// Compute formation slots for the given number of drones.
    ///
    /// Returns a vector of slots, where index 0 is typically the leader position.
    pub fn compute_slots(&self, drone_count: usize) -> Vec<FormationSlot> {
        match *self {
            FormationType::Line { spacing } => {
                // Drones arranged in a line perpendicular to heading
                let half = (drone_count as f32 - 1.0) / 2.0;
                (0..drone_count)
                    .map(|i| FormationSlot::new(
                        Vec2::new(0.0, (i as f32 - half) * spacing),
                        i as u8,
                    ))
                    .collect()
            }
            FormationType::Vee { spacing, angle } => {
                // Leader at front, others trail back in V shape
                let mut slots = Vec::with_capacity(drone_count);
                // Leader at origin
                slots.push(FormationSlot::new(Vec2::new(0.0, 0.0), 0));

                for i in 1..drone_count {
                    let side = if i % 2 == 1 { 1.0 } else { -1.0 };
                    let row = i.div_ceil(2) as f32;
                    let x = -row * spacing * angle.cos();
                    let y = side * row * spacing * angle.sin();
                    slots.push(FormationSlot::new(Vec2::new(x, y), i as u8));
                }
                slots
            }
            FormationType::Diamond { spacing } => {
                // Diamond/rhombus formation — rows expand then contract:
                //         *           row 0: 1 (leader)
                //        * *          row 1: 2
                //       * * *         row 2: 3  (widest, W=3 example)
                //        * *          row 3: 2
                //         *           row 4: 1
                // Complete diamond with max-width W has W^2 slots.
                // For N drones, fill front-to-back until all are placed.
                let mut slots = Vec::with_capacity(drone_count);
                if drone_count == 0 { return slots; }

                // Leader at front
                slots.push(FormationSlot::new(Vec2::new(0.0, 0.0), 0));
                if drone_count == 1 { return slots; }

                // Max row width W so that W^2 >= drone_count
                let w = (drone_count as f32).sqrt().ceil() as usize;

                let mut idx = 1u8;
                let mut row_num = 1usize;

                // Expanding rows: width 2, 3, ..., w
                for width in 2..=w {
                    if idx as usize >= drone_count { break; }
                    let x = -(row_num as f32) * spacing;
                    let count = width.min(drone_count - idx as usize);
                    let half = (count as f32 - 1.0) / 2.0;
                    for i in 0..count {
                        if idx as usize >= drone_count { break; }
                        let y = (i as f32 - half) * spacing;
                        slots.push(FormationSlot::new(Vec2::new(x, y), idx));
                        idx += 1;
                    }
                    row_num += 1;
                }

                // Contracting rows: width w-1, w-2, ..., 1
                for width in (1..w).rev() {
                    if idx as usize >= drone_count { break; }
                    let x = -(row_num as f32) * spacing;
                    let count = width.min(drone_count - idx as usize);
                    let half = (count as f32 - 1.0) / 2.0;
                    for i in 0..count {
                        if idx as usize >= drone_count { break; }
                        let y = (i as f32 - half) * spacing;
                        slots.push(FormationSlot::new(Vec2::new(x, y), idx));
                        idx += 1;
                    }
                    row_num += 1;
                }

                slots
            }
            FormationType::Circle { radius } => {
                // Drones arranged in a circle
                let angle_step = std::f32::consts::TAU / drone_count as f32;
                (0..drone_count)
                    .map(|i| {
                        let angle = i as f32 * angle_step;
                        FormationSlot::new(
                            Vec2::new(radius * angle.cos(), radius * angle.sin()),
                            i as u8,
                        )
                    })
                    .collect()
            }
            FormationType::Grid { spacing, cols } => {
                // Drones in a grid, leader at front-center
                let mut slots: Vec<FormationSlot> = (0..drone_count)
                    .map(|i| {
                        let col = i % cols;
                        let row = i / cols;
                        let x = -(row as f32) * spacing;
                        let y = (col as f32 - (cols as f32 - 1.0) / 2.0) * spacing;
                        FormationSlot::new(Vec2::new(x, y), i as u8)
                    })
                    .collect();

                // Ensure the leader (slot[0]) gets the center-most position.
                // Without this, odd column counts place (0,0) on a follower index,
                // causing that follower to fly onto the leader's position.
                if let Some(center_idx) = slots.iter().enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let da = a.offset.x * a.offset.x + a.offset.y * a.offset.y;
                        let db = b.offset.x * b.offset.x + b.offset.y * b.offset.y;
                        da.partial_cmp(&db).unwrap()
                    })
                    .map(|(i, _)| i)
                {
                    if center_idx != 0 {
                        slots.swap(0, center_idx);
                        slots[0].priority = 0;
                        slots[center_idx].priority = center_idx as u8;
                    }
                }

                slots
            }
            FormationType::Chevron { spacing, angle } => {
                // Stacked Vees: each sub-Vee has 7 drones (1 center + 3 pairs).
                // Multiple Vees are placed behind each other with a gap.
                // Leader is the center of the first Vee at (0,0).
                let mut slots = Vec::with_capacity(drone_count);
                if drone_count == 0 { return slots; }

                let row_spacing = spacing * angle.cos();
                let col_spacing = spacing * angle.sin();
                let vee_arms = 3usize; // 3 arm rows per Vee → 7 drones
                // Stride between Vee centers: depth + one gap
                let vee_stride = (vee_arms as f32 + 1.0) * row_spacing;

                let mut idx = 0u8;
                let mut vee_group = 0usize;

                while (idx as usize) < drone_count {
                    let center_x = -(vee_group as f32) * vee_stride;

                    // Center of this Vee
                    slots.push(FormationSlot::new(Vec2::new(center_x, 0.0), idx));
                    idx += 1;

                    // Arm rows (pairs fanning out)
                    for row in 1..=vee_arms {
                        if idx as usize >= drone_count { break; }
                        let x = center_x - (row as f32) * row_spacing;
                        let y = (row as f32) * col_spacing;
                        slots.push(FormationSlot::new(Vec2::new(x, -y), idx));
                        idx += 1;
                        if idx as usize >= drone_count { break; }
                        slots.push(FormationSlot::new(Vec2::new(x, y), idx));
                        idx += 1;
                    }

                    vee_group += 1;
                }

                slots
            }
        }
    }
}

// =============================================================================
// Formation Stability
// =============================================================================

/// Overall formation stability status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FormationStatus {
    /// All drones are at or near their formation slots.
    #[default]
    Nominal,
    /// Some drones are lagging but formation is recoverable.
    Degraded,
    /// Formation has broken down, emergency slowdown needed.
    Critical,
}

/// Per-drone stability metrics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneStabilityMetrics {
    /// Drone identifier.
    pub drone_id: usize,
    /// Distance from assigned slot (meters).
    pub slot_deviation: f32,
    /// Arc-length behind leader on path (meters). Negative means ahead.
    pub parameter_lag: f32,
    /// Estimated time to reach slot (seconds).
    pub time_to_slot: f32,
    /// Individual drone status based on thresholds.
    pub status: FormationStatus,
}

impl DroneStabilityMetrics {
    /// Create new drone stability metrics.
    pub fn new(drone_id: usize, slot_deviation: f32, parameter_lag: f32, time_to_slot: f32) -> Self {
        // Determine status based on thresholds
        let status = if slot_deviation > defaults::CRITICAL_DEVIATION
            || parameter_lag > defaults::CRITICAL_LAG
        {
            FormationStatus::Critical
        } else if slot_deviation > defaults::DEGRADED_DEVIATION
            || parameter_lag > defaults::DEGRADED_LAG
        {
            FormationStatus::Degraded
        } else {
            FormationStatus::Nominal
        };

        DroneStabilityMetrics {
            drone_id,
            slot_deviation,
            parameter_lag,
            time_to_slot,
            status,
        }
    }
}

/// Aggregate formation stability metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct FormationStabilityMetrics {
    /// Overall formation status.
    pub status: FormationStatus,
    /// Per-drone metrics.
    pub drone_metrics: Vec<DroneStabilityMetrics>,
    /// Maximum slot deviation across all drones (meters).
    pub max_deviation: f32,
    /// Maximum path lag across all drones (meters).
    pub max_lag: f32,
    /// Number of drones in critical state.
    pub critical_count: usize,
    /// Number of drones in degraded state.
    pub degraded_count: usize,
}

impl Default for FormationStabilityMetrics {
    fn default() -> Self {
        FormationStabilityMetrics {
            status: FormationStatus::Nominal,
            drone_metrics: Vec::new(),
            max_deviation: 0.0,
            max_lag: 0.0,
            critical_count: 0,
            degraded_count: 0,
        }
    }
}

// =============================================================================
// Formation Coordinator
// =============================================================================

/// Formation coordinator that directly controls drones.
///
/// Unlike message-based coordination, this uses direct method calls
/// for immediate, reliable formation control.
#[derive(Debug)]
pub struct FormationCoordinator {
    formation_type: FormationType,
    leader_id: Option<usize>,
    /// Cached slot assignments: (drone_id, slot)
    slot_assignments: Vec<(usize, FormationSlot)>,
    /// Formation center position
    center: Position,
    /// Formation heading (radians)
    heading: f32,
    /// Formation center velocity (for lead pursuit)
    velocity: Vec2,
    /// Whether formation needs to be re-assigned
    dirty: bool,
    /// Velocity consensus for synchronized formation movement
    velocity_consensus: VelocityConsensus,
    /// Formation stability metrics
    stability_metrics: FormationStabilityMetrics,
    /// Previous heading for turn detection (radians)
    prev_heading: f32,
    /// Whether dynamic slot reassignment is enabled
    dynamic_reassignment: bool,
}

impl FormationCoordinator {
    /// Create a new formation coordinator.
    pub fn new(formation_type: FormationType) -> Self {
        FormationCoordinator {
            formation_type,
            leader_id: None,
            slot_assignments: Vec::new(),
            center: Position::new(0.0, 0.0),
            heading: 0.0,
            velocity: Vec2::ZERO,
            dirty: true,
            velocity_consensus: VelocityConsensus::default(),
            stability_metrics: FormationStabilityMetrics::default(),
            prev_heading: 0.0,
            dynamic_reassignment: true,
        }
    }

    /// Enable or disable dynamic slot reassignment.
    ///
    /// When enabled, followers will be reassigned to nearest slots
    /// when the formation makes significant turns.
    pub fn set_dynamic_reassignment(&mut self, enabled: bool) {
        self.dynamic_reassignment = enabled;
    }

    /// Set the formation type.
    pub fn set_formation_type(&mut self, formation_type: FormationType) {
        self.formation_type = formation_type;
        self.dirty = true;
    }

    /// Get the formation type.
    pub fn formation_type(&self) -> FormationType {
        self.formation_type
    }

    /// Set the formation leader.
    pub fn set_leader(&mut self, leader_id: usize) {
        self.leader_id = Some(leader_id);
        self.dirty = true;
    }

    /// Get the leader ID.
    pub fn leader_id(&self) -> Option<usize> {
        self.leader_id
    }

    /// Get the formation center.
    pub fn center(&self) -> Position {
        self.center
    }

    /// Get the formation heading.
    pub fn heading(&self) -> f32 {
        self.heading
    }

    /// Assign formation slots to all drones.
    ///
    /// Call this when drones are added/removed or formation type changes.
    pub fn assign_slots(&mut self, agents: &mut [DroneAgent]) {
        if agents.is_empty() {
            self.slot_assignments.clear();
            return;
        }

        // Get drone IDs and sort (leader first if set)
        let mut drone_ids: Vec<usize> = agents.iter().map(|a| a.uid()).collect();

        if let Some(lid) = self.leader_id {
            // Move leader to front
            if let Some(pos) = drone_ids.iter().position(|&id| id == lid) {
                drone_ids.remove(pos);
                drone_ids.insert(0, lid);
            }
        } else {
            // Sort by ID, lowest first (becomes leader)
            drone_ids.sort();
            self.leader_id = drone_ids.first().copied();
        }

        // Compute slots for this formation
        let slots = self.formation_type.compute_slots(drone_ids.len());

        // Store assignments
        self.slot_assignments = drone_ids
            .into_iter()
            .zip(slots)
            .collect();

        // Update formation center from leader position
        self.update_center_from_leader(agents);

        // Apply to all drones
        self.broadcast_slots(agents);

        self.dirty = false;
    }

    /// Update formation center based on leader position.
    ///
    /// Formation heading is based on the leader's velocity direction with heavy
    /// smoothing to filter out oscillations during curved path following.
    /// This keeps the formation stable even when the leader follows smooth curves.
    fn update_center_from_leader(&mut self, agents: &[DroneAgent]) {
        if let Some(leader_id) = self.leader_id {
            if let Some(leader) = agents.iter().find(|a| a.uid() == leader_id) {
                let state = leader.state();
                self.center = state.pos;
                self.velocity = state.vel.as_vec2();

                // Compute target heading from velocity direction (actual travel direction)
                let vel = state.vel.as_vec2();
                let speed = vel.magnitude();

                let target_heading = if speed > 1.0 {
                    vel.y.atan2(vel.x)
                } else {
                    // Stationary - keep current heading
                    self.heading
                };

                // Smoothly interpolate formation heading toward target
                // This filters out oscillations during curved path following
                const SMOOTHING: f32 = 0.05; // Very slow - only ~5% per tick

                // Handle angle wrapping for smooth interpolation
                let mut diff = target_heading - self.heading;
                while diff > std::f32::consts::PI {
                    diff -= std::f32::consts::TAU;
                }
                while diff < -std::f32::consts::PI {
                    diff += std::f32::consts::TAU;
                }

                self.heading += diff * SMOOTHING;

                // Normalize heading
                while self.heading > std::f32::consts::PI {
                    self.heading -= std::f32::consts::TAU;
                }
                while self.heading < -std::f32::consts::PI {
                    self.heading += std::f32::consts::TAU;
                }
            }
        } else if !agents.is_empty() {
            // Compute centroid
            let sum: Vec2 = agents
                .iter()
                .map(|a| a.state().pos.as_vec2())
                .fold(Vec2::new(0.0, 0.0), |acc, v| {
                    Vec2::new(acc.x + v.x, acc.y + v.y)
                });
            let n = agents.len() as f32;
            self.center = Position::new(sum.x / n, sum.y / n);
            self.velocity = Vec2::new(0.0, 0.0);
        }
    }

    /// Broadcast slot assignments to all drones except the leader.
    ///
    /// The leader continues following its current waypoint/mission,
    /// while followers maintain formation around the leader.
    fn broadcast_slots(&self, agents: &mut [DroneAgent]) {
        for (drone_id, slot) in &self.slot_assignments {
            // Skip the leader - it keeps its current waypoint
            if Some(*drone_id) == self.leader_id {
                continue;
            }
            if let Some(agent) = agents.iter_mut().find(|a| a.uid() == *drone_id) {
                agent.set_formation_slot(*slot, self.center, self.heading);
            }
        }
    }

    /// Optimize slot assignments based on current drone positions.
    ///
    /// Called when formation makes a significant turn. Reassigns followers
    /// to nearest available slots to minimize total travel distance.
    fn optimize_slot_assignments(&mut self, agents: &[DroneAgent]) {
        if !self.dynamic_reassignment {
            return;
        }

        // Check for significant heading change (> 30 degrees)
        let heading_change = (self.heading - self.prev_heading).abs();
        let heading_change = if heading_change > std::f32::consts::PI {
            std::f32::consts::TAU - heading_change
        } else {
            heading_change
        };

        // Threshold: ~30 degrees
        const TURN_THRESHOLD: f32 = 0.52;
        if heading_change < TURN_THRESHOLD {
            return;
        }

        // Update prev_heading
        self.prev_heading = self.heading;

        // Get all slots (excluding leader's slot)
        let slots: Vec<FormationSlot> = self.slot_assignments
            .iter()
            .filter(|(id, _)| Some(*id) != self.leader_id)
            .map(|(_, slot)| *slot)
            .collect();

        if slots.is_empty() {
            return;
        }

        // Get follower IDs and positions
        let followers: Vec<(usize, Vec2)> = self.slot_assignments
            .iter()
            .filter(|(id, _)| Some(*id) != self.leader_id)
            .filter_map(|(id, _)| {
                agents.iter()
                    .find(|a| a.uid() == *id)
                    .map(|a| (*id, a.state().pos.as_vec2()))
            })
            .collect();

        if followers.is_empty() {
            return;
        }

        // Compute slot world positions
        let slot_positions: Vec<Vec2> = slots.iter()
            .map(|slot| compute_slot_world_position(self.center, self.heading, slot.offset))
            .collect();

        // Greedy assignment: for each slot, find nearest unassigned follower
        let mut new_assignments: Vec<(usize, FormationSlot)> = Vec::new();
        let mut assigned_followers: Vec<bool> = vec![false; followers.len()];

        for (slot_idx, slot_pos) in slot_positions.iter().enumerate() {
            let mut best_follower_idx = None;
            let mut best_dist = f32::MAX;

            for (f_idx, (_, follower_pos)) in followers.iter().enumerate() {
                if assigned_followers[f_idx] {
                    continue;
                }
                let dist = (follower_pos.x - slot_pos.x).hypot(follower_pos.y - slot_pos.y);
                if dist < best_dist {
                    best_dist = dist;
                    best_follower_idx = Some(f_idx);
                }
            }

            if let Some(f_idx) = best_follower_idx {
                assigned_followers[f_idx] = true;
                new_assignments.push((followers[f_idx].0, slots[slot_idx]));
            }
        }

        // Keep leader's slot assignment
        if let Some(leader_id) = self.leader_id {
            if let Some((_, leader_slot)) = self.slot_assignments.iter().find(|(id, _)| *id == leader_id) {
                new_assignments.push((leader_id, *leader_slot));
            }
        }

        // Update slot assignments
        self.slot_assignments = new_assignments;
    }

    /// Update all drones with current formation reference.
    ///
    /// Call this each tick to keep formation moving with leader.
    /// The leader continues its mission; followers update their positions.
    pub fn update_formation(&mut self, agents: &mut [DroneAgent], _dt: f32) {
        if self.dirty {
            self.assign_slots(agents);
            return;
        }

        // Update center from leader
        self.update_center_from_leader(agents);

        // Check for significant turn and reassign slots if needed
        self.optimize_slot_assignments(agents);

        // Broadcast updated slot assignments to agents
        self.broadcast_slots(agents);

        // Compute velocity consensus from leader
        self.update_velocity_consensus(agents);

        // Compute stability metrics (before applying speed adjustment)
        self.compute_stability_metrics(agents);

        // Apply automatic speed adjustment based on formation status
        let speed_factor = match self.stability_metrics.status {
            FormationStatus::Nominal => 1.0,
            FormationStatus::Degraded => defaults::DEGRADED_SPEED_FACTOR,
            FormationStatus::Critical => defaults::CRITICAL_SPEED_FACTOR,
        };

        // Scale velocity consensus for followers (allows formation to recover)
        let adjusted_consensus = self.velocity_consensus.scaled(speed_factor);

        // Update all follower drones with new reference (skip leader)
        for agent in agents.iter_mut() {
            // Skip the leader - it follows its own waypoint
            if Some(agent.uid()) == self.leader_id {
                continue;
            }
            if agent.in_formation() {
                agent.update_formation_reference(self.center, self.heading, self.velocity);
                agent.set_velocity_consensus(adjusted_consensus);
            }
        }
    }

    /// Update velocity consensus based on leader state.
    fn update_velocity_consensus(&mut self, agents: &[DroneAgent]) {
        if let Some(leader_id) = self.leader_id {
            if let Some(leader) = agents.iter().find(|a| a.uid() == leader_id) {
                let leader_vel = leader.state().vel.as_vec2();
                let is_moving = leader_vel.magnitude() > 0.5; // 0.5 m/s threshold
                self.velocity_consensus = VelocityConsensus::new(leader_vel, is_moving);
            }
        } else {
            self.velocity_consensus = VelocityConsensus::stopped();
        }
    }

    /// Get the current velocity consensus.
    pub fn velocity_consensus(&self) -> &VelocityConsensus {
        &self.velocity_consensus
    }


    /// Compute stability metrics for all formation drones.
    ///
    /// Calculates per-drone deviation from slots and path lag,
    /// then determines overall formation status.
    fn compute_stability_metrics(&mut self, agents: &[DroneAgent]) {
        let mut drone_metrics = Vec::with_capacity(self.slot_assignments.len());
        let mut max_deviation = 0.0_f32;
        let mut max_lag = 0.0_f32;
        let mut critical_count = 0;
        let mut degraded_count = 0;

        for (drone_id, slot) in &self.slot_assignments {
            // Skip leader - leader defines the reference
            if Some(*drone_id) == self.leader_id {
                continue;
            }

            // Find the agent
            let agent = match agents.iter().find(|a| a.uid() == *drone_id) {
                Some(a) => a,
                None => continue,
            };

            let agent_pos = agent.state().pos.as_vec2();
            let agent_vel = agent.state().vel.as_vec2();

            // Compute slot position (rotated by formation heading)
            let slot_pos = compute_slot_world_position(self.center, self.heading, slot.offset);

            // Compute slot deviation
            let slot_deviation = (agent_pos.x - slot_pos.x).hypot(agent_pos.y - slot_pos.y);

            // Estimate time to slot
            let closing_speed = agent_vel.magnitude().max(1.0);
            let time_to_slot = slot_deviation / closing_speed;

            let metrics = DroneStabilityMetrics::new(*drone_id, slot_deviation, 0.0, time_to_slot);

            // Track maximums and counts
            max_deviation = max_deviation.max(slot_deviation);
            max_lag = max_lag.max(0.0);

            match metrics.status {
                FormationStatus::Critical => critical_count += 1,
                FormationStatus::Degraded => degraded_count += 1,
                FormationStatus::Nominal => {}
            }

            drone_metrics.push(metrics);
        }

        // Determine overall status based on fractions
        let follower_count = drone_metrics.len();
        let status = if follower_count == 0 {
            FormationStatus::Nominal
        } else {
            let critical_fraction = critical_count as f32 / follower_count as f32;
            let degraded_fraction = (critical_count + degraded_count) as f32 / follower_count as f32;

            if critical_fraction >= defaults::CRITICAL_FRACTION {
                FormationStatus::Critical
            } else if degraded_fraction >= defaults::DEGRADED_FRACTION {
                FormationStatus::Degraded
            } else {
                FormationStatus::Nominal
            }
        };

        self.stability_metrics = FormationStabilityMetrics {
            status,
            drone_metrics,
            max_deviation,
            max_lag,
            critical_count,
            degraded_count,
        };
    }

    /// Get the current formation status.
    pub fn status(&self) -> FormationStatus {
        self.stability_metrics.status
    }

    /// Get the current stability metrics.
    pub fn stability_metrics(&self) -> &FormationStabilityMetrics {
        &self.stability_metrics
    }

    /// Get the current speed factor based on formation status.
    ///
    /// Returns the factor applied to velocity consensus:
    /// - Nominal: 1.0 (full speed)
    /// - Degraded: 0.7 (70% speed)
    /// - Critical: 0.3 (30% speed)
    pub fn speed_factor(&self) -> f32 {
        match self.stability_metrics.status {
            FormationStatus::Nominal => 1.0,
            FormationStatus::Degraded => defaults::DEGRADED_SPEED_FACTOR,
            FormationStatus::Critical => defaults::CRITICAL_SPEED_FACTOR,
        }
    }

    /// Issue a formation command to all drones.
    pub fn issue_command(&mut self, cmd: FormationCommand, agents: &mut [DroneAgent]) {
        match cmd {
            FormationCommand::Disperse => {
                // Clear all formation assignments
                for agent in agents.iter_mut() {
                    agent.handle_formation_command(cmd);
                }
                self.slot_assignments.clear();
                self.leader_id = None;
            }
            FormationCommand::Contract => {
                // Reduce spacing by 20%
                self.scale_spacing(0.8);
                self.dirty = true;
            }
            FormationCommand::Expand => {
                // Increase spacing by 20%
                self.scale_spacing(1.2);
                self.dirty = true;
            }
            _ => {
                // Pass command to all drones
                for agent in agents.iter_mut() {
                    agent.handle_formation_command(cmd);
                }
            }
        }
    }

    /// Scale the formation spacing.
    fn scale_spacing(&mut self, factor: f32) {
        self.formation_type = match self.formation_type {
            FormationType::Line { spacing } => FormationType::Line {
                spacing: spacing * factor,
            },
            FormationType::Vee { spacing, angle } => FormationType::Vee {
                spacing: spacing * factor,
                angle,
            },
            FormationType::Diamond { spacing } => FormationType::Diamond {
                spacing: spacing * factor,
            },
            FormationType::Circle { radius } => FormationType::Circle {
                radius: radius * factor,
            },
            FormationType::Grid { spacing, cols } => FormationType::Grid {
                spacing: spacing * factor,
                cols,
            },
            FormationType::Chevron { spacing, angle } => FormationType::Chevron {
                spacing: spacing * factor,
                angle,
            },
        };
    }

    /// Get the slot assignment for a specific drone.
    pub fn get_slot(&self, drone_id: usize) -> Option<&FormationSlot> {
        self.slot_assignments
            .iter()
            .find(|(id, _)| *id == drone_id)
            .map(|(_, slot)| slot)
    }

    /// Check if a drone is in this formation.
    pub fn contains(&self, drone_id: usize) -> bool {
        self.slot_assignments.iter().any(|(id, _)| *id == drone_id)
    }

    /// Get the number of drones in the formation.
    pub fn drone_count(&self) -> usize {
        self.slot_assignments.len()
    }
}

impl SwarmBehavior for FormationCoordinator {
    fn pre_tick(&mut self, agents: &[DroneAgent], _dt: f32) {
        // Auto-assign leader if not set
        if self.leader_id.is_none() && !agents.is_empty() {
            self.leader_id = agents.iter().map(|a| a.uid()).min();
        }
    }

    fn get_update_order(&self, agents: &[DroneAgent]) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..agents.len()).collect();

        // Leader first, then by slot priority
        if let Some(leader_id) = self.leader_id {
            indices.sort_by_key(|&i| {
                let uid = agents[i].uid();
                if uid == leader_id {
                    0
                } else {
                    self.slot_assignments
                        .iter()
                        .find(|(id, _)| *id == uid)
                        .map(|(_, slot)| slot.priority as usize + 1)
                        .unwrap_or(usize::MAX)
                }
            });
        } else {
            indices.sort_by_key(|&i| agents[i].uid());
        }

        indices
    }

    fn post_tick(&mut self, _agents: &[DroneAgent], _dt: f32) {
        // Could check formation metrics here
    }

    fn get_swarm_info(&self, agents: &[DroneAgent]) -> Vec<DroneInfo> {
        agents.iter().map(|a| a.get_info()).collect()
    }
}

// Keep the old FormationBehavior for backwards compatibility
/// Legacy formation behavior (deprecated, use FormationCoordinator).
#[derive(Debug)]
pub struct FormationBehavior {
    coordinator: FormationCoordinator,
}

impl FormationBehavior {
    /// Create a new formation behavior.
    pub fn new(formation_type: FormationType, _spacing: f32) -> Self {
        FormationBehavior {
            coordinator: FormationCoordinator::new(formation_type),
        }
    }

    /// Set the formation leader.
    pub fn set_leader(&mut self, leader_id: usize) {
        self.coordinator.set_leader(leader_id);
    }

    /// Get the formation type.
    pub fn formation_type(&self) -> FormationType {
        self.coordinator.formation_type()
    }

    /// Get the spacing between drones.
    pub fn spacing(&self) -> f32 {
        match self.coordinator.formation_type() {
            FormationType::Line { spacing } => spacing,
            FormationType::Vee { spacing, .. } => spacing,
            FormationType::Diamond { spacing } => spacing,
            FormationType::Circle { radius } => radius,
            FormationType::Grid { spacing, .. } => spacing,
            FormationType::Chevron { spacing, .. } => spacing,
        }
    }
}

impl SwarmBehavior for FormationBehavior {
    fn pre_tick(&mut self, agents: &[DroneAgent], dt: f32) {
        self.coordinator.pre_tick(agents, dt);
    }

    fn get_update_order(&self, agents: &[DroneAgent]) -> Vec<usize> {
        self.coordinator.get_update_order(agents)
    }

    fn post_tick(&mut self, agents: &[DroneAgent], dt: f32) {
        self.coordinator.post_tick(agents, dt);
    }

    fn get_swarm_info(&self, agents: &[DroneAgent]) -> Vec<DroneInfo> {
        self.coordinator.get_swarm_info(agents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Bounds, Heading};

    fn create_test_bounds() -> Bounds {
        Bounds::new(1000.0, 1000.0).unwrap()
    }

    fn create_test_agents(count: usize) -> Vec<DroneAgent> {
        (0..count)
            .map(|i| {
                DroneAgent::new(
                    i,
                    Position::new(500.0 + (i as f32) * 50.0, 500.0),
                    Heading::new(0.0),
                    create_test_bounds(),
                )
            })
            .collect()
    }

    #[test]
    fn test_line_formation_slots() {
        let formation = FormationType::Line { spacing: 50.0 };
        let slots = formation.compute_slots(3);

        assert_eq!(slots.len(), 3);
        // Center drone at y=0
        assert!((slots[1].offset.y).abs() < 0.1);
        // Side drones at y=-50 and y=50
        assert!((slots[0].offset.y - (-50.0)).abs() < 0.1);
        assert!((slots[2].offset.y - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_vee_formation_slots() {
        let formation = FormationType::Vee {
            spacing: 50.0,
            angle: std::f32::consts::FRAC_PI_4, // 45 degrees
        };
        let slots = formation.compute_slots(5);

        assert_eq!(slots.len(), 5);
        // Leader at front
        assert_eq!(slots[0].offset.x, 0.0);
        assert_eq!(slots[0].offset.y, 0.0);
        // Others trail back (negative x)
        assert!(slots[1].offset.x < 0.0);
        assert!(slots[2].offset.x < 0.0);
    }

    #[test]
    fn test_circle_formation_slots() {
        let formation = FormationType::Circle { radius: 100.0 };
        let slots = formation.compute_slots(4);

        assert_eq!(slots.len(), 4);
        // All slots should be at radius distance from origin
        for slot in &slots {
            let dist = (slot.offset.x.powi(2) + slot.offset.y.powi(2)).sqrt();
            assert!((dist - 100.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_grid_formation_slots() {
        let formation = FormationType::Grid {
            spacing: 50.0,
            cols: 2,
        };
        let slots = formation.compute_slots(4);

        assert_eq!(slots.len(), 4);
        // First row at x=0
        assert_eq!(slots[0].offset.x, 0.0);
        assert_eq!(slots[1].offset.x, 0.0);
        // Second row at x=-50
        assert!((slots[2].offset.x - (-50.0)).abs() < 0.1);
    }

    #[test]
    fn test_coordinator_new() {
        let coord = FormationCoordinator::new(FormationType::Line { spacing: 50.0 });
        assert_eq!(coord.drone_count(), 0);
        assert!(coord.leader_id().is_none());
    }

    #[test]
    fn test_coordinator_assign_slots() {
        let mut agents = create_test_agents(3);
        let mut coord = FormationCoordinator::new(FormationType::Line { spacing: 50.0 });

        coord.assign_slots(&mut agents);

        assert_eq!(coord.drone_count(), 3);
        let leader_id = coord.leader_id();
        assert!(leader_id.is_some());

        // Leader should NOT be in formation (it follows waypoints)
        // Followers should be in formation
        for agent in &agents {
            if Some(agent.uid()) == leader_id {
                assert!(!agent.in_formation(), "Leader should not be in formation");
            } else {
                assert!(agent.in_formation(), "Follower should be in formation");
            }
        }
    }

    #[test]
    fn test_coordinator_leader_first() {
        let agents = create_test_agents(3);
        let mut coord = FormationCoordinator::new(FormationType::Vee {
            spacing: 50.0,
            angle: 0.5,
        });
        coord.set_leader(1); // Set middle agent as leader

        coord.pre_tick(&agents, 0.016);
        let order = coord.get_update_order(&agents);

        // Leader (UID 1, at index 1) should be first in update order
        assert_eq!(agents[order[0]].uid(), 1);
    }

    #[test]
    fn test_coordinator_disperse() {
        let mut agents = create_test_agents(3);
        let mut coord = FormationCoordinator::new(FormationType::Line { spacing: 50.0 });

        coord.assign_slots(&mut agents);
        // Leader (agent[0] with uid 0) is NOT in formation, but followers are
        assert!(agents[1].in_formation()); // Follower should be in formation

        coord.issue_command(FormationCommand::Disperse, &mut agents);

        // All agents should be out of formation
        for agent in &agents {
            assert!(!agent.in_formation());
        }
        assert_eq!(coord.drone_count(), 0);
    }

    #[test]
    fn test_coordinator_contract_expand() {
        let mut coord = FormationCoordinator::new(FormationType::Line { spacing: 100.0 });

        coord.issue_command(FormationCommand::Contract, &mut []);
        match coord.formation_type() {
            FormationType::Line { spacing } => assert!((spacing - 80.0).abs() < 0.1),
            _ => panic!("Wrong formation type"),
        }

        coord.issue_command(FormationCommand::Expand, &mut []);
        match coord.formation_type() {
            FormationType::Line { spacing } => assert!((spacing - 96.0).abs() < 0.1),
            _ => panic!("Wrong formation type"),
        }
    }

    #[test]
    fn test_coordinator_update_formation() {
        let mut agents = create_test_agents(2);
        let mut coord = FormationCoordinator::new(FormationType::Line { spacing: 50.0 });

        coord.assign_slots(&mut agents);
        let _initial_center = coord.center();

        // Move leader
        // Note: We can't directly move the agent, but we can check that
        // update_formation reads from leader position
        coord.update_formation(&mut agents, 0.016);

        // Center should match leader position
        assert_eq!(coord.center().x(), agents[0].state().pos.x());
    }

    // Keep old test for backwards compatibility
    #[test]
    fn test_formation_behavior_new() {
        let fb = FormationBehavior::new(FormationType::Line { spacing: 50.0 }, 50.0);
        match fb.formation_type() {
            FormationType::Line { spacing } => assert_eq!(spacing, 50.0),
            _ => panic!("Wrong formation type"),
        }
    }


    // ========== Stability Metrics Tests ==========

    #[test]
    fn test_stability_metrics_computed() {
        // Agents created by create_test_agents are NOT at slot positions,
        // so metrics will show deviations
        let mut agents = create_test_agents(3);
        let mut coord = FormationCoordinator::new(FormationType::Line { spacing: 50.0 });

        coord.assign_slots(&mut agents);
        coord.update_formation(&mut agents, 0.016);

        // Metrics should be computed (followers have metrics)
        let metrics = coord.stability_metrics();
        // Leader is skipped, so we have 2 follower metrics
        assert_eq!(metrics.drone_metrics.len(), 2);
        // Followers are not at their slots, so deviation > 0
        assert!(metrics.max_deviation > 0.0);
    }

    #[test]
    fn test_stability_metrics_at_slots() {
        // Create agents at their actual slot positions for nominal status
        let bounds = create_test_bounds();

        // Line formation with spacing 50: slots at y = -25, 0, +25 from center
        // We'll put leader at (500, 500), followers at their slots
        let leader = DroneAgent::new(0, Position::new(500.0, 500.0), Heading::new(0.0), bounds);
        let follower1 = DroneAgent::new(1, Position::new(500.0, 475.0), Heading::new(0.0), bounds); // slot y = -25
        let follower2 = DroneAgent::new(2, Position::new(500.0, 525.0), Heading::new(0.0), bounds); // slot y = +25

        let mut agents = vec![leader, follower1, follower2];
        let mut coord = FormationCoordinator::new(FormationType::Line { spacing: 50.0 });

        coord.assign_slots(&mut agents);
        coord.update_formation(&mut agents, 0.016);

        // Agents are at slots, status should be nominal
        assert_eq!(coord.status(), FormationStatus::Nominal);
        assert_eq!(coord.stability_metrics().critical_count, 0);
    }

    #[test]
    fn test_drone_stability_metrics_thresholds() {
        // Test the per-drone metric thresholds
        let nominal = DroneStabilityMetrics::new(0, 5.0, 10.0, 1.0);
        assert_eq!(nominal.status, FormationStatus::Nominal);

        let degraded = DroneStabilityMetrics::new(1, 35.0, 10.0, 2.0);
        assert_eq!(degraded.status, FormationStatus::Degraded);

        let critical = DroneStabilityMetrics::new(2, 80.0, 10.0, 5.0);
        assert_eq!(critical.status, FormationStatus::Critical);

        // Also test lag thresholds
        let degraded_lag = DroneStabilityMetrics::new(3, 5.0, 45.0, 2.0);
        assert_eq!(degraded_lag.status, FormationStatus::Degraded);

        let critical_lag = DroneStabilityMetrics::new(4, 5.0, 110.0, 5.0);
        assert_eq!(critical_lag.status, FormationStatus::Critical);
    }

    #[test]
    fn test_stability_metrics_default() {
        let metrics = FormationStabilityMetrics::default();
        assert_eq!(metrics.status, FormationStatus::Nominal);
        assert!(metrics.drone_metrics.is_empty());
        assert_eq!(metrics.critical_count, 0);
        assert_eq!(metrics.degraded_count, 0);
    }

    #[test]
    fn test_formation_status_default() {
        let status = FormationStatus::default();
        assert_eq!(status, FormationStatus::Nominal);
    }

    // ========== Automatic Response Tests ==========

    #[test]
    fn test_speed_factor_nominal() {
        let bounds = create_test_bounds();
        let leader = DroneAgent::new(0, Position::new(500.0, 500.0), Heading::new(0.0), bounds);
        let follower1 = DroneAgent::new(1, Position::new(500.0, 475.0), Heading::new(0.0), bounds);
        let follower2 = DroneAgent::new(2, Position::new(500.0, 525.0), Heading::new(0.0), bounds);

        let mut agents = vec![leader, follower1, follower2];
        let mut coord = FormationCoordinator::new(FormationType::Line { spacing: 50.0 });

        coord.assign_slots(&mut agents);
        coord.update_formation(&mut agents, 0.016);

        // Agents are at slots, speed factor should be 1.0
        assert!((coord.speed_factor() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_speed_factor_degraded() {
        // Verify degraded speed factor constant
        assert!((defaults::DEGRADED_SPEED_FACTOR - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_speed_factor_critical() {
        // Verify critical speed factor constant
        assert!((defaults::CRITICAL_SPEED_FACTOR - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_velocity_consensus_scaled() {
        // Test that VelocityConsensus.scaled() works correctly
        let consensus = VelocityConsensus::new(Vec2::new(10.0, 0.0), true);
        let scaled = consensus.scaled(0.7);

        assert!((scaled.target_speed - 7.0).abs() < 0.001);
        assert!((scaled.target_velocity.x - 7.0).abs() < 0.001);
        assert!(scaled.is_moving);
    }

    // ========== Offset Route Tests ==========

    #[test]
    fn test_offset_route_empty() {
        let bounds = create_test_bounds();
        let result = compute_offset_route(&[], Vec2::new(-50.0, -30.0), &bounds);
        assert!(result.is_empty());
    }

    #[test]
    fn test_offset_route_single_waypoint() {
        let bounds = create_test_bounds();
        let waypoints = [Position::new(500.0, 500.0)];
        let result = compute_offset_route(&waypoints, Vec2::new(50.0, 30.0), &bounds);
        assert_eq!(result.len(), 1);
        // With heading 0, offset (50, 30) → world position (550, 530)
        assert!((result[0].x() - 550.0).abs() < 0.1);
        assert!((result[0].y() - 530.0).abs() < 0.1);
    }

    #[test]
    fn test_offset_route_straight_line() {
        // Racetrack-shaped loop: all mid-segment waypoints have heading 0
        let bounds = create_test_bounds();
        let waypoints = [
            Position::new(200.0, 500.0),
            Position::new(300.0, 500.0),
            Position::new(400.0, 500.0),
            Position::new(400.0, 600.0), // Turn around
            Position::new(300.0, 600.0),
            Position::new(200.0, 600.0),
        ];
        // Offset: 0m along-track, 30m left (positive y in local frame)
        // For the top straight segments (waypoints 0-2, heading ~0), offset should be +30 in y
        let result = compute_offset_route(&waypoints, Vec2::new(0.0, 30.0), &bounds);
        assert_eq!(result.len(), 6);
        // Middle waypoint (index 1) has clean heading 0 (both incoming and outgoing go right)
        assert!(
            (result[1].x() - 300.0).abs() < 1.0,
            "Waypoint 1 x: {} vs 300.0", result[1].x()
        );
        assert!(
            (result[1].y() - 530.0).abs() < 1.0,
            "Waypoint 1 y: {} vs 530.0", result[1].y()
        );
    }

    #[test]
    fn test_offset_route_right_angle_turn() {
        // L-shaped route: right then up
        let bounds = create_test_bounds();
        let waypoints = [
            Position::new(100.0, 500.0),
            Position::new(300.0, 500.0), // Corner
            Position::new(300.0, 700.0),
        ];
        // Offset: behind (-50m along-track), left (30m cross-track)
        let offset = Vec2::new(-50.0, 30.0);
        let result = compute_offset_route(&waypoints, offset, &bounds);
        assert_eq!(result.len(), 3);

        // At the corner (waypoint 1), the bisector heading should be ~45 degrees
        // (bisecting 0 and PI/2). The offset should be rotated accordingly.
        // Just verify the corner waypoint is offset from (300, 500) in a reasonable direction.
        let corner = &result[1];
        let dx = corner.x() - 300.0;
        let dy = corner.y() - 500.0;
        let dist = (dx * dx + dy * dy).sqrt();
        // Offset magnitude should be roughly sqrt(50^2 + 30^2) ≈ 58.3
        assert!(
            (dist - 58.3).abs() < 5.0,
            "Corner offset distance {} should be near 58.3", dist
        );
    }

    #[test]
    fn test_offset_route_closed_loop_square() {
        // Square route: right → down → left → up (CW in screen coords)
        let bounds = create_test_bounds();
        let waypoints = [
            Position::new(200.0, 200.0), // Top-left
            Position::new(400.0, 200.0), // Top-right
            Position::new(400.0, 400.0), // Bottom-right
            Position::new(200.0, 400.0), // Bottom-left
        ];
        // Offset: 30m to the left (positive y in local frame)
        // For a CW square, left offset should push waypoints inward
        let offset = Vec2::new(0.0, 30.0);
        let result = compute_offset_route(&waypoints, offset, &bounds);
        assert_eq!(result.len(), 4);

        // Each offset waypoint should be closer to the center (300, 300) than
        // the original waypoints. Original corners are ~141m from center.
        let center = Vec2::new(300.0, 300.0);
        for (i, wp) in result.iter().enumerate() {
            let orig_dist = ((waypoints[i].x() - center.x).powi(2)
                + (waypoints[i].y() - center.y).powi(2)).sqrt();
            let offset_dist = ((wp.x() - center.x).powi(2)
                + (wp.y() - center.y).powi(2)).sqrt();
            assert!(
                offset_dist < orig_dist,
                "Offset waypoint {} ({:.1}, {:.1}) dist {:.1} should be closer to center than original ({:.1}, {:.1}) dist {:.1}",
                i, wp.x(), wp.y(), offset_dist,
                waypoints[i].x(), waypoints[i].y(), orig_dist
            );
        }
    }
}
