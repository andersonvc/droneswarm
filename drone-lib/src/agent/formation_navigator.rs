//! Formation navigation for drones in a swarm formation.
//!
//! Manages formation slot tracking, intercept navigation, and formation
//! visualization paths. Extracted from DroneAgent to improve separation
//! of concerns.

use std::collections::VecDeque;

use crate::messages::{FormationCommand, FormationSlot, VelocityConsensus};
use crate::missions::{ApproachGateConfig, PathPlanner, Task, WaypointMission};
use crate::swarm::formation::{compute_slot_world_position, defaults};
use crate::types::{Bounds, DroneInfo, Heading, Position, State, Vec2};

use super::drone_agent::FormationApproachMode;

/// Manages formation following for a single drone.
///
/// Handles slot assignment, intercept navigation computation,
/// and formation visualization paths.
#[derive(Debug)]
pub struct FormationNavigator {
    /// Current formation slot assignment.
    slot: Option<FormationSlot>,
    /// Formation center position.
    center: Option<Position>,
    /// Formation heading (radians).
    heading: Option<f32>,
    /// Formation center velocity (for lead pursuit).
    velocity: Option<Vec2>,
    /// Whether this drone is the formation leader.
    is_leader: bool,
    /// Speed multiplier for formation synchronization.
    speed_multiplier: f32,
    /// Velocity consensus from formation leader.
    velocity_consensus: Option<VelocityConsensus>,
}

impl Default for FormationNavigator {
    fn default() -> Self {
        Self::new()
    }
}

impl FormationNavigator {
    /// Create a new formation navigator (not in formation).
    pub fn new() -> Self {
        FormationNavigator {
            slot: None,
            center: None,
            heading: None,
            velocity: None,
            is_leader: false,
            speed_multiplier: 1.0,
            velocity_consensus: None,
        }
    }

    // ===== Slot management =====

    /// Assign a formation slot to this drone.
    pub fn set_slot(
        &mut self,
        slot: FormationSlot,
        center: Position,
        heading: f32,
        mission: &mut WaypointMission,
    ) {
        self.slot = Some(slot);
        self.center = Some(center);
        self.heading = Some(heading);
        self.velocity = None;
        self.update_waypoint(mission);
    }

    /// Update the formation reference point (center, heading, and velocity).
    ///
    /// Also overwrites the mission waypoint to the slot world position.
    /// Use `update_reference_no_waypoint` in route-following mode to avoid
    /// destroying the follower's offset route.
    pub fn update_reference(
        &mut self,
        center: Position,
        heading: f32,
        velocity: Vec2,
        mission: &mut WaypointMission,
    ) {
        self.center = Some(center);
        self.heading = Some(heading);
        self.velocity = Some(velocity);
        if self.slot.is_some() {
            self.update_waypoint(mission);
        }
    }

    /// Update the formation reference point without overwriting mission waypoints.
    ///
    /// Used in route-following mode where followers independently follow their
    /// own offset routes. The center/heading/velocity are stored for reference
    /// (e.g. speed factor computation) but the mission is not touched.
    pub fn update_reference_no_waypoint(
        &mut self,
        center: Position,
        heading: f32,
        velocity: Vec2,
    ) {
        self.center = Some(center);
        self.heading = Some(heading);
        self.velocity = Some(velocity);
    }

    /// Clear the formation slot assignment.
    pub fn clear_slot(&mut self) {
        self.slot = None;
        self.center = None;
        self.heading = None;
        self.velocity = None;
    }

    /// Handle a formation command.
    pub fn handle_command(&mut self, cmd: FormationCommand, mission: &mut WaypointMission) {
        match cmd {
            FormationCommand::Hold => {
                if self.slot.is_some() {
                    self.update_waypoint(mission);
                }
            }
            FormationCommand::Disperse => {
                self.clear_slot();
                mission.set_waypoints(Task::Loiter, VecDeque::new(), None);
            }
            FormationCommand::Advance | FormationCommand::Contract | FormationCommand::Expand => {
                if self.slot.is_some() {
                    self.update_waypoint(mission);
                }
            }
        }
    }

    // ===== Getters/setters =====

    /// Get the current formation slot (if assigned).
    pub fn slot(&self) -> Option<&FormationSlot> {
        self.slot.as_ref()
    }

    /// Check if drone is in a formation.
    pub fn in_formation(&self) -> bool {
        self.slot.is_some()
    }

    /// Check if this drone is the formation leader.
    pub fn is_leader(&self) -> bool {
        self.is_leader
    }

    /// Set whether this drone is the formation leader.
    pub fn set_leader(&mut self, is_leader: bool) {
        self.is_leader = is_leader;
    }

    /// Set the formation speed multiplier for synchronized arrival.
    pub fn set_speed_multiplier(&mut self, multiplier: f32) {
        self.speed_multiplier = multiplier.clamp(0.5, 2.0);
    }

    /// Get the current formation speed multiplier.
    pub fn speed_multiplier(&self) -> f32 {
        self.speed_multiplier
    }

    /// Set velocity consensus from formation leader.
    pub fn set_velocity_consensus(&mut self, consensus: VelocityConsensus) {
        self.velocity_consensus = Some(consensus);
    }

    /// Clear velocity consensus.
    pub fn clear_velocity_consensus(&mut self) {
        self.velocity_consensus = None;
    }

    /// Get the current velocity consensus (if any).
    pub fn velocity_consensus(&self) -> Option<&VelocityConsensus> {
        self.velocity_consensus.as_ref()
    }


    // ===== Speed-based formation distance control =====

    /// Compute a speed factor for formation distance control.
    ///
    /// Uses **along-track** distance (ahead/behind the leader in the direction
    /// of travel) rather than absolute distance. This correctly distinguishes
    /// a follower that is 50m ahead (should slow down) from one that is 50m
    /// behind (should speed up).
    ///
    /// Returns `Some(factor)` clamped to [0.3, 1.5], or `None` if not applicable.
    pub fn compute_formation_speed_factor(
        &self,
        my_pos: Vec2,
        swarm: &[DroneInfo],
        bounds: &Bounds,
    ) -> Option<f32> {
        let slot = self.slot?;
        if self.is_leader {
            return None;
        }

        let leader = swarm.iter().find(|d| d.is_formation_leader)?;
        let leader_pos = leader.pos.as_vec2();
        let leader_vel = leader.vel.as_vec2();
        let leader_speed = leader_vel.magnitude();

        // If leader is barely moving, followers should crawl
        if leader_speed < 0.5 {
            return Some(0.3);
        }

        let leader_dir = Vec2::new(leader_vel.x / leader_speed, leader_vel.y / leader_speed);

        // Vector from leader to follower (toroidal-aware)
        let to_follower = bounds.delta(leader_pos, my_pos);

        // Along-track position of follower (positive = ahead of leader)
        let along_track = to_follower.dot(leader_dir);

        // Desired along-track: project the slot's world offset onto the leader direction.
        // For V formation slots with negative x (behind leader), this is negative.
        let formation_heading = self.heading.unwrap_or(0.0);
        let (sin_h, cos_h) = formation_heading.sin_cos();
        let slot_world = Vec2::new(
            slot.offset.x * cos_h - slot.offset.y * sin_h,
            slot.offset.x * sin_h + slot.offset.y * cos_h,
        );
        let desired_along_track = slot_world.dot(leader_dir);

        // Error: positive = follower is too far ahead, should slow down
        let along_track_error = along_track - desired_along_track;

        let reference_dist = slot.offset.magnitude().max(1.0);

        const K: f32 = 0.5;
        // Negate: positive error (ahead) → factor < 1 (slow down)
        let speed_factor = 1.0 - K * along_track_error / reference_dist;

        Some(speed_factor.clamp(0.3, 1.5))
    }

    // ===== Navigation computation =====

    /// Default approach gate configuration for formation joining.
    fn default_gate_config() -> ApproachGateConfig {
        ApproachGateConfig {
            ball_radius: defaults::APPROACH_BALL_RADIUS,
            gate_offset: defaults::APPROACH_GATE_OFFSET,
            angle_tolerance: defaults::APPROACH_ANGLE_TOLERANCE,
        }
    }

    /// Compute the world position of the current formation slot.
    ///
    /// Returns `None` if not fully configured (missing slot, center, or heading).
    fn slot_world_position(&self) -> Option<Vec2> {
        match (self.slot, self.center, self.heading) {
            (Some(slot), Some(center), Some(heading)) => {
                Some(compute_slot_world_position(center, heading, slot.offset))
            }
            _ => None,
        }
    }

    /// Compute intercept navigation for formation following.
    ///
    /// Returns `(desired_heading, speed_override)` if in formation,
    /// or `(None, None)` if not in formation.
    pub fn compute_intercept_navigation(
        &self,
        state: &State,
        max_vel: f32,
        bounds: &Bounds,
    ) -> (Option<Heading>, Option<f32>) {
        let (slot_pos, formation_hdg) = match self.slot_world_position() {
            Some(p) => (p, self.heading.unwrap()),
            None => return (None, None),
        };

        let my_pos = state.pos.as_vec2();
        let my_vel = state.vel.as_vec2();
        let pos_error = bounds.delta(my_pos, slot_pos);
        let dist = pos_error.magnitude();

        // Get target velocity from velocity consensus (preferred) or formation velocity (fallback)
        let (target_vel, target_speed, _) = if let Some(consensus) = &self.velocity_consensus {
            (consensus.target_velocity, consensus.target_speed, consensus.is_moving)
        } else {
            let vel = self.velocity.unwrap_or(Vec2::ZERO);
            let speed = vel.magnitude();
            (vel, speed, speed > 0.5)
        };

        // Formation distance thresholds
        let station_keeping_dist = defaults::STATION_KEEPING_DIST;
        let at_position_dist = defaults::AT_POSITION_DIST;

        if dist < station_keeping_dist {
            // STATION-KEEPING MODE — velocity matching + proportional correction
            //
            // Instead of aiming at the slot (which causes zigzag when the leader
            // moves), match the leader's velocity and add a gentle correction
            // toward the slot position.
            let target_vel = self.velocity.unwrap_or(Vec2::ZERO);

            // Proportional gain: stronger correction when farther from slot
            let kp = if dist < at_position_dist { 0.8 } else { 1.5 };
            let correction = Vec2::new(pos_error.x * kp, pos_error.y * kp);

            // Derivative damping: resist velocity that differs from leader
            let vel_error = Vec2::new(my_vel.x - target_vel.x, my_vel.y - target_vel.y);
            let kd = 0.3;
            let damping = Vec2::new(-vel_error.x * kd, -vel_error.y * kd);

            // Desired velocity = leader velocity + correction + damping
            let desired = Vec2::new(
                target_vel.x + correction.x + damping.x,
                target_vel.y + correction.y + damping.y,
            );

            let speed = desired.magnitude().min(max_vel);
            if speed < 0.5 {
                (Some(state.hdg), Some(0.0))
            } else {
                (Some(Heading::new(desired.heading())), Some(speed))
            }
        } else {
            // INTERCEPT MODE
            let gate_config = Self::default_gate_config();

            let needs_gate = PathPlanner::needs_approach_gate(
                my_pos,
                slot_pos,
                formation_hdg,
                &gate_config,
                bounds,
            );

            if needs_gate {
                // PHASE 1: APPROACH - Gated spline to rear gate
                let approach_speed = max_vel;
                let time_to_intercept = (dist / max_vel).min(3.0);
                let intercept_pos = Vec2::new(
                    slot_pos.x + target_vel.x * time_to_intercept,
                    slot_pos.y + target_vel.y * time_to_intercept,
                );

                let planner = PathPlanner::new();
                let spline_heading = planner.compute_gated_approach_heading(
                    state,
                    intercept_pos,
                    formation_hdg,
                    &gate_config,
                    bounds,
                );

                (Some(spline_heading), Some(approach_speed))
            } else {
                // PHASE 2: PURSUIT - Aggressive lead pursuit
                let pursuit_speed = if dist > station_keeping_dist {
                    max_vel
                } else {
                    let blend = dist / station_keeping_dist;
                    let fast = max_vel;
                    let slow = target_speed.max(max_vel * 0.5);
                    slow + (fast - slow) * blend
                };

                let relative_vel = Vec2::new(my_vel.x - target_vel.x, my_vel.y - target_vel.y);
                let closing_speed = if dist > 0.1 {
                    let error_dir = Vec2::new(pos_error.x / dist, pos_error.y / dist);
                    -(relative_vel.x * error_dir.x + relative_vel.y * error_dir.y)
                } else {
                    pursuit_speed
                };

                let time_to_intercept = if closing_speed > 2.0 {
                    (dist / closing_speed).min(1.5)
                } else {
                    (dist / pursuit_speed).min(1.5)
                };

                let lead_pos = Vec2::new(
                    slot_pos.x + target_vel.x * time_to_intercept,
                    slot_pos.y + target_vel.y * time_to_intercept,
                );

                let to_lead = bounds.delta(my_pos, lead_pos);
                let pursuit_heading = Heading::new(to_lead.heading());

                (Some(pursuit_heading), Some(pursuit_speed))
            }
        }
    }

    // ===== Visualization =====

    /// Get formation approach path for visualization.
    pub fn get_spline_path(
        &self,
        state: &State,
        planner: &PathPlanner,
        bounds: &Bounds,
        num_points: usize,
    ) -> Vec<Vec2> {
        if self.is_leader {
            return Vec::new();
        }

        let (slot_pos, formation_hdg) = match self.slot_world_position() {
            Some(p) => (p, self.heading.unwrap()),
            None => return Vec::new(),
        };

        let my_pos = state.pos.as_vec2();
        let pos_error = bounds.delta(my_pos, slot_pos);
        let dist = pos_error.magnitude();

        if dist < defaults::STATION_KEEPING_DIST {
            return Vec::new();
        }

        let gate_config = ApproachGateConfig {
            ball_radius: defaults::APPROACH_BALL_RADIUS,
            gate_offset: defaults::APPROACH_GATE_OFFSET,
            angle_tolerance: defaults::APPROACH_ANGLE_TOLERANCE,
        };

        let needs_gate = PathPlanner::needs_approach_gate(
            my_pos,
            slot_pos,
            formation_hdg,
            &gate_config,
            bounds,
        );

        // Compute intercept position
        let target_vel = self.velocity.unwrap_or(Vec2::ZERO);
        let my_vel = state.vel.as_vec2();
        let max_speed = 20.0; // Fallback; callers can use platform perf

        let relative_vel = Vec2::new(my_vel.x - target_vel.x, my_vel.y - target_vel.y);
        let closing_speed = if dist > 0.1 {
            let error_dir = Vec2::new(pos_error.x / dist, pos_error.y / dist);
            -(relative_vel.x * error_dir.x + relative_vel.y * error_dir.y)
        } else {
            max_speed
        };

        let time_to_intercept = if closing_speed > 2.0 {
            (dist / closing_speed).min(2.0)
        } else {
            (dist / max_speed).min(2.0)
        };

        let lead_pos = Vec2::new(
            slot_pos.x + target_vel.x * time_to_intercept,
            slot_pos.y + target_vel.y * time_to_intercept,
        );

        if needs_gate {
            planner.get_gated_approach_path(
                state,
                lead_pos,
                formation_hdg,
                &gate_config,
                bounds,
                num_points,
            )
        } else {
            vec![my_pos, lead_pos]
        }
    }

    /// Get the current formation approach mode for visualization.
    pub fn get_approach_mode(&self, state: &State, bounds: &Bounds) -> FormationApproachMode {
        if self.is_leader {
            return FormationApproachMode::None;
        }

        let (slot_pos, formation_hdg) = match self.slot_world_position() {
            Some(p) => (p, self.heading.unwrap()),
            None => return FormationApproachMode::None,
        };

        let my_pos = state.pos.as_vec2();
        let pos_error = bounds.delta(my_pos, slot_pos);
        let dist = pos_error.magnitude();

        let at_position_dist = defaults::AT_POSITION_DIST;
        let station_keeping_dist = defaults::STATION_KEEPING_DIST;

        if dist < at_position_dist {
            FormationApproachMode::StationKeeping
        } else if dist < station_keeping_dist {
            FormationApproachMode::Correction
        } else {
            let gate_config = Self::default_gate_config();

            let needs_gate = PathPlanner::needs_approach_gate(
                my_pos,
                slot_pos,
                formation_hdg,
                &gate_config,
                bounds,
            );

            if needs_gate {
                FormationApproachMode::Approach
            } else {
                FormationApproachMode::Pursuit
            }
        }
    }

    // ===== Private =====

    /// Compute target position from formation slot and set as waypoint.
    fn update_waypoint(&self, mission: &mut WaypointMission) {
        if let Some(slot_pos) = self.slot_world_position() {
            let target = Position::new(slot_pos.x, slot_pos.y);
            let mut waypoints = VecDeque::new();
            waypoints.push_back(target);
            mission.set_waypoints(Task::ReachWaypoint, waypoints, None);
        }
    }
}
