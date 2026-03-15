use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use drone_lib::{FormationCommand, FormationSlot, FormationType, Objective, Position};

use crate::types::Point;
use crate::{FormationState, Swarm};

impl Swarm {
    // ========================================================================
    // Formation Control
    // ========================================================================

    /// Set formation for all drones (legacy API - clears all existing formations).
    pub fn set_formation(
        &mut self,
        formation_type: FormationType,
        leader_id: Option<usize>,
    ) {
        self.clear_formation();
        let all_ids: HashSet<u32> = self.drones.iter().map(|d| d.id).collect();
        self.set_formation_for_group(formation_type, all_ids, leader_id);
    }

    /// Set formation for a specific group of drones identified by their IDs.
    pub fn set_formation_for_group(
        &mut self,
        formation_type: FormationType,
        group_ids: HashSet<u32>,
        leader_id: Option<usize>,
    ) {
        if group_ids.is_empty() {
            return;
        }

        // Clear formation state for drones in this group
        for drone in &mut self.drones {
            if group_ids.contains(&drone.id) {
                drone.agent.set_formation_leader(false);
                drone.agent.clear_formation_slot();
            }
        }

        // Remove any existing formation that overlaps with this group
        self.formations.retain(|f| f.drone_ids.is_disjoint(&group_ids));

        // Determine leader (lowest ID in group if not specified)
        let resolved_leader = leader_id.or_else(|| {
            group_ids.iter().map(|&id| id as usize).min()
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

        // Get drone IDs in this group, with leader first
        let mut drone_ids: Vec<usize> = self.drones.iter()
            .filter(|d| group_ids.contains(&d.id))
            .map(|d| d.id as usize)
            .collect();
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
            if Some(*drone_id) == resolved_leader {
                continue;
            }
            if let Some(drone) = self.drones.iter_mut().find(|d| d.id == *drone_id as u32) {
                drone.agent.set_formation_slot(*slot, center, heading);
            }
        }

        self.formations.push(FormationState {
            formation_type,
            leader_id: resolved_leader,
            slot_assignments,
            drone_ids: group_ids,
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
        self.formations.clear();
    }

    /// Remove a drone from its formation, promoting a new leader if it was the leader.
    pub(crate) fn remove_drone_from_formation(&mut self, drone_id: u32) {
        // Find which formation this drone belongs to
        let formation_idx = self.formations.iter().position(|f| f.drone_ids.contains(&drone_id));
        let Some(idx) = formation_idx else { return };

        let was_leader = self.formations[idx].leader_id == Some(drone_id as usize);

        // Remove from formation and clear agent state
        self.formations[idx].drone_ids.remove(&drone_id);
        if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
            drone.agent.clear_formation_slot();
            drone.agent.set_formation_leader(false);
        }

        if !was_leader {
            return;
        }

        // Leader was pulled out — promote a successor
        let remaining_ids: HashSet<u32> = self.formations[idx].drone_ids.iter()
            .copied()
            .filter(|id| self.drones.iter().any(|d| d.id == *id))
            .collect();

        if remaining_ids.is_empty() {
            self.formations.remove(idx);
            return;
        }

        // Save the formation state before removing it
        let formation_type = self.formations[idx].formation_type;
        let route = self.formations[idx].leader_route.clone();
        let was_route_mode = self.formations[idx].route_mode;
        self.formations.remove(idx);

        // Re-create formation with new leader (lowest ID)
        let new_leader = remaining_ids.iter().map(|&id| id as usize).min();
        self.set_formation_for_group(formation_type, remaining_ids, new_leader);

        // Restore route to new leader
        if let Some(route) = route {
            if let Some(formation) = self.formations.last_mut() {
                formation.leader_route = Some(route.clone());
                formation.route_mode = was_route_mode;

                if let Some(leader_id) = formation.leader_id {
                    if let Some(leader_drone) = self.drones.iter_mut().find(|d| d.id == leader_id as u32) {
                        let deque: VecDeque<Position> = route.iter().copied().collect();
                        leader_drone.agent.set_objective(Objective::FollowRoute {
                            waypoints: deque,
                            route,
                        });
                    }
                }
            }
        }
    }

    /// Return a drone to its original formation group.
    /// Re-adds it to the formation's drone_ids, recomputes slots, and clears
    /// any attack/intercept assignments.
    pub fn return_to_formation(&mut self, drone_id: u32, group_start: u32, group_end: u32) {
        if !self.drones.iter().any(|d| d.id == drone_id) { return; }

        // Clear from attack/intercept tracking and clear active task
        self.attack_targets.remove(&drone_id);
        self.intercept_targets.remove(&drone_id);
        if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
            drone.agent.clear_task();
        }

        // Find the formation for this group range
        let formation_idx = self.formations.iter().position(|f| {
            // Match formation by checking if it contains any drone in the group range
            f.drone_ids.iter().any(|&id| id >= group_start && id < group_end)
                || (drone_id >= group_start && drone_id < group_end)
        });

        let Some(idx) = formation_idx else {
            // No formation found, just set drone to sleep
            if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                drone.agent.set_objective(Objective::Sleep);
            }
            return;
        };

        // Re-add drone to formation group
        self.formations[idx].drone_ids.insert(drone_id);

        // Recompute all slot assignments for this formation
        let formation = &self.formations[idx];
        let leader_id = formation.leader_id;
        let formation_type = formation.formation_type.clone();
        let center = formation.center;
        let heading = formation.heading;

        // Build ordered drone list (leader first)
        let mut drone_ids: Vec<usize> = self.drones.iter()
            .filter(|d| self.formations[idx].drone_ids.contains(&d.id))
            .map(|d| d.id as usize)
            .collect();
        if let Some(lid) = leader_id {
            if let Some(pos) = drone_ids.iter().position(|&id| id == lid) {
                drone_ids.remove(pos);
                drone_ids.insert(0, lid);
            }
        } else {
            drone_ids.sort();
        }

        let slots = formation_type.compute_slots(drone_ids.len());
        let slot_assignments: Vec<(usize, FormationSlot)> = drone_ids
            .into_iter()
            .zip(slots)
            .collect();

        // Apply the returning drone's slot (don't disturb others)
        for (did, slot) in &slot_assignments {
            if *did == drone_id as usize && Some(*did) != leader_id {
                if let Some(drone) = self.drones.iter_mut().find(|d| d.id == drone_id) {
                    drone.agent.set_formation_slot(*slot, center, heading);
                }
            }
        }

        self.formations[idx].slot_assignments = slot_assignments;
    }

    /// Save leader routes before leaders are destroyed.
    /// Returns a vec of (formation_index, route) for destroyed leaders.
    pub(crate) fn save_leader_routes_if_destroyed(&self, destroyed_ids: &HashSet<u32>) -> Vec<(usize, Arc<[Position]>)> {
        let mut saved = Vec::new();
        for (idx, state) in self.formations.iter().enumerate() {
            let Some(leader_id) = state.leader_id else { continue };
            if !destroyed_ids.contains(&(leader_id as u32)) {
                continue;
            }
            if let Some(route) = self.drones
                .iter()
                .find(|d| d.id == leader_id as u32)
                .and_then(|d| match d.agent.objective() {
                    Objective::FollowRoute { route, .. } => Some(route),
                    _ => None,
                })
            {
                saved.push((idx, route));
            }
        }
        saved
    }

    /// Check all formations for leader succession after drone destruction.
    pub(crate) fn check_leader_successions(&mut self, saved_routes: Vec<(usize, Arc<[Position]>)>) {
        // Collect formation info that needs succession
        let mut successions: Vec<(FormationType, HashSet<u32>, Option<Arc<[Position]>>, bool)> = Vec::new();

        let mut to_remove = Vec::new();
        for (idx, state) in self.formations.iter().enumerate() {
            let current_leader = state.leader_id;
            let leader_alive = current_leader
                .map(|lid| self.drones.iter().any(|d| d.id == lid as u32))
                .unwrap_or(false);

            if leader_alive {
                continue;
            }

            // Check if any drones in this group are still alive
            let remaining_ids: HashSet<u32> = state.drone_ids.iter()
                .copied()
                .filter(|id| self.drones.iter().any(|d| d.id == *id))
                .collect();

            if remaining_ids.is_empty() {
                to_remove.push(idx);
                continue;
            }

            let route = saved_routes.iter()
                .find(|(i, _)| *i == idx)
                .map(|(_, r)| r.clone())
                .or_else(|| state.leader_route.clone());

            successions.push((state.formation_type, remaining_ids, route, state.route_mode));
            to_remove.push(idx);
        }

        // Remove old formations (in reverse order to preserve indices)
        for idx in to_remove.into_iter().rev() {
            self.formations.remove(idx);
        }

        // Re-create formations with new leaders
        for (formation_type, group_ids, route, was_route_mode) in successions {
            let new_leader = group_ids.iter().map(|&id| id as usize).min();
            self.set_formation_for_group(formation_type, group_ids, new_leader);

            if let Some(route) = route {
                let waypoints_px: Vec<Point> = route.iter().map(|p| {
                    self.world_scale.position_to_point_px(*p)
                }).collect();

                // Find the formation we just created and assign route to its leader
                if let Some(formation) = self.formations.last_mut() {
                    formation.leader_route = Some(Arc::from(
                        waypoints_px.iter()
                            .map(|p| self.world_scale.point_px_to_position(*p))
                            .collect::<Vec<_>>()
                    ));
                    formation.route_mode = was_route_mode;

                    if let Some(leader_id) = formation.leader_id {
                        let route_m: Vec<Position> = waypoints_px.iter()
                            .map(|p| self.world_scale.point_px_to_position(*p))
                            .collect();
                        let route_arc: Arc<[Position]> = Arc::from(route_m);

                        if let Some(leader_drone) = self.drones.iter_mut().find(|d| d.id == leader_id as u32) {
                            let deque: VecDeque<Position> = route_arc.iter().copied().collect();
                            leader_drone.agent.set_objective(Objective::FollowRoute {
                                waypoints: deque,
                                route: route_arc,
                            });
                        }
                    }
                }
            }
        }
    }

    pub fn formation_command(&mut self, cmd: FormationCommand) {
        if self.formations.is_empty() { return };

        match cmd {
            FormationCommand::Disperse => {
                for drone in &mut self.drones {
                    drone.agent.handle_formation_command(cmd);
                }
                self.formations.clear();
            }
            FormationCommand::Contract | FormationCommand::Expand => {
                let scale = if matches!(cmd, FormationCommand::Contract) { 0.8 } else { 1.2 };
                // Collect formation info, then recreate
                let infos: Vec<_> = self.formations.iter()
                    .map(|f| (Self::scale_formation_type(&f.formation_type, scale), f.drone_ids.clone(), f.leader_id))
                    .collect();
                self.formations.clear();
                for (new_type, group_ids, leader_id) in infos {
                    self.set_formation_for_group(new_type, group_ids, leader_id);
                }
            }
            FormationCommand::Hold | FormationCommand::Advance => {
                for drone in &mut self.drones {
                    drone.agent.handle_formation_command(cmd);
                }
            }
        }
    }

    pub fn update_formation(&mut self, _dt: f32) {
        for state in &mut self.formations {
            let Some(lid) = state.leader_id else { continue };

            let (leader_pos, leader_heading, leader_velocity) = {
                if let Some(leader) = self.drones.iter().find(|d| d.id == lid as u32) {
                    let pos = leader.agent.state().pos;
                    let hdg = leader.agent.state().hdg.radians();
                    let vel = leader.agent.state().vel.as_vec2();
                    (pos, hdg, vel)
                } else {
                    continue;
                }
            };

            state.center = leader_pos;

            let target_heading = leader_heading;
            state.heading = target_heading;

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
            let leader_id = state.leader_id;
            for drone in &mut self.drones {
                if !state.drone_ids.contains(&drone.id) {
                    continue;
                }
                if leader_id == Some(drone.id as usize) {
                    continue;
                }
                if drone.agent.in_formation() {
                    if is_route_mode {
                        drone.agent.update_formation_reference_no_waypoint(
                            state.center, smoothed_heading, leader_velocity,
                        );
                    } else {
                        drone.agent.update_formation_reference(
                            state.center, smoothed_heading, leader_velocity,
                        );
                    }
                }
            }
        }
    }

    pub(crate) fn scale_formation_type(formation_type: &FormationType, scale: f32) -> FormationType {
        match *formation_type {
            FormationType::Line { spacing } => FormationType::Line { spacing: spacing * scale },
            FormationType::Vee { spacing, angle } => FormationType::Vee { spacing: spacing * scale, angle },
            FormationType::Diamond { spacing } => FormationType::Diamond { spacing: spacing * scale },
            FormationType::Circle { radius } => FormationType::Circle { radius: radius * scale },
            FormationType::Grid { spacing, cols } => FormationType::Grid { spacing: spacing * scale, cols },
            FormationType::Chevron { spacing, angle } => FormationType::Chevron { spacing: spacing * scale, angle },
        }
    }

    pub(crate) fn calculate_centroid(&self) -> Position {
        if self.drones.is_empty() {
            return Position::new(0.0, 0.0);
        }

        let sum_x: f32 = self.drones.iter().map(|d| d.agent.state().pos.x()).sum();
        let sum_y: f32 = self.drones.iter().map(|d| d.agent.state().pos.y()).sum();
        let count = self.drones.len() as f32;

        Position::new(sum_x / count, sum_y / count)
    }
}
