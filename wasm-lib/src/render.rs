use drone_lib::{FormationApproachMode, Objective, PathPlanner};

use crate::types::{DroneRenderData, Point, SwarmStatus};
use crate::Swarm;

impl Swarm {
    pub fn get_render_state(&self) -> Vec<DroneRenderData> {
        self.drones
            .iter()
            .map(|d| {
                let state = d.agent.state();
                let objective = d.agent.objective();

                // Hide all waypoint/path visualization for formation followers
                let is_follower = self.formations.iter()
                    .any(|f| f.drone_ids.contains(&d.id) && f.leader_id != Some(d.id as usize));

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
}
