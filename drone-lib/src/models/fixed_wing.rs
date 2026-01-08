use log::trace;

use crate::behaviors::{
    calculate_separation, calculate_velocity_obstacle, SeparationConfig, VelocityObstacleConfig,
};
use crate::models::drone::Drone;
use crate::types::{
    Acceleration, Bounds, DroneInfo, DronePerfFeatures, Heading, Objective, ObjectiveType,
    Position, State, Vec2, Velocity,
};

const DEFAULT_MAX_VELOCITY: f32 = 120.0;
const DEFAULT_MAX_ACCELERATION: f32 = 21.0;
const DEFAULT_MAX_TURN_RATE: f32 = 4.0; // radians per second at max velocity
const MIN_TURN_RATE: f32 = 0.15; // minimum turn rate even at zero velocity (rad/s)
const DEFAULT_WAYPOINT_CLEARANCE: f32 = 10.0;

#[derive(Debug)]
pub struct FixedWing {
    id: usize,
    state: State,
    objective: Objective,
    perf: DronePerfFeatures,
    bounds: Bounds,
    separation: SeparationConfig,
    velocity_obstacle: VelocityObstacleConfig,
    waypoint_clearance: f32,
}

impl FixedWing {
    /// Create a new FixedWing drone
    ///
    /// # Errors
    /// Returns `DroneError::InvalidBounds` if width or height <= 0
    pub fn new(id: usize, pos: Position, hdg: Heading, bounds: Bounds) -> FixedWing {
        let state = State {
            hdg,
            pos,
            vel: Velocity::zero(),
            acc: Acceleration::zero(),
        };

        FixedWing {
            id,
            state,
            objective: Objective::default(),
            perf: DronePerfFeatures::new_unchecked(
                DEFAULT_MAX_VELOCITY,
                DEFAULT_MAX_ACCELERATION,
                DEFAULT_MAX_TURN_RATE,
            ),
            bounds,
            separation: SeparationConfig::default(),
            velocity_obstacle: VelocityObstacleConfig::default(),
            waypoint_clearance: DEFAULT_WAYPOINT_CLEARANCE,
        }
    }

    /// Set the waypoint clearance distance (how close to consider "arrived").
    pub fn set_waypoint_clearance(&mut self, clearance: f32) {
        self.waypoint_clearance = clearance.max(1.0); // Minimum 1 unit
    }

    /// Get the current waypoint clearance distance.
    pub fn waypoint_clearance(&self) -> f32 {
        self.waypoint_clearance
    }

    /// Set the separation behavior configuration for collision avoidance.
    pub fn set_separation_config(&mut self, config: SeparationConfig) {
        self.separation = config;
    }

    /// Get the current separation configuration.
    pub fn separation_config(&self) -> &SeparationConfig {
        &self.separation
    }

    /// Set the velocity obstacle configuration for predictive collision avoidance.
    pub fn set_velocity_obstacle_config(&mut self, config: VelocityObstacleConfig) {
        self.velocity_obstacle = config;
    }

    /// Get the current velocity obstacle configuration.
    pub fn velocity_obstacle_config(&self) -> &VelocityObstacleConfig {
        &self.velocity_obstacle
    }

    /// Move toward a specific heading using velocity obstacle avoidance.
    ///
    /// # Arguments
    /// * `desired_heading` - The heading we want to reach (toward waypoint)
    /// * `swarm` - All drones for VO calculations
    /// * `dt` - Delta time in seconds
    fn move_to_heading(&mut self, desired_heading: Heading, swarm: &[DroneInfo], dt: f32) {
        let current_speed = self.state.vel.as_vec2().magnitude();

        // Use velocity obstacles for collision avoidance
        let vo_result = calculate_velocity_obstacle(
            &self.state,
            self.id,
            desired_heading,
            self.perf.max_turn_rate,
            swarm,
            &self.bounds,
            &self.velocity_obstacle,
        );

        // Apply the heading adjustment from VO to the desired heading
        let target_heading = Heading::new(desired_heading.radians() + vo_result.heading_adjustment);

        // Calculate heading error (shortest angular distance)
        let hdg_error = self.state.hdg.difference(target_heading);

        // Turn rate scales with velocity squared - need momentum to turn effectively
        let speed_ratio = current_speed / self.perf.max_vel;
        let turn_rate = (self.perf.max_turn_rate * speed_ratio * speed_ratio).max(MIN_TURN_RATE);
        let max_turn = turn_rate * dt;
        let turn = hdg_error.clamp(-max_turn, max_turn);
        self.state.hdg += turn;

        // Speed control - more aggressive slowdown when collision is imminent
        // At urgency 0.3+, start slowing; at urgency 1.0, reduce to 40% speed
        let speed_factor = if vo_result.urgency > 0.3 {
            let urgency_above_threshold = (vo_result.urgency - 0.3) / 0.7; // 0 to 1
            1.0 - urgency_above_threshold * 0.6 // Slow down to 40% at max urgency
        } else {
            1.0
        };
        let target_speed = self.perf.max_vel * speed_factor;

        // Calculate acceleration
        let speed_error = target_speed - current_speed;
        let accel = speed_error.clamp(-self.perf.max_acc, self.perf.max_acc);

        // Apply acceleration
        let new_speed = (current_speed + accel * dt).clamp(0.0, self.perf.max_vel);
        self.state.vel = Velocity::from_heading_and_speed(self.state.hdg, new_speed);
        self.state.acc = Acceleration::from_heading_and_magnitude(self.state.hdg, accel);

        // Update position (unbounded - can move beyond screen)
        self.state.pos += self.state.vel.scaled(dt);
    }

    /// Move toward a target delta while incorporating avoidance steering.
    /// Used for emergency separation when not following waypoints.
    ///
    /// # Arguments
    /// * `target_delta` - Vector pointing toward the waypoint target (can be zero for pure avoidance)
    /// * `avoidance` - Steering vector from separation behavior (can be zero)
    /// * `dt` - Delta time in seconds
    fn move_to_pos(&mut self, target_delta: Vec2, avoidance: Vec2, dt: f32) {
        let dist = target_delta.magnitude();
        let avoidance_mag = avoidance.magnitude();

        // Blend target seeking with avoidance
        // Avoidance weight scales with its magnitude relative to strength
        // At full strength avoidance, it dominates; at zero, target dominates
        let combined = if avoidance_mag > f32::EPSILON || dist > f32::EPSILON {
            // Calculate blend weight: avoidance becomes dominant as magnitude increases
            // Using sigmoid-like curve: weight = mag / (mag + strength/2)
            let avoidance_weight = if avoidance_mag > f32::EPSILON {
                let half_strength = self.separation.strength / 2.0;
                (avoidance_mag / (avoidance_mag + half_strength)).min(0.95)
            } else {
                0.0
            };
            let target_weight = 1.0 - avoidance_weight;

            // Normalize target delta for blending (if non-zero)
            let target_dir = if dist > f32::EPSILON {
                Vec2::new(target_delta.x / dist, target_delta.y / dist)
            } else {
                Vec2::new(0.0, 0.0)
            };

            // Normalize avoidance for blending (if non-zero)
            let avoid_dir = if avoidance_mag > f32::EPSILON {
                Vec2::new(avoidance.x / avoidance_mag, avoidance.y / avoidance_mag)
            } else {
                Vec2::new(0.0, 0.0)
            };

            Vec2::new(
                target_dir.x * target_weight + avoid_dir.x * avoidance_weight,
                target_dir.y * target_weight + avoid_dir.y * avoidance_weight,
            )
        } else {
            // No steering input at all
            Vec2::new(0.0, 0.0)
        };

        // Current speed squared (avoids sqrt)
        let speed_sq = self.state.vel.speed_squared();
        let current_speed = speed_sq.sqrt();

        // Calculate desired heading from combined steering
        let combined_mag = combined.magnitude();
        let desired_hdg = if combined_mag > f32::EPSILON {
            Heading::new(combined.heading())
        } else {
            self.state.hdg // Keep current heading if no steering input
        };

        // Calculate heading error (shortest angular distance)
        let hdg_error = self.state.hdg.difference(desired_hdg);

        // Turn rate scales with velocity squared - need momentum to turn effectively
        // At zero speed, use minimum turn rate. At max speed, max turn rate.
        let speed_ratio = current_speed / self.perf.max_vel;
        let turn_rate = (self.perf.max_turn_rate * speed_ratio * speed_ratio).max(MIN_TURN_RATE);
        let max_turn = turn_rate * dt;
        let turn = hdg_error.clamp(-max_turn, max_turn);
        self.state.hdg += turn;

        // Determine desired speed
        // Must maintain speed to be able to turn, so always try to move
        // Use speed_squared to avoid extra sqrt
        let stopping_dist = speed_sq / (2.0 * self.perf.max_acc);

        let desired_speed = if dist < stopping_dist + 5.0 && avoidance_mag < f32::EPSILON {
            // Close to target and no avoidance needed - slow down to stop
            (2.0 * self.perf.max_acc * dist.max(0.0)).sqrt() * 0.8
        } else {
            // Need to keep moving to maintain turn authority
            // Cruise at max speed
            self.perf.max_vel
        };

        // Calculate acceleration along heading
        let speed_error = desired_speed - current_speed;
        let accel = speed_error.clamp(-self.perf.max_acc, self.perf.max_acc);

        // Apply acceleration along heading direction
        let new_speed = (current_speed + accel * dt).clamp(0.0, self.perf.max_vel);

        // Update velocity using heading direction
        self.state.vel = Velocity::from_heading_and_speed(self.state.hdg, new_speed);

        // Store acceleration for debugging/display (along heading)
        self.state.acc = Acceleration::from_heading_and_magnitude(self.state.hdg, accel);

        // Update position: pos += vel * dt (unbounded)
        self.state.pos += self.state.vel.scaled(dt);
    }

    /// Get immutable reference to drone objective
    pub fn objective(&self) -> &Objective {
        &self.objective
    }

    /// Process current waypoint: check arrival, advance to next, and move toward it.
    /// Returns true if there's a waypoint to move toward.
    ///
    /// # Arguments
    /// * `dt` - Delta time in seconds
    /// * `swarm` - All drones for velocity obstacle calculations
    /// * `use_smoothing` - If true, uses smoothed 3-waypoint lookahead (for route mode)
    fn process_waypoints(&mut self, dt: f32, swarm: &[DroneInfo], use_smoothing: bool) -> bool {
        // Pop all waypoints that are within clearance (handles fast movement / close waypoints)
        while let Some(&waypoint) = self.objective.waypoints.front() {
            let delta = self
                .bounds
                .delta(self.state.pos.as_vec2(), waypoint.as_vec2());
            let dist_to_waypt = delta.magnitude();

            if dist_to_waypt < self.waypoint_clearance {
                self.objective.waypoints.pop_front();
            } else {
                break; // Next waypoint is not yet reached
            }
        }

        if self.objective.waypoints.is_empty() {
            return false;
        }

        // Compute heading - smoothed for routes, direct for waypoints
        let desired_heading = if use_smoothing {
            self.compute_smoothed_heading()
        } else {
            // Direct heading to next waypoint
            let next_waypoint = self.objective.waypoints.front().unwrap();
            let next_delta = self
                .bounds
                .delta(self.state.pos.as_vec2(), next_waypoint.as_vec2());
            Heading::new(next_delta.heading())
        };
        self.move_to_heading(desired_heading, swarm, dt);
        true
    }

    /// Compute a smoothed heading using Stanley controller on a Hermite spline.
    ///
    /// Stanley controller: steering = heading_error + atan(k * cross_track_error / speed)
    /// - heading_error: angle between drone heading and path tangent
    /// - cross_track_error: signed perpendicular distance from drone to path
    fn compute_smoothed_heading(&self) -> Heading {
        let drone_pos = self.state.pos.as_vec2();
        let mut waypoints: Vec<Position> = self.objective.waypoints.iter().take(2).copied().collect();

        if waypoints.is_empty() {
            return self.state.hdg;
        }

        // Get WP1 position
        let wp1 = waypoints[0].as_vec2();
        let to_wp1 = self.bounds.delta(drone_pos, wp1);
        let dist_to_wp1 = to_wp1.magnitude();

        if dist_to_wp1 < f32::EPSILON {
            return self.state.hdg;
        }

        // If only one waypoint but we have a route, use route[0] as second waypoint for spline
        if waypoints.len() == 1 {
            if let Some(route) = &self.objective.route {
                if !route.is_empty() {
                    waypoints.push(route[0]);
                }
            }
        }

        // If still only one waypoint, head straight to it
        if waypoints.len() == 1 {
            return Heading::new(to_wp1.heading());
        }

        // Build Hermite spline (relative to drone at origin)
        let waypoint_refs: Vec<&Position> = waypoints.iter().collect();
        let (p0, p1, t0, t1) = self.build_spline_params(&waypoint_refs);

        let speed = self.state.vel.speed();

        // Speed-adaptive lookahead: faster drones look further ahead
        // This gives more time to turn, preventing overshoot on curves
        const BASE_LOOKAHEAD: f32 = 30.0;
        const SPEED_LOOKAHEAD_FACTOR: f32 = 1.5; // seconds of travel time
        let lookahead_dist = BASE_LOOKAHEAD + speed * SPEED_LOOKAHEAD_FACTOR;
        let lookahead_t = (lookahead_dist / dist_to_wp1).clamp(0.05, 0.8);

        // Get the lookahead point and tangent
        let lookahead_point = Self::hermite_point(p0, p1, t0, t1, lookahead_t);
        let path_tangent = Self::hermite_tangent(p0, p1, t0, t1, lookahead_t);

        // Check how much we need to turn to face the lookahead point
        let pursuit_heading = Heading::new(lookahead_point.heading());
        let heading_to_target = self.state.hdg.difference(pursuit_heading);

        // If the turn is very sharp (> 90 degrees), we may need to consider
        // that we're overshooting. For now, just use the pursuit heading
        // but could add logic to reduce speed or turn harder.

        // Cross-track error for Stanley correction
        let lookahead_to_drone = Vec2::new(-lookahead_point.x, -lookahead_point.y);
        let path_tangent_norm = if path_tangent.magnitude() > f32::EPSILON {
            path_tangent.normalized()
        } else {
            lookahead_point.normalized()
        };
        let cross_track = path_tangent_norm.perp().dot(lookahead_to_drone);

        // Stanley correction (scaled down at high speed to prevent oscillation)
        const STANLEY_K: f32 = 2.0;
        const MIN_SPEED: f32 = 5.0;
        let effective_speed = speed.max(MIN_SPEED);
        let cross_track_correction = (STANLEY_K * cross_track / effective_speed).atan();

        // Final heading: pursuit direction with cross-track correction
        let desired_heading = pursuit_heading.radians() - cross_track_correction;

        Heading::new(desired_heading)
    }

    /// Build Hermite spline parameters for the current route.
    /// Returns (p0, p1, t0, t1) where the spline runs from the path start toward WP1.
    fn build_spline_params(&self, waypoints: &[&Position]) -> (Vec2, Vec2, Vec2, Vec2) {
        let drone_pos = self.state.pos.as_vec2();
        let wp1 = waypoints[0].as_vec2();
        let to_wp1 = self.bounds.delta(drone_pos, wp1);
        let dist_to_wp1 = to_wp1.magnitude();

        // P0: spline starts at drone's current position (origin in relative coords)
        let p0 = Vec2::new(0.0, 0.0);
        // P1: spline ends at WP1
        let p1 = to_wp1;

        // Tangent scale - use a fraction of distance, capped to reasonable value
        let tangent_scale = (dist_to_wp1 * 0.5).min(150.0);

        // T0: start tangent based on drone's velocity direction (or heading if not moving)
        let speed = self.state.vel.speed();
        let t0 = if speed > 1.0 {
            self.state.vel.as_vec2().normalized() * tangent_scale
        } else {
            self.state.hdg.to_vec2() * tangent_scale
        };

        // T1: end tangent toward WP2
        let wp1_to_wp2 = self.bounds.delta(wp1, waypoints[1].as_vec2());
        let t1 = if wp1_to_wp2.magnitude() > f32::EPSILON {
            wp1_to_wp2.normalized() * tangent_scale
        } else {
            to_wp1.normalized() * tangent_scale
        };

        (p0, p1, t0, t1)
    }

    /// Evaluate a point on a cubic Hermite spline at parameter t (0-1).
    /// - p0: start position (at t=0)
    /// - p1: end position (at t=1)
    /// - t0: start tangent
    /// - t1: end tangent
    fn hermite_point(p0: Vec2, p1: Vec2, t0: Vec2, t1: Vec2, t: f32) -> Vec2 {
        let t2 = t * t;
        let t3 = t2 * t;

        // Hermite basis functions
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0; // p0 coefficient
        let h10 = t3 - 2.0 * t2 + t;          // t0 coefficient
        let h01 = -2.0 * t3 + 3.0 * t2;       // p1 coefficient
        let h11 = t3 - t2;                     // t1 coefficient

        Vec2::new(
            h00 * p0.x + h10 * t0.x + h01 * p1.x + h11 * t1.x,
            h00 * p0.y + h10 * t0.y + h01 * p1.y + h11 * t1.y,
        )
    }

    /// Evaluate the tangent (derivative) of a cubic Hermite spline at parameter t (0-1).
    fn hermite_tangent(p0: Vec2, p1: Vec2, t0: Vec2, t1: Vec2, t: f32) -> Vec2 {
        let t2 = t * t;

        // Derivatives of Hermite basis functions
        let dh00 = 6.0 * t2 - 6.0 * t;        // d/dt(2t³ - 3t² + 1)
        let dh10 = 3.0 * t2 - 4.0 * t + 1.0;  // d/dt(t³ - 2t² + t)
        let dh01 = -6.0 * t2 + 6.0 * t;       // d/dt(-2t³ + 3t²)
        let dh11 = 3.0 * t2 - 2.0 * t;        // d/dt(t³ - t²)

        Vec2::new(
            dh00 * p0.x + dh10 * t0.x + dh01 * p1.x + dh11 * t1.x,
            dh00 * p0.y + dh10 * t0.y + dh01 * p1.y + dh11 * t1.y,
        )
    }

    /// Get spline path points for visualization (only in route mode with 2+ waypoints).
    /// Returns world coordinates for each point along the spline.
    pub fn get_spline_path(&self, num_points: usize) -> Vec<Vec2> {
        // Only compute spline for route mode
        if !matches!(self.objective.task, ObjectiveType::FollowRoute) {
            return Vec::new();
        }

        let drone_pos = self.state.pos.as_vec2();
        let mut waypoints: Vec<Position> = self.objective.waypoints.iter().take(2).copied().collect();

        if waypoints.is_empty() {
            return Vec::new();
        }

        // If only one waypoint but we have a route, use route[0] as second waypoint for spline
        if waypoints.len() == 1 {
            if let Some(route) = &self.objective.route {
                if !route.is_empty() {
                    waypoints.push(route[0]);
                }
            }
        }

        if waypoints.len() < 2 {
            return Vec::new();
        }

        let wp1 = waypoints[0].as_vec2();
        let to_wp1 = self.bounds.delta(drone_pos, wp1);
        let dist_to_wp1 = to_wp1.magnitude();

        if dist_to_wp1 < f32::EPSILON {
            return Vec::new();
        }

        // Hermite spline setup (relative to drone)
        let p0 = Vec2::new(0.0, 0.0);
        let p1 = to_wp1;

        // Tangent scale - use a fraction of distance, capped to reasonable value
        let tangent_scale = (dist_to_wp1 * 0.5).min(150.0);

        // Start tangent: use drone's velocity direction (or heading if not moving)
        let speed = self.state.vel.speed();
        let t0 = if speed > 1.0 {
            self.state.vel.as_vec2().normalized() * tangent_scale
        } else {
            self.state.hdg.to_vec2() * tangent_scale
        };

        // End tangent: direction toward WP2
        let wp2 = waypoints[1].as_vec2();
        let wp1_to_wp2 = self.bounds.delta(wp1, wp2);
        let t1 = if wp1_to_wp2.magnitude() > f32::EPSILON {
            wp1_to_wp2.normalized() * tangent_scale
        } else {
            to_wp1.normalized() * tangent_scale
        };

        // Generate points along the spline
        let mut points = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let t = i as f32 / (num_points - 1) as f32;
            let relative_point = Self::hermite_point(p0, p1, t0, t1, t);
            // Convert back to world coordinates
            let world_point = Vec2::new(
                drone_pos.x + relative_point.x,
                drone_pos.y + relative_point.y,
            );
            points.push(world_point);
        }

        points
    }
}

impl Drone for FixedWing {
    fn uid(&self) -> usize {
        self.id
    }

    fn state_update(&mut self, dt: f32, swarm: &[DroneInfo]) {
        match self.objective.task {
            ObjectiveType::ReachWaypoint => {
                // Direct waypoint navigation (no path smoothing)
                if !self.process_waypoints(dt, swarm, false) && self.objective.waypoints.is_empty() {
                    self.objective.task = ObjectiveType::Sleep;
                }
            }
            ObjectiveType::FollowRoute => {
                // Reload waypoints from route when exhausted
                if self.objective.waypoints.is_empty() {
                    if let Some(route) = &self.objective.route {
                        self.objective.waypoints = route.iter().copied().collect();
                    }
                }
                // Smoothed path planning with 3-waypoint lookahead for routes
                self.process_waypoints(dt, swarm, true);
            }
            _ => {
                // Sleep, Loiter, FollowTarget - use separation as fallback
                let avoidance = calculate_separation(
                    self.state.pos,
                    self.id,
                    swarm,
                    &self.bounds,
                    &self.separation,
                );
                if avoidance.magnitude() > f32::EPSILON {
                    // Move to avoid collision even without a waypoint target
                    self.move_to_pos(Vec2::new(0.0, 0.0), avoidance, dt);
                } else {
                    trace!("Drone {} in {:?} state", self.id, self.objective.task);
                }
            }
        }
    }

    fn set_objective(&mut self, objective: Objective) {
        self.objective = objective;
    }

    fn clear_objective(&mut self) {
        self.objective = Objective::default();
    }

    fn state(&self) -> &State {
        &self.state
    }

    fn get_info(&self) -> DroneInfo {
        DroneInfo::new(self.id, &self.state)
    }

    fn set_flight_params(&mut self, params: DronePerfFeatures) {
        self.perf = params;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Arc;

    fn create_test_bounds() -> Bounds {
        Bounds::new(1000.0, 1000.0).unwrap()
    }

    fn create_test_drone() -> FixedWing {
        FixedWing::new(
            0,
            Position::new(500.0, 500.0),
            Heading::new(0.0),
            create_test_bounds(),
        )
    }

    #[test]
    fn test_new_drone_valid() {
        let drone = FixedWing::new(
            1,
            Position::new(100.0, 200.0),
            Heading::new(0.5),
            create_test_bounds(),
        );
        assert_eq!(drone.uid(), 1);
        assert!((drone.state().pos.x() - 100.0).abs() < 1e-6);
        assert!((drone.state().pos.y() - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_flight_params() {
        let mut drone = create_test_drone();
        let params = DronePerfFeatures::new(100.0, 10.0, 2.0).unwrap();
        drone.set_flight_params(params);
        // Validation happens at DronePerfFeatures::new(), not set_flight_params
    }

    #[test]
    fn test_follow_route_loops() {
        let mut drone = create_test_drone();

        let route: Arc<[Position]> = Arc::from(vec![
            Position::new(100.0, 100.0),
            Position::new(200.0, 200.0),
        ]);

        drone.set_objective(Objective {
            task: ObjectiveType::FollowRoute,
            waypoints: route.iter().copied().collect(),
            route: Some(route.clone()),
            targets: None,
        });

        // Simulate arriving at all waypoints by clearing waypoints
        drone.objective.waypoints.clear();

        // Update should reset waypoints from route
        drone.state_update(0.016, &[]);

        assert_eq!(drone.objective.waypoints.len(), 2);
    }

    #[test]
    fn test_clear_objective() {
        let mut drone = create_test_drone();

        let mut waypoints = VecDeque::new();
        waypoints.push_back(Position::new(100.0, 100.0));

        drone.set_objective(Objective {
            task: ObjectiveType::ReachWaypoint,
            waypoints,
            route: None,
            targets: None,
        });

        drone.clear_objective();

        assert_eq!(drone.objective.task, ObjectiveType::Sleep);
        assert!(drone.objective.waypoints.is_empty());
    }

    #[test]
    fn test_get_info() {
        let drone = FixedWing::new(
            42,
            Position::new(100.0, 200.0),
            Heading::new(1.5),
            create_test_bounds(),
        );

        let info = drone.get_info();
        assert_eq!(info.uid, 42);
        assert!((info.pos.x() - 100.0).abs() < 1e-6);
        assert!((info.pos.y() - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_velocity_from_heading() {
        let mut drone = FixedWing::new(
            0,
            Position::new(100.0, 100.0),
            Heading::new(0.0), // Facing right (+x)
            create_test_bounds(),
        );

        // Set a waypoint far to the right
        let mut waypoints = VecDeque::new();
        waypoints.push_back(Position::new(900.0, 100.0));

        drone.set_objective(Objective {
            task: ObjectiveType::ReachWaypoint,
            waypoints,
            route: None,
            targets: None,
        });

        // Run a few updates to build up speed
        for _ in 0..100 {
            drone.state_update(0.016, &[]);
        }

        // Velocity should be primarily in +x direction
        let vel = drone.state().vel;
        assert!(vel.as_vec2().x > 0.0);
        assert!(vel.as_vec2().x.abs() > vel.as_vec2().y.abs());
    }

}
