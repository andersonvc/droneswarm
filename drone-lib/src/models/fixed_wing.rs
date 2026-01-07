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

        // Apply the heading adjustment from VO
        let target_heading = Heading::new(self.state.hdg.radians() + vo_result.heading_adjustment);

        // Calculate heading error (shortest angular distance)
        let hdg_error = self.state.hdg.difference(target_heading);

        // Turn rate scales with velocity squared - need momentum to turn effectively
        let speed_ratio = current_speed / self.perf.max_vel;
        let turn_rate = (self.perf.max_turn_rate * speed_ratio * speed_ratio).max(MIN_TURN_RATE);
        let max_turn = turn_rate * dt;
        let turn = hdg_error.clamp(-max_turn, max_turn);
        self.state.hdg += turn;

        // Speed control - go slower if urgent avoidance needed
        let speed_factor = if vo_result.urgency > 0.5 {
            1.0 - (vo_result.urgency - 0.5) * 0.5 // Slow down by up to 25% at max urgency
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

        // Update position with wrapping
        self.state.pos += self.state.vel.scaled(dt);
        self.state.pos = self.bounds.wrap_position(self.state.pos);
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

        // Update position: pos += vel * dt, then wrap to bounds
        self.state.pos += self.state.vel.scaled(dt);
        self.state.pos = self.bounds.wrap_position(self.state.pos);
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
    fn process_waypoints(&mut self, dt: f32, swarm: &[DroneInfo]) -> bool {
        if let Some(&waypoint) = self.objective.waypoints.front() {
            let delta = self
                .bounds
                .toroidal_delta(self.state.pos.as_vec2(), waypoint.as_vec2());
            let dist_to_waypt = delta.magnitude();

            if dist_to_waypt < self.waypoint_clearance {
                self.objective.waypoints.pop_front();
            }

            if let Some(&next_waypoint) = self.objective.waypoints.front() {
                let next_delta = self
                    .bounds
                    .toroidal_delta(self.state.pos.as_vec2(), next_waypoint.as_vec2());
                // Compute desired heading toward waypoint
                let desired_heading = Heading::new(next_delta.heading());
                self.move_to_heading(desired_heading, swarm, dt);
                return true;
            }
        }
        false
    }
}

impl Drone for FixedWing {
    fn uid(&self) -> usize {
        self.id
    }

    fn state_update(&mut self, dt: f32, swarm: &[DroneInfo]) {
        match self.objective.task {
            ObjectiveType::ReachWaypoint => {
                // VO handles collision avoidance during waypoint following
                if !self.process_waypoints(dt, swarm) && self.objective.waypoints.is_empty() {
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
                // VO handles collision avoidance during route following
                self.process_waypoints(dt, swarm);
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
    fn test_waypoint_arrival_uses_toroidal_distance() {
        // Drone at edge of world, waypoint just across the wrap boundary
        let mut drone = FixedWing::new(
            0,
            Position::new(999.0, 500.0),
            Heading::new(0.0),
            create_test_bounds(),
        );

        // Set waypoint at x=1, which is only 2 units away toroidally
        let mut waypoints = VecDeque::new();
        waypoints.push_back(Position::new(1.0, 500.0));

        drone.set_objective(Objective {
            task: ObjectiveType::ReachWaypoint,
            waypoints,
            route: None,
            targets: None,
        });

        // The waypoint should be detected as "arrived" since toroidal distance is 2 < default clearance (10)
        drone.state_update(0.016, &[]);

        // Waypoint should have been removed and task should be Sleep
        assert!(drone.objective.waypoints.is_empty());
        assert_eq!(drone.objective.task, ObjectiveType::Sleep);
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

    #[test]
    fn test_position_wraps_automatically() {
        let bounds = Bounds::new(100.0, 100.0).unwrap();
        let mut drone = FixedWing::new(
            0,
            Position::new(99.0, 50.0),
            Heading::new(0.0), // Facing right
            bounds,
        );

        // Set waypoint far to the right (will cause position to exceed bounds)
        let mut waypoints = VecDeque::new();
        waypoints.push_back(Position::new(50.0, 50.0));

        drone.set_objective(Objective {
            task: ObjectiveType::ReachWaypoint,
            waypoints,
            route: None,
            targets: None,
        });

        // Run updates until drone crosses the boundary
        for _ in 0..100 {
            drone.state_update(0.016, &[]);
        }

        // Position should be within bounds
        assert!(drone.state().pos.x() >= 0.0 && drone.state().pos.x() < 100.0);
        assert!(drone.state().pos.y() >= 0.0 && drone.state().pos.y() < 100.0);
    }
}
