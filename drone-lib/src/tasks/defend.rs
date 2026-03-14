//! Defend task: orbit a point and engage threats that enter the zone.

use crate::tasks::{DroneTask, SafetyFeedback, TaskOutput, TaskStatus};
use crate::types::{Bounds, DroneInfo, DronePerfFeatures, Heading, Position, State, Vec2};

/// Phase of the defend task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefendPhase {
    /// Circling the defense point at the orbit radius.
    Orbit,
    /// Pursuing a threat that entered the defense zone.
    Engage,
    /// Returning to the orbit after engagement.
    Return,
}

/// Defend task: orbit a point, engage threats that enter the zone, return after.
///
/// The drone circles the defense point at `orbit_radius`. When an enemy drone
/// enters the `engage_radius`, the drone breaks orbit to pursue it with lead
/// pursuit. If the target is destroyed or leaves the zone, the drone returns
/// to orbit.
#[derive(Debug)]
pub struct DefendTask {
    /// Center of the defense zone.
    center: Position,
    /// Radius to orbit at.
    orbit_radius: f32,
    /// Radius within which to engage threats.
    engage_radius: f32,
    /// Current phase.
    phase: DefendPhase,
    /// Task status.
    status: TaskStatus,
    /// This drone's own ID.
    self_id: usize,
    /// This drone's group (for identifying enemies).
    self_group: u32,
    /// Target drone being engaged (if any).
    engage_target_id: Option<usize>,
    /// Detonation radius — detonate when this close to engaged target.
    detonation_radius: f32,
    /// Orbit direction: true = counterclockwise, false = clockwise.
    orbit_ccw: bool,
}

impl DefendTask {
    /// Create a new defend task.
    pub fn new(
        self_id: usize,
        self_group: u32,
        center: Position,
        orbit_radius: f32,
        engage_radius: f32,
        detonation_radius: f32,
    ) -> Self {
        DefendTask {
            center,
            orbit_radius,
            engage_radius,
            phase: DefendPhase::Orbit,
            status: TaskStatus::Active,
            self_id,
            self_group,
            engage_target_id: None,
            detonation_radius,
            orbit_ccw: true,
        }
    }

    /// Get the current phase.
    pub fn phase(&self) -> DefendPhase {
        self.phase
    }

    /// Compute orbit velocity — tangent to the circle around center.
    ///
    /// The drone is steered toward the orbit circle and then along it.
    /// If inside the orbit radius, it steers outward + tangent.
    /// If outside, it steers inward + tangent.
    fn orbit_velocity(
        &self,
        state: &State,
        bounds: &Bounds,
        max_speed: f32,
    ) -> Vec2 {
        // Vector from center to drone
        let to_drone = bounds.delta(self.center.as_vec2(), state.pos.as_vec2());
        let dist = to_drone.magnitude();

        if dist < 1.0 {
            // At center — move outward in any direction
            return Vec2::new(max_speed * 0.5, 0.0);
        }

        // Radial unit vector (center → drone)
        let radial = Vec2::new(to_drone.x / dist, to_drone.y / dist);

        // Tangent vector (perpendicular to radial)
        let tangent = if self.orbit_ccw {
            Vec2::new(-radial.y, radial.x) // CCW
        } else {
            Vec2::new(radial.y, -radial.x) // CW
        };

        // Radial correction: steer toward the orbit circle
        // Positive = too far out, need to move inward
        // Negative = too close, need to move outward
        let radial_error = dist - self.orbit_radius;
        // Correction strength proportional to error, saturating at orbit_radius
        let correction = (radial_error / self.orbit_radius).clamp(-1.0, 1.0);

        // Blend tangent (movement along orbit) with radial correction (toward circle)
        // More tangent when on the circle, more radial when off it
        let radial_weight = correction.abs().min(0.8);
        let tangent_weight = 1.0 - radial_weight;

        let blended = Vec2::new(
            tangent.x * tangent_weight - radial.x * correction * radial_weight,
            tangent.y * tangent_weight - radial.y * correction * radial_weight,
        );

        let mag = blended.magnitude();
        if mag > f32::EPSILON {
            // Slow down when far from the circle, cruise at full speed on it
            let speed = max_speed * (1.0 - radial_weight * 0.3);
            Vec2::new(blended.x / mag * speed, blended.y / mag * speed)
        } else {
            Vec2::new(tangent.x * max_speed, tangent.y * max_speed)
        }
    }

    /// Compute pursue velocity toward an enemy drone with lead pursuit.
    fn pursue_velocity(
        &self,
        state: &State,
        target: &DroneInfo,
        bounds: &Bounds,
        max_speed: f32,
    ) -> Vec2 {
        let delta = bounds.delta(state.pos.as_vec2(), target.pos.as_vec2());
        let dist = delta.magnitude();

        if dist < 0.1 {
            return Vec2::new(0.0, 0.0);
        }

        // Lead pursuit: predict where target will be
        let target_vel = target.vel.as_vec2();
        let closing_speed = max_speed + target_vel.magnitude();
        let intercept_time = if closing_speed > 0.1 {
            (dist / closing_speed).min(3.0)
        } else {
            0.0
        };

        let predicted = Vec2::new(
            delta.x + target_vel.x * intercept_time,
            delta.y + target_vel.y * intercept_time,
        );

        let pred_dist = predicted.magnitude();
        if pred_dist < 0.1 {
            return Vec2::new(delta.x / dist * max_speed, delta.y / dist * max_speed);
        }

        Vec2::new(
            predicted.x / pred_dist * max_speed,
            predicted.y / pred_dist * max_speed,
        )
    }

    /// Compute return velocity — navigate back toward the orbit circle.
    fn return_velocity(
        &self,
        state: &State,
        bounds: &Bounds,
        max_speed: f32,
    ) -> Vec2 {
        // Aim for the nearest point on the orbit circle
        let to_drone = bounds.delta(self.center.as_vec2(), state.pos.as_vec2());
        let dist = to_drone.magnitude();

        if dist < 1.0 {
            // At center — move outward
            return Vec2::new(max_speed * 0.5, 0.0);
        }

        // Point on orbit circle closest to current position
        let radial = Vec2::new(to_drone.x / dist, to_drone.y / dist);
        let target_point = Vec2::new(
            self.center.x() + radial.x * self.orbit_radius,
            self.center.y() + radial.y * self.orbit_radius,
        );

        let delta = bounds.delta(state.pos.as_vec2(), target_point);
        let target_dist = delta.magnitude();

        if target_dist < 1.0 {
            return Vec2::new(0.0, 0.0);
        }

        Vec2::new(
            delta.x / target_dist * max_speed,
            delta.y / target_dist * max_speed,
        )
    }

    /// Find the closest enemy drone within the engage radius.
    fn find_threat<'a>(&self, state: &State, swarm: &'a [DroneInfo], bounds: &Bounds) -> Option<&'a DroneInfo> {
        let mut closest: Option<(&DroneInfo, f32)> = None;

        for drone in swarm {
            if drone.uid == self.self_id || drone.group == self.self_group {
                continue;
            }

            // Check distance from the defense center
            let dist_to_center = bounds.distance(self.center.as_vec2(), drone.pos.as_vec2());
            if dist_to_center > self.engage_radius {
                continue;
            }

            // Pick the closest to us
            let dist_to_self = bounds.distance(state.pos.as_vec2(), drone.pos.as_vec2());
            match closest {
                None => closest = Some((drone, dist_to_self)),
                Some((_, best_dist)) if dist_to_self < best_dist => {
                    closest = Some((drone, dist_to_self));
                }
                _ => {}
            }
        }

        closest.map(|(d, _)| d)
    }
}

impl DroneTask for DefendTask {
    fn tick(
        &mut self,
        state: &State,
        swarm: &[DroneInfo],
        bounds: &Bounds,
        perf: &DronePerfFeatures,
        _dt: f32,
    ) -> TaskOutput {
        let max_speed = perf.max_vel;

        // Scan for threats in the defense zone
        let threat = self.find_threat(state, swarm, bounds);

        // Phase transitions
        match self.phase {
            DefendPhase::Orbit => {
                if let Some(t) = threat {
                    self.phase = DefendPhase::Engage;
                    self.engage_target_id = Some(t.uid);
                }
            }
            DefendPhase::Engage => {
                // Check if engaged target still exists and is in zone
                let target_alive = self.engage_target_id
                    .and_then(|tid| swarm.iter().find(|d| d.uid == tid));

                match target_alive {
                    Some(target) => {
                        // Check if within detonation range
                        let dist = bounds.distance(state.pos.as_vec2(), target.pos.as_vec2());
                        if dist <= self.detonation_radius {
                            self.status = TaskStatus::Complete;
                            return TaskOutput {
                                desired_velocity: Vec2::new(0.0, 0.0),
                                desired_heading: None,
                                detonate: true,
                                exclude_from_ca: vec![],
                            };
                        }

                        // Check if target left the engagement zone
                        let target_dist_to_center = bounds.distance(
                            self.center.as_vec2(), target.pos.as_vec2()
                        );
                        if target_dist_to_center > self.engage_radius * 1.5 {
                            // Target fled — return to orbit
                            self.phase = DefendPhase::Return;
                            self.engage_target_id = None;
                        }
                    }
                    None => {
                        // Target destroyed or gone
                        self.phase = DefendPhase::Return;
                        self.engage_target_id = None;
                    }
                }
            }
            DefendPhase::Return => {
                // Check if a new threat appeared while returning
                if let Some(t) = threat {
                    self.phase = DefendPhase::Engage;
                    self.engage_target_id = Some(t.uid);
                } else {
                    // Check if we've reached the orbit circle
                    let dist_to_center = bounds.distance(
                        self.center.as_vec2(), state.pos.as_vec2()
                    );
                    let orbit_error = (dist_to_center - self.orbit_radius).abs();
                    if orbit_error < self.orbit_radius * 0.3 {
                        self.phase = DefendPhase::Orbit;
                    }
                }
            }
        }

        // Compute velocity based on phase
        let desired_vel = match self.phase {
            DefendPhase::Orbit => self.orbit_velocity(state, bounds, max_speed),
            DefendPhase::Engage => {
                if let Some(target) = self.engage_target_id
                    .and_then(|tid| swarm.iter().find(|d| d.uid == tid))
                {
                    self.pursue_velocity(state, target, bounds, max_speed)
                } else {
                    // Target lost during this tick — return
                    self.phase = DefendPhase::Return;
                    self.engage_target_id = None;
                    self.return_velocity(state, bounds, max_speed)
                }
            }
            DefendPhase::Return => self.return_velocity(state, bounds, max_speed),
        };

        let heading = if desired_vel.magnitude() > f32::EPSILON {
            Some(Heading::new(desired_vel.y.atan2(desired_vel.x)))
        } else {
            None
        };

        TaskOutput {
            desired_velocity: desired_vel,
            desired_heading: heading,
            detonate: false,
            exclude_from_ca: vec![],
        }
    }

    fn process_feedback(&mut self, feedback: &SafetyFeedback) {
        // In orbit or return, tolerate safety adjustments
        // During engage, high urgency threat from non-target could force disengage
        if self.phase == DefendPhase::Engage && feedback.urgency > 0.8 {
            // Extremely high urgency during engagement — stay committed
            // (unlike InterceptTask which evades, defenders hold their ground)
        }
    }

    fn phase_name(&self) -> &str {
        match self.phase {
            DefendPhase::Orbit => "orbit",
            DefendPhase::Engage => "engage",
            DefendPhase::Return => "return",
        }
    }

    fn status(&self) -> TaskStatus {
        self.status
    }

    fn name(&self) -> &str {
        "Defend"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Acceleration, Heading, Velocity};

    fn test_perf() -> DronePerfFeatures {
        DronePerfFeatures {
            max_vel: 120.0,
            max_acc: 100.0,
            max_turn_rate: 0.6,
        }
    }

    fn make_state(x: f32, y: f32, vx: f32, vy: f32) -> State {
        State {
            pos: Position::new(x, y),
            hdg: Heading::new(vy.atan2(vx)),
            vel: Velocity::new(vx, vy),
            acc: Acceleration::zero(),
        }
    }

    fn make_drone(uid: usize, x: f32, y: f32, vx: f32, vy: f32, group: u32) -> DroneInfo {
        DroneInfo {
            uid,
            pos: Position::new(x, y),
            hdg: Heading::new(vy.atan2(vx)),
            vel: Velocity::new(vx, vy),
            is_formation_leader: false,
            group,
        }
    }

    #[test]
    fn test_orbit_produces_tangent_velocity() {
        let mut task = DefendTask::new(0, 0, Position::new(0.0, 0.0), 300.0, 500.0, 187.5);
        // Place drone on the orbit circle (300m east of center)
        let state = make_state(300.0, 0.0, 0.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        let output = task.tick(&state, &[], &bounds, &test_perf(), 0.016);

        // On the circle facing east, CCW orbit should produce mostly northward velocity
        assert!(
            output.desired_velocity.y > 50.0,
            "Expected tangent (northward) velocity, got y={}",
            output.desired_velocity.y
        );
        assert_eq!(task.phase(), DefendPhase::Orbit);
    }

    #[test]
    fn test_orbit_corrects_inward_when_too_far() {
        let task = DefendTask::new(0, 0, Position::new(0.0, 0.0), 300.0, 500.0, 187.5);
        // Place drone 600m east (double the orbit radius)
        let state = make_state(600.0, 0.0, 0.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        let vel = task.orbit_velocity(&state, &bounds, 120.0);

        // Should have inward (negative x) component to correct toward orbit circle
        assert!(vel.x < 0.0, "Expected inward correction, got x={}", vel.x);
    }

    #[test]
    fn test_engage_on_threat() {
        let mut task = DefendTask::new(0, 0, Position::new(0.0, 0.0), 300.0, 500.0, 187.5);
        let state = make_state(300.0, 0.0, 0.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();
        let swarm = vec![
            make_drone(0, 300.0, 0.0, 0.0, 0.0, 0), // self
            make_drone(1, 200.0, 0.0, -60.0, 0.0, 1), // enemy inside engage zone
        ];

        task.tick(&state, &swarm, &bounds, &test_perf(), 0.016);

        assert_eq!(task.phase(), DefendPhase::Engage);
    }

    #[test]
    fn test_ignore_threat_outside_zone() {
        let mut task = DefendTask::new(0, 0, Position::new(0.0, 0.0), 300.0, 500.0, 187.5);
        let state = make_state(300.0, 0.0, 0.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();
        let swarm = vec![
            make_drone(0, 300.0, 0.0, 0.0, 0.0, 0),
            make_drone(1, 800.0, 0.0, -60.0, 0.0, 1), // enemy outside engage zone
        ];

        task.tick(&state, &swarm, &bounds, &test_perf(), 0.016);

        assert_eq!(task.phase(), DefendPhase::Orbit);
    }

    #[test]
    fn test_return_after_target_lost() {
        let mut task = DefendTask::new(0, 0, Position::new(0.0, 0.0), 300.0, 500.0, 187.5);
        let state = make_state(400.0, 0.0, 60.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        // Engage a target
        let swarm_with_enemy = vec![
            make_drone(0, 400.0, 0.0, 60.0, 0.0, 0),
            make_drone(1, 200.0, 0.0, -60.0, 0.0, 1),
        ];
        task.tick(&state, &swarm_with_enemy, &bounds, &test_perf(), 0.016);
        assert_eq!(task.phase(), DefendPhase::Engage);

        // Target disappears
        let swarm_empty = vec![make_drone(0, 400.0, 0.0, 60.0, 0.0, 0)];
        task.tick(&state, &swarm_empty, &bounds, &test_perf(), 0.016);
        assert_eq!(task.phase(), DefendPhase::Return);
    }

    #[test]
    fn test_detonate_on_target() {
        let mut task = DefendTask::new(0, 0, Position::new(0.0, 0.0), 300.0, 500.0, 187.5);
        let state = make_state(100.0, 0.0, 60.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();

        // Enemy right next to us (within detonation radius)
        let swarm = vec![
            make_drone(0, 100.0, 0.0, 60.0, 0.0, 0),
            make_drone(1, 150.0, 0.0, -60.0, 0.0, 1), // 50m away < 187.5m
        ];

        // First tick: detect and engage
        task.tick(&state, &swarm, &bounds, &test_perf(), 0.016);
        assert_eq!(task.phase(), DefendPhase::Engage);

        // Second tick: close enough to detonate
        let output = task.tick(&state, &swarm, &bounds, &test_perf(), 0.016);
        assert!(output.detonate);
        assert_eq!(task.status(), TaskStatus::Complete);
    }

    #[test]
    fn test_ignore_friendly_drones() {
        let mut task = DefendTask::new(0, 0, Position::new(0.0, 0.0), 300.0, 500.0, 187.5);
        let state = make_state(300.0, 0.0, 0.0, 0.0);
        let bounds = Bounds::new(2500.0, 2500.0).unwrap();
        let swarm = vec![
            make_drone(0, 300.0, 0.0, 0.0, 0.0, 0),
            make_drone(2, 100.0, 0.0, 60.0, 0.0, 0), // friendly inside zone
        ];

        task.tick(&state, &swarm, &bounds, &test_perf(), 0.016);

        assert_eq!(task.phase(), DefendPhase::Orbit, "Should not engage friendlies");
    }
}
