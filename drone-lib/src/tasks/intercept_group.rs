//! Intercept group task: fly into a cluster of enemy drones and detonate.
//!
//! Unlike regular intercept (which targets a single drone), this task
//! scans for the densest nearby cluster of 2+ enemy drones and steers
//! toward its centroid. The goal is to maximize splash damage by
//! detonating inside a group.

use crate::tasks::{DroneTask, SafetyFeedback, TaskOutput, TaskStatus};
use crate::types::{Bounds, DroneInfo, DronePerfFeatures, Heading, State, Vec2};

/// Intercept group task — fly into a cluster of enemies and detonate.
///
/// Finds the densest cluster of enemy drones (2+ within detonation_radius
/// of each other), steers toward the centroid, and detonates once inside.
/// Collision avoidance is disabled for all enemy drones.
#[derive(Debug)]
pub struct InterceptGroupTask {
    /// The group this drone belongs to (enemies are anything else).
    self_group: u32,
    /// Task status.
    status: TaskStatus,
    /// Blast radius — also used as the cluster search radius.
    detonation_radius: f32,
    /// Current cluster centroid target (updated each tick).
    cluster_target: Option<Vec2>,
    /// IDs of enemies in current target cluster (for CA filtering).
    cluster_enemy_ids: Vec<usize>,
}

impl InterceptGroupTask {
    pub fn new(self_group: u32, detonation_radius: f32) -> Self {
        InterceptGroupTask {
            self_group,
            status: TaskStatus::Active,
            detonation_radius,
            cluster_target: None,
            cluster_enemy_ids: Vec::new(),
        }
    }

    /// Find the densest cluster of enemy drones and return (centroid, enemy_ids).
    /// A cluster is enemies within `detonation_radius` of each other.
    /// We pick the seed enemy with the most neighbors, then use that group.
    fn find_best_cluster(
        &self,
        _state: &State,
        swarm: &[DroneInfo],
        bounds: &Bounds,
    ) -> Option<(Vec2, Vec<usize>)> {
        let enemies: Vec<&DroneInfo> = swarm
            .iter()
            .filter(|d| d.group != self.self_group)
            .collect();

        if enemies.len() < 2 {
            return None;
        }

        // For each enemy, count how many other enemies are within detonation_radius
        let cluster_radius = self.detonation_radius;
        let mut best_seed = 0;
        let mut best_count = 0;

        for (i, enemy) in enemies.iter().enumerate() {
            let count = enemies
                .iter()
                .filter(|other| {
                    other.uid != enemy.uid
                        && bounds.distance(enemy.pos.as_vec2(), other.pos.as_vec2())
                            <= cluster_radius
                })
                .count();
            if count > best_count {
                best_count = count;
                best_seed = i;
            }
        }

        // Need at least 1 neighbor (so 2+ in cluster)
        if best_count == 0 {
            return None;
        }

        let seed = enemies[best_seed];
        let mut cluster_ids = vec![seed.uid];
        let mut cx = seed.pos.x();
        let mut cy = seed.pos.y();

        for enemy in &enemies {
            if enemy.uid != seed.uid
                && bounds.distance(seed.pos.as_vec2(), enemy.pos.as_vec2()) <= cluster_radius
            {
                cluster_ids.push(enemy.uid);
                cx += enemy.pos.x();
                cy += enemy.pos.y();
            }
        }

        let n = cluster_ids.len() as f32;
        let centroid = Vec2::new(cx / n, cy / n);
        Some((centroid, cluster_ids))
    }

    /// Compute pursuit velocity toward the cluster centroid with lead prediction.
    fn pursue_centroid(
        &self,
        state: &State,
        centroid: Vec2,
        cluster: &[&DroneInfo],
        bounds: &Bounds,
        max_speed: f32,
    ) -> Vec2 {
        let delta = bounds.delta(state.pos.as_vec2(), centroid);
        let dist = delta.magnitude();

        if dist < 0.1 {
            return Vec2::new(0.0, 0.0);
        }

        // Average velocity of cluster for lead prediction
        let avg_vel = if !cluster.is_empty() {
            let n = cluster.len() as f32;
            Vec2::new(
                cluster.iter().map(|d| d.vel.as_vec2().x).sum::<f32>() / n,
                cluster.iter().map(|d| d.vel.as_vec2().y).sum::<f32>() / n,
            )
        } else {
            Vec2::new(0.0, 0.0)
        };

        let closing_speed = max_speed + avg_vel.magnitude();
        let intercept_time = if closing_speed > 0.1 {
            (dist / closing_speed).min(3.0)
        } else {
            0.0
        };

        let predicted = Vec2::new(
            delta.x + avg_vel.x * intercept_time,
            delta.y + avg_vel.y * intercept_time,
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
}

impl DroneTask for InterceptGroupTask {
    fn tick(
        &mut self,
        state: &State,
        swarm: &[DroneInfo],
        bounds: &Bounds,
        perf: &DronePerfFeatures,
        _dt: f32,
    ) -> TaskOutput {
        let max_speed = perf.max_vel;

        // Find best enemy cluster
        match self.find_best_cluster(state, swarm, bounds) {
            Some((centroid, enemy_ids)) => {
                self.cluster_target = Some(centroid);
                self.cluster_enemy_ids = enemy_ids;
            }
            None => {
                // No cluster of 2+ found — fall back to nearest enemy
                let nearest = swarm
                    .iter()
                    .filter(|d| d.group != self.self_group)
                    .min_by(|a, b| {
                        let da = bounds.distance(state.pos.as_vec2(), a.pos.as_vec2());
                        let db = bounds.distance(state.pos.as_vec2(), b.pos.as_vec2());
                        da.partial_cmp(&db).unwrap()
                    });

                match nearest {
                    Some(enemy) => {
                        self.cluster_target = Some(enemy.pos.as_vec2());
                        self.cluster_enemy_ids = vec![enemy.uid];
                    }
                    None => {
                        // No enemies at all
                        self.status = TaskStatus::Failed;
                        return TaskOutput {
                            desired_velocity: Vec2::new(0.0, 0.0),
                            desired_heading: None,
                            detonate: false,
                            exclude_from_ca: vec![],
                        };
                    }
                }
            }
        }

        let centroid = self.cluster_target.unwrap();

        // Check detonation: within blast radius of centroid AND
        // at least 2 enemies within blast radius of us
        let enemies_in_range = swarm
            .iter()
            .filter(|d| {
                d.group != self.self_group
                    && bounds.distance(state.pos.as_vec2(), d.pos.as_vec2())
                        <= self.detonation_radius
            })
            .count();

        if enemies_in_range >= 2 {
            self.status = TaskStatus::Complete;
            return TaskOutput {
                desired_velocity: Vec2::new(0.0, 0.0),
                desired_heading: None,
                detonate: true,
                exclude_from_ca: vec![],
            };
        }

        // Also detonate if close to centroid (cluster may have shifted)
        let dist_to_centroid = bounds.distance(state.pos.as_vec2(), centroid);
        if dist_to_centroid <= self.detonation_radius * 0.5 && enemies_in_range >= 1 {
            self.status = TaskStatus::Complete;
            return TaskOutput {
                desired_velocity: Vec2::new(0.0, 0.0),
                desired_heading: None,
                detonate: true,
                exclude_from_ca: vec![],
            };
        }

        // Pursue cluster centroid
        let cluster_drones: Vec<&DroneInfo> = swarm
            .iter()
            .filter(|d| self.cluster_enemy_ids.contains(&d.uid))
            .collect();

        let desired_vel = self.pursue_centroid(state, centroid, &cluster_drones, bounds, max_speed);

        let heading = if desired_vel.magnitude() > f32::EPSILON {
            Some(Heading::new(desired_vel.y.atan2(desired_vel.x)))
        } else {
            None
        };

        TaskOutput {
            desired_velocity: desired_vel,
            desired_heading: heading,
            detonate: false,
            // Disable CA for cluster enemies so we can fly into them
            exclude_from_ca: self.cluster_enemy_ids.clone(),
        }
    }

    fn process_feedback(&mut self, _feedback: &SafetyFeedback) {
        // Kamikaze — ignore safety feedback
    }

    fn phase_name(&self) -> &str {
        "pursue_cluster"
    }

    fn status(&self) -> TaskStatus {
        self.status
    }

    fn name(&self) -> &str {
        "InterceptGroup"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Acceleration, Heading, Position, Velocity};

    fn make_state(x: f32, y: f32) -> State {
        State {
            pos: Position::new(x, y),
            hdg: Heading::new(0.0),
            vel: Velocity::new(60.0, 0.0),
            acc: Acceleration::zero(),
        }
    }

    fn make_drone(uid: usize, x: f32, y: f32, group: u32) -> DroneInfo {
        DroneInfo {
            uid,
            pos: Position::new(x, y),
            hdg: Heading::new(0.0),
            vel: Velocity::zero(),
            is_formation_leader: false,
            group,
        }
    }

    fn perf() -> DronePerfFeatures {
        DronePerfFeatures {
            max_vel: 120.0,
            max_acc: 100.0,
            max_turn_rate: 0.6,
        }
    }

    fn bounds() -> Bounds {
        Bounds::new(5000.0, 5000.0).unwrap()
    }

    #[test]
    fn test_finds_cluster_and_pursues_centroid() {
        let mut task = InterceptGroupTask::new(0, 187.5);
        let state = make_state(0.0, 0.0);
        let swarm = vec![
            make_drone(0, 0.0, 0.0, 0),       // self
            make_drone(1, 1000.0, 0.0, 1),     // enemy cluster
            make_drone(2, 1050.0, 0.0, 1),     // enemy cluster (within 187.5 of drone 1)
            make_drone(3, 3000.0, 3000.0, 1),  // lone enemy, far away
        ];

        let output = task.tick(&state, &swarm, &bounds(), &perf(), 0.016);

        // Should pursue toward the cluster (around x=1025)
        assert!(output.desired_velocity.x > 0.0, "Should pursue toward cluster");
        assert!(!output.detonate, "Not close enough to detonate");
        assert_eq!(task.status(), TaskStatus::Active);
    }

    #[test]
    fn test_detonates_when_2_enemies_in_range() {
        let mut task = InterceptGroupTask::new(0, 187.5);
        let state = make_state(500.0, 0.0);
        let swarm = vec![
            make_drone(0, 500.0, 0.0, 0),   // self
            make_drone(1, 550.0, 0.0, 1),    // within 187.5
            make_drone(2, 600.0, 0.0, 1),    // within 187.5
        ];

        let output = task.tick(&state, &swarm, &bounds(), &perf(), 0.016);

        assert!(output.detonate, "Should detonate when 2+ enemies in range");
        assert_eq!(task.status(), TaskStatus::Complete);
    }

    #[test]
    fn test_no_detonate_with_single_enemy_far() {
        let mut task = InterceptGroupTask::new(0, 187.5);
        let state = make_state(0.0, 0.0);
        let swarm = vec![
            make_drone(0, 0.0, 0.0, 0),
            make_drone(1, 1000.0, 0.0, 1), // only 1 enemy, far away
        ];

        let output = task.tick(&state, &swarm, &bounds(), &perf(), 0.016);

        assert!(!output.detonate, "Should not detonate with only 1 enemy far away");
        assert_eq!(task.status(), TaskStatus::Active);
    }

    #[test]
    fn test_fails_when_no_enemies() {
        let mut task = InterceptGroupTask::new(0, 187.5);
        let state = make_state(0.0, 0.0);
        let swarm = vec![make_drone(0, 0.0, 0.0, 0)]; // only self

        let output = task.tick(&state, &swarm, &bounds(), &perf(), 0.016);

        assert_eq!(task.status(), TaskStatus::Failed);
        assert!(!output.detonate);
    }

    #[test]
    fn test_fallback_to_nearest_when_no_cluster() {
        let mut task = InterceptGroupTask::new(0, 187.5);
        let state = make_state(0.0, 0.0);
        // Two enemies but too far apart to form a cluster
        let swarm = vec![
            make_drone(0, 0.0, 0.0, 0),
            make_drone(1, 500.0, 0.0, 1),
            make_drone(2, 2000.0, 2000.0, 1),
        ];

        let output = task.tick(&state, &swarm, &bounds(), &perf(), 0.016);

        // Should still pursue (nearest enemy)
        assert!(output.desired_velocity.x > 0.0, "Should pursue nearest enemy as fallback");
        assert_eq!(task.status(), TaskStatus::Active);
    }
}
