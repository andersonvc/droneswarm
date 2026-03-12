//! Path planning using Hermite splines and NLGL (Nonlinear Guidance Law).

use crate::types::{Bounds, Heading, Position, State, Vec2};

/// Configuration for approach gate (rear-entry formation joining).
#[derive(Debug, Clone, Copy)]
pub struct ApproachGateConfig {
    /// Radius of the "capture ball" around each formation slot (meters).
    pub ball_radius: f32,
    /// Distance behind the slot where the approach gate is located (meters).
    pub gate_offset: f32,
    /// Angle tolerance for rear approach (radians).
    pub angle_tolerance: f32,
}

/// Path planner using Hermite splines and NLGL guidance.
///
/// NLGL (Nonlinear Guidance Law) is a geometric path-following algorithm
/// commonly used in fixed-wing UAV autopilots. It works by:
///
/// 1. Computing a lookahead point on the desired path
/// 2. Steering toward that lookahead point
/// 3. The lateral acceleration naturally causes path convergence
///
/// Key advantages over Stanley controller:
/// - Simpler: only one tuning parameter (lookahead distance)
/// - Designed for aircraft, not ground vehicles
/// - Exponential convergence with fewer oscillations
/// - Better handling of curved paths
///
/// Reference: Park, S., Deyst, J., & How, J. P. (2007).
/// "Performance and Lyapunov stability of a nonlinear path-following guidance method"
#[derive(Debug, Clone)]
pub struct PathPlanner {
    /// Base lookahead distance in units.
    /// Larger values = smoother but slower convergence.
    /// Smaller values = tighter tracking but may oscillate.
    pub base_lookahead: f32,
    /// Additional lookahead per unit of speed (seconds of travel time).
    /// This makes faster drones look further ahead for stability.
    pub speed_lookahead_factor: f32,
    /// Minimum speed for calculations (prevents edge cases).
    pub min_speed: f32,
}

impl Default for PathPlanner {
    fn default() -> Self {
        PathPlanner {
            base_lookahead: 50.0,           // Increased from 30 for smoother tracking
            speed_lookahead_factor: 1.0,    // 1 second of travel time ahead
            min_speed: 5.0,
        }
    }
}

impl PathPlanner {
    /// Create a new path planner with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute desired heading using NLGL (Nonlinear Guidance Law).
    ///
    /// NLGL simply steers toward a lookahead point on the path.
    /// The path is smoothed using Hermite splines when multiple waypoints exist.
    ///
    /// # Arguments
    /// * `state` - Current drone state
    /// * `waypoints` - At least 1 waypoint; 2+ enables spline smoothing
    /// * `bounds` - World bounds for toroidal distance calculations
    ///
    /// # Returns
    /// Desired heading that follows the path toward the first waypoint.
    pub fn compute_smoothed_heading(
        &self,
        state: &State,
        waypoints: &[Position],
        bounds: &Bounds,
    ) -> Heading {
        if waypoints.is_empty() {
            return state.hdg;
        }

        let drone_pos = state.pos.as_vec2();
        let wp1 = waypoints[0].as_vec2();
        let to_wp1 = bounds.delta(drone_pos, wp1);
        let dist_to_wp1 = to_wp1.magnitude();

        if dist_to_wp1 < f32::EPSILON {
            return state.hdg;
        }

        // Compute speed-adaptive lookahead distance
        let speed = state.vel.speed().max(self.min_speed);
        let lookahead_dist = self.base_lookahead + speed * self.speed_lookahead_factor;

        // If only one waypoint or very close, head straight to it
        if waypoints.len() == 1 || dist_to_wp1 < lookahead_dist * 0.5 {
            return Heading::new(to_wp1.heading());
        }

        // Use Hermite spline for smooth path between waypoints
        let (p0, p1, t0, t1) = self.build_spline_params(state, waypoints, bounds);

        // Convert lookahead distance to spline parameter t
        // Clamp to [0.1, 0.9] to stay within reasonable spline bounds
        let lookahead_t = (lookahead_dist / dist_to_wp1).clamp(0.1, 0.9);

        // Get the lookahead point on the spline
        let lookahead_point = Self::hermite_point(p0, p1, t0, t1, lookahead_t);

        // NLGL: simply steer toward the lookahead point
        // The platform's turn rate limiting handles the rest
        Heading::new(lookahead_point.heading())
    }

    /// Build Hermite spline parameters for the path.
    ///
    /// # Returns
    /// (p0, p1, t0, t1) where the spline runs from drone position toward WP1.
    fn build_spline_params(
        &self,
        state: &State,
        waypoints: &[Position],
        bounds: &Bounds,
    ) -> (Vec2, Vec2, Vec2, Vec2) {
        let drone_pos = state.pos.as_vec2();
        let wp1 = waypoints[0].as_vec2();
        let to_wp1 = bounds.delta(drone_pos, wp1);
        let dist_to_wp1 = to_wp1.magnitude();

        // P0: spline starts at drone's current position (origin in relative coords)
        let p0 = Vec2::new(0.0, 0.0);
        // P1: spline ends at WP1
        let p1 = to_wp1;

        // Tangent scale - use a fraction of distance, capped to reasonable value
        let tangent_scale = (dist_to_wp1 * 0.4).min(120.0);

        // T0: start tangent based on drone's velocity direction (or heading if not moving)
        let speed = state.vel.speed();
        let t0 = if speed > 1.0 {
            state.vel.as_vec2().normalized() * tangent_scale
        } else {
            state.hdg.to_vec2() * tangent_scale
        };

        // T1: end tangent toward WP2 (or continue toward WP1 if no WP2)
        let t1 = if waypoints.len() > 1 {
            let wp2 = waypoints[1].as_vec2();
            let wp1_to_wp2 = bounds.delta(wp1, wp2);
            if wp1_to_wp2.magnitude() > f32::EPSILON {
                wp1_to_wp2.normalized() * tangent_scale
            } else {
                to_wp1.normalized() * tangent_scale
            }
        } else {
            to_wp1.normalized() * tangent_scale
        };

        (p0, p1, t0, t1)
    }

    /// Evaluate a point on a cubic Hermite spline at parameter t (0-1).
    ///
    /// # Arguments
    /// * `p0` - Start position (at t=0)
    /// * `p1` - End position (at t=1)
    /// * `t0` - Start tangent
    /// * `t1` - End tangent
    /// * `t` - Parameter value (0 to 1)
    pub fn hermite_point(p0: Vec2, p1: Vec2, t0: Vec2, t1: Vec2, t: f32) -> Vec2 {
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
    pub fn hermite_tangent(p0: Vec2, p1: Vec2, t0: Vec2, t1: Vec2, t: f32) -> Vec2 {
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

    /// Build approach spline parameters for a target position with arrival heading.
    ///
    /// # Returns
    /// `(p0, p1, t0, t1)` in drone-relative coordinates, or `None` if too close.
    fn build_approach_spline(
        state: &State,
        to_target: Vec2,
        target_heading: f32,
    ) -> (Vec2, Vec2, Vec2, Vec2) {
        let dist = to_target.magnitude();

        let p0 = Vec2::new(0.0, 0.0);
        let p1 = to_target;

        // 25% of distance for aggressive approach, capped
        let tangent_scale = (dist * 0.25).min(50.0);

        let speed = state.vel.speed();
        let t0 = if speed > 1.0 {
            state.vel.as_vec2().normalized() * tangent_scale
        } else {
            state.hdg.to_vec2() * tangent_scale
        };

        let t1 = Vec2::new(
            target_heading.cos() * tangent_scale,
            target_heading.sin() * tangent_scale,
        );

        (p0, p1, t0, t1)
    }

    /// Compute desired heading using a Hermite spline approach to a target position.
    ///
    /// This creates a smooth curved path from the drone's current position/heading
    /// to a target position with a specified arrival heading. Useful for formation
    /// followers to smoothly approach their slots.
    pub fn compute_approach_heading(
        &self,
        state: &State,
        target_pos: Vec2,
        target_heading: f32,
        bounds: &Bounds,
    ) -> Heading {
        let drone_pos = state.pos.as_vec2();
        let to_target = bounds.delta(drone_pos, target_pos);
        let dist = to_target.magnitude();

        if dist < 5.0 {
            return Heading::new(to_target.heading());
        }

        let (p0, p1, t0, t1) = Self::build_approach_spline(state, to_target, target_heading);

        let speed = state.vel.speed();
        let effective_speed = speed.max(self.min_speed);
        let lookahead_dist = self.base_lookahead * 0.5 + effective_speed * self.speed_lookahead_factor * 0.5;
        let lookahead_t = (lookahead_dist / dist).clamp(0.2, 0.9);

        let lookahead_point = Self::hermite_point(p0, p1, t0, t1, lookahead_t);
        Heading::new(lookahead_point.heading())
    }

    /// Generate points along the approach spline for visualization.
    ///
    /// Creates a smooth curved path from the drone's current position/heading
    /// to a target position with a specified arrival heading.
    pub fn get_approach_spline_path(
        &self,
        state: &State,
        target_pos: Vec2,
        target_heading: f32,
        bounds: &Bounds,
        num_points: usize,
    ) -> Vec<Vec2> {
        if num_points < 2 {
            return Vec::new();
        }

        let drone_pos = state.pos.as_vec2();
        let to_target = bounds.delta(drone_pos, target_pos);

        if to_target.magnitude() < 1.0 {
            return Vec::new();
        }

        let (p0, p1, t0, t1) = Self::build_approach_spline(state, to_target, target_heading);

        let mut points = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let t = i as f32 / (num_points - 1) as f32;
            let relative_point = Self::hermite_point(p0, p1, t0, t1, t);
            points.push(Vec2::new(
                drone_pos.x + relative_point.x,
                drone_pos.y + relative_point.y,
            ));
        }

        points
    }

    /// Compute the approach gate point for rear-entry formation joining.
    ///
    /// The gate is positioned behind the slot (opposite to formation heading),
    /// forcing drones to approach from the rear and curve into position.
    ///
    /// # Arguments
    /// * `slot_pos` - Target slot position
    /// * `formation_heading` - Formation heading (radians)
    /// * `gate_offset` - Distance behind slot for gate
    ///
    /// # Returns
    /// Gate position (behind the slot)
    pub fn compute_approach_gate(
        slot_pos: Vec2,
        formation_heading: f32,
        gate_offset: f32,
    ) -> Vec2 {
        // Gate is behind the slot (opposite to formation heading)
        Vec2::new(
            slot_pos.x - formation_heading.cos() * gate_offset,
            slot_pos.y - formation_heading.sin() * gate_offset,
        )
    }

    /// Check if a drone needs to route through the approach gate.
    ///
    /// A drone needs the gate if:
    /// 1. It's outside the approach ball, AND
    /// 2. It's not already approaching from the rear
    ///
    /// # Returns
    /// true if drone should route through gate first
    pub fn needs_approach_gate(
        drone_pos: Vec2,
        slot_pos: Vec2,
        formation_heading: f32,
        gate_config: &ApproachGateConfig,
        bounds: &Bounds,
    ) -> bool {
        let to_slot = bounds.delta(drone_pos, slot_pos);
        let dist = to_slot.magnitude();

        // Inside the ball - no gate needed
        if dist < gate_config.ball_radius {
            return false;
        }

        // Check approach angle - is drone coming from the rear?
        let approach_heading = to_slot.heading();
        let angle_diff = crate::types::normalize_angle(approach_heading - formation_heading);

        // If approaching roughly from behind (within tolerance of formation heading),
        // no gate needed
        angle_diff.abs() > gate_config.angle_tolerance
    }

    /// Compute desired heading for gated approach (rear-entry).
    ///
    /// If drone needs to route through gate, returns heading toward gate.
    /// Otherwise, returns heading toward slot using standard spline.
    pub fn compute_gated_approach_heading(
        &self,
        state: &State,
        slot_pos: Vec2,
        formation_heading: f32,
        gate_config: &ApproachGateConfig,
        bounds: &Bounds,
    ) -> Heading {
        let drone_pos = state.pos.as_vec2();

        if Self::needs_approach_gate(
            drone_pos,
            slot_pos,
            formation_heading,
            gate_config,
            bounds,
        ) {
            // Route through gate first
            let gate_pos = Self::compute_approach_gate(slot_pos, formation_heading, gate_config.gate_offset);

            // Spline to gate, with arrival heading toward slot
            self.compute_approach_heading(state, gate_pos, formation_heading, bounds)
        } else {
            // Direct approach to slot
            self.compute_approach_heading(state, slot_pos, formation_heading, bounds)
        }
    }

    /// Generate points along the gated approach path for visualization.
    ///
    /// If drone needs to route through gate, generates a 2-segment path:
    /// drone → gate → slot. Otherwise, generates direct spline to slot.
    pub fn get_gated_approach_path(
        &self,
        state: &State,
        slot_pos: Vec2,
        formation_heading: f32,
        gate_config: &ApproachGateConfig,
        bounds: &Bounds,
        num_points: usize,
    ) -> Vec<Vec2> {
        if num_points < 2 {
            return Vec::new();
        }

        let drone_pos = state.pos.as_vec2();

        if Self::needs_approach_gate(
            drone_pos,
            slot_pos,
            formation_heading,
            gate_config,
            bounds,
        ) {
            // Two-segment path: drone → gate → slot
            let gate_pos = Self::compute_approach_gate(slot_pos, formation_heading, gate_config.gate_offset);

            let to_gate = bounds.delta(drone_pos, gate_pos);
            let gate_dist = to_gate.magnitude();
            let to_slot = bounds.delta(gate_pos, slot_pos);
            let slot_dist = to_slot.magnitude();
            let total_dist = gate_dist + slot_dist;

            if total_dist < 1.0 {
                return Vec::new();
            }

            // Proportion of path to gate vs to slot
            let gate_ratio = gate_dist / total_dist;
            let gate_points = ((num_points as f32) * gate_ratio).max(2.0) as usize;
            let slot_points = num_points.saturating_sub(gate_points).max(2);

            let mut points = Vec::with_capacity(num_points);

            // First segment: drone → gate (with arrival heading toward slot = formation_heading)
            let seg1 = self.get_approach_spline_path(
                state,
                gate_pos,
                formation_heading,
                bounds,
                gate_points,
            );
            points.extend(seg1);

            // Create a synthetic state at the gate for the second segment
            let gate_state = State {
                pos: crate::types::Position::new(gate_pos.x, gate_pos.y),
                hdg: Heading::new(formation_heading),
                vel: crate::types::Velocity::from_heading_and_speed(
                    Heading::new(formation_heading),
                    state.vel.speed(),
                ),
                acc: crate::types::Acceleration::zero(),
            };

            // Second segment: gate → slot
            let seg2 = self.get_approach_spline_path(
                &gate_state,
                slot_pos,
                formation_heading,
                bounds,
                slot_points,
            );

            // Skip first point of seg2 to avoid duplicate at gate
            if seg2.len() > 1 {
                points.extend(seg2.into_iter().skip(1));
            }

            points
        } else {
            // Direct approach to slot
            self.get_approach_spline_path(state, slot_pos, formation_heading, bounds, num_points)
        }
    }

    /// Generate a full-route Hermite spline for visualization.
    ///
    /// Computes Hermite spline segments between every consecutive pair of
    /// waypoints in a closed-loop route, using route-direction tangents.
    /// This matches the smoothing that `compute_smoothed_heading` uses at
    /// runtime, giving a faithful preview of the path each drone will fly.
    ///
    /// # Arguments
    /// * `route` - Closed-loop route waypoints (last connects back to first)
    /// * `bounds` - World bounds for toroidal distance calculations
    /// * `points_per_segment` - Samples per waypoint pair (e.g. 10)
    ///
    /// # Returns
    /// Densely sampled world-coordinate points along the full route loop.
    pub fn get_full_route_spline(
        route: &[Position],
        bounds: &Bounds,
        points_per_segment: usize,
    ) -> Vec<Vec2> {
        Self::sample_route_spline(route, bounds, points_per_segment, None)
    }

    /// Compute a follower's offset route by offsetting the leader's smoothed
    /// Hermite spline curve.
    ///
    /// Instead of offsetting raw waypoints and smoothing independently, this:
    /// 1. Computes the leader's Hermite spline through all waypoints
    /// 2. At each sample point, computes the tangent heading
    /// 3. Offsets perpendicular to the tangent by the slot offset
    ///
    /// This produces a smooth follower curve that faithfully mirrors the
    /// leader's actual smoothed path, maintaining consistent formation shape
    /// through curves and corners.
    ///
    /// # Arguments
    /// * `route` - Leader's closed-loop route waypoints
    /// * `offset` - Formation slot offset (x=forward, y=left in body frame)
    /// * `bounds` - World bounds for toroidal distance calculations
    /// * `points_per_segment` - Samples per waypoint pair (e.g. 10)
    ///
    /// # Returns
    /// Densely sampled offset route as `Vec<Position>` suitable for `FollowRoute`.
    pub fn compute_offset_spline_route(
        route: &[Position],
        offset: Vec2,
        bounds: &Bounds,
        points_per_segment: usize,
    ) -> Vec<Position> {
        Self::sample_route_spline(route, bounds, points_per_segment, Some(offset))
            .into_iter()
            .map(|v| Position::new(v.x, v.y))
            .collect()
    }

    /// Core spline sampling: generates points along the Hermite spline of a
    /// closed-loop route. Optionally applies a formation offset at each point
    /// using the local tangent heading.
    fn sample_route_spline(
        route: &[Position],
        bounds: &Bounds,
        points_per_segment: usize,
        offset: Option<Vec2>,
    ) -> Vec<Vec2> {
        let n = route.len();
        if n < 2 || points_per_segment < 2 {
            return Vec::new();
        }

        let mut points = Vec::with_capacity(n * points_per_segment);

        for i in 0..n {
            let i_next = (i + 1) % n;
            let i_prev = if i == 0 { n - 1 } else { i - 1 };
            let i_next2 = (i + 2) % n;

            let p0 = route[i].as_vec2();
            let p1 = route[i_next].as_vec2();

            let seg = bounds.delta(p0, p1);
            let seg_len = seg.magnitude();
            if seg_len < f32::EPSILON {
                continue;
            }

            // Tangent scale matches PathPlanner: 40% of segment length, capped
            let tangent_scale = (seg_len * 0.4).min(120.0);

            // T0: bisector of incoming and outgoing directions at waypoint[i]
            let incoming = bounds.delta(route[i_prev].as_vec2(), p0);
            let outgoing = seg;
            let t0 = Self::bisector_tangent(incoming, outgoing, tangent_scale);

            // T1: bisector of incoming and outgoing directions at waypoint[i+1]
            let next_outgoing = bounds.delta(p1, route[i_next2].as_vec2());
            let next_incoming = seg;
            let t1 = Self::bisector_tangent(next_incoming, next_outgoing, tangent_scale);

            // Sample this segment (skip last point to avoid duplicates at joins)
            let samples = if i == n - 1 { points_per_segment } else { points_per_segment - 1 };
            let p0_rel = Vec2::ZERO;
            let p1_rel = seg;

            for s in 0..samples {
                let t = s as f32 / (points_per_segment - 1) as f32;
                let rel = Self::hermite_point(p0_rel, p1_rel, t0, t1, t);
                let mut world = Vec2::new(p0.x + rel.x, p0.y + rel.y);

                // Apply formation offset using local tangent heading
                if let Some(off) = offset {
                    let tangent = Self::hermite_tangent(p0_rel, p1_rel, t0, t1, t);
                    let heading = tangent.y.atan2(tangent.x);
                    let (sin_h, cos_h) = heading.sin_cos();
                    world.x += off.x * cos_h - off.y * sin_h;
                    world.y += off.x * sin_h + off.y * cos_h;
                }

                points.push(world);
            }
        }

        points
    }

    /// Compute a bisector tangent vector from incoming and outgoing directions.
    fn bisector_tangent(incoming: Vec2, outgoing: Vec2, scale: f32) -> Vec2 {
        if incoming.magnitude() > f32::EPSILON {
            let in_dir = incoming.normalized();
            let out_dir = outgoing.normalized();
            let bisector = Vec2::new(in_dir.x + out_dir.x, in_dir.y + out_dir.y);
            if bisector.magnitude() > f32::EPSILON {
                bisector.normalized() * scale
            } else {
                out_dir * scale
            }
        } else {
            outgoing.normalized() * scale
        }
    }

    /// Generate points along the spline path for visualization.
    ///
    /// # Arguments
    /// * `state` - Current drone state
    /// * `waypoints` - At least 2 waypoints for spline construction
    /// * `bounds` - World bounds for toroidal distance calculations
    /// * `num_points` - Number of points to generate
    ///
    /// # Returns
    /// World coordinates for each point along the spline.
    pub fn get_spline_path(
        &self,
        state: &State,
        waypoints: &[Position],
        bounds: &Bounds,
        num_points: usize,
    ) -> Vec<Vec2> {
        if waypoints.len() < 2 || num_points < 2 {
            return Vec::new();
        }

        let drone_pos = state.pos.as_vec2();
        let wp1 = waypoints[0].as_vec2();
        let to_wp1 = bounds.delta(drone_pos, wp1);
        let dist_to_wp1 = to_wp1.magnitude();

        if dist_to_wp1 < f32::EPSILON {
            return Vec::new();
        }

        let (p0, p1, t0, t1) = self.build_spline_params(state, waypoints, bounds);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Acceleration, Velocity};

    fn create_test_bounds() -> Bounds {
        Bounds::new(1000.0, 1000.0).unwrap()
    }

    fn create_test_state(x: f32, y: f32, hdg: f32, speed: f32) -> State {
        State {
            pos: Position::new(x, y),
            hdg: Heading::new(hdg),
            vel: Velocity::from_heading_and_speed(Heading::new(hdg), speed),
            acc: Acceleration::zero(),
        }
    }

    #[test]
    fn test_hermite_point_endpoints() {
        let p0 = Vec2::new(0.0, 0.0);
        let p1 = Vec2::new(100.0, 0.0);
        let t0 = Vec2::new(50.0, 0.0);
        let t1 = Vec2::new(50.0, 0.0);

        // At t=0, should be at p0
        let start = PathPlanner::hermite_point(p0, p1, t0, t1, 0.0);
        assert!((start.x - p0.x).abs() < 0.001);
        assert!((start.y - p0.y).abs() < 0.001);

        // At t=1, should be at p1
        let end = PathPlanner::hermite_point(p0, p1, t0, t1, 1.0);
        assert!((end.x - p1.x).abs() < 0.001);
        assert!((end.y - p1.y).abs() < 0.001);
    }

    #[test]
    fn test_hermite_point_midpoint() {
        let p0 = Vec2::new(0.0, 0.0);
        let p1 = Vec2::new(100.0, 0.0);
        let t0 = Vec2::new(50.0, 0.0);
        let t1 = Vec2::new(50.0, 0.0);

        // With straight tangents, midpoint should be roughly in the middle
        let mid = PathPlanner::hermite_point(p0, p1, t0, t1, 0.5);
        assert!((mid.x - 50.0).abs() < 5.0);
    }

    #[test]
    fn test_compute_smoothed_heading_returns_current_when_empty() {
        let planner = PathPlanner::new();
        let state = create_test_state(500.0, 500.0, 0.5, 60.0);
        let bounds = create_test_bounds();

        let heading = planner.compute_smoothed_heading(&state, &[], &bounds);
        assert!((heading.radians() - state.hdg.radians()).abs() < 0.001);
    }

    #[test]
    fn test_compute_smoothed_heading_single_waypoint() {
        let planner = PathPlanner::new();
        let state = create_test_state(500.0, 500.0, 0.0, 60.0);
        let bounds = create_test_bounds();

        let waypoints = vec![Position::new(600.0, 500.0)]; // Directly ahead

        let heading = planner.compute_smoothed_heading(&state, &waypoints, &bounds);
        // Should head roughly toward the waypoint (heading 0)
        assert!(heading.radians().abs() < 0.3);
    }

    #[test]
    fn test_nlgl_converges_to_path() {
        let planner = PathPlanner::new();
        let bounds = create_test_bounds();

        // Drone is off to the side of the path
        let state = create_test_state(500.0, 550.0, 0.0, 60.0); // 50 units above
        let waypoints = vec![
            Position::new(700.0, 500.0),
            Position::new(800.0, 500.0),
        ];

        let heading = planner.compute_smoothed_heading(&state, &waypoints, &bounds);

        // NLGL should steer slightly down (negative y direction) to converge
        // Heading should be between 0 and -π/4 (toward waypoint but also down)
        assert!(heading.radians() < 0.1); // Not too far up
        assert!(heading.radians() > -0.8); // Not too sharply down
    }

    #[test]
    fn test_get_spline_path_returns_correct_count() {
        let planner = PathPlanner::new();
        let state = create_test_state(500.0, 500.0, 0.0, 60.0);
        let bounds = create_test_bounds();

        let waypoints = vec![
            Position::new(600.0, 500.0),
            Position::new(700.0, 600.0),
        ];

        let path = planner.get_spline_path(&state, &waypoints, &bounds, 20);
        assert_eq!(path.len(), 20);
    }

    #[test]
    fn test_get_spline_path_starts_at_drone() {
        let planner = PathPlanner::new();
        let state = create_test_state(500.0, 500.0, 0.0, 60.0);
        let bounds = create_test_bounds();

        let waypoints = vec![
            Position::new(600.0, 500.0),
            Position::new(700.0, 600.0),
        ];

        let path = planner.get_spline_path(&state, &waypoints, &bounds, 20);
        assert!((path[0].x - 500.0).abs() < 1.0);
        assert!((path[0].y - 500.0).abs() < 1.0);
    }
}
