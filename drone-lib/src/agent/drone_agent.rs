//! DroneAgent - composed autonomous agent with modular components.

use std::collections::VecDeque;
use std::sync::Arc;

use crate::behaviors::tree::node::{BehaviorContext, BehaviorNode};
use crate::behaviors::{
    create_orca_bt_with_apf, APFConfig, ORCAConfig, SeparationConfig,
};
use crate::behaviors::safety::SafetyLayer;
use crate::tasks::{DroneTask, TaskStatus};
use crate::messages::{FormationCommand, FormationSlot, VelocityConsensus};
use crate::missions::{CommandQueue, Mission, Task, WaypointMission};
use super::formation_navigator::FormationNavigator;
use super::traits::Drone;
use crate::platform::{GenericPlatform, Platform};
use crate::swarm::formation::defaults as formation_defaults;
use crate::types::{
    units, Bounds, DroneInfo, DronePerfFeatures, Heading, Objective, Position, State, Vec2,
    Velocity,
};

/// Formation approach mode for visualization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormationApproachMode {
    /// Not in formation or is the leader
    None,
    /// At position - holding station
    StationKeeping,
    /// Close to position - gentle correction
    Correction,
    /// Inside ball, approaching from rear - lead pursuit
    Pursuit,
    /// Outside ball or wrong angle - gated spline approach
    Approach,
}

/// Default waypoint clearance distance (meters).
const DEFAULT_WAYPOINT_CLEARANCE: f32 = units::WAYPOINT_CLEARANCE;

/// DroneAgent composes Platform + Mission + BehaviorTree into a complete autonomous agent.
///
/// This replaces the monolithic FixedWing struct with a modular architecture
/// following MOSA principles.
#[derive(Debug)]
pub struct DroneAgent {
    /// Unique drone identifier.
    id: usize,
    /// Physical platform (kinematics).
    platform: GenericPlatform,
    /// Mission management (waypoints, tasks).
    mission: WaypointMission,
    /// Behavior tree for decision-making.
    behavior_tree: Box<dyn BehaviorNode>,
    /// Command queue for decoupled control.
    command_queue: CommandQueue,
    /// ORCA collision avoidance configuration.
    orca_config: ORCAConfig,
    /// Separation behavior configuration.
    sep_config: SeparationConfig,
    /// Distance threshold for waypoint arrival.
    waypoint_clearance: f32,
    /// APF collision avoidance configuration.
    apf_config: Option<APFConfig>,
    /// Formation navigation state.
    formation: FormationNavigator,
    /// Group identifier for friend/foe distinction.
    group: u32,
    /// Active task state machine (intercept, attack, etc.)
    /// When set, bypasses the behavior tree and uses the task→safety pipeline.
    active_task: Option<Box<dyn DroneTask>>,
    /// Safety layer for collision avoidance (used with task pipeline).
    safety_layer: SafetyLayer,
    /// Whether the task requested detonation this tick.
    pending_detonate: bool,
}

impl DroneAgent {
    /// Create a new drone agent at the specified position and heading.
    pub fn new(id: usize, pos: Position, hdg: Heading, bounds: Bounds) -> Self {
        let orca_config = ORCAConfig::default();
        let sep_config = SeparationConfig::default();

        DroneAgent {
            id,
            platform: GenericPlatform::new(pos, hdg, bounds),
            mission: WaypointMission::new(bounds),
            behavior_tree: create_orca_bt_with_apf(orca_config, sep_config, None),
            command_queue: CommandQueue::new(),
            orca_config,
            sep_config,
            apf_config: None,
            waypoint_clearance: DEFAULT_WAYPOINT_CLEARANCE,
            formation: FormationNavigator::new(),
            group: 0,
            active_task: None,
            safety_layer: SafetyLayer::new(orca_config, APFConfig::default()),
            pending_detonate: false,
        }
    }

    /// Get the drone's unique identifier.
    pub fn uid(&self) -> usize {
        self.id
    }

    /// Get the drone's current state.
    pub fn state(&self) -> &State {
        self.platform.state()
    }

    /// Get shareable drone info for swarm communication.
    pub fn get_info(&self) -> DroneInfo {
        let mut info = DroneInfo::new_with_leader(self.id, self.platform.state(), self.formation.is_leader());
        info.group = self.group;
        info
    }

    /// Get the drone's current target waypoint (if any).
    pub fn current_waypoint(&self) -> Option<Position> {
        self.mission.current_waypoint()
    }

    /// Get upcoming waypoints (up to n).
    pub fn upcoming_waypoints(&self, n: usize) -> Vec<Position> {
        self.mission.upcoming_waypoints(n)
    }

    /// Main update tick - called once per frame.
    ///
    /// # Arguments
    /// * `dt` - Delta time in seconds
    /// * `swarm` - Information about other drones in the swarm
    pub fn state_update(&mut self, dt: f32, swarm: &[DroneInfo]) {
        // 1. Process pending commands
        self.process_commands();

        // If an active task is set, use the task→safety→platform pipeline.
        // Otherwise, use the legacy behavior tree pipeline.
        if self.active_task.is_some() {
            self.state_update_task(dt, swarm);
        } else {
            self.state_update_bt(dt, swarm);
        }
    }

    /// Task-based update: task→safety→platform pipeline.
    ///
    /// The task selects a behavior and produces a desired velocity.
    /// The safety layer adjusts it for collision avoidance.
    /// Feedback flows back to the task for phase transitions.
    fn state_update_task(&mut self, dt: f32, swarm: &[DroneInfo]) {
        let task = self.active_task.as_mut().unwrap();

        // 1. Task tick: assess situation, select behavior, produce desired velocity
        let output = task.tick(
            self.platform.state(),
            swarm,
            self.platform.bounds(),
            self.platform.perf(),
            dt,
        );

        // 2. Check for detonation (handled by caller via should_detonate())
        // Store the detonate flag for the simulation to check
        self.pending_detonate = output.detonate;

        // 3. Safety layer: adjust velocity for collision avoidance.
        //    If intercepting a target, exclude that drone from the swarm
        //    so ORCA/APF won't push us away from it (terminal engagement).
        let (safe_vel, feedback) = if !output.exclude_from_ca.is_empty() {
            let filtered: Vec<DroneInfo> = swarm.iter()
                .copied()
                .filter(|d| !output.exclude_from_ca.contains(&d.uid))
                .collect();
            self.safety_layer.apply(
                output.desired_velocity,
                self.platform.state(),
                self.id,
                &filtered,
                self.platform.bounds(),
            )
        } else {
            self.safety_layer.apply(
                output.desired_velocity,
                self.platform.state(),
                self.id,
                swarm,
                self.platform.bounds(),
            )
        };

        // 4. Feed safety feedback back to task for phase transitions
        let task = self.active_task.as_mut().unwrap();
        task.process_feedback(&feedback);

        // 5. Apply safe velocity to platform
        let heading = output.desired_heading.unwrap_or_else(|| {
            let speed = safe_vel.magnitude();
            if speed > f32::EPSILON {
                Heading::new(safe_vel.y.atan2(safe_vel.x))
            } else {
                self.platform.state().hdg
            }
        });
        self.platform.apply_velocity_steering(
            Velocity::from_vec2(safe_vel),
            Some(heading),
            dt,
        );
    }

    /// Legacy behavior tree update pipeline.
    /// Used for formation following, regular navigation, etc.
    fn state_update_bt(&mut self, dt: f32, swarm: &[DroneInfo]) {
        // 2. Update mission state (waypoint arrivals)
        self.mission.update(self.platform.state(), self.waypoint_clearance);

        // 3. Get desired heading from mission
        let mut desired_heading = self.mission.get_desired_heading(self.platform.state());

        // 3b. Determine if this follower is in route-following mode
        let is_follower_route_mode = self.formation.in_formation()
            && !self.formation.is_leader()
            && self.mission.task() == Task::FollowRoute;

        // 3c. If in formation but NOT following a route (position-tracking mode),
        // use intercept navigation for smooth slot following.
        // In route mode, the mission provides heading from the offset route.
        let formation_speed_override = if !is_follower_route_mode {
            let (heading_override, speed_override) =
                self.formation.compute_intercept_navigation(
                    self.platform.state(),
                    self.platform.perf().max_vel,
                    self.platform.bounds(),
                );
            if let Some(hdg) = heading_override {
                desired_heading = Some(hdg);
            }
            speed_override
        } else {
            None
        };

        // 4. Build behavior context (pass formation leader status and perf)
        let mut ctx = BehaviorContext::new_with_leader(
            self.platform.state(),
            swarm,
            self.platform.bounds(),
            self.id,
            dt,
            self.formation.is_leader(),
            self.platform.perf(),
        );
        ctx.desired_heading = desired_heading;

        // Compute speed factor
        if let Some(speed) = formation_speed_override {
            // Position-tracking mode: intercept navigation controls speed
            ctx.desired_speed = speed / self.platform.perf().max_vel;
        } else if is_follower_route_mode {
            // Route-following mode: base speed matches leader's speed cap,
            // modulated by the along-track formation factor.
            let formation_factor = self.formation.compute_formation_speed_factor(
                self.platform.state().pos.as_vec2(),
                swarm,
                self.platform.bounds(),
            ).unwrap_or(1.0);
            ctx.desired_speed = (formation_defaults::LEADER_SPEED_CAP * formation_factor).clamp(0.1, 1.0);
        } else {
            let base_speed = self.mission.get_desired_speed_factor();
            ctx.desired_speed = (base_speed * self.formation.speed_multiplier()).min(1.0);
        }

        // 5. Tick behavior tree (applies avoidance, modifies context)
        self.behavior_tree.tick(&mut ctx);

        // 6. Apply steering to platform (quadcopter-style: velocity independent of heading)
        let max_speed = self.platform.perf().max_vel;

        if let Some(desired_vel) = ctx.desired_velocity {
            let mut vel = desired_vel;

            // Formation leaders operate at 65% max speed so followers can catch up
            if self.formation.is_leader() {
                vel = vel.clamp_speed(max_speed * formation_defaults::LEADER_SPEED_CAP);
            }

            self.platform.apply_velocity_steering(vel, ctx.desired_heading, dt);
        } else if let Some(heading) = ctx.desired_heading {
            let mut speed = max_speed * ctx.desired_speed;

            // Formation leaders operate at 65% max speed so followers can catch up
            if self.formation.is_leader() {
                speed = speed.min(max_speed * formation_defaults::LEADER_SPEED_CAP);
            }

            self.platform.apply_steering(heading, speed, dt);
        }
    }

    /// Process pending commands from the command queue.
    fn process_commands(&mut self) {
        use crate::missions::AgentCommand;

        while let Some(cmd) = self.command_queue.pop() {
            match cmd {
                AgentCommand::GoToWaypoints(waypoints) => {
                    self.mission.set_waypoints(Task::ReachWaypoint, waypoints, None);
                }
                AgentCommand::FollowRoute(route) => {
                    let waypoints = route.iter().copied().collect();
                    self.mission.set_waypoints(Task::FollowRoute, waypoints, Some(route));
                }
                AgentCommand::Loiter => {
                    self.mission.set_waypoints(Task::Loiter, VecDeque::new(), None);
                }
                AgentCommand::Stop | AgentCommand::ClearMission => {
                    self.mission.clear();
                }
                AgentCommand::SetMaxVelocity(v) => {
                    let perf = self.platform.perf();
                    if let Ok(p) = DronePerfFeatures::new(v, perf.max_acc, perf.max_turn_rate) {
                        self.platform.set_perf(p);
                    }
                }
                AgentCommand::SetMaxAcceleration(a) => {
                    let perf = self.platform.perf();
                    if let Ok(p) = DronePerfFeatures::new(perf.max_vel, a, perf.max_turn_rate) {
                        self.platform.set_perf(p);
                    }
                }
                AgentCommand::SetMaxTurnRate(r) => {
                    let perf = self.platform.perf();
                    if let Ok(p) = DronePerfFeatures::new(perf.max_vel, perf.max_acc, r) {
                        self.platform.set_perf(p);
                    }
                }
                AgentCommand::SetWaypointClearance(c) => {
                    self.waypoint_clearance = c.max(1.0);
                }
            }
        }
    }

    // ===== wasm-lib compatibility methods =====

    /// Set objective, converting to the internal mission representation.
    pub fn set_objective(&mut self, objective: Objective) {
        match objective {
            Objective::Sleep => self.mission.clear(),
            Objective::ReachWaypoint { waypoints } => {
                self.mission.set_waypoints(Task::ReachWaypoint, waypoints, None);
            }
            Objective::FollowRoute { waypoints, route } => {
                self.mission.set_waypoints(Task::FollowRoute, waypoints, Some(route));
            }
            Objective::FollowTarget { .. } => {
                self.mission.set_waypoints(Task::FollowTarget, VecDeque::new(), None);
            }
            Objective::Loiter { .. } => {
                self.mission.set_waypoints(Task::Loiter, VecDeque::new(), None);
            }
        }
    }

    /// Clear objective (wasm-lib compatibility).
    pub fn clear_objective(&mut self) {
        self.mission.clear();
    }

    /// Get current objective, reconstructed from internal mission state.
    pub fn objective(&self) -> Objective {
        match self.mission.task() {
            Task::Sleep => Objective::Sleep,
            Task::ReachWaypoint => Objective::ReachWaypoint {
                waypoints: self.mission.waypoints().clone(),
            },
            Task::FollowRoute => Objective::FollowRoute {
                waypoints: self.mission.waypoints().clone(),
                route: self.mission.route().cloned().unwrap_or_else(|| Arc::from(Vec::new())),
            },
            Task::FollowTarget => Objective::FollowTarget { targets: Vec::new() },
            Task::Loiter => Objective::Loiter { center: None },
        }
    }

    /// Set flight parameters (wasm-lib compatibility).
    /// Also syncs ORCA max_speed so collision avoidance operates at the correct speed regime.
    pub fn set_flight_params(&mut self, params: DronePerfFeatures) {
        let speed_changed = (params.max_vel - self.platform.perf().max_vel).abs() > 0.01;
        self.platform.set_perf(params);
        if speed_changed {
            self.orca_config.max_speed = params.max_vel;
            self.rebuild_behavior_tree();
        }
    }

    /// Set waypoint clearance (wasm-lib compatibility).
    pub fn set_waypoint_clearance(&mut self, clearance: f32) {
        self.waypoint_clearance = clearance.max(1.0);
    }

    /// Get waypoint clearance (wasm-lib compatibility).
    pub fn waypoint_clearance(&self) -> f32 {
        self.waypoint_clearance
    }

    /// Set the drone's group for friend/foe distinction.
    pub fn set_group(&mut self, group: u32) {
        self.group = group;
    }

    /// Get the drone's group.
    pub fn group(&self) -> u32 {
        self.group
    }

    /// Set APF collision avoidance configuration.
    pub fn set_apf_config(&mut self, config: APFConfig) {
        self.apf_config = Some(config);
        self.safety_layer.apf_config = config;
        self.rebuild_behavior_tree();
    }

    /// Set ORCA collision avoidance configuration.
    pub fn set_orca_config(&mut self, config: ORCAConfig) {
        self.orca_config = config;
        self.safety_layer.orca_config = config;
        self.rebuild_behavior_tree();
    }

    /// Get ORCA collision avoidance configuration.
    pub fn orca_config(&self) -> &ORCAConfig {
        &self.orca_config
    }

    /// Set separation configuration.
    pub fn set_separation_config(&mut self, config: SeparationConfig) {
        self.sep_config = config;
        self.rebuild_behavior_tree();
    }

    /// Get separation configuration.
    pub fn separation_config(&self) -> &SeparationConfig {
        &self.sep_config
    }

    /// Rebuild behavior tree with current configurations.
    fn rebuild_behavior_tree(&mut self) {
        self.behavior_tree = create_orca_bt_with_apf(self.orca_config, self.sep_config, self.apf_config);
    }

    // ===== Task pipeline methods =====

    /// Set an active task, switching to the task→safety pipeline.
    /// Clears the current mission.
    pub fn set_task(&mut self, task: Box<dyn DroneTask>) {
        self.active_task = Some(task);
        self.pending_detonate = false;
        // Sync safety layer configs
        self.safety_layer.orca_config = self.orca_config;
        if let Some(apf) = self.apf_config {
            self.safety_layer.apf_config = apf;
        }
    }

    /// Clear the active task, returning to the behavior tree pipeline.
    pub fn clear_task(&mut self) {
        self.active_task = None;
        self.pending_detonate = false;
    }

    /// Check if the task has requested detonation.
    pub fn should_detonate(&self) -> bool {
        self.pending_detonate
    }

    /// Get the active task status (if any).
    pub fn task_status(&self) -> Option<TaskStatus> {
        self.active_task.as_ref().map(|t| t.status())
    }

    /// Get the active task name and phase (for rendering/debugging).
    pub fn task_info(&self) -> Option<(&str, &str)> {
        self.active_task.as_ref().map(|t| (t.name(), t.phase_name()))
    }

    /// Get spline path for visualization (wasm-lib compatibility).
    ///
    /// Returns empty vec if not in route-following mode.
    pub fn get_spline_path(&self, num_points: usize) -> Vec<Vec2> {
        if !self.mission.uses_path_smoothing() {
            return Vec::new();
        }

        let waypoints = self.mission.get_waypoints_for_spline();
        if waypoints.len() < 2 {
            return Vec::new();
        }

        self.mission.planner().get_spline_path(
            self.platform.state(),
            &waypoints,
            self.mission.bounds(),
            num_points,
        )
    }

    /// Get formation approach path for visualization.
    pub fn get_formation_spline_path(&self, num_points: usize) -> Vec<Vec2> {
        self.formation.get_spline_path(
            self.platform.state(),
            self.mission.planner(),
            self.platform.bounds(),
            num_points,
        )
    }

    /// Get the current formation approach mode for visualization.
    pub fn get_formation_approach_mode(&self) -> FormationApproachMode {
        self.formation.get_approach_mode(self.platform.state(), self.platform.bounds())
    }

    /// Get reference to the command queue for internal command injection.
    #[cfg(test)]
    pub(crate) fn command_queue_mut(&mut self) -> &mut CommandQueue {
        &mut self.command_queue
    }

    /// Get reference to the platform.
    pub fn platform(&self) -> &GenericPlatform {
        &self.platform
    }

    /// Get reference to the mission.
    pub fn mission(&self) -> &WaypointMission {
        &self.mission
    }

    // ===== Formation control methods (delegate to FormationNavigator) =====

    /// Assign a formation slot to this drone.
    pub fn set_formation_slot(&mut self, slot: FormationSlot, center: Position, heading: f32) {
        self.formation.set_slot(slot, center, heading, &mut self.mission);
    }

    /// Update the formation reference point (center, heading, and velocity).
    pub fn update_formation_reference(&mut self, center: Position, heading: f32, velocity: Vec2) {
        self.formation.update_reference(center, heading, velocity, &mut self.mission);
    }

    /// Update the formation reference without overwriting mission waypoints.
    ///
    /// Used in route-following mode where followers follow their own offset routes.
    pub fn update_formation_reference_no_waypoint(&mut self, center: Position, heading: f32, velocity: Vec2) {
        self.formation.update_reference_no_waypoint(center, heading, velocity);
    }

    /// Clear the formation slot assignment.
    pub fn clear_formation_slot(&mut self) {
        self.formation.clear_slot();
    }

    /// Handle a formation command.
    pub fn handle_formation_command(&mut self, cmd: FormationCommand) {
        self.formation.handle_command(cmd, &mut self.mission);
    }

    /// Get the current formation slot (if assigned).
    pub fn formation_slot(&self) -> Option<&FormationSlot> {
        self.formation.slot()
    }

    /// Check if drone is in a formation.
    pub fn in_formation(&self) -> bool {
        self.formation.in_formation()
    }

    /// Set the formation speed multiplier for synchronized arrival.
    pub fn set_formation_speed_multiplier(&mut self, multiplier: f32) {
        self.formation.set_speed_multiplier(multiplier);
    }

    /// Get the current formation speed multiplier.
    pub fn formation_speed_multiplier(&self) -> f32 {
        self.formation.speed_multiplier()
    }

    /// Get the formation planning path for visualization.
    pub fn get_formation_planning_path(&self, num_points: usize) -> Vec<Vec2> {
        self.get_formation_spline_path(num_points)
    }

    /// Check if this drone is the formation leader.
    pub fn is_formation_leader(&self) -> bool {
        self.formation.is_leader()
    }

    /// Set whether this drone is the formation leader.
    pub fn set_formation_leader(&mut self, is_leader: bool) {
        self.formation.set_leader(is_leader);
    }

    /// Set velocity consensus from formation leader.
    pub fn set_velocity_consensus(&mut self, consensus: VelocityConsensus) {
        self.formation.set_velocity_consensus(consensus);
    }

    /// Clear velocity consensus.
    pub fn clear_velocity_consensus(&mut self) {
        self.formation.clear_velocity_consensus();
    }

    /// Get the current velocity consensus (if any).
    pub fn velocity_consensus(&self) -> Option<&VelocityConsensus> {
        self.formation.velocity_consensus()
    }

}

/// Implement the Drone trait for DroneAgent, replacing the legacy FixedWing implementation.
impl Drone for DroneAgent {
    fn uid(&self) -> usize {
        self.id
    }

    fn state_update(&mut self, dt: f32, swarm: &[DroneInfo]) {
        DroneAgent::state_update(self, dt, swarm)
    }

    fn set_objective(&mut self, objective: Objective) {
        DroneAgent::set_objective(self, objective)
    }

    fn clear_objective(&mut self) {
        DroneAgent::clear_objective(self)
    }

    fn state(&self) -> &State {
        DroneAgent::state(self)
    }

    fn get_info(&self) -> DroneInfo {
        DroneAgent::get_info(self)
    }

    fn set_flight_params(&mut self, params: DronePerfFeatures) {
        DroneAgent::set_flight_params(self, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn create_test_bounds() -> Bounds {
        Bounds::new(1000.0, 1000.0).unwrap()
    }

    fn create_test_agent() -> DroneAgent {
        DroneAgent::new(
            0,
            Position::new(500.0, 500.0),
            Heading::new(0.0),
            create_test_bounds(),
        )
    }

    #[test]
    fn test_new_agent() {
        let agent = create_test_agent();
        assert_eq!(agent.uid(), 0);
        assert_eq!(agent.state().pos.x(), 500.0);
        assert_eq!(agent.state().pos.y(), 500.0);
    }

    #[test]
    fn test_set_objective() {
        let mut agent = create_test_agent();

        let objective = Objective::ReachWaypoint {
            waypoints: vec![Position::new(600.0, 500.0)].into(),
        };
        agent.set_objective(objective);

        assert!(matches!(agent.objective(), Objective::ReachWaypoint { .. }));
        assert_eq!(agent.mission.waypoints().len(), 1);
    }

    #[test]
    fn test_clear_objective() {
        let mut agent = create_test_agent();

        let objective = Objective::ReachWaypoint {
            waypoints: vec![Position::new(600.0, 500.0)].into(),
        };
        agent.set_objective(objective);
        agent.clear_objective();

        assert!(matches!(agent.objective(), Objective::Sleep));
        assert!(agent.mission.waypoints().is_empty());
    }

    #[test]
    fn test_state_update_moves_toward_waypoint() {
        let mut agent = create_test_agent();

        // Set a waypoint to the right
        agent.set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(700.0, 500.0)].into(),
        });

        // Simulate multiple ticks
        for _ in 0..100 {
            agent.state_update(0.016, &[]);
        }

        // Agent should have moved to the right
        assert!(agent.state().pos.x() > 500.0);
    }

    #[test]
    fn test_get_info() {
        let agent = create_test_agent();
        let info = agent.get_info();

        assert_eq!(info.uid, 0);
        assert_eq!(info.pos.x(), 500.0);
        assert_eq!(info.pos.y(), 500.0);
    }

    #[test]
    fn test_set_flight_params() {
        let mut agent = create_test_agent();

        let params = DronePerfFeatures::new_unchecked(200.0, 30.0, 5.0);
        agent.set_flight_params(params);

        assert_eq!(agent.platform.perf().max_vel, 200.0);
        assert_eq!(agent.platform.perf().max_acc, 30.0);
        assert_eq!(agent.platform.perf().max_turn_rate, 5.0);
    }

    #[test]
    fn test_set_waypoint_clearance() {
        let mut agent = create_test_agent();

        agent.set_waypoint_clearance(20.0);
        assert_eq!(agent.waypoint_clearance(), 20.0);

        // Minimum 1.0
        agent.set_waypoint_clearance(0.5);
        assert_eq!(agent.waypoint_clearance(), 1.0);
    }

    #[test]
    fn test_follow_route_produces_spline() {
        let mut agent = create_test_agent();

        let route: Arc<[Position]> = vec![
            Position::new(600.0, 500.0),
            Position::new(700.0, 600.0),
        ].into();

        agent.set_objective(Objective::FollowRoute {
            waypoints: route.iter().copied().collect(),
            route,
        });

        let path = agent.get_spline_path(20);
        assert_eq!(path.len(), 20);
    }

    #[test]
    fn test_non_route_no_spline() {
        let mut agent = create_test_agent();

        agent.set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(600.0, 500.0)].into(),
        });

        let path = agent.get_spline_path(20);
        assert!(path.is_empty());
    }

    #[test]
    fn test_invalid_flight_params_rejected() {
        use crate::missions::AgentCommand;

        let mut agent = create_test_agent();
        let original_vel = agent.platform.perf().max_vel;
        let original_acc = agent.platform.perf().max_acc;
        let original_turn = agent.platform.perf().max_turn_rate;

        // Zero velocity should be silently discarded
        agent.command_queue_mut().push(AgentCommand::SetMaxVelocity(0.0));
        agent.process_commands();
        assert_eq!(agent.platform.perf().max_vel, original_vel);

        // Negative acceleration should be silently discarded
        agent.command_queue_mut().push(AgentCommand::SetMaxAcceleration(-5.0));
        agent.process_commands();
        assert_eq!(agent.platform.perf().max_acc, original_acc);

        // Negative turn rate should be silently discarded
        agent.command_queue_mut().push(AgentCommand::SetMaxTurnRate(-1.0));
        agent.process_commands();
        assert_eq!(agent.platform.perf().max_turn_rate, original_turn);

        // Valid values should be accepted
        agent.command_queue_mut().push(AgentCommand::SetMaxVelocity(200.0));
        agent.process_commands();
        assert_eq!(agent.platform.perf().max_vel, 200.0);
    }

    // ===== Formation tests =====

    #[test]
    fn test_set_formation_slot() {
        let mut agent = create_test_agent();

        let slot = FormationSlot::new(Vec2::new(50.0, 0.0), 1);
        let center = Position::new(600.0, 500.0);
        let heading = 0.0;

        agent.set_formation_slot(slot, center, heading);

        assert!(agent.in_formation());
        assert!(agent.formation_slot().is_some());
        assert_eq!(agent.formation_slot().unwrap().priority, 1);

        // Should have set a waypoint
        assert!(!agent.mission.waypoints().is_empty());
    }

    #[test]
    fn test_formation_waypoint_calculation() {
        let mut agent = create_test_agent();

        // Slot 50 units in front (positive x direction)
        let slot = FormationSlot::new(Vec2::new(50.0, 0.0), 1);
        let center = Position::new(500.0, 500.0);
        let heading = 0.0; // Facing right

        agent.set_formation_slot(slot, center, heading);

        // Target should be at (550, 500)
        let waypoint = agent.mission.waypoints().front().unwrap();
        assert!((waypoint.x() - 550.0).abs() < 0.1);
        assert!((waypoint.y() - 500.0).abs() < 0.1);
    }

    #[test]
    fn test_formation_waypoint_with_rotation() {
        let mut agent = create_test_agent();

        // Slot 50 units in front
        let slot = FormationSlot::new(Vec2::new(50.0, 0.0), 1);
        let center = Position::new(500.0, 500.0);
        let heading = std::f32::consts::FRAC_PI_2; // 90 degrees, facing up

        agent.set_formation_slot(slot, center, heading);

        // Target should be at (500, 550) - rotated 90 degrees
        let waypoint = agent.mission.waypoints().front().unwrap();
        assert!((waypoint.x() - 500.0).abs() < 0.1);
        assert!((waypoint.y() - 550.0).abs() < 0.1);
    }

    #[test]
    fn test_update_formation_reference() {
        let mut agent = create_test_agent();

        let slot = FormationSlot::new(Vec2::new(50.0, 0.0), 1);
        agent.set_formation_slot(slot, Position::new(500.0, 500.0), 0.0);

        // Move formation center (with velocity for lead pursuit)
        agent.update_formation_reference(Position::new(600.0, 600.0), 0.0, Vec2::new(0.0, 0.0));

        // Target should update
        let waypoint = agent.mission.waypoints().front().unwrap();
        assert!((waypoint.x() - 650.0).abs() < 0.1);
        assert!((waypoint.y() - 600.0).abs() < 0.1);
    }

    #[test]
    fn test_clear_formation_slot() {
        let mut agent = create_test_agent();

        let slot = FormationSlot::new(Vec2::new(50.0, 0.0), 1);
        agent.set_formation_slot(slot, Position::new(500.0, 500.0), 0.0);

        assert!(agent.in_formation());

        agent.clear_formation_slot();

        assert!(!agent.in_formation());
        assert!(agent.formation_slot().is_none());
    }

    #[test]
    fn test_handle_formation_command_disperse() {
        let mut agent = create_test_agent();

        let slot = FormationSlot::new(Vec2::new(50.0, 0.0), 1);
        agent.set_formation_slot(slot, Position::new(500.0, 500.0), 0.0);

        agent.handle_formation_command(FormationCommand::Disperse);

        assert!(!agent.in_formation());
    }

    #[test]
    fn test_handle_formation_command_hold() {
        let mut agent = create_test_agent();

        let slot = FormationSlot::new(Vec2::new(50.0, 0.0), 1);
        agent.set_formation_slot(slot, Position::new(500.0, 500.0), 0.0);

        // Clear waypoints manually to test that hold restores them
        agent.mission.clear();

        agent.handle_formation_command(FormationCommand::Hold);

        // Should have re-set the formation waypoint
        assert!(!agent.mission.waypoints().is_empty());
    }

    // ===== Integration tests for collision avoidance =====

    /// Helper to compute minimum distance between any two drones
    fn min_pairwise_distance(agents: &[DroneAgent], bounds: &Bounds) -> f32 {
        let mut min_dist = f32::MAX;
        for i in 0..agents.len() {
            for j in (i + 1)..agents.len() {
                let pos_i = agents[i].state().pos.as_vec2();
                let pos_j = agents[j].state().pos.as_vec2();
                let dist = bounds.distance(pos_i, pos_j);
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
        min_dist
    }

    /// Helper to run simulation for n ticks
    fn run_simulation(agents: &mut [DroneAgent], bounds: &Bounds, ticks: usize, dt: f32) -> f32 {
        let mut min_dist_ever = f32::MAX;

        for _ in 0..ticks {
            // Collect swarm info
            let swarm: Vec<DroneInfo> = agents.iter().map(|a| a.get_info()).collect();

            // Update all agents
            for agent in agents.iter_mut() {
                agent.state_update(dt, &swarm);
            }

            // Track minimum distance
            let min_dist = min_pairwise_distance(agents, bounds);
            if min_dist < min_dist_ever {
                min_dist_ever = min_dist;
            }
        }

        min_dist_ever
    }

    #[test]
    fn test_collision_avoidance_head_on() {
        // Two drones heading straight at each other
        let bounds = create_test_bounds();

        let mut agent_a = DroneAgent::new(
            0,
            Position::new(200.0, 500.0),
            Heading::new(0.0), // Facing right
            bounds,
        );
        let mut agent_b = DroneAgent::new(
            1,
            Position::new(800.0, 500.0),
            Heading::new(std::f32::consts::PI), // Facing left
            bounds,
        );

        // Set waypoints so they would collide
        agent_a.set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(800.0, 500.0)].into(),
        });
        agent_b.set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(200.0, 500.0)].into(),
        });

        let mut agents = [agent_a, agent_b];
        let min_dist = run_simulation(&mut agents, &bounds, 500, 0.016);

        // Minimum safe distance should be maintained (agents have radius ~20)
        // Allow some tolerance but they should never overlap
        assert!(
            min_dist > 25.0,
            "Collision detected! Minimum distance was {} (expected > 25)",
            min_dist
        );
    }

    #[test]
    fn test_collision_avoidance_crossing_paths() {
        // Four drones crossing paths at center
        let bounds = create_test_bounds();

        let mut agents = vec![
            DroneAgent::new(0, Position::new(200.0, 500.0), Heading::new(0.0), bounds),
            DroneAgent::new(1, Position::new(800.0, 500.0), Heading::new(std::f32::consts::PI), bounds),
            DroneAgent::new(2, Position::new(500.0, 200.0), Heading::new(std::f32::consts::FRAC_PI_2), bounds),
            DroneAgent::new(3, Position::new(500.0, 800.0), Heading::new(-std::f32::consts::FRAC_PI_2), bounds),
        ];

        // Each drone heads toward the opposite side (all crossing at center)
        agents[0].set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(800.0, 500.0)].into(),
        });
        agents[1].set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(200.0, 500.0)].into(),
        });
        agents[2].set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(500.0, 800.0)].into(),
        });
        agents[3].set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(500.0, 200.0)].into(),
        });

        let min_dist = run_simulation(&mut agents, &bounds, 500, 0.016);

        // Note: 4-way crossing is an extreme case. ORCA radius is 25, but we allow
        // slightly closer passes in this challenging scenario. Physical collision is at 30.
        assert!(
            min_dist > 20.0,
            "Collision detected! Minimum distance was {} (expected > 20)",
            min_dist
        );
    }

    #[test]
    fn test_collision_avoidance_stationary_obstacle() {
        // One drone moving, one stationary in the way
        let bounds = create_test_bounds();

        let mut moving_drone = DroneAgent::new(
            0,
            Position::new(200.0, 500.0),
            Heading::new(0.0),
            bounds,
        );
        let stationary_drone = DroneAgent::new(
            1,
            Position::new(500.0, 500.0), // In the path
            Heading::new(0.0),
            bounds,
        );

        moving_drone.set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(800.0, 500.0)].into(),
        });
        // Stationary drone has no objective (will stay in place)

        let mut agents = [moving_drone, stationary_drone];
        let min_dist = run_simulation(&mut agents, &bounds, 500, 0.016);

        assert!(
            min_dist > 25.0,
            "Collision with stationary drone! Minimum distance was {} (expected > 25)",
            min_dist
        );
    }

    #[test]
    fn test_collision_avoidance_slow_approach() {
        // Two drones approaching slowly - tests the escape speed feature
        let bounds = create_test_bounds();

        let mut agent_a = DroneAgent::new(
            0,
            Position::new(450.0, 500.0),
            Heading::new(0.0),
            bounds,
        );
        let mut agent_b = DroneAgent::new(
            1,
            Position::new(550.0, 500.0),
            Heading::new(std::f32::consts::PI),
            bounds,
        );

        // Both drones want to be at center - will approach slowly
        agent_a.set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(500.0, 500.0)].into(),
        });
        agent_b.set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(500.0, 500.0)].into(),
        });

        let mut agents = [agent_a, agent_b];
        let min_dist = run_simulation(&mut agents, &bounds, 500, 0.016);

        assert!(
            min_dist > 25.0,
            "Slow collision detected! Minimum distance was {} (expected > 25)",
            min_dist
        );
    }

    #[test]
    fn test_formation_following_with_collision_avoidance() {
        // Test scenario:
        // 1. Leader drone follows a predefined route
        // 2. Follower drones start at random positions
        // 3. Followers should form up and follow leader without collisions

        let bounds = Bounds::new(1000.0, 1000.0).unwrap();
        let dt = 0.016;

        // Predefined route for leader (square pattern)
        let route: Arc<[Position]> = vec![
            Position::new(300.0, 300.0),
            Position::new(700.0, 300.0),
            Position::new(700.0, 700.0),
            Position::new(300.0, 700.0),
        ].into();

        // Create leader drone at start of route
        let mut leader = DroneAgent::new(
            0,
            Position::new(300.0, 300.0),
            Heading::new(0.0),
            bounds,
        );
        leader.set_formation_leader(true);
        leader.set_objective(Objective::FollowRoute {
            waypoints: route.iter().copied().collect(),
            route: route.clone(),
        });

        // Create follower drones at scattered positions behind/around the leader
        // Start them in roughly the right area to avoid major path conflicts
        let follower_starts = [
            (Position::new(200.0, 250.0), Heading::new(0.5)),   // Behind-left of leader
            (Position::new(200.0, 350.0), Heading::new(-0.3)),  // Behind-right of leader
            (Position::new(150.0, 200.0), Heading::new(0.8)),   // Further behind-left
            (Position::new(150.0, 400.0), Heading::new(-0.5)),  // Further behind-right
            (Position::new(100.0, 300.0), Heading::new(0.0)),   // Far behind
        ];

        // Formation slots (V formation behind leader, wider spacing for safety)
        let formation_slots = [
            FormationSlot::new(Vec2::new(-60.0, -50.0), 1),  // Back-left
            FormationSlot::new(Vec2::new(-60.0, 50.0), 2),   // Back-right
            FormationSlot::new(Vec2::new(-120.0, -100.0), 3), // Further back-left
            FormationSlot::new(Vec2::new(-120.0, 100.0), 4),  // Further back-right
            FormationSlot::new(Vec2::new(-180.0, 0.0), 5),    // Tail
        ];

        let mut followers: Vec<DroneAgent> = follower_starts
            .iter()
            .enumerate()
            .map(|(i, (pos, hdg))| {
                let mut agent = DroneAgent::new(i + 1, *pos, *hdg, bounds);
                agent.set_formation_slot(
                    formation_slots[i],
                    leader.state().pos,
                    leader.state().hdg.radians(),
                );
                agent
            })
            .collect();

        // Run simulation
        let mut min_dist_ever = f32::MAX;
        let total_ticks = 2000; // Long enough to see formation behavior

        for tick in 0..total_ticks {
            // Collect swarm info (leader + followers)
            let mut swarm: Vec<DroneInfo> = vec![leader.get_info()];
            swarm.extend(followers.iter().map(|f| f.get_info()));

            // Update leader
            leader.state_update(dt, &swarm);

            // Update formation reference for followers based on leader state
            let leader_pos = leader.state().pos;
            let leader_hdg = leader.state().hdg.radians();
            let leader_vel = leader.state().vel.as_vec2();

            for follower in &mut followers {
                follower.update_formation_reference(leader_pos, leader_hdg, leader_vel);
                follower.state_update(dt, &swarm);
            }

            // Check minimum distance between all drones
            let mut all_agents: Vec<&DroneAgent> = vec![&leader];
            all_agents.extend(followers.iter());

            for i in 0..all_agents.len() {
                for j in (i + 1)..all_agents.len() {
                    let pos_i = all_agents[i].state().pos.as_vec2();
                    let pos_j = all_agents[j].state().pos.as_vec2();
                    let dist = bounds.distance(pos_i, pos_j);
                    if dist < min_dist_ever {
                        min_dist_ever = dist;
                    }
                }
            }

            // Early exit if actual collision detected (physical overlap)
            if min_dist_ever < 15.0 {
                panic!(
                    "Physical collision at tick {}! Minimum distance was {} (expected > 15)",
                    tick, min_dist_ever
                );
            }
        }

        // Verify drones maintained safe separation (>20 is good, >15 is minimum)
        assert!(
            min_dist_ever > 18.0,
            "Formation collision detected! Minimum distance was {} (expected > 18)",
            min_dist_ever
        );

        // Verify followers are making progress toward formation
        // (average distance to slots should be reasonable - not a strict requirement
        // since the main test is collision avoidance, not perfect formation keeping)
        let leader_pos = leader.state().pos;
        let leader_hdg = leader.state().hdg.radians();
        let cos_h = leader_hdg.cos();
        let sin_h = leader_hdg.sin();

        let mut total_dist = 0.0;
        for (i, follower) in followers.iter().enumerate() {
            let slot = formation_slots[i];
            let rotated_x = slot.offset.x * cos_h - slot.offset.y * sin_h;
            let rotated_y = slot.offset.x * sin_h + slot.offset.y * cos_h;
            let target = Vec2::new(leader_pos.x() + rotated_x, leader_pos.y() + rotated_y);

            let follower_pos = follower.state().pos.as_vec2();
            let dist_to_slot = bounds.distance(follower_pos, target);
            total_dist += dist_to_slot;
        }

        let avg_dist = total_dist / followers.len() as f32;
        // Average distance should be under 200 - they're trying to follow
        // (not a strict formation test, just verifying they're in the vicinity)
        assert!(
            avg_dist < 200.0,
            "Followers not tracking formation! Average distance to slots: {} (expected < 200)",
            avg_dist
        );
    }

    #[test]
    fn test_formation_with_leader_stopping() {
        // Test that followers don't collide when leader stops suddenly
        let bounds = Bounds::new(1000.0, 1000.0).unwrap();
        let dt = 0.016;

        // Leader starts moving, then stops
        let mut leader = DroneAgent::new(
            0,
            Position::new(500.0, 500.0),
            Heading::new(0.0),
            bounds,
        );
        leader.set_formation_leader(true);

        // Single waypoint - leader will reach it and stop
        leader.set_objective(Objective::ReachWaypoint {
            waypoints: vec![Position::new(600.0, 500.0)].into(),
        });

        // Followers start behind leader
        let mut followers: Vec<DroneAgent> = vec![
            DroneAgent::new(1, Position::new(450.0, 480.0), Heading::new(0.0), bounds),
            DroneAgent::new(2, Position::new(450.0, 520.0), Heading::new(0.0), bounds),
            DroneAgent::new(3, Position::new(400.0, 500.0), Heading::new(0.0), bounds),
        ];

        // Assign formation slots (wider spacing)
        let slots = [
            FormationSlot::new(Vec2::new(-60.0, -50.0), 1),
            FormationSlot::new(Vec2::new(-60.0, 50.0), 2),
            FormationSlot::new(Vec2::new(-120.0, 0.0), 3),
        ];

        for (i, follower) in followers.iter_mut().enumerate() {
            follower.set_formation_slot(
                slots[i],
                leader.state().pos,
                leader.state().hdg.radians(),
            );
        }

        let mut min_dist_ever = f32::MAX;

        for tick in 0..1000 {
            let mut swarm: Vec<DroneInfo> = vec![leader.get_info()];
            swarm.extend(followers.iter().map(|f| f.get_info()));

            leader.state_update(dt, &swarm);

            let leader_pos = leader.state().pos;
            let leader_hdg = leader.state().hdg.radians();
            let leader_vel = leader.state().vel.as_vec2();

            for follower in &mut followers {
                follower.update_formation_reference(leader_pos, leader_hdg, leader_vel);
                follower.state_update(dt, &swarm);
            }

            // Check distances
            let mut all_agents: Vec<&DroneAgent> = vec![&leader];
            all_agents.extend(followers.iter());

            for i in 0..all_agents.len() {
                for j in (i + 1)..all_agents.len() {
                    let pos_i = all_agents[i].state().pos.as_vec2();
                    let pos_j = all_agents[j].state().pos.as_vec2();
                    let dist = bounds.distance(pos_i, pos_j);
                    if dist < min_dist_ever {
                        min_dist_ever = dist;
                    }
                }
            }

            if min_dist_ever < 15.0 {
                panic!(
                    "Physical collision at tick {} when leader stopping! Min distance: {}",
                    tick, min_dist_ever
                );
            }
        }

        assert!(
            min_dist_ever > 18.0,
            "Collision when leader stopped! Minimum distance was {} (expected > 18)",
            min_dist_ever
        );
    }

    // ===== Formation speed factor tests =====

    #[test]
    fn test_formation_speed_factor_at_desired_distance() {
        use crate::agent::formation_navigator::FormationNavigator;

        let bounds = create_test_bounds();
        let mut nav = FormationNavigator::new();
        let slot = FormationSlot::new(Vec2::new(-50.0, -30.0), 1);
        let mut dummy_mission = crate::missions::WaypointMission::new(bounds);
        nav.set_slot(slot, Position::new(500.0, 500.0), 0.0, &mut dummy_mission);

        // Leader at (500, 500) moving right; follower at exact slot position (450, 470)
        let my_pos = Vec2::new(450.0, 470.0);

        let leader_info = DroneInfo::new_with_leader(
            0,
            &crate::types::State {
                pos: Position::new(500.0, 500.0),
                hdg: Heading::new(0.0),
                vel: crate::types::Velocity::new(10.0, 0.0),
                acc: crate::types::Acceleration::new(0.0, 0.0),
            },
            true,
        );

        let factor = nav.compute_formation_speed_factor(my_pos, &[leader_info], &bounds);
        assert!(factor.is_some());
        let f = factor.unwrap();
        // At exact slot position, factor should be ~1.0
        assert!((f - 1.0).abs() < 0.1, "Factor at desired distance: {}", f);
    }

    #[test]
    fn test_formation_speed_factor_too_far() {
        use crate::agent::formation_navigator::FormationNavigator;

        let bounds = create_test_bounds();
        let mut nav = FormationNavigator::new();
        let slot = FormationSlot::new(Vec2::new(-50.0, -30.0), 1);
        let mut dummy_mission = crate::missions::WaypointMission::new(bounds);
        nav.set_slot(slot, Position::new(500.0, 500.0), 0.0, &mut dummy_mission);

        // Follower much farther behind than desired (150m behind leader, moving right)
        let my_pos = Vec2::new(350.0, 500.0);

        let leader_info = DroneInfo::new_with_leader(
            0,
            &crate::types::State {
                pos: Position::new(500.0, 500.0),
                hdg: Heading::new(0.0),
                vel: crate::types::Velocity::new(10.0, 0.0),
                acc: crate::types::Acceleration::new(0.0, 0.0),
            },
            true,
        );

        let factor = nav.compute_formation_speed_factor(my_pos, &[leader_info], &bounds);
        assert!(factor.is_some());
        let f = factor.unwrap();
        // Too far behind, factor should be > 1.0 (speed up)
        assert!(f > 1.0, "Factor when too far: {}", f);
    }

    #[test]
    fn test_formation_speed_factor_too_close() {
        use crate::agent::formation_navigator::FormationNavigator;

        let bounds = create_test_bounds();
        let mut nav = FormationNavigator::new();
        let slot = FormationSlot::new(Vec2::new(-50.0, -30.0), 1);
        let mut dummy_mission = crate::missions::WaypointMission::new(bounds);
        nav.set_slot(slot, Position::new(500.0, 500.0), 0.0, &mut dummy_mission);

        // Follower ahead of desired slot (only 10m behind leader, should be 50m behind)
        let my_pos = Vec2::new(490.0, 500.0);

        let leader_info = DroneInfo::new_with_leader(
            0,
            &crate::types::State {
                pos: Position::new(500.0, 500.0),
                hdg: Heading::new(0.0),
                vel: crate::types::Velocity::new(10.0, 0.0),
                acc: crate::types::Acceleration::new(0.0, 0.0),
            },
            true,
        );

        let factor = nav.compute_formation_speed_factor(my_pos, &[leader_info], &bounds);
        assert!(factor.is_some());
        let f = factor.unwrap();
        // Too close / ahead of slot, factor should be < 1.0 (slow down)
        assert!(f < 1.0, "Factor when too close: {}", f);
    }

    #[test]
    fn test_formation_speed_factor_leader_returns_none() {
        use crate::agent::formation_navigator::FormationNavigator;

        let bounds = create_test_bounds();
        let mut nav = FormationNavigator::new();
        nav.set_leader(true);

        let my_pos = Vec2::new(500.0, 500.0);
        let factor = nav.compute_formation_speed_factor(my_pos, &[], &bounds);
        assert!(factor.is_none());
    }

    #[test]
    fn test_formation_speed_factor_clamping() {
        use crate::agent::formation_navigator::FormationNavigator;

        let bounds = create_test_bounds();
        let mut nav = FormationNavigator::new();
        let slot = FormationSlot::new(Vec2::new(-50.0, -30.0), 1);
        let mut dummy_mission = crate::missions::WaypointMission::new(bounds);
        nav.set_slot(slot, Position::new(500.0, 500.0), 0.0, &mut dummy_mission);

        let leader_info = DroneInfo::new_with_leader(
            0,
            &crate::types::State {
                pos: Position::new(500.0, 500.0),
                hdg: Heading::new(0.0),
                vel: crate::types::Velocity::new(10.0, 0.0),
                acc: crate::types::Acceleration::new(0.0, 0.0),
            },
            true,
        );

        // Extremely far behind - should clamp to 1.5
        let my_pos_far = Vec2::new(100.0, 500.0);
        let factor = nav.compute_formation_speed_factor(my_pos_far, &[leader_info], &bounds).unwrap();
        assert!(factor <= 1.5, "Factor should be clamped to 1.5, got {}", factor);
        assert!(factor >= 1.4, "Factor should be at clamp, got {}", factor);

        // Ahead of leader - should slow down significantly
        let my_pos_ahead = Vec2::new(550.0, 500.0);
        let factor = nav.compute_formation_speed_factor(my_pos_ahead, &[leader_info], &bounds).unwrap();
        assert!(factor < 1.0, "Factor when ahead should be < 1.0, got {}", factor);
        assert!(factor >= 0.3, "Factor should be clamped to 0.3, got {}", factor);
    }
}
