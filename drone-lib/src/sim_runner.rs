//! Headless simulation runner for RL training.
//!
//! Replicates the full game loop (wasm-lib tick + webapp target destruction +
//! win condition) in pure Rust, with no WASM or browser dependencies.
//! Exposes `observe()` / `step()` / reward for RL training.
//!
//! Also provides a multi-agent API (`observe_multi()` / `step_multi()`) for
//! per-drone RL training with shared policy networks.

use std::collections::HashMap;

use crate::agent::DroneAgent;
use crate::behaviors::APFConfig;
use crate::doctrine::{DoctrineMode, SwarmDoctrine};
use crate::game::engine::GameEngine;
use crate::game::obs_layout::obs_idx;
use crate::game::patrol::build_patrol_route;
use crate::game::rng::lcg_f32;
use crate::game::state::{GameDrone, TargetState};
use crate::strategies::{StrategyDroneState, SwarmStrategy, TaskAssignment};
use crate::tasks::attack::AttackTask;
use crate::tasks::defend::DefendTask;
use crate::tasks::evade::EvadeTask;
use crate::tasks::intercept::InterceptTask;
use crate::tasks::intercept_group::InterceptGroupTask;
use crate::tasks::loiter::LoiterTask;
use crate::tasks::patrol::PatrolTask;
use crate::tasks::TaskStatus;
use crate::types::{Bounds, DroneInfo, Heading, Position, Vec2};

// Re-export game types for backward compatibility with rl-train.
pub use crate::game::result::GameResult;
pub use crate::game::state::TargetState as _TargetStateAlias;

// ============================================================================
// Constants
// ============================================================================

const DRONE_LENGTH_METERS: f32 = 37.5;
const DETONATION_RADIUS: f32 = DRONE_LENGTH_METERS * 5.0; // 187.5m
/// How often the RL agent makes decisions (in ticks).
const DECISION_INTERVAL: u32 = 60;
/// Patrol route standoff distance from target centroid.
const PATROL_STANDOFF: f32 = 200.0;
/// Max nearby threats for observation normalization.
const MAX_NEARBY_THREATS: f32 = 20.0;
/// Threat detection radius multiplier (same as doctrine).
const THREAT_RADIUS_MULTIPLIER: f32 = 5.0;

// -- Multi-agent constants (re-exported from obs_layout for backward compat) --

pub use crate::game::obs_layout::{OBS_DIM, ACT_DIM, MAX_VELOCITY};
/// How often the multi-agent RL makes decisions (in ticks).
const MULTI_DECISION_INTERVAL: u32 = 20;

// World layout fractions (derived from 10000m world proportions).
const TARGET_A_MIN_FRAC: f32 = 0.225;
const TARGET_A_MAX_FRAC: f32 = 0.4;
const TARGET_B_MIN_FRAC: f32 = 0.6;
const TARGET_B_MAX_FRAC: f32 = 0.775;
const CLUSTER_RADIUS_FRAC: f32 = 0.06;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the headless simulation.
#[derive(Debug, Clone)]
pub struct SimConfig {
    pub drones_per_side: u32,
    pub targets_per_side: u32,
    pub world_size: f32,
    pub max_ticks: u32,
    pub seed: u64,
    pub dt: f32,
    pub speed_multiplier: f32,
    /// If true, randomize Group B doctrine between Aggressive/Defensive each reset.
    pub randomize_opponent: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            drones_per_side: 24,
            targets_per_side: 6,
            world_size: 2500.0,
            max_ticks: 10000,
            seed: 42,
            dt: 0.05,
            speed_multiplier: 8.0,
            randomize_opponent: false,
        }
    }
}

// ============================================================================
// Step Result (RL interface)
// ============================================================================

/// Result of a single RL step (doctrine-level).
#[derive(Debug, Clone)]
pub struct StepResult {
    pub observation: [f32; 8],
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub game_result: GameResult,
}

/// Result of a multi-agent RL step (per-drone level).
#[derive(Debug, Clone)]
pub struct MultiStepResult {
    /// One observation per alive Group A drone.
    pub observations: Vec<[f32; OBS_DIM]>,
    /// Which drone each observation belongs to.
    pub drone_ids: Vec<usize>,
    /// Shared team reward (Group A perspective).
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub game_result: GameResult,
}

/// Result of a self-play RL step (both groups RL-controlled).
#[derive(Debug, Clone)]
pub struct SelfPlayStepResult {
    /// Observations for Group A drones.
    pub obs_a: Vec<[f32; OBS_DIM]>,
    pub drone_ids_a: Vec<usize>,
    /// Observations for Group B drones.
    pub obs_b: Vec<[f32; OBS_DIM]>,
    pub drone_ids_b: Vec<usize>,
    /// Reward from Group A's perspective (negate for Group B).
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub game_result: GameResult,
}


// ============================================================================
// Tick Mode
// ============================================================================

/// Which doctrine mode to use after the engine tick.
enum TickMode {
    /// Both doctrines run (standard simulation).
    Doctrine,
    /// Only Group B doctrine runs (Group A is RL-controlled).
    RlVsDoctrine,
    /// No doctrines run (both groups are RL-controlled).
    SelfPlay,
}

// ============================================================================
// SimRunner
// ============================================================================

/// Headless simulation runner for RL training.
///
/// Mirrors the wasm-lib `Swarm` + webapp game logic in pure Rust.
/// The RL agent controls Group A's doctrine; Group B uses a fixed policy.
pub struct SimRunner {
    /// Core game engine handling physics, detonation, collision, targets.
    engine: GameEngine,

    // Doctrine strategies (kept separate for direct access).
    doctrine_a: SwarmDoctrine,
    doctrine_b: SwarmDoctrine,

    config: SimConfig,
    rng: u32,

    // RL tracking: initial counts.
    initial_drones_per_side: u32,
    initial_targets_per_side: u32,

    // Per-step delta tracking (snapshot before step).
    snap_targets_a_alive: usize,
    snap_targets_b_alive: usize,
    snap_drones_a_alive: usize,
    snap_drones_b_alive: usize,
    snap_avg_enemy_dist: f32,

    /// Last RL action applied to each drone (for observation encoding).
    last_actions: HashMap<usize, u32>,
}

impl SimRunner {
    /// Read-only access to the simulation config.
    pub fn config(&self) -> &SimConfig {
        &self.config
    }

    /// Create a new simulation from config. Call `reset()` before first `step()`.
    pub fn new(config: SimConfig) -> Self {
        let bounds = Bounds::new(config.world_size, config.world_size).unwrap();
        let group_split_id = config.drones_per_side as usize;

        let mut runner = SimRunner {
            engine: GameEngine::new(bounds, group_split_id),
            doctrine_a: SwarmDoctrine::new(vec![], 0, vec![], vec![], vec![], DETONATION_RADIUS, DoctrineMode::Aggressive),
            doctrine_b: SwarmDoctrine::new(vec![], 1, vec![], vec![], vec![], DETONATION_RADIUS, DoctrineMode::Defensive),
            initial_drones_per_side: config.drones_per_side,
            initial_targets_per_side: config.targets_per_side,
            snap_targets_a_alive: 0,
            snap_targets_b_alive: 0,
            snap_drones_a_alive: 0,
            snap_drones_b_alive: 0,
            snap_avg_enemy_dist: 0.0,
            last_actions: HashMap::new(),
            rng: 0,
            config,
        };
        runner.reset_with_seed(runner.config.seed);
        runner
    }

    /// Reset the simulation with a new seed. Returns initial observation.
    pub fn reset_with_seed(&mut self, seed: u64) -> [f32; 8] {
        self.rng = seed as u32;
        self.engine.tick_count = 0;
        self.engine.pending_detonations.clear();
        self.engine.attack_targets.clear();
        self.engine.intercept_targets.clear();
        self.last_actions.clear();
        self.engine.protected_zones.clear();

        let w = self.config.world_size;

        // Generate targets.
        self.engine.targets_a = self.generate_targets(
            w * TARGET_A_MIN_FRAC, w * TARGET_A_MAX_FRAC,
            w * TARGET_A_MIN_FRAC, w * TARGET_A_MAX_FRAC,
        );
        self.engine.targets_b = self.generate_targets(
            w * TARGET_B_MIN_FRAC, w * TARGET_B_MAX_FRAC,
            w * TARGET_B_MIN_FRAC, w * TARGET_B_MAX_FRAC,
        );

        // Generate drones.
        let centroid_a = self.target_centroid(&self.engine.targets_a);
        let centroid_b = self.target_centroid(&self.engine.targets_b);
        let cluster_radius = w * CLUSTER_RADIUS_FRAC;

        self.engine.bounds = Bounds::new(w, w).unwrap();
        self.engine.group_split_id = self.config.drones_per_side as usize;

        let mut drones = Vec::new();
        // Group A drones (IDs 0..drones_per_side).
        for i in 0..self.config.drones_per_side {
            let pos = self.random_in_cluster(centroid_a, cluster_radius);
            let hdg = Heading::new(lcg_f32(&mut self.rng) * std::f32::consts::TAU - std::f32::consts::PI);
            let mut agent = DroneAgent::new(i as usize, pos, hdg, self.engine.bounds);
            agent.set_group(0);
            self.configure_combat_safety(&mut agent);
            drones.push(GameDrone { id: i as usize, agent, group: 0 });
        }
        // Group B drones (IDs drones_per_side..2*drones_per_side).
        for i in 0..self.config.drones_per_side {
            let id = (self.config.drones_per_side + i) as usize;
            let pos = self.random_in_cluster(centroid_b, cluster_radius);
            let hdg = Heading::new(lcg_f32(&mut self.rng) * std::f32::consts::TAU - std::f32::consts::PI);
            let mut agent = DroneAgent::new(id, pos, hdg, self.engine.bounds);
            agent.set_group(1);
            self.configure_combat_safety(&mut agent);
            drones.push(GameDrone { id, agent, group: 1 });
        }
        self.engine.drones = drones;

        // Set up protected zones.
        self.engine.update_protected_zones();

        // Build patrol routes and doctrines.
        let friendly_a: Vec<Position> = self.engine.targets_a.iter().map(|t| t.pos).collect();
        let enemy_a: Vec<Position> = self.engine.targets_b.iter().map(|t| t.pos).collect();
        let patrol_a = build_patrol_route(&friendly_a, PATROL_STANDOFF);

        let friendly_b: Vec<Position> = self.engine.targets_b.iter().map(|t| t.pos).collect();
        let enemy_b: Vec<Position> = self.engine.targets_a.iter().map(|t| t.pos).collect();
        let patrol_b = build_patrol_route(&friendly_b, PATROL_STANDOFF);

        let ids_a: Vec<usize> = (0..self.config.drones_per_side as usize).collect();
        let ids_b: Vec<usize> = (self.engine.group_split_id..self.engine.group_split_id + self.config.drones_per_side as usize).collect();

        self.doctrine_a = SwarmDoctrine::new(
            ids_a, 0, friendly_a, enemy_a, patrol_a, DETONATION_RADIUS, DoctrineMode::Aggressive,
        );
        let doctrine_b_mode = if self.config.randomize_opponent {
            if lcg_f32(&mut self.rng) < 0.5 {
                DoctrineMode::Aggressive
            } else {
                DoctrineMode::Defensive
            }
        } else {
            DoctrineMode::Defensive
        };
        self.doctrine_b = SwarmDoctrine::new(
            ids_b, 1, friendly_b, enemy_b, patrol_b, DETONATION_RADIUS, doctrine_b_mode,
        );

        // Init RL tracking snapshots.
        self.snap_targets_a_alive = self.engine.targets_a.len();
        self.snap_targets_b_alive = self.engine.targets_b.len();
        self.snap_drones_a_alive = self.config.drones_per_side as usize;
        self.snap_drones_b_alive = self.config.drones_per_side as usize;
        self.snap_avg_enemy_dist = self.avg_min_enemy_target_dist(0);

        self.observe()
    }

    /// Reset with the config's default seed.
    pub fn reset(&mut self) -> [f32; 8] {
        let seed = self.config.seed;
        self.reset_with_seed(seed)
    }

    // ========================================================================
    // RL Interface
    // ========================================================================

    /// Compute the 8-dim observation vector (all normalized 0-1).
    ///
    /// From Group A's perspective:
    /// `[own_drones, enemy_drones, friendly_targets, enemy_targets,
    ///   nearby_threats, time_fraction, defend_fraction, attack_fraction]`
    pub fn observe(&self) -> [f32; 8] {
        let initial_d = self.initial_drones_per_side as f32;
        let initial_t = self.initial_targets_per_side as f32;

        let own_drones = self.engine.count_drones(0) as f32 / initial_d;
        let enemy_drones = self.engine.count_drones(1) as f32 / initial_d;
        let friendly_targets = self.engine.alive_targets_a() as f32 / initial_t;
        let enemy_targets = self.engine.alive_targets_b() as f32 / initial_t;

        let threats = self.count_nearby_threats(0) as f32 / MAX_NEARBY_THREATS;
        let time_frac = self.engine.tick_count as f32 / self.config.max_ticks as f32;

        let total_a = self.engine.count_drones(0) as f32;
        let (defend_frac, attack_frac) = if total_a > 0.0 {
            let d = self.doctrine_a.defend_count() as f32;
            let a = self.doctrine_a.attack_count() as f32;
            (d / total_a, a / total_a)
        } else {
            (0.0, 0.0)
        };

        [
            own_drones.clamp(0.0, 1.0),
            enemy_drones.clamp(0.0, 1.0),
            friendly_targets.clamp(0.0, 1.0),
            enemy_targets.clamp(0.0, 1.0),
            threats.clamp(0.0, 1.0),
            time_frac.clamp(0.0, 1.0),
            defend_frac.clamp(0.0, 1.0),
            attack_frac.clamp(0.0, 1.0),
        ]
    }

    /// Take one RL step: apply action, advance DECISION_INTERVAL ticks, return result.
    ///
    /// Actions: 0 = Aggressive, 1 = Defensive, 2 = Hold (keep current mode).
    pub fn step(&mut self, action: u32) -> StepResult {
        // Snapshot for reward computation.
        self.snap_targets_a_alive = self.engine.alive_targets_a();
        self.snap_targets_b_alive = self.engine.alive_targets_b();
        self.snap_drones_a_alive = self.engine.count_drones(0);
        self.snap_drones_b_alive = self.engine.count_drones(1);
        self.snap_avg_enemy_dist = self.avg_min_enemy_target_dist(0);

        // Apply action to Group A's doctrine.
        match action {
            0 => self.doctrine_a.set_mode(DoctrineMode::Aggressive),
            1 => self.doctrine_a.set_mode(DoctrineMode::Defensive),
            _ => {} // Hold: keep current mode.
        }

        // Advance simulation for DECISION_INTERVAL ticks.
        let mut result = GameResult::InProgress;
        for _ in 0..DECISION_INTERVAL {
            self.tick_with_mode(TickMode::Doctrine);
            result = self.engine.check_result();
            if result != GameResult::InProgress {
                break;
            }
        }

        // Check for truncation (time limit).
        let truncated = result == GameResult::InProgress && self.engine.tick_count >= self.config.max_ticks;
        if truncated {
            result = GameResult::Draw;
        }
        let terminated = result != GameResult::InProgress;

        let reward = self.compute_reward(result);
        let observation = self.observe();

        StepResult {
            observation,
            reward,
            terminated,
            truncated,
            game_result: result,
        }
    }

    // ========================================================================
    // Multi-Agent RL Interface (per-drone)
    // ========================================================================

    /// Compute per-drone observations for all alive Group A drones.
    /// Returns (observations, drone_ids).
    pub fn observe_multi(&self) -> (Vec<[f32; OBS_DIM]>, Vec<usize>) {
        self.observe_multi_group(0)
    }

    /// Observe all alive drones in the given group.
    /// Returns group-relative observations (friendly/enemy are relative to the drone's group).
    pub fn observe_multi_group(&self, group: u32) -> (Vec<[f32; OBS_DIM]>, Vec<usize>) {
        let mut observations = Vec::new();
        let mut drone_ids = Vec::new();

        for drone in &self.engine.drones {
            if drone.group != group {
                continue;
            }
            observations.push(self.observe_drone(drone));
            drone_ids.push(drone.id);
        }

        (observations, drone_ids)
    }

    /// Take one multi-agent RL step: apply per-drone actions, advance
    /// MULTI_DECISION_INTERVAL ticks, return result.
    ///
    /// Actions: (drone_id, action) pairs where action is 0-13.
    /// Group A doctrine is bypassed; Group B still uses fixed Defensive doctrine.
    pub fn step_multi(&mut self, actions: &[(usize, u32)]) -> MultiStepResult {
        // Snapshot for reward computation.
        self.snap_targets_a_alive = self.engine.alive_targets_a();
        self.snap_targets_b_alive = self.engine.alive_targets_b();
        self.snap_drones_a_alive = self.engine.count_drones(0);
        self.snap_drones_b_alive = self.engine.count_drones(1);
        self.snap_avg_enemy_dist = self.avg_min_enemy_target_dist(0);

        // Apply per-drone actions.
        for &(drone_id, action) in actions {
            self.apply_rl_action(drone_id, action);
        }

        // Advance simulation for MULTI_DECISION_INTERVAL ticks.
        // Group B uses doctrine; Group A is RL-controlled.
        let mut result = GameResult::InProgress;
        for _ in 0..MULTI_DECISION_INTERVAL {
            self.tick_with_mode(TickMode::RlVsDoctrine);
            result = self.engine.check_result();
            if result != GameResult::InProgress {
                break;
            }
        }

        // Check for truncation (time limit).
        let truncated = result == GameResult::InProgress && self.engine.tick_count >= self.config.max_ticks;
        if truncated {
            result = GameResult::Draw;
        }
        let terminated = result != GameResult::InProgress;

        let reward = self.compute_multi_reward(result);
        let (observations, drone_ids) = self.observe_multi();

        MultiStepResult {
            observations,
            drone_ids,
            reward,
            terminated,
            truncated,
            game_result: result,
        }
    }

    /// Reset with seed and return initial multi-agent observations.
    pub fn reset_multi_with_seed(&mut self, seed: u64) -> (Vec<[f32; OBS_DIM]>, Vec<usize>) {
        self.reset_with_seed(seed);
        self.observe_multi()
    }

    /// Self-play step: both groups controlled by RL policies.
    /// No doctrine is used for either group.
    pub fn step_multi_selfplay(
        &mut self,
        actions_a: &[(usize, u32)],
        actions_b: &[(usize, u32)],
    ) -> SelfPlayStepResult {
        // Snapshot for reward computation.
        self.snap_targets_a_alive = self.engine.alive_targets_a();
        self.snap_targets_b_alive = self.engine.alive_targets_b();
        self.snap_drones_a_alive = self.engine.count_drones(0);
        self.snap_drones_b_alive = self.engine.count_drones(1);
        self.snap_avg_enemy_dist = self.avg_min_enemy_target_dist(0);

        // Apply per-drone actions for both groups.
        for &(drone_id, action) in actions_a {
            self.apply_rl_action(drone_id, action);
        }
        for &(drone_id, action) in actions_b {
            self.apply_rl_action(drone_id, action);
        }

        // Advance simulation — no doctrine for either group.
        let mut result = GameResult::InProgress;
        for _ in 0..MULTI_DECISION_INTERVAL {
            self.tick_with_mode(TickMode::SelfPlay);
            result = self.engine.check_result();
            if result != GameResult::InProgress {
                break;
            }
        }

        let truncated = result == GameResult::InProgress && self.engine.tick_count >= self.config.max_ticks;
        if truncated {
            result = GameResult::Draw;
        }
        let terminated = result != GameResult::InProgress;

        let reward = self.compute_multi_reward(result);
        let (obs_a, drone_ids_a) = self.observe_multi_group(0);
        let (obs_b, drone_ids_b) = self.observe_multi_group(1);

        SelfPlayStepResult {
            obs_a,
            drone_ids_a,
            obs_b,
            drone_ids_b,
            reward,
            terminated,
            truncated,
            game_result: result,
        }
    }

    /// Reset for self-play: returns observations for both groups.
    pub fn reset_selfplay_with_seed(&mut self, seed: u64) -> ((Vec<[f32; OBS_DIM]>, Vec<usize>), (Vec<[f32; OBS_DIM]>, Vec<usize>)) {
        self.reset_with_seed(seed);
        (self.observe_multi_group(0), self.observe_multi_group(1))
    }

    /// Compute 64-dim per-drone observation vector.
    ///
    /// Layout:
    ///   [0..5]   Ego: x/w, y/w, vx/max_v, vy/max_v, last_action/ACT_DIM
    ///   [5..25]  4 nearest enemies: (dx/w, dy/w, dist/diag, vx/max_v, vy/max_v) × 4
    ///   [25..40] 3 nearest friendlies: (dx/w, dy/w, dist/diag, vx/max_v, vy/max_v) × 3
    ///   [40..46] 3 nearest enemy targets: (dx/w, dy/w) × 3
    ///   [46..52] 3 nearest friendly targets: (dx/w, dy/w) × 3
    ///   [52..60] Global: own_d/init, enemy_d/init, own_t/init, enemy_t/init, time/max, threats/max,
    ///            nearest_friendly_dist/diag, friendlies_in_blast_radius/8
    ///   [60]     Agent ID: drone.id / initial_drones_per_side (unique per drone, breaks symmetry)
    ///   [61..64] Last actions of 3 nearest friendly drones: action/ACT_DIM × 3 (enables coordination)
    fn observe_drone(&self, drone: &GameDrone) -> [f32; OBS_DIM] {
        let w = self.config.world_size;
        let diag = (w * w + w * w).sqrt();
        let state = drone.agent.state();
        let my_pos = state.pos.as_vec2();

        let mut obs = [0.0f32; OBS_DIM];

        // [0..5] Ego state
        obs[obs_idx::EGO_X] = state.pos.x() / w;
        obs[obs_idx::EGO_Y] = state.pos.y() / w;
        obs[obs_idx::EGO_VX] = state.vel.as_vec2().x / MAX_VELOCITY;
        obs[obs_idx::EGO_VY] = state.vel.as_vec2().y / MAX_VELOCITY;
        let last_action = self.last_actions.get(&drone.id).copied().unwrap_or(13) as f32; // 13 = Hold
        obs[obs_idx::EGO_LAST_ACTION] = last_action / ACT_DIM as f32;

        // [5..25] 4 nearest enemy drones: (dx/w, dy/w, dist/diag, vx/max_v, vy/max_v) × 4
        let mut enemy_drones: Vec<(f32, f32, f32, f32, f32)> = self.engine.drones.iter()
            .filter(|d| d.group != drone.group)
            .map(|d| {
                let es = d.agent.state();
                let epos = es.pos.as_vec2();
                let dist = self.engine.bounds.distance(my_pos, epos);
                (
                    (epos.x - my_pos.x) / w,
                    (epos.y - my_pos.y) / w,
                    dist / diag,
                    es.vel.as_vec2().x / MAX_VELOCITY,
                    es.vel.as_vec2().y / MAX_VELOCITY,
                )
            })
            .collect();
        enemy_drones.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        for i in 0..obs_idx::ENEMY_DRONES_COUNT {
            let base = obs_idx::ENEMY_DRONES_START + i * obs_idx::ENEMY_DRONE_STRIDE;
            if i < enemy_drones.len() {
                let e = &enemy_drones[i];
                obs[base + obs_idx::REL_DX] = e.0;
                obs[base + obs_idx::REL_DY] = e.1;
                obs[base + obs_idx::REL_DIST] = e.2;
                obs[base + obs_idx::REL_VX] = e.3;
                obs[base + obs_idx::REL_VY] = e.4;
            } else {
                obs[base + obs_idx::REL_DX] = 1.0;
                obs[base + obs_idx::REL_DY] = 1.0;
                obs[base + obs_idx::REL_DIST] = 1.0;
            }
        }

        // [25..40] 3 nearest friendly drones: (dx/w, dy/w, dist/diag, vx/max_v, vy/max_v) × 3
        let mut friendly_drones: Vec<(f32, f32, f32, f32, f32)> = self.engine.drones.iter()
            .filter(|d| d.group == drone.group && d.id != drone.id)
            .map(|d| {
                let fs = d.agent.state();
                let fpos = fs.pos.as_vec2();
                let dist = self.engine.bounds.distance(my_pos, fpos);
                (
                    (fpos.x - my_pos.x) / w,
                    (fpos.y - my_pos.y) / w,
                    dist / diag,
                    fs.vel.as_vec2().x / MAX_VELOCITY,
                    fs.vel.as_vec2().y / MAX_VELOCITY,
                )
            })
            .collect();
        friendly_drones.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        for i in 0..obs_idx::FRIENDLY_DRONES_COUNT {
            let base = obs_idx::FRIENDLY_DRONES_START + i * obs_idx::FRIENDLY_DRONE_STRIDE;
            if i < friendly_drones.len() {
                let f = &friendly_drones[i];
                obs[base + obs_idx::REL_DX] = f.0;
                obs[base + obs_idx::REL_DY] = f.1;
                obs[base + obs_idx::REL_DIST] = f.2;
                obs[base + obs_idx::REL_VX] = f.3;
                obs[base + obs_idx::REL_VY] = f.4;
            } else {
                obs[base + obs_idx::REL_DX] = 1.0;
                obs[base + obs_idx::REL_DY] = 1.0;
                obs[base + obs_idx::REL_DIST] = 1.0;
            }
        }

        // [40..46] 3 nearest enemy targets: (dx/w, dy/w) × 3
        // Group-aware: group 0's enemy targets are targets_b, group 1's are targets_a.
        let enemy_target_list = if drone.group == 0 { &self.engine.targets_b } else { &self.engine.targets_a };
        let mut enemy_targets: Vec<(f32, f32, f32)> = enemy_target_list.iter()
            .filter(|t| !t.destroyed)
            .map(|t| {
                let dist = self.engine.bounds.distance(my_pos, t.pos.as_vec2());
                ((t.pos.x() - state.pos.x()) / w, (t.pos.y() - state.pos.y()) / w, dist)
            })
            .collect();
        enemy_targets.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        for i in 0..obs_idx::ENEMY_TARGETS_COUNT {
            let base = obs_idx::ENEMY_TARGETS_START + i * obs_idx::ENEMY_TARGET_STRIDE;
            if i < enemy_targets.len() {
                obs[base] = enemy_targets[i].0;
                obs[base + 1] = enemy_targets[i].1;
            }
        }

        // [46..52] 3 nearest friendly targets: (dx/w, dy/w) × 3
        let friendly_target_list = if drone.group == 0 { &self.engine.targets_a } else { &self.engine.targets_b };
        let mut friendly_targets: Vec<(f32, f32, f32)> = friendly_target_list.iter()
            .filter(|t| !t.destroyed)
            .map(|t| {
                let dist = self.engine.bounds.distance(my_pos, t.pos.as_vec2());
                ((t.pos.x() - state.pos.x()) / w, (t.pos.y() - state.pos.y()) / w, dist)
            })
            .collect();
        friendly_targets.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        for i in 0..obs_idx::FRIENDLY_TARGETS_COUNT {
            let base = obs_idx::FRIENDLY_TARGETS_START + i * obs_idx::FRIENDLY_TARGET_STRIDE;
            if i < friendly_targets.len() {
                obs[base] = friendly_targets[i].0;
                obs[base + 1] = friendly_targets[i].1;
            }
        }

        // [52..60] Global features (group-aware: own vs enemy perspective)
        let initial_d = self.initial_drones_per_side as f32;
        let initial_t = self.initial_targets_per_side as f32;
        let own_group = drone.group;
        let enemy_group = 1 - drone.group;
        obs[obs_idx::GLOBAL_OWN_DRONES] = self.engine.count_drones(own_group) as f32 / initial_d;
        obs[obs_idx::GLOBAL_ENEMY_DRONES] = self.engine.count_drones(enemy_group) as f32 / initial_d;
        let (own_targets_alive, enemy_targets_alive) = if own_group == 0 {
            (self.engine.alive_targets_a(), self.engine.alive_targets_b())
        } else {
            (self.engine.alive_targets_b(), self.engine.alive_targets_a())
        };
        obs[obs_idx::GLOBAL_OWN_TARGETS] = own_targets_alive as f32 / initial_t;
        obs[obs_idx::GLOBAL_ENEMY_TARGETS] = enemy_targets_alive as f32 / initial_t;
        obs[obs_idx::GLOBAL_TIME_FRAC] = self.engine.tick_count as f32 / self.config.max_ticks as f32;
        let threats = self.count_nearby_threats(own_group) as f32 / MAX_NEARBY_THREATS;
        obs[obs_idx::GLOBAL_THREATS] = threats.clamp(0.0, 1.0);

        // Nearest friendly distance (scalar) — safety-critical for avoiding clumping.
        let nearest_friendly_dist = self.engine.drones.iter()
            .filter(|d| d.group == drone.group && d.id != drone.id)
            .map(|d| self.engine.bounds.distance(my_pos, d.agent.state().pos.as_vec2()))
            .fold(f32::INFINITY, f32::min);
        obs[obs_idx::GLOBAL_NEAREST_FRIENDLY_DIST] = if nearest_friendly_dist.is_finite() { nearest_friendly_dist / diag } else { 1.0 };

        // Friendly density within detonation radius — cluster danger signal.
        let friendlies_in_blast = self.engine.drones.iter()
            .filter(|d| d.group == drone.group && d.id != drone.id)
            .filter(|d| self.engine.bounds.distance(my_pos, d.agent.state().pos.as_vec2()) <= DETONATION_RADIUS)
            .count() as f32;
        obs[obs_idx::GLOBAL_FRIENDLIES_IN_BLAST] = (friendlies_in_blast / 8.0).clamp(0.0, 1.0);

        // [60] Agent ID — unique per drone, breaks symmetry for parameter-shared policy.
        obs[obs_idx::AGENT_ID] = drone.id as f32 / self.initial_drones_per_side as f32;

        // [61..64] Last actions of 3 nearest friendly drones (for coordination).
        let mut friendly_ids_sorted: Vec<(f32, usize)> = self.engine.drones.iter()
            .filter(|d| d.group == drone.group && d.id != drone.id)
            .map(|d| (self.engine.bounds.distance(my_pos, d.agent.state().pos.as_vec2()), d.id))
            .collect();
        friendly_ids_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for i in 0..obs_idx::FRIENDLY_ACTIONS_COUNT {
            if i < friendly_ids_sorted.len() {
                let fid = friendly_ids_sorted[i].1;
                let action = self.last_actions.get(&fid).copied().unwrap_or(13) as f32;
                obs[obs_idx::FRIENDLY_ACTIONS_START + i] = action / ACT_DIM as f32;
            }
        }

        obs
    }

    /// Translate an RL action (0-13) into a task assignment for a drone.
    ///
    /// Action space:
    ///   0  = Attack nearest enemy target (direct)
    ///   1  = Attack 2nd nearest enemy target (direct)
    ///   2  = Attack 3rd nearest enemy target (direct)
    ///   3  = Attack nearest enemy target (evasive)
    ///   4  = Attack 2nd nearest enemy target (evasive)
    ///   5  = Attack 3rd nearest enemy target (evasive)
    ///   6  = Intercept nearest enemy drone
    ///   7  = Intercept 2nd nearest enemy drone
    ///   8  = Intercept enemy cluster (InterceptGroup)
    ///   9  = Defend nearest friendly target (tight: 100m orbit, 300m engage)
    ///   10 = Defend nearest friendly target (wide: 250m orbit, 600m engage)
    ///   11 = Patrol perimeter
    ///   12 = Evade nearest threat
    ///   13 = Hold
    fn apply_rl_action(&mut self, drone_id: usize, action: u32) {
        // Record action for observation encoding.
        self.last_actions.insert(drone_id, action);

        // 13 = Hold: keep current task
        if action == 13 {
            return;
        }

        // Find the drone's group and position.
        let (group, drone_pos) = match self.engine.drones.iter().find(|d| d.id == drone_id) {
            Some(d) => (d.group, d.agent.state().pos.as_vec2()),
            None => return,
        };

        match action {
            // --- Attack (direct): target by nth-nearest ---
            0 | 1 | 2 => {
                let nth = action as usize;
                let target_pos = self.nth_nearest_enemy_target(drone_pos, group, nth);
                if let Some(target) = target_pos {
                    self.clear_drone_tasks(drone_id);
                    if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(AttackTask::new(target, DETONATION_RADIUS)));
                    }
                    self.engine.attack_targets.insert(drone_id, target);
                }
            }

            // --- Attack (evasive): target by nth-nearest ---
            3 | 4 | 5 => {
                let nth = (action - 3) as usize;
                let target_pos = self.nth_nearest_enemy_target(drone_pos, group, nth);
                if let Some(target) = target_pos {
                    self.clear_drone_tasks(drone_id);
                    if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(AttackTask::new_evasive(target, DETONATION_RADIUS)));
                    }
                    self.engine.attack_targets.insert(drone_id, target);
                }
            }

            // --- Intercept nearest / 2nd nearest enemy drone ---
            6 | 7 => {
                let nth = (action - 6) as usize;
                let enemy_id = self.nth_nearest_enemy_drone(drone_pos, group, nth);
                if let Some(eid) = enemy_id {
                    self.clear_drone_tasks(drone_id);
                    self.intercept_drone(drone_id, eid);
                }
            }

            // --- Intercept enemy cluster ---
            8 => {
                self.clear_drone_tasks(drone_id);
                if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == drone_id) {
                    drone.agent.set_task(Box::new(InterceptGroupTask::new(group, DETONATION_RADIUS)));
                }
            }

            // --- Defend nearest friendly target (tight: 100m orbit, 300m engage) ---
            9 => {
                let target_pos = self.nth_nearest_friendly_target(drone_pos, group, 0);
                if let Some(center) = target_pos {
                    self.clear_drone_tasks(drone_id);
                    if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(DefendTask::new(
                            drone_id, group, center, 100.0, 300.0, DETONATION_RADIUS,
                        )));
                    }
                }
            }

            // --- Defend nearest friendly target (wide: 250m orbit, 600m engage) ---
            10 => {
                let target_pos = self.nth_nearest_friendly_target(drone_pos, group, 0);
                if let Some(center) = target_pos {
                    self.clear_drone_tasks(drone_id);
                    if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(DefendTask::new(
                            drone_id, group, center, 250.0, 600.0, DETONATION_RADIUS,
                        )));
                    }
                }
            }

            // --- Patrol perimeter ---
            11 => {
                let friendly_targets = if group == 0 { &self.engine.targets_a } else { &self.engine.targets_b };
                let friendly_positions: Vec<Position> = friendly_targets.iter()
                    .filter(|t| !t.destroyed)
                    .map(|t| t.pos)
                    .collect();
                if !friendly_positions.is_empty() {
                    let waypoints = build_patrol_route(&friendly_positions, PATROL_STANDOFF);
                    self.clear_drone_tasks(drone_id);
                    if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == drone_id) {
                        drone.agent.set_task(Box::new(PatrolTask::new(waypoints, 50.0, 2.0)));
                    }
                }
            }

            // --- Evade nearest threat ---
            12 => {
                self.clear_drone_tasks(drone_id);
                if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == drone_id) {
                    drone.agent.set_task(Box::new(EvadeTask::new(group)));
                }
            }

            _ => {} // Unknown action — treat as hold
        }
    }

    /// Clear attack/intercept tracking for a drone before assigning a new task.
    fn clear_drone_tasks(&mut self, drone_id: usize) {
        self.engine.attack_targets.remove(&drone_id);
        self.engine.intercept_targets.remove(&drone_id);
    }

    /// Get the Nth nearest alive enemy target position (0-indexed, group-aware).
    fn nth_nearest_enemy_target(&self, drone_pos: Vec2, group: u32, nth: usize) -> Option<Position> {
        let enemy_targets = if group == 0 { &self.engine.targets_b } else { &self.engine.targets_a };
        let mut targets: Vec<(f32, Position)> = enemy_targets.iter()
            .filter(|t| !t.destroyed)
            .map(|t| (self.engine.bounds.distance(drone_pos, t.pos.as_vec2()), t.pos))
            .collect();
        targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        targets.get(nth).map(|(_, pos)| *pos)
    }

    /// Get the Nth nearest alive friendly target position (0-indexed, group-aware).
    fn nth_nearest_friendly_target(&self, drone_pos: Vec2, group: u32, nth: usize) -> Option<Position> {
        let friendly_targets = if group == 0 { &self.engine.targets_a } else { &self.engine.targets_b };
        let mut targets: Vec<(f32, Position)> = friendly_targets.iter()
            .filter(|t| !t.destroyed)
            .map(|t| (self.engine.bounds.distance(drone_pos, t.pos.as_vec2()), t.pos))
            .collect();
        targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        targets.get(nth).map(|(_, pos)| *pos)
    }

    /// Get the ID of the Nth nearest enemy drone (0-indexed).
    fn nth_nearest_enemy_drone(&self, drone_pos: Vec2, group: u32, nth: usize) -> Option<usize> {
        let mut enemies: Vec<(f32, usize)> = self.engine.drones.iter()
            .filter(|d| d.group != group)
            .map(|d| (self.engine.bounds.distance(drone_pos, d.agent.state().pos.as_vec2()), d.id))
            .collect();
        enemies.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        enemies.get(nth).map(|(_, id)| *id)
    }

    // ========================================================================
    // Core Simulation Tick
    // ========================================================================

    /// Unified tick: delegates physics to GameEngine, then runs doctrine per mode.
    fn tick_with_mode(&mut self, mode: TickMode) {
        let effective_dt = self.config.dt * self.config.speed_multiplier;
        let result = self.engine.tick(effective_dt);

        // Update doctrine targets when targets change (not needed in selfplay).
        if result.targets_changed {
            match mode {
                TickMode::Doctrine | TickMode::RlVsDoctrine => self.update_doctrine_targets(),
                TickMode::SelfPlay => {}
            }
        }

        // Process doctrine strategies based on mode.
        match mode {
            TickMode::Doctrine => self.process_doctrines(effective_dt),
            TickMode::RlVsDoctrine => self.process_doctrine_b(effective_dt),
            TickMode::SelfPlay => {}
        }
    }

    /// Tick only Group B's doctrine strategy (Group A is RL-controlled).
    fn process_doctrine_b(&mut self, dt: f32) {
        let swarm_info: Vec<DroneInfo> = self.engine.drones.iter()
            .map(|d| d.agent.get_info())
            .collect();

        let own_drones: Vec<StrategyDroneState> = self.doctrine_b
            .drone_ids()
            .iter()
            .filter_map(|&id| {
                self.engine.drones.iter().find(|d| d.id == id).map(|d| {
                    let available = match d.agent.task_status() {
                        None => true,
                        Some(TaskStatus::Active) => false,
                        Some(TaskStatus::Complete) | Some(TaskStatus::Failed) => true,
                    };
                    StrategyDroneState {
                        id,
                        pos: d.agent.state().pos,
                        vel: d.agent.state().vel,
                        available,
                    }
                })
            })
            .collect();

        let assignments = self.doctrine_b.tick(&own_drones, &swarm_info, &self.engine.bounds, dt);
        for assignment in assignments {
            self.apply_task_assignment(assignment);
        }
    }

    /// Compute reward for multi-agent step (shared team reward, Group A perspective).
    fn compute_multi_reward(&self, result: GameResult) -> f32 {
        let mut reward = 0.0;

        // Terminal reward.
        match result {
            GameResult::AWins => reward += 10.0,
            GameResult::BWins => reward -= 10.0,
            GameResult::Draw => reward -= 5.0,
            GameResult::InProgress => {}
        }

        // Potential-based reward shaping using advantage ratios.
        const W_TARGETS: f32 = 5.0;
        const W_DRONES: f32 = 1.0;

        let phi_before =
            W_TARGETS * (self.snap_targets_a_alive as f32 / (1.0 + self.snap_targets_b_alive as f32))
            + W_DRONES * (self.snap_drones_a_alive as f32 / (1.0 + self.snap_drones_b_alive as f32));

        let phi_after =
            W_TARGETS * (self.engine.alive_targets_a() as f32 / (1.0 + self.engine.alive_targets_b() as f32))
            + W_DRONES * (self.engine.count_drones(0) as f32 / (1.0 + self.engine.count_drones(1) as f32));

        reward += phi_after - phi_before;

        // Approach reward: reward drones for moving closer to enemy targets.
        let current_avg_dist = self.avg_min_enemy_target_dist(0);
        let world_diag = (self.config.world_size * self.config.world_size * 2.0).sqrt();
        if world_diag > 0.0 {
            let dist_delta = (self.snap_avg_enemy_dist - current_avg_dist) / world_diag;
            reward += 2.0 * dist_delta; // positive when drones moved closer
        }

        // Time pressure: step penalty ramps up as the game progresses.
        // Early game (~0%): -0.001/step. Late game (~100%): -0.01/step.
        // Quadratic ramp so agents feel increasing urgency to finish.
        let time_frac = self.engine.tick_count as f32 / self.config.max_ticks as f32;
        let step_penalty = 0.001 + 0.009 * time_frac * time_frac;
        reward -= step_penalty;

        // Cluster density penalty.
        let group_a_drones: Vec<_> = self.engine.drones.iter()
            .filter(|d| d.group == 0)
            .collect();
        let mut cluster_penalty = 0.0f32;
        for drone in &group_a_drones {
            let pos = drone.agent.state().pos.as_vec2();
            let nearby = group_a_drones.iter()
                .filter(|d| d.id != drone.id
                    && self.engine.bounds.distance(pos, d.agent.state().pos.as_vec2()) <= DETONATION_RADIUS)
                .count() as f32;
            cluster_penalty += (nearby - 2.0).max(0.0);
        }
        let n_drones = group_a_drones.len().max(1) as f32;
        reward -= 0.01 * cluster_penalty / n_drones;

        reward
    }

    /// Current game result.
    pub fn game_result(&self) -> GameResult {
        self.engine.check_result()
    }

    /// Current tick count.
    pub fn tick_count(&self) -> u32 {
        self.engine.tick_count
    }

    /// Number of alive drones for a group.
    pub fn count_drones(&self, group: u32) -> usize {
        self.engine.count_drones(group)
    }

    /// Number of alive Group A targets.
    pub fn alive_targets_a(&self) -> usize {
        self.engine.alive_targets_a()
    }

    /// Number of alive Group B targets.
    pub fn alive_targets_b(&self) -> usize {
        self.engine.alive_targets_b()
    }

    /// Access targets (for testing).
    pub fn targets_a(&self) -> &[TargetState] {
        &self.engine.targets_a
    }

    /// Access targets (for testing).
    pub fn targets_b(&self) -> &[TargetState] {
        &self.engine.targets_b
    }

    /// Tick both doctrines and apply their task assignments.
    fn process_doctrines(&mut self, dt: f32) {
        let swarm_info: Vec<DroneInfo> = self.engine.drones.iter()
            .map(|d| d.agent.get_info())
            .collect();

        let mut all_assignments: Vec<TaskAssignment> = Vec::new();

        // Doctrine A.
        {
            let own_drones: Vec<StrategyDroneState> = self.doctrine_a
                .drone_ids()
                .iter()
                .filter_map(|&id| {
                    self.engine.drones.iter().find(|d| d.id == id).map(|d| {
                        let available = match d.agent.task_status() {
                            None => true,
                            Some(TaskStatus::Active) => false,
                            Some(TaskStatus::Complete) | Some(TaskStatus::Failed) => true,
                        };
                        StrategyDroneState {
                            id,
                            pos: d.agent.state().pos,
                            vel: d.agent.state().vel,
                            available,
                        }
                    })
                })
                .collect();
            all_assignments.extend(
                self.doctrine_a.tick(&own_drones, &swarm_info, &self.engine.bounds, dt)
            );
        }

        // Doctrine B.
        {
            let own_drones: Vec<StrategyDroneState> = self.doctrine_b
                .drone_ids()
                .iter()
                .filter_map(|&id| {
                    self.engine.drones.iter().find(|d| d.id == id).map(|d| {
                        let available = match d.agent.task_status() {
                            None => true,
                            Some(TaskStatus::Active) => false,
                            Some(TaskStatus::Complete) | Some(TaskStatus::Failed) => true,
                        };
                        StrategyDroneState {
                            id,
                            pos: d.agent.state().pos,
                            vel: d.agent.state().vel,
                            available,
                        }
                    })
                })
                .collect();
            all_assignments.extend(
                self.doctrine_b.tick(&own_drones, &swarm_info, &self.engine.bounds, dt)
            );
        }

        // Apply assignments.
        for assignment in all_assignments {
            self.apply_task_assignment(assignment);
        }
    }

    /// Apply a strategy's task assignment to a drone.
    fn apply_task_assignment(&mut self, assignment: TaskAssignment) {
        // Handle PatrolFormation specially.
        if let TaskAssignment::PatrolFormation {
            leader_id,
            follower_ids,
            waypoints,
            loiter_duration,
            ..
        } = assignment
        {
            self.apply_patrol_formation(leader_id, follower_ids, waypoints, loiter_duration);
            return;
        }

        let drone_id = match &assignment {
            TaskAssignment::Intercept { drone_id, .. }
            | TaskAssignment::InterceptGroup { drone_id, .. }
            | TaskAssignment::Attack { drone_id, .. }
            | TaskAssignment::Defend { drone_id, .. }
            | TaskAssignment::Patrol { drone_id, .. }
            | TaskAssignment::Loiter { drone_id, .. } => *drone_id,
            TaskAssignment::PatrolFormation { .. } => unreachable!(),
        };

        self.engine.attack_targets.remove(&drone_id);
        self.engine.intercept_targets.remove(&drone_id);

        match assignment {
            TaskAssignment::Intercept { drone_id, target_id } => {
                self.intercept_drone(drone_id, target_id);
            }
            TaskAssignment::InterceptGroup { drone_id } => {
                let group = self.drone_group(drone_id);
                if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == drone_id) {
                    let task = InterceptGroupTask::new(group, DETONATION_RADIUS);
                    drone.agent.set_task(Box::new(task));
                }
            }
            TaskAssignment::Attack { drone_id, target } => {
                if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == drone_id) {
                    let task = AttackTask::new(target, DETONATION_RADIUS);
                    drone.agent.set_task(Box::new(task));
                }
                self.engine.attack_targets.insert(drone_id, target);
            }
            TaskAssignment::Defend { drone_id, center, orbit_radius, engage_radius } => {
                let group = self.drone_group(drone_id);
                if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == drone_id) {
                    let task = DefendTask::new(
                        drone_id, group, center, orbit_radius, engage_radius, DETONATION_RADIUS,
                    );
                    drone.agent.set_task(Box::new(task));
                }
            }
            TaskAssignment::Patrol { drone_id, waypoints, loiter_duration } => {
                if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == drone_id) {
                    let task = PatrolTask::new(waypoints, 50.0, loiter_duration);
                    drone.agent.set_task(Box::new(task));
                }
            }
            TaskAssignment::Loiter { drone_id, position } => {
                if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == drone_id) {
                    let task = LoiterTask::new(position, 50.0);
                    drone.agent.set_task(Box::new(task));
                }
            }
            TaskAssignment::PatrolFormation { .. } => unreachable!(),
        }
    }

    /// Set up intercept: one drone chases another.
    fn intercept_drone(&mut self, attacker_id: usize, target_id: usize) {
        if attacker_id == target_id { return; }
        if !self.engine.drones.iter().any(|d| d.id == attacker_id) { return; }
        if !self.engine.drones.iter().any(|d| d.id == target_id) { return; }

        let group = self.drone_group(attacker_id);
        if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == attacker_id) {
            let task = InterceptTask::new(attacker_id, target_id, group, DETONATION_RADIUS);
            drone.agent.set_task(Box::new(task));
        }
        self.engine.intercept_targets.insert(attacker_id, target_id);
    }

    /// Simplified PatrolFormation: leader gets patrol task, followers get
    /// individual patrol tasks at staggered waypoints.
    fn apply_patrol_formation(
        &mut self,
        leader_id: usize,
        follower_ids: Vec<usize>,
        waypoints: Vec<Position>,
        loiter_duration: f32,
    ) {
        if waypoints.is_empty() { return; }

        // Leader gets the patrol task.
        self.engine.attack_targets.remove(&leader_id);
        self.engine.intercept_targets.remove(&leader_id);
        if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == leader_id) {
            let task = PatrolTask::new(waypoints.clone(), 50.0, loiter_duration);
            drone.agent.set_task(Box::new(task));
        }

        // Followers get patrol tasks with staggered starting waypoints.
        let n_waypoints = waypoints.len();
        for (i, &fid) in follower_ids.iter().enumerate() {
            self.engine.attack_targets.remove(&fid);
            self.engine.intercept_targets.remove(&fid);
            if let Some(drone) = self.engine.drones.iter_mut().find(|d| d.id == fid) {
                let offset = (i + 1) % n_waypoints;
                let mut staggered = waypoints.clone();
                staggered.rotate_left(offset);
                let task = PatrolTask::new(staggered, 50.0, loiter_duration);
                drone.agent.set_task(Box::new(task));
            }
        }
    }

    // ========================================================================
    // Helpers
    // ========================================================================

    /// Get a drone's group (0 or 1).
    fn drone_group(&self, drone_id: usize) -> u32 {
        if drone_id < self.engine.group_split_id { 0 } else { 1 }
    }

    /// Configure combat safety layer on a drone (mirrors wasm-lib set_group_split).
    fn configure_combat_safety(&self, agent: &mut DroneAgent) {
        let mut orca_config = *agent.orca_config();
        orca_config.enemy_avoidance_radius = DETONATION_RADIUS;
        agent.set_orca_config(orca_config);

        agent.set_apf_config(APFConfig {
            influence_distance: DETONATION_RADIUS * 1.2,
            repulsion_strength: 10000000.0,
            min_distance: 35.0,
            max_force: 80.0,
            enemy_influence_distance: DETONATION_RADIUS * 10.0,
        });
    }

    /// Generate random targets in a rectangular region.
    fn generate_targets(&mut self, x_min: f32, x_max: f32, y_min: f32, y_max: f32) -> Vec<TargetState> {
        (0..self.config.targets_per_side)
            .map(|_| {
                let x = x_min + lcg_f32(&mut self.rng) * (x_max - x_min);
                let y = y_min + lcg_f32(&mut self.rng) * (y_max - y_min);
                TargetState { pos: Position::new(x, y), destroyed: false }
            })
            .collect()
    }

    /// Generate a random position within a circle.
    fn random_in_cluster(&mut self, center: Position, radius: f32) -> Position {
        let angle = lcg_f32(&mut self.rng) * std::f32::consts::TAU;
        let r = lcg_f32(&mut self.rng).sqrt() * radius;
        Position::new(center.x() + r * angle.cos(), center.y() + r * angle.sin())
    }

    /// Centroid of alive targets.
    fn target_centroid(&self, targets: &[TargetState]) -> Position {
        let alive: Vec<&TargetState> = targets.iter().filter(|t| !t.destroyed).collect();
        if alive.is_empty() {
            return Position::new(self.config.world_size / 2.0, self.config.world_size / 2.0);
        }
        let cx = alive.iter().map(|t| t.pos.x()).sum::<f32>() / alive.len() as f32;
        let cy = alive.iter().map(|t| t.pos.y()).sum::<f32>() / alive.len() as f32;
        Position::new(cx, cy)
    }

    /// Average distance from group's drones to nearest enemy target.
    fn avg_min_enemy_target_dist(&self, group: u32) -> f32 {
        let enemy_targets = if group == 0 { &self.engine.targets_b } else { &self.engine.targets_a };
        let alive_targets: Vec<_> = enemy_targets.iter().filter(|t| !t.destroyed).collect();
        if alive_targets.is_empty() { return 0.0; }

        let drones: Vec<_> = self.engine.drones.iter().filter(|d| d.group == group).collect();
        if drones.is_empty() { return 0.0; }

        let total: f32 = drones.iter().map(|d| {
            let pos = d.agent.state().pos.as_vec2();
            alive_targets.iter()
                .map(|t| self.engine.bounds.distance(pos, t.pos.as_vec2()))
                .fold(f32::INFINITY, f32::min)
        }).sum();

        total / drones.len() as f32
    }

    /// Update doctrine targets from current engine target state.
    fn update_doctrine_targets(&mut self) {
        let friendly_a: Vec<Position> = self.engine.targets_a.iter().filter(|t| !t.destroyed).map(|t| t.pos).collect();
        let enemy_a: Vec<Position> = self.engine.targets_b.iter().filter(|t| !t.destroyed).map(|t| t.pos).collect();
        self.doctrine_a.update_targets(&friendly_a, &enemy_a);

        let friendly_b: Vec<Position> = self.engine.targets_b.iter().filter(|t| !t.destroyed).map(|t| t.pos).collect();
        let enemy_b: Vec<Position> = self.engine.targets_a.iter().filter(|t| !t.destroyed).map(|t| t.pos).collect();
        self.doctrine_b.update_targets(&friendly_b, &enemy_b);
    }

    /// Count enemy drones near friendly target centroid (for observation).
    fn count_nearby_threats(&self, group: u32) -> usize {
        let targets = if group == 0 { &self.engine.targets_a } else { &self.engine.targets_b };
        let alive: Vec<Position> = targets.iter().filter(|t| !t.destroyed).map(|t| t.pos).collect();
        if alive.is_empty() { return 0; }

        let cx = alive.iter().map(|p| p.x()).sum::<f32>() / alive.len() as f32;
        let cy = alive.iter().map(|p| p.y()).sum::<f32>() / alive.len() as f32;
        let centroid = Vec2::new(cx, cy);
        let threat_radius = DETONATION_RADIUS * THREAT_RADIUS_MULTIPLIER;

        self.engine.drones.iter()
            .filter(|d| d.group != group && self.engine.bounds.distance(centroid, d.agent.state().pos.as_vec2()) <= threat_radius)
            .count()
    }

    /// Compute reward for the RL step (from Group A's perspective).
    fn compute_reward(&self, result: GameResult) -> f32 {
        let mut reward = 0.0;

        // Terminal reward.
        match result {
            GameResult::AWins => reward += 10.0,
            GameResult::BWins => reward -= 10.0,
            GameResult::Draw => {}
            GameResult::InProgress => {}
        }

        // Per-step: enemy targets destroyed this step.
        let enemy_destroyed = self.snap_targets_b_alive.saturating_sub(self.engine.alive_targets_b());
        reward += enemy_destroyed as f32 * 1.0;

        // Per-step: friendly targets lost this step.
        let friendly_lost = self.snap_targets_a_alive.saturating_sub(self.engine.alive_targets_a());
        reward -= friendly_lost as f32 * 2.0;

        // Step penalty.
        reward -= 0.001;

        // Drone advantage this step.
        let enemy_drones_killed = self.snap_drones_b_alive.saturating_sub(self.engine.count_drones(1));
        let own_drones_killed = self.snap_drones_a_alive.saturating_sub(self.engine.count_drones(0));
        reward += 0.1 * (enemy_drones_killed as f32 - own_drones_killed as f32);

        reward
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_correctness() {
        let config = SimConfig {
            drones_per_side: 12,
            targets_per_side: 3,
            ..Default::default()
        };
        let runner = SimRunner::new(config);

        // Correct number of drones per group.
        assert_eq!(runner.engine.count_drones(0), 12);
        assert_eq!(runner.engine.count_drones(1), 12);
        assert_eq!(runner.engine.drones.len(), 24);

        // Correct number of targets.
        assert_eq!(runner.engine.targets_a.len(), 3);
        assert_eq!(runner.engine.targets_b.len(), 3);

        // Group A drones have IDs < group_split_id.
        for d in runner.engine.drones.iter().filter(|d| d.group == 0) {
            assert!(d.id < runner.engine.group_split_id);
        }
        // Group B drones have IDs >= group_split_id.
        for d in runner.engine.drones.iter().filter(|d| d.group == 1) {
            assert!(d.id >= runner.engine.group_split_id);
        }

        // Targets A are in the A region.
        let w = runner.config.world_size;
        for t in &runner.engine.targets_a {
            assert!(t.pos.x() >= w * TARGET_A_MIN_FRAC - 1.0);
            assert!(t.pos.x() <= w * TARGET_A_MAX_FRAC + 1.0);
        }
        // Targets B are in the B region.
        for t in &runner.engine.targets_b {
            assert!(t.pos.x() >= w * TARGET_B_MIN_FRAC - 1.0);
            assert!(t.pos.x() <= w * TARGET_B_MAX_FRAC + 1.0);
        }
    }

    #[test]
    fn test_detonation_destroys_drones_and_targets() {
        let config = SimConfig {
            drones_per_side: 4,
            targets_per_side: 2,
            seed: 123,
            ..Default::default()
        };
        let mut runner = SimRunner::new(config);

        // Place a drone right on top of a target B.
        let target_pos = runner.engine.targets_b[0].pos;

        // Move drone 0 (group A) to the target position by giving it an attack task.
        if let Some(drone) = runner.engine.drones.iter_mut().find(|d| d.id == 0) {
            let task = AttackTask::new(target_pos, DETONATION_RADIUS);
            drone.agent.set_task(Box::new(task));
        }

        // Run enough ticks for the drone to reach and detonate.
        let initial_drones = runner.engine.drones.len();
        for _ in 0..2000 {
            runner.tick_with_mode(TickMode::Doctrine);
            if runner.engine.drones.len() < initial_drones {
                break;
            }
        }

        // Something should have been destroyed (detonation or collision).
        assert!(runner.engine.drones.len() < initial_drones, "Some drones should be destroyed");
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let config1 = SimConfig { seed: 99, drones_per_side: 6, targets_per_side: 2, ..Default::default() };
        let config2 = config1.clone();

        let mut runner1 = SimRunner::new(config1);
        let mut runner2 = SimRunner::new(config2);

        // Run 100 ticks on each.
        for _ in 0..100 {
            runner1.tick_with_mode(TickMode::Doctrine);
            runner2.tick_with_mode(TickMode::Doctrine);
        }

        // Same drone count.
        assert_eq!(runner1.engine.drones.len(), runner2.engine.drones.len());
        // Same positions.
        for (d1, d2) in runner1.engine.drones.iter().zip(runner2.engine.drones.iter()) {
            assert_eq!(d1.id, d2.id);
            let p1 = d1.agent.state().pos;
            let p2 = d2.agent.state().pos;
            assert!((p1.x() - p2.x()).abs() < 0.01, "Positions diverged");
            assert!((p1.y() - p2.y()).abs() < 0.01, "Positions diverged");
        }
    }

    #[test]
    fn test_full_episode_terminates() {
        let config = SimConfig {
            drones_per_side: 6,
            targets_per_side: 2,
            max_ticks: 5000,
            seed: 42,
            ..Default::default()
        };
        let mut runner = SimRunner::new(config);
        let mut result = GameResult::InProgress;

        for _ in 0..200 {
            let step = runner.step(0); // Aggressive
            if step.terminated || step.truncated {
                result = step.game_result;
                break;
            }
        }

        assert_ne!(result, GameResult::InProgress, "Episode should terminate");
    }

    #[test]
    fn test_observe_returns_valid_range() {
        let runner = SimRunner::new(SimConfig::default());
        let obs = runner.observe();
        for (i, &v) in obs.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0, "obs[{}] = {} out of range", i, v);
        }
    }

    #[test]
    fn test_step_returns_reward() {
        let mut runner = SimRunner::new(SimConfig {
            drones_per_side: 6,
            targets_per_side: 2,
            seed: 42,
            ..Default::default()
        });
        let result = runner.step(0);
        // Just verify it's a finite number.
        assert!(result.reward.is_finite(), "Reward should be finite");
        assert_eq!(result.observation.len(), 8);
    }

    #[test]
    fn test_win_condition_all_targets_destroyed() {
        let config = SimConfig {
            drones_per_side: 2,
            targets_per_side: 1,
            seed: 1,
            ..Default::default()
        };
        let mut runner = SimRunner::new(config);

        // Manually destroy all B targets.
        for t in &mut runner.engine.targets_b {
            t.destroyed = true;
        }
        assert_eq!(runner.engine.check_result(), GameResult::AWins);

        // Manually destroy all A targets too.
        for t in &mut runner.engine.targets_a {
            t.destroyed = true;
        }
        assert_eq!(runner.engine.check_result(), GameResult::Draw);
    }

    // ====================================================================
    // Multi-agent API tests
    // ====================================================================

    #[test]
    fn test_observe_multi_returns_correct_count() {
        let config = SimConfig {
            drones_per_side: 12,
            targets_per_side: 3,
            ..Default::default()
        };
        let runner = SimRunner::new(config);
        let (obs, ids) = runner.observe_multi();

        assert_eq!(obs.len(), 12, "Should have one obs per Group A drone");
        assert_eq!(ids.len(), 12, "Should have one id per Group A drone");

        // Each obs should be OBS_DIM long and have finite values.
        for o in &obs {
            assert_eq!(o.len(), OBS_DIM);
            for &v in o.iter() {
                assert!(v.is_finite(), "Obs value should be finite: {}", v);
            }
        }

        // All IDs should be < group_split_id (Group A).
        for &id in &ids {
            assert!(id < runner.engine.group_split_id, "ID {} should be Group A", id);
        }
    }

    #[test]
    fn test_step_multi_hold_actions() {
        let config = SimConfig {
            drones_per_side: 6,
            targets_per_side: 2,
            seed: 42,
            ..Default::default()
        };
        let mut runner = SimRunner::new(config);
        let (_, ids) = runner.observe_multi();

        // All Hold (action=13) — should not crash.
        let actions: Vec<(usize, u32)> = ids.iter().map(|&id| (id, 13u32)).collect();
        let result = runner.step_multi(&actions);

        assert!(result.reward.is_finite(), "Reward should be finite");
        assert!(!result.observations.is_empty(), "Should have observations");
    }

    #[test]
    fn test_step_multi_attack_actions() {
        let config = SimConfig {
            drones_per_side: 6,
            targets_per_side: 2,
            seed: 42,
            ..Default::default()
        };
        let mut runner = SimRunner::new(config);
        let (_, ids) = runner.observe_multi();

        // All Attack (action=0).
        let actions: Vec<(usize, u32)> = ids.iter().map(|&id| (id, 0u32)).collect();
        let result = runner.step_multi(&actions);

        assert!(result.reward.is_finite());
        // Drones should have attack tasks now.
        assert!(
            runner.engine.attack_targets.len() > 0,
            "Some drones should have attack targets"
        );
    }

    #[test]
    fn test_step_multi_full_episode() {
        let config = SimConfig {
            drones_per_side: 6,
            targets_per_side: 2,
            max_ticks: 5000,
            seed: 42,
            ..Default::default()
        };
        let mut runner = SimRunner::new(config);
        let mut game_result = GameResult::InProgress;

        for _ in 0..500 {
            let (_, ids) = runner.observe_multi();
            if ids.is_empty() { break; }
            // Alternate between Attack(0) and Defend-tight(9)
            let actions: Vec<(usize, u32)> = ids.iter().enumerate()
                .map(|(i, &id)| (id, if i % 2 == 0 { 0 } else { 9 }))
                .collect();
            let result = runner.step_multi(&actions);
            if result.terminated || result.truncated {
                game_result = result.game_result;
                break;
            }
        }

        assert_ne!(game_result, GameResult::InProgress, "Episode should terminate");
    }
}
