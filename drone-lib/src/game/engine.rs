use std::collections::{HashMap, HashSet};

use crate::types::{Bounds, DroneInfo, Position};

use super::collision::{check_target_hits, detect_collisions, process_detonations};
use super::config::GameConfig;
use super::result::{self, GameResult};
use super::state::{GameDrone, TargetState};
use super::task_processing::{apply_task_results, process_task_status};

/// Result of a single engine tick.
pub struct TickResult {
    pub destroyed_ids: HashSet<usize>,
    pub blast_positions: Vec<Position>,
    pub targets_changed: bool,
    pub game_result: GameResult,
}

/// Core game engine: owns drones, targets, and the authoritative game loop.
///
/// Both `SimRunner` (RL training) and `Swarm` (WASM) delegate to this engine
/// for physics, detonation, collision, and win-condition logic.
pub struct GameEngine {
    pub drones: Vec<GameDrone>,
    pub targets_a: Vec<TargetState>,
    pub targets_b: Vec<TargetState>,
    pub pending_detonations: HashSet<usize>,
    pub attack_targets: HashMap<usize, Position>,
    pub intercept_targets: HashMap<usize, usize>,
    pub protected_zones: HashMap<u32, Vec<Position>>,
    pub bounds: Bounds,
    pub group_split_id: usize,
    pub config: GameConfig,
    pub tick_count: u32,
}

impl GameEngine {
    /// Create a new engine with the given bounds and default game config.
    pub fn new(bounds: Bounds, group_split_id: usize) -> Self {
        Self::with_config(bounds, group_split_id, GameConfig::default())
    }

    /// Create a new engine with a custom game config.
    pub fn with_config(bounds: Bounds, group_split_id: usize, config: GameConfig) -> Self {
        GameEngine {
            drones: Vec::new(),
            targets_a: Vec::new(),
            targets_b: Vec::new(),
            pending_detonations: HashSet::new(),
            attack_targets: HashMap::new(),
            intercept_targets: HashMap::new(),
            protected_zones: HashMap::new(),
            bounds,
            group_split_id,
            config,
            tick_count: 0,
        }
    }

    /// Authoritative game tick. Updates drones in ID-priority order by default.
    pub fn tick(&mut self, dt: f32) -> TickResult {
        self.tick_with_ordering(dt, None)
    }

    /// Game tick with optional custom drone update ordering.
    ///
    /// If `ordering` is `None`, drones are updated in ascending ID order.
    /// If `Some(indices)`, drones are updated in the given index order.
    pub fn tick_with_ordering(&mut self, dt: f32, ordering: Option<&[usize]>) -> TickResult {
        self.tick_count += 1;

        // 1. Build swarm info.
        let mut swarm_info: Vec<DroneInfo> = self.drones.iter().map(|d| d.agent.get_info()).collect();

        // 2. Update each drone in priority order.
        let default_ordering: Vec<usize>;
        let indices = match ordering {
            Some(order) => order,
            None => {
                default_ordering = {
                    let mut v: Vec<usize> = (0..self.drones.len()).collect();
                    v.sort_by_key(|&i| self.drones[i].id);
                    v
                };
                &default_ordering
            }
        };

        for &idx in indices {
            if idx < self.drones.len() {
                self.drones[idx].agent.state_update(dt, &swarm_info);
                swarm_info[idx] = self.drones[idx].agent.get_info();
            }
        }

        // 3. Process task status (detonations, completions, failures).
        let task_result = process_task_status(
            &self.drones,
            &self.intercept_targets,
            &self.protected_zones,
            self.config.detonation_radius,
            &self.bounds,
        );
        apply_task_results(
            &task_result,
            &mut self.drones,
            &mut self.pending_detonations,
            &mut self.attack_targets,
            &mut self.intercept_targets,
        );

        // 4. Process detonations.
        let det_result = process_detonations(
            &self.drones,
            &mut self.pending_detonations,
            self.config.detonation_radius,
            &self.bounds,
        );

        // 5. Collision detection.
        let collided = detect_collisions(
            &self.drones,
            self.config.collision_distance,
            &self.bounds,
        );

        // Merge all destroyed IDs.
        let mut destroyed_ids = det_result.destroyed_ids;
        destroyed_ids.extend(collided);

        // 6. Remove destroyed drones + cleanup tracking maps.
        if !destroyed_ids.is_empty() {
            self.drones.retain(|d| !destroyed_ids.contains(&d.id));
            self.attack_targets.retain(|id, _| !destroyed_ids.contains(id));
            self.intercept_targets.retain(|id, _| !destroyed_ids.contains(id));
        }

        // 7. Check target hits.
        let mut targets_changed = false;
        if !det_result.blast_positions.is_empty() {
            let hit_a = check_target_hits(
                &mut self.targets_a,
                &det_result.blast_positions,
                self.config.target_hit_radius,
                &self.bounds,
            );
            let hit_b = check_target_hits(
                &mut self.targets_b,
                &det_result.blast_positions,
                self.config.target_hit_radius,
                &self.bounds,
            );
            targets_changed = hit_a || hit_b;
        }

        // 8. Update protected zones if targets changed.
        if targets_changed {
            self.update_protected_zones();
        }

        // 9. Check win condition.
        let game_result = self.check_result();

        TickResult {
            destroyed_ids,
            blast_positions: det_result.blast_positions,
            targets_changed,
            game_result,
        }
    }

    /// Check current game result based on alive counts.
    pub fn check_result(&self) -> GameResult {
        result::check_win_condition(
            self.alive_targets_a(),
            self.alive_targets_b(),
            self.count_drones(0),
            self.count_drones(1),
        )
    }

    /// Number of alive drones for a group.
    pub fn count_drones(&self, group: u32) -> usize {
        self.drones.iter().filter(|d| d.group == group).count()
    }

    /// Number of alive Group A targets.
    pub fn alive_targets_a(&self) -> usize {
        self.targets_a.iter().filter(|t| !t.destroyed).count()
    }

    /// Number of alive Group B targets.
    pub fn alive_targets_b(&self) -> usize {
        self.targets_b.iter().filter(|t| !t.destroyed).count()
    }

    /// Update protected zones from current target state.
    pub fn update_protected_zones(&mut self) {
        let zone_a: Vec<Position> = self.targets_a.iter()
            .filter(|t| !t.destroyed).map(|t| t.pos).collect();
        let zone_b: Vec<Position> = self.targets_b.iter()
            .filter(|t| !t.destroyed).map(|t| t.pos).collect();
        self.protected_zones.insert(0, zone_a);
        self.protected_zones.insert(1, zone_b);
    }
}
