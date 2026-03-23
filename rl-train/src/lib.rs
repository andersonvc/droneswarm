//! PyO3 bindings for the droneswarm simulation environment.
//!
//! Exposes `VecSimRunner` — a vectorized environment wrapper that manages
//! multiple `SimRunner` instances in parallel, returning batched numpy arrays
//! suitable for PyTorch RL training.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use drone_lib::sim_runner::{
    DroneObsV2, SimConfig, SimRunner, SelfPlayStepResult,
    EGO_DIM, ENTITY_DIM, MAX_ENTITIES, ACT_DIM,
};

// ---------------------------------------------------------------------------
// Flattened observation arrays (shared between reset and step)
// ---------------------------------------------------------------------------

struct FlatObs {
    ego: Vec<f32>,
    entities: Vec<f32>,
    n_entities: Vec<i32>,
    drone_ids: Vec<i32>,
    env_indices: Vec<i32>,
    total_drones: usize,
}

fn flatten_obs(all_obs: &[(Vec<DroneObsV2>, Vec<usize>)]) -> FlatObs {
    let total_drones: usize = all_obs.iter().map(|(obs, _)| obs.len()).sum();

    let mut ego = Vec::with_capacity(total_drones * EGO_DIM);
    let mut entities = vec![0.0f32; total_drones * MAX_ENTITIES * ENTITY_DIM];
    let mut n_entities = Vec::with_capacity(total_drones);
    let mut drone_ids = Vec::with_capacity(total_drones);
    let mut env_indices = Vec::with_capacity(total_drones);

    let mut offset = 0usize;
    for (env_idx, (obs_list, id_list)) in all_obs.iter().enumerate() {
        for (obs, &drone_id) in obs_list.iter().zip(id_list.iter()) {
            ego.extend_from_slice(&obs.ego);

            for i in 0..obs.n_entities {
                let src_start = i * ENTITY_DIM;
                let dst_start = offset * MAX_ENTITIES * ENTITY_DIM + i * ENTITY_DIM;
                entities[dst_start..dst_start + ENTITY_DIM]
                    .copy_from_slice(&obs.entities[src_start..src_start + ENTITY_DIM]);
            }

            n_entities.push(obs.n_entities as i32);
            drone_ids.push(drone_id as i32);
            env_indices.push(env_idx as i32);
            offset += 1;
        }
    }

    FlatObs { ego, entities, n_entities, drone_ids, env_indices, total_drones }
}

// ---------------------------------------------------------------------------
// VecSimRunner
// ---------------------------------------------------------------------------

/// Vectorized simulation runner for batched RL training.
///
/// Manages N independent `SimRunner` instances and returns observations
/// as flat numpy arrays indexed by (drone, feature).
#[pyclass]
struct VecSimRunner {
    envs: Vec<SimRunner>,
    configs: Vec<SimConfig>,
    /// Per-env drone IDs from the last observation (Group A only).
    drone_ids_a: Vec<Vec<usize>>,
    /// Monotonically increasing seed counter for auto-resets.
    next_seed: u64,
}

#[pymethods]
impl VecSimRunner {
    /// Create a new vectorized environment.
    ///
    /// Each env receives a distinct seed (0..n_envs).
    #[new]
    #[pyo3(signature = (n_envs, drones_per_side=24, targets_per_side=6, world_size=2500.0, max_ticks=10000, speed_multiplier=4.0, skip_orca=true))]
    fn __new__(
        n_envs: usize,
        drones_per_side: u32,
        targets_per_side: u32,
        world_size: f32,
        max_ticks: u32,
        speed_multiplier: f32,
        skip_orca: bool,
    ) -> Self {
        let mut envs = Vec::with_capacity(n_envs);
        let mut configs = Vec::with_capacity(n_envs);

        for i in 0..n_envs {
            let config = SimConfig {
                drones_per_side,
                targets_per_side,
                world_size,
                max_ticks,
                seed: i as u64,
                dt: 0.05,
                speed_multiplier,
                randomize_opponent: true,
                skip_orca,
            };
            envs.push(SimRunner::new(config.clone()));
            configs.push(config);
        }

        VecSimRunner {
            drone_ids_a: vec![Vec::new(); n_envs],
            next_seed: n_envs as u64,
            envs,
            configs,
        }
    }

    /// Reset all environments and return batched observations.
    ///
    /// Returns a dict with numpy arrays:
    ///   - "ego_obs":      [total_drones, EGO_DIM] f32
    ///   - "entity_obs":   [total_drones, MAX_ENTITIES, ENTITY_DIM] f32
    ///   - "n_entities":   [total_drones] i32
    ///   - "drone_ids":    [total_drones] i32
    ///   - "env_indices":  [total_drones] i32
    #[pyo3(signature = (seeds=None))]
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
        seeds: Option<Vec<u64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n_envs = self.envs.len();
        let seeds: Vec<u64> = seeds.unwrap_or_else(|| (0..n_envs as u64).collect());

        let all_obs: Vec<(Vec<DroneObsV2>, Vec<usize>)> = self
            .envs
            .iter_mut()
            .zip(seeds.iter())
            .map(|(env, &seed)| {
                let ((obs_a, ids_a), _) = env.reset_selfplay_with_seed(seed);
                (obs_a, ids_a)
            })
            .collect();

        for (i, (_, ids)) in all_obs.iter().enumerate() {
            self.drone_ids_a[i] = ids.clone();
        }

        let flat = flatten_obs(&all_obs);
        build_obs_dict(py, &flat)
    }

    /// Step all environments with the given actions.
    ///
    /// `actions` is a flat i32 array of length = total alive Group A drones
    /// (same ordering as the last observation output).
    ///
    /// Auto-resets terminated/truncated environments.
    ///
    /// Returns a dict with:
    ///   - All observation arrays (post-reset for terminated envs)
    ///   - "rewards":      [prev_total_drones] f32
    ///   - "team_rewards": [prev_total_drones] f32
    ///   - "dones":        [prev_total_drones] bool
    ///   - "truncated":    [prev_total_drones] bool
    ///   - "drone_died":   [prev_total_drones] bool
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: numpy::PyReadonlyArray1<'py, i32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let actions_slice = actions.as_slice()?;

        let expected_len: usize = self.drone_ids_a.iter().map(|ids| ids.len()).sum();
        if actions_slice.len() != expected_len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "actions length {} does not match expected {} (total alive drones)",
                actions_slice.len(),
                expected_len,
            )));
        }

        // Map flat action array back to per-env (drone_id, action) pairs.
        let n_envs = self.envs.len();
        let mut per_env_actions: Vec<Vec<(usize, u32)>> = Vec::with_capacity(n_envs);
        let mut action_offset = 0usize;
        for ids in &self.drone_ids_a {
            let mut env_actions = Vec::with_capacity(ids.len());
            for &drone_id in ids {
                let action = actions_slice[action_offset] as u32;
                env_actions.push((drone_id, action));
                action_offset += 1;
            }
            per_env_actions.push(env_actions);
        }

        // Step all envs in parallel (release GIL).
        let results: Vec<SelfPlayStepResult> = py.allow_threads(|| {
            self.envs
                .par_iter_mut()
                .zip(per_env_actions.par_iter())
                .map(|(env, actions)| env.step_rl_vs_doctrine_v2(actions))
                .collect()
        });

        // Build reward/done arrays from results (indexed by previous drone ordering).
        let mut rewards_flat = Vec::with_capacity(expected_len);
        let mut team_rewards_flat = Vec::with_capacity(expected_len);
        let mut dones_flat = Vec::with_capacity(expected_len);
        let mut truncated_flat = Vec::with_capacity(expected_len);
        let mut drone_died_flat = Vec::with_capacity(expected_len);

        for (env_idx, result) in results.iter().enumerate() {
            let team_reward = result.reward;
            for &drone_id in &self.drone_ids_a[env_idx] {
                let individual = result
                    .individual_rewards_a
                    .get(&drone_id)
                    .copied()
                    .unwrap_or(0.0);
                rewards_flat.push(team_reward + individual);
                team_rewards_flat.push(team_reward);
                dones_flat.push(result.terminated || result.truncated);
                truncated_flat.push(result.truncated);
                drone_died_flat.push(result.drone_deaths_a.contains(&drone_id));
            }
        }

        // Auto-reset terminated/truncated envs and collect new observations.
        let mut all_obs: Vec<(Vec<DroneObsV2>, Vec<usize>)> = Vec::with_capacity(n_envs);

        for (env_idx, result) in results.iter().enumerate() {
            if result.terminated || result.truncated {
                let seed = self.next_seed;
                self.next_seed += 1;
                let ((obs_a, ids_a), _) =
                    self.envs[env_idx].reset_selfplay_with_seed(seed);
                all_obs.push((obs_a, ids_a));
            } else {
                all_obs.push((result.obs_a.clone(), result.drone_ids_a.clone()));
            }
        }

        for (i, (_, ids)) in all_obs.iter().enumerate() {
            self.drone_ids_a[i] = ids.clone();
        }

        let flat = flatten_obs(&all_obs);
        let dict = build_obs_dict(py, &flat)?;

        dict.set_item("rewards", PyArray1::from_slice(py, &rewards_flat))?;
        dict.set_item("team_rewards", PyArray1::from_slice(py, &team_rewards_flat))?;
        dict.set_item("dones", PyArray1::from_slice(py, &dones_flat))?;
        dict.set_item("truncated", PyArray1::from_slice(py, &truncated_flat))?;
        dict.set_item("drone_died", PyArray1::from_slice(py, &drone_died_flat))?;

        Ok(dict)
    }

    /// Reconfigure specific environments for curriculum changes.
    ///
    /// Updates the config for the given env indices and resets them.
    #[pyo3(signature = (env_indices, drones_per_side, targets_per_side, world_size))]
    fn reconfigure(
        &mut self,
        _py: Python<'_>,
        env_indices: Vec<usize>,
        drones_per_side: u32,
        targets_per_side: u32,
        world_size: f32,
    ) {
        let n_envs = self.envs.len();
        for &idx in &env_indices {
            if idx >= n_envs {
                continue;
            }
            self.configs[idx].drones_per_side = drones_per_side;
            self.configs[idx].targets_per_side = targets_per_side;
            self.configs[idx].world_size = world_size;

            self.envs[idx] = SimRunner::new(self.configs[idx].clone());
            let ((_, ids_a), _) = self.envs[idx].reset_selfplay_with_seed(idx as u64);
            self.drone_ids_a[idx] = ids_a;
        }
    }

    /// Number of environments.
    #[getter]
    fn n_envs(&self) -> usize {
        self.envs.len()
    }

    /// Total number of alive Group A drones across all envs.
    #[getter]
    fn total_drones(&self) -> usize {
        self.drone_ids_a.iter().map(|ids| ids.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Helper: build observation dict from flat arrays
// ---------------------------------------------------------------------------

fn build_obs_dict<'py>(
    py: Python<'py>,
    flat: &FlatObs,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let n = flat.total_drones;

    let ego_array = PyArray1::from_slice(py, &flat.ego)
        .reshape([n, EGO_DIM])?;
    dict.set_item("ego_obs", ego_array)?;

    let ent_array = PyArray1::from_slice(py, &flat.entities)
        .reshape([n, MAX_ENTITIES, ENTITY_DIM])?;
    dict.set_item("entity_obs", ent_array)?;

    dict.set_item("n_entities", PyArray1::from_slice(py, &flat.n_entities))?;
    dict.set_item("drone_ids", PyArray1::from_slice(py, &flat.drone_ids))?;
    dict.set_item("env_indices", PyArray1::from_slice(py, &flat.env_indices))?;

    Ok(dict)
}

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------

/// droneswarm_env Python module.
#[pymodule]
fn droneswarm_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecSimRunner>()?;

    // Export observation layout constants for Python side.
    m.add("EGO_DIM", EGO_DIM)?;
    m.add("ENTITY_DIM", ENTITY_DIM)?;
    m.add("MAX_ENTITIES", MAX_ENTITIES)?;
    m.add("ACT_DIM", ACT_DIM)?;

    Ok(())
}
