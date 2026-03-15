//! Multi-agent self-play PPO training with entity-attention policy.
//!
//! Architecture: EgoEncoder + EntityEncoder + MultiHeadAttention + Trunk MLP.
//! Features: running obs normalization, checkpoint pool, smooth curriculum.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use drone_lib::sim_runner::{
    GameResult, SimConfig, SimRunner, DroneObsV2,
    EGO_DIM, ENTITY_DIM, ACT_DIM,
};
use rl_train::network::policy_v2::PolicyNetV2;
use rl_train::network::critic::CentralizedCritic;
use rl_train::network::mixer::QMIXMixer;
use rl_train::network::layer::matmul_bias_relu;
use rl_train::ppo::{compute_gae, Transition};
use rl_train::ppo::backward::PolicyNetV2PPOExt;
use rl_train::normalize::RunningMeanStd;
use rl_train::checkpoint_pool::CheckpointPool;
use rl_train::curriculum::curriculum_config_smooth;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

const EMBED_DIM: usize = 64;
const CHECKPOINT_POOL_SIZE: usize = 20;
const CHECKPOINT_SAVE_FREQ: usize = 5;
const MAX_AGENTS: usize = 24;
const QMIX_COEF: f32 = 0.5;
const MAPPO_VF_COEF: f32 = 0.5;
/// Global state dim for QMIX = hidden_dim + 8 global obs features.
const GLOBAL_OBS_FEATURES: usize = 8;

/// PPO training config.
struct TrainConfig {
    total_timesteps: u64,
    n_steps: usize,
    n_epochs: usize,
    batch_size: usize,
    lr: f32,
    gamma: f32,
    lam: f32,
    clip_range: f32,
    ent_coef_start: f32,
    ent_coef_end: f32,
    vf_coef: f32,
    max_grad_norm: f32,
    n_envs: usize,
    hidden_size: usize,
    drones_per_side: u32,
    targets_per_side: u32,
    max_ticks: u32,
    eval_freq: usize,
    eval_episodes: usize,
    opponent_update_freq: usize,
    output_dir: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            total_timesteps: 50_000_000,
            n_steps: 512,
            n_epochs: 4,
            batch_size: 4096,
            lr: 1e-4,
            gamma: 0.99,
            lam: 0.95,
            clip_range: 0.2,
            ent_coef_start: 0.02,
            ent_coef_end: 0.001,
            vf_coef: 0.5,
            max_grad_norm: 0.5,
            n_envs: 1024,
            hidden_size: 256,
            drones_per_side: 24,
            targets_per_side: 6,
            max_ticks: 6000,
            eval_freq: 50,
            eval_episodes: 50,
            opponent_update_freq: 30,
            output_dir: "results".to_string(),
        }
    }
}

fn make_sim_config(tc: &TrainConfig) -> SimConfig {
    SimConfig {
        drones_per_side: tc.drones_per_side,
        targets_per_side: tc.targets_per_side,
        max_ticks: tc.max_ticks,
        seed: 0,
        randomize_opponent: false,
        ..Default::default()
    }
}

/// Normalize a DroneObsV2 in-place using running normalizers.
fn normalize_obs(obs: &DroneObsV2, ego_norm: &RunningMeanStd, ent_norm: &RunningMeanStd) -> DroneObsV2 {
    let ego = ego_norm.normalize(&obs.ego);
    let mut entities = Vec::with_capacity(obs.entities.len());
    for i in 0..obs.n_entities {
        let start = i * ENTITY_DIM;
        let end = start + ENTITY_DIM;
        entities.extend_from_slice(&ent_norm.normalize(&obs.entities[start..end]));
    }
    DroneObsV2 { ego, entities, n_entities: obs.n_entities }
}

/// Run self-play evaluation.
fn evaluate(policy: &PolicyNetV2, ego_norm: &RunningMeanStd, ent_norm: &RunningMeanStd, config: &TrainConfig, base_seed: u64) -> (f32, f32, f32) {
    let results: Vec<(u32, f32, u64)> = (0..config.eval_episodes)
        .into_par_iter()
        .map(|ep| {
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed + ep as u64);
            let sim_config = make_sim_config(config);
            let mut env = SimRunner::new(sim_config);
            let ((mut obs_a, mut ids_a), (mut obs_b, mut ids_b)) =
                env.reset_selfplay_with_seed(10000 + ep as u64);
            let mut ep_reward = 0.0f32;
            let mut ep_len = 0u64;
            let mut win = 0u32;

            loop {
                // Normalize and build batched inputs.
                let norm_a: Vec<DroneObsV2> = obs_a.iter().map(|o| normalize_obs(o, ego_norm, ent_norm)).collect();
                let egos_a: Vec<&[f32]> = norm_a.iter().map(|o| o.ego.as_slice()).collect();
                let ents_a: Vec<&[f32]> = norm_a.iter().map(|o| o.entities.as_slice()).collect();
                let nents_a: Vec<usize> = norm_a.iter().map(|o| o.n_entities).collect();
                let results_a = policy.act_batch(&egos_a, &ents_a, &nents_a, &mut rng);
                let actions_a: Vec<(usize, u32)> = ids_a.iter()
                    .zip(results_a.iter())
                    .map(|(&id, &(act, _, _))| (id, act))
                    .collect();

                let norm_b: Vec<DroneObsV2> = obs_b.iter().map(|o| normalize_obs(o, ego_norm, ent_norm)).collect();
                let egos_b: Vec<&[f32]> = norm_b.iter().map(|o| o.ego.as_slice()).collect();
                let ents_b: Vec<&[f32]> = norm_b.iter().map(|o| o.entities.as_slice()).collect();
                let nents_b: Vec<usize> = norm_b.iter().map(|o| o.n_entities).collect();
                let results_b = policy.act_batch(&egos_b, &ents_b, &nents_b, &mut rng);
                let actions_b: Vec<(usize, u32)> = ids_b.iter()
                    .zip(results_b.iter())
                    .map(|(&id, &(act, _, _))| (id, act))
                    .collect();

                let result = env.step_multi_selfplay(&actions_a, &actions_b);
                ep_reward += result.reward;
                ep_len += 1;

                if result.game_result == GameResult::AWins { win = 1; }
                if result.terminated || result.truncated { break; }

                obs_a = result.obs_a;
                ids_a = result.drone_ids_a;
                obs_b = result.obs_b;
                ids_b = result.drone_ids_b;
            }
            (win, ep_reward, ep_len)
        })
        .collect();

    let wins: u32 = results.iter().map(|(w, _, _)| w).sum();
    let total_reward: f64 = results.iter().map(|(_, r, _)| *r as f64).sum();
    let total_length: u64 = results.iter().map(|(_, _, l)| l).sum();
    let n = config.eval_episodes as f32;
    (wins as f32 / n, total_reward as f32 / n, total_length as f32 / n)
}

/// Per-env state for self-play rollout collection.
struct EnvState {
    obs_a: Vec<DroneObsV2>,
    drone_ids_a: Vec<usize>,
    obs_b: Vec<DroneObsV2>,
    drone_ids_b: Vec<usize>,
}

/// Per-env action mapping into the flat batch.
struct EnvActionSlice {
    a_start: usize,
    a_count: usize,
    b_start: usize,
    b_count: usize,
}

fn main() {
    let config = parse_args();

    let num_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .expect("Failed to configure rayon thread pool");

    std::fs::create_dir_all(&config.output_dir).ok();
    let best_model_path = format!("{}/best_model.json", config.output_dir);
    let final_model_path = format!("{}/final_model.json", config.output_dir);

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // PolicyNetV2: entity-attention architecture.
    let mut policy = PolicyNetV2::new(EGO_DIM, ENTITY_DIM, ACT_DIM, EMBED_DIM, config.hidden_size, &mut rng);
    let mut opponent = policy.clone();

    // MAPPO centralized critic + QMIX mixer.
    let global_state_dim = config.hidden_size + GLOBAL_OBS_FEATURES;
    let mut critic = CentralizedCritic::new(config.hidden_size, &mut rng);
    let mut mixer = QMIXMixer::new(global_state_dim, MAX_AGENTS, &mut rng);

    // Running observation normalizers (Welford's online algorithm).
    let mut ego_normalizer = RunningMeanStd::new(EGO_DIM);
    let mut entity_normalizer = RunningMeanStd::new(ENTITY_DIM);

    // Checkpoint pool for diverse self-play opponents.
    let mut checkpoint_pool = CheckpointPool::new(CHECKPOINT_POOL_SIZE);

    // Curriculum init (smooth blending).
    let (init_drones, init_targets) = curriculum_config_smooth(0.0, 0, config.drones_per_side, config.targets_per_side);

    let mut envs: Vec<SimRunner> = (0..config.n_envs)
        .map(|i| {
            let mut sc = make_sim_config(&config);
            sc.drones_per_side = init_drones;
            sc.targets_per_side = init_targets;
            sc.seed = i as u64;
            SimRunner::new(sc)
        })
        .collect();

    let mut env_states: Vec<EnvState> = envs.iter().map(|e| {
        let (obs_a, drone_ids_a) = e.observe_multi_group_v2(0);
        let (obs_b, drone_ids_b) = e.observe_multi_group_v2(1);
        EnvState { obs_a, drone_ids_a, obs_b, drone_ids_b }
    }).collect();

    let mut total_steps = 0u64;
    let mut update_count = 0u64;
    let mut adam_step_count = 0usize;
    let mut best_win_rate = 0.0f32;
    let mut ep_rewards: Vec<f32> = Vec::new();
    let mut ep_reward_accum = vec![0.0f32; config.n_envs];
    let mut curr_drones = init_drones;
    let mut curr_targets = init_targets;

    let start_time = Instant::now();
    let n_threads = rayon::current_num_threads();

    println!("Starting self-play PPO training (V2 entity-attention)");
    println!("  Ego dim: {}, Entity dim: {}, Embed dim: {}", EGO_DIM, ENTITY_DIM, EMBED_DIM);
    println!("  Act dim: {}, Hidden: {}, Attn heads: 4", ACT_DIM, config.hidden_size);
    println!("  Total timesteps: {}", config.total_timesteps);
    println!("  Envs: {}, Steps/rollout: {}, Epochs: {}, Batch: {}",
        config.n_envs, config.n_steps, config.n_epochs, config.batch_size);
    println!("  LR: {} (annealed), Gamma: {}, Clip: {}", config.lr, config.gamma, config.clip_range);
    println!("  Entropy coef: {} → {}", config.ent_coef_start, config.ent_coef_end);
    println!("  Checkpoint pool size: {}, Opponent update freq: {}", CHECKPOINT_POOL_SIZE, config.opponent_update_freq);
    println!("  Smooth curriculum: 4v4 → {}v{}", config.drones_per_side, config.drones_per_side);
    println!("  CPUs: {}, Rayon: {}, Accelerate: {}", num_cpus, n_threads, cfg!(target_os = "macos"));
    println!();

    while total_steps < config.total_timesteps {
        // Smooth curriculum: per-env stage with blended transitions.
        let progress = (total_steps as f32 / config.total_timesteps as f32).min(1.0);

        // Check if any env needs a curriculum update (sample env 0 as proxy).
        let (new_drones, new_targets) = curriculum_config_smooth(
            progress, 0, config.drones_per_side, config.targets_per_side,
        );
        if new_drones != curr_drones || new_targets != curr_targets {
            println!("  Curriculum: {}v{} → {}v{} at {:.1}%",
                curr_drones, curr_drones, new_drones, new_drones, progress * 100.0);
            curr_drones = new_drones;
            curr_targets = new_targets;
            // Recreate envs with per-env smooth curriculum.
            for i in 0..config.n_envs {
                let (d, t) = curriculum_config_smooth(
                    progress, i as u64, config.drones_per_side, config.targets_per_side,
                );
                let mut sc = make_sim_config(&config);
                sc.drones_per_side = d;
                sc.targets_per_side = t;
                sc.seed = total_steps + i as u64;
                envs[i] = SimRunner::new(sc);
                let (obs_a, drone_ids_a) = envs[i].observe_multi_group_v2(0);
                let (obs_b, drone_ids_b) = envs[i].observe_multi_group_v2(1);
                env_states[i] = EnvState { obs_a, drone_ids_a, obs_b, drone_ids_b };
                ep_reward_accum[i] = 0.0;
            }
        }

        // ================================================================
        // Collect rollout
        // ================================================================
        let est_transitions = config.n_steps * config.n_envs * curr_drones as usize;
        let mut rollout: Vec<Transition> = Vec::with_capacity(est_transitions);
        let mut rollout_env_indices: Vec<usize> = Vec::with_capacity(est_transitions);

        // Buffers for raw obs (for normalizer update).
        let mut raw_ego_batch: Vec<f32> = Vec::with_capacity(est_transitions * EGO_DIM);
        let mut raw_ent_batch: Vec<f32> = Vec::new();
        let mut raw_ent_count: usize = 0;

        // Cached h2 and local_value per transition (avoids re-forwarding in critic/mixer).
        let mut cached_h2: Vec<Vec<f32>> = Vec::with_capacity(est_transitions);
        let mut cached_local_values: Vec<f32> = Vec::with_capacity(est_transitions);

        // Buffers for batched inference.
        let mut ego_refs_a: Vec<Vec<f32>> = Vec::new();
        let mut ent_refs_a: Vec<Vec<f32>> = Vec::new();
        let mut nent_a: Vec<usize> = Vec::new();
        let mut ego_refs_b: Vec<Vec<f32>> = Vec::new();
        let mut ent_refs_b: Vec<Vec<f32>> = Vec::new();
        let mut nent_b: Vec<usize> = Vec::new();
        let mut env_slices: Vec<EnvActionSlice> = Vec::with_capacity(config.n_envs);
        let mut flat_ids_a: Vec<usize> = Vec::new();
        let mut flat_ids_b: Vec<usize> = Vec::new();

        for _step in 0..config.n_steps {
            // --- 1. Normalize obs in parallel, then flatten ---
            // Collect all raw obs for normalizer (sequential but fast — just memcpy).
            for env_st in &env_states {
                for obs in &env_st.obs_a {
                    raw_ego_batch.extend_from_slice(&obs.ego);
                    for i in 0..obs.n_entities {
                        raw_ent_batch.extend_from_slice(&obs.entities[i * ENTITY_DIM..(i + 1) * ENTITY_DIM]);
                        raw_ent_count += 1;
                    }
                }
            }

            // Normalize all group A obs in parallel.
            let all_obs_a: Vec<&DroneObsV2> = env_states.iter()
                .flat_map(|es| es.obs_a.iter())
                .collect();
            let norm_a: Vec<DroneObsV2> = all_obs_a.par_iter()
                .map(|obs| normalize_obs(obs, &ego_normalizer, &entity_normalizer))
                .collect();

            // Normalize all group B obs in parallel.
            let all_obs_b: Vec<&DroneObsV2> = env_states.iter()
                .flat_map(|es| es.obs_b.iter())
                .collect();
            let norm_b: Vec<DroneObsV2> = all_obs_b.par_iter()
                .map(|obs| normalize_obs(obs, &ego_normalizer, &entity_normalizer))
                .collect();

            // Build flat buffers and slices from normalized obs.
            ego_refs_a.clear(); ent_refs_a.clear(); nent_a.clear();
            ego_refs_b.clear(); ent_refs_b.clear(); nent_b.clear();
            env_slices.clear(); flat_ids_a.clear(); flat_ids_b.clear();

            let mut a_idx = 0usize;
            let mut b_idx = 0usize;
            for env_st in &env_states {
                let a_start = ego_refs_a.len();
                for _ in 0..env_st.obs_a.len() {
                    ego_refs_a.push(norm_a[a_idx].ego.clone());
                    ent_refs_a.push(norm_a[a_idx].entities.clone());
                    nent_a.push(norm_a[a_idx].n_entities);
                    a_idx += 1;
                }
                flat_ids_a.extend_from_slice(&env_st.drone_ids_a);

                let b_start = ego_refs_b.len();
                for _ in 0..env_st.obs_b.len() {
                    ego_refs_b.push(norm_b[b_idx].ego.clone());
                    ent_refs_b.push(norm_b[b_idx].entities.clone());
                    nent_b.push(norm_b[b_idx].n_entities);
                    b_idx += 1;
                }
                flat_ids_b.extend_from_slice(&env_st.drone_ids_b);

                env_slices.push(EnvActionSlice {
                    a_start,
                    a_count: env_st.obs_a.len(),
                    b_start,
                    b_count: env_st.obs_b.len(),
                });
            }

            // --- 2. Batched forward pass (with h2 for MAPPO/QMIX) ---
            let egos_a_sl: Vec<&[f32]> = ego_refs_a.iter().map(|e| e.as_slice()).collect();
            let ents_a_sl: Vec<&[f32]> = ent_refs_a.iter().map(|e| e.as_slice()).collect();
            let (batch_a, h2_flat_a) = policy.act_batch_with_h2(&egos_a_sl, &ents_a_sl, &nent_a, &mut rng);

            let egos_b_sl: Vec<&[f32]> = ego_refs_b.iter().map(|e| e.as_slice()).collect();
            let ents_b_sl: Vec<&[f32]> = ent_refs_b.iter().map(|e| e.as_slice()).collect();
            let batch_b = opponent.act_batch(&egos_b_sl, &ents_b_sl, &nent_b, &mut rng);

            // Compute centralized values via batched critic forward.
            // Build flat input: concat(h2_i, mean_pool_env) for each drone.
            let hidden = config.hidden_size;
            let total_a = batch_a.len();
            let critic_in_dim = 2 * hidden;
            let mut critic_input = vec![0.0f32; total_a * critic_in_dim];

            // Compute per-env mean_pool in parallel, then build critic input.
            let mean_pools: Vec<Vec<f32>> = env_slices.par_iter().map(|slice| {
                if slice.a_count == 0 { return vec![0.0f32; hidden]; }
                let mut pool = vec![0.0f32; hidden];
                for i in 0..slice.a_count {
                    let idx = slice.a_start + i;
                    for j in 0..hidden { pool[j] += h2_flat_a[idx * hidden + j]; }
                }
                let inv = 1.0 / slice.a_count as f32;
                for j in 0..hidden { pool[j] *= inv; }
                pool
            }).collect();

            for (env_idx, slice) in env_slices.iter().enumerate() {
                for i in 0..slice.a_count {
                    let idx = slice.a_start + i;
                    let dst = idx * critic_in_dim;
                    critic_input[dst..dst + hidden].copy_from_slice(
                        &h2_flat_a[idx * hidden..(idx + 1) * hidden]);
                    critic_input[dst + hidden..dst + critic_in_dim].copy_from_slice(
                        &mean_pools[env_idx]);
                }
            }

            // Batched critic forward: 3 matmuls instead of N individual forwards.
            let c_h1 = matmul_bias_relu(&critic_input, &critic.fc1.weights, &critic.fc1.biases,
                total_a, critic.fc1.out_dim, critic_in_dim, true);
            let c_h2 = matmul_bias_relu(&c_h1, &critic.fc2.weights, &critic.fc2.biases,
                total_a, critic.fc2.out_dim, critic.fc2.in_dim, true);
            let centralized_values_flat = matmul_bias_relu(&c_h2, &critic.value_head.weights, &critic.value_head.biases,
                total_a, 1, critic.value_head.in_dim, false);
            let centralized_values = centralized_values_flat;

            // --- 3. Step envs in parallel ---
            let per_env_results: Vec<_> = envs.par_iter_mut()
                .zip(env_slices.par_iter())
                .map(|(env, slice)| {
                    let actions_a: Vec<(usize, u32)> = (0..slice.a_count)
                        .map(|i| (flat_ids_a[slice.a_start + i], batch_a[slice.a_start + i].0))
                        .collect();
                    let actions_b: Vec<(usize, u32)> = (0..slice.b_count)
                        .map(|i| (flat_ids_b[slice.b_start + i], batch_b[slice.b_start + i].0))
                        .collect();
                    env.step_multi_selfplay(&actions_a, &actions_b)
                })
                .collect();

            // --- 4. Collect transitions + handle resets ---
            for env_idx in 0..config.n_envs {
                let result = &per_env_results[env_idx];
                let slice = &env_slices[env_idx];
                ep_reward_accum[env_idx] += result.reward;
                let done = result.terminated || result.truncated;

                for i in 0..slice.a_count {
                    let batch_idx = slice.a_start + i;
                    let (action, log_prob, local_value) = batch_a[batch_idx];
                    let drone_id = flat_ids_a[batch_idx];
                    let individual_reward = result.individual_rewards_a.get(&drone_id).copied().unwrap_or(0.0);
                    let drone_died = result.drone_deaths_a.contains(&drone_id);
                    // Use centralized value for GAE (better value estimates).
                    let value = centralized_values[batch_idx];

                    // Cache h2 and local_value for critic/mixer update (avoid re-forwarding).
                    let h2_start = batch_idx * hidden;
                    cached_h2.push(h2_flat_a[h2_start..h2_start + hidden].to_vec());
                    cached_local_values.push(local_value);

                    rollout.push(Transition {
                        ego_obs: ego_refs_a[batch_idx].clone(),
                        entity_obs: ent_refs_a[batch_idx].clone(),
                        n_entities: nent_a[batch_idx],
                        action,
                        reward: result.reward + individual_reward,
                        value,
                        log_prob,
                        done,
                        drone_died,
                        team_value_at_death: if drone_died { value } else { 0.0 },
                    });
                    rollout_env_indices.push(env_idx);
                }

                if done {
                    ep_rewards.push(ep_reward_accum[env_idx]);
                    ep_reward_accum[env_idx] = 0.0;
                    let new_seed = total_steps + env_idx as u64;

                    // Per-env smooth curriculum on reset.
                    let (d, t) = curriculum_config_smooth(
                        progress, new_seed, config.drones_per_side, config.targets_per_side,
                    );
                    let env_cfg = envs[env_idx].config();
                    if env_cfg.drones_per_side != d || env_cfg.targets_per_side != t {
                        let mut sc = make_sim_config(&config);
                        sc.drones_per_side = d;
                        sc.targets_per_side = t;
                        sc.seed = new_seed;
                        envs[env_idx] = SimRunner::new(sc);
                    }
                    let ((obs_a, drone_ids_a), (obs_b, drone_ids_b)) =
                        envs[env_idx].reset_selfplay_with_seed(new_seed);
                    env_states[env_idx] = EnvState { obs_a, drone_ids_a, obs_b, drone_ids_b };
                } else {
                    env_states[env_idx] = EnvState {
                        obs_a: result.obs_a.clone(),
                        drone_ids_a: result.drone_ids_a.clone(),
                        obs_b: result.obs_b.clone(),
                        drone_ids_b: result.drone_ids_b.clone(),
                    };
                }
                total_steps += 1;
            }
        }

        if rollout.is_empty() { continue; }

        // Update running normalizers with this rollout's raw observations.
        let ego_batch_size = raw_ego_batch.len() / EGO_DIM;
        if ego_batch_size > 0 {
            ego_normalizer.update_batch(&raw_ego_batch, ego_batch_size);
        }
        if raw_ent_count > 0 {
            entity_normalizer.update_batch(&raw_ent_batch, raw_ent_count);
        }
        raw_ego_batch.clear();
        raw_ent_batch.clear();
        raw_ent_count = 0;

        // Compute last values for GAE using centralized critic.
        let hidden = config.hidden_size;
        let mut last_values = vec![0.0f32; config.n_envs];
        {
            let mut boot_egos: Vec<Vec<f32>> = Vec::new();
            let mut boot_ents: Vec<Vec<f32>> = Vec::new();
            let mut boot_nents: Vec<usize> = Vec::new();
            let mut boot_map: Vec<(usize, usize)> = Vec::new();
            for (env_idx, env_st) in env_states.iter().enumerate() {
                if env_st.obs_a.is_empty() { continue; }
                let start = boot_egos.len();
                for obs in &env_st.obs_a {
                    let norm = normalize_obs(obs, &ego_normalizer, &entity_normalizer);
                    boot_egos.push(norm.ego);
                    boot_ents.push(norm.entities);
                    boot_nents.push(norm.n_entities);
                }
                boot_map.push((env_idx, boot_egos.len() - start));
            }
            if !boot_egos.is_empty() {
                let e_sl: Vec<&[f32]> = boot_egos.iter().map(|e| e.as_slice()).collect();
                let ent_sl: Vec<&[f32]> = boot_ents.iter().map(|e| e.as_slice()).collect();
                let (results, h2_boot) = policy.act_batch_with_h2(&e_sl, &ent_sl, &boot_nents, &mut rng);
                let mut offset = 0;
                for &(env_idx, count) in &boot_map {
                    // Compute mean pool of h2 for this env's bootstrap obs.
                    let mut mean_pool = vec![0.0f32; hidden];
                    for i in 0..count {
                        for j in 0..hidden { mean_pool[j] += h2_boot[(offset + i) * hidden + j]; }
                    }
                    let inv_n = 1.0 / count as f32;
                    for j in 0..hidden { mean_pool[j] *= inv_n; }
                    // Centralized value for each drone, averaged.
                    let mut sum = 0.0f32;
                    for i in 0..count {
                        let h2_i = &h2_boot[(offset + i) * hidden..(offset + i + 1) * hidden];
                        sum += critic.forward(h2_i, &mean_pool);
                    }
                    last_values[env_idx] = sum / count as f32;
                    offset += count;
                }
            }
        }

        // Split rollout by env for GAE.
        let mut env_rollouts: Vec<Vec<(usize, &Transition)>> = vec![Vec::new(); config.n_envs];
        for (i, (trans, &env_idx)) in rollout.iter().zip(rollout_env_indices.iter()).enumerate() {
            env_rollouts[env_idx].push((i, trans));
        }
        let mut advantages = vec![0.0f32; rollout.len()];
        let mut returns = vec![0.0f32; rollout.len()];
        for env_idx in 0..config.n_envs {
            let env_trans: Vec<Transition> = env_rollouts[env_idx].iter().map(|(_, t)| (*t).clone()).collect();
            if env_trans.is_empty() { continue; }
            let (adv, ret) = compute_gae(&env_trans, last_values[env_idx], config.gamma, config.lam);
            for (j, &(global_idx, _)) in env_rollouts[env_idx].iter().enumerate() {
                advantages[global_idx] = adv[j];
                returns[global_idx] = ret[j];
            }
        }

        // Normalize advantages.
        let n = rollout.len();
        let adv_mean: f32 = advantages.iter().sum::<f32>() / n as f32;
        let adv_var: f32 = advantages.iter().map(|a| (a - adv_mean).powi(2)).sum::<f32>() / n as f32;
        let adv_std = adv_var.sqrt().max(1e-8);
        for a in &mut advantages { *a = (*a - adv_mean) / adv_std; }

        // Anneal LR and entropy.
        let progress = (total_steps as f32 / config.total_timesteps as f32).min(1.0);
        let current_lr = config.lr * (1.0 - progress);
        let current_ent_coef = config.ent_coef_start + (config.ent_coef_end - config.ent_coef_start) * progress;

        // PPO update with parallel backward.
        let chunk_size = (config.batch_size + n_threads - 1) / n_threads;
        let mut thread_policies: Vec<PolicyNetV2> = (0..n_threads).map(|_| policy.clone()).collect();

        for _epoch in 0..config.n_epochs {
            let mut indices: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }

            for batch_start in (0..n).step_by(config.batch_size) {
                let batch_end = (batch_start + config.batch_size).min(n);
                let batch_size = batch_end - batch_start;
                let batch_indices = &indices[batch_start..batch_end];
                let chunks: Vec<&[usize]> = batch_indices.chunks(chunk_size).collect();
                let n_chunks = chunks.len();

                for tp in &mut thread_policies[..n_chunks] {
                    tp.copy_weights_from(&policy);
                }

                thread_policies[..n_chunks]
                    .par_iter_mut()
                    .zip(chunks.into_par_iter())
                    .for_each(|(local, chunk)| {
                        local.zero_grad();
                        for &idx in chunk {
                            let t = &rollout[idx];
                            local.backward_ppo_v2(
                                &t.ego_obs,
                                &t.entity_obs,
                                t.n_entities,
                                t.action,
                                advantages[idx],
                                returns[idx],
                                t.log_prob,
                                config.clip_range,
                                config.vf_coef,
                                current_ent_coef,
                            );
                        }
                    });

                policy.zero_grad();
                for tp in &thread_policies[..n_chunks] {
                    policy.add_grads_from(tp);
                }
                policy.clip_grad_norm(config.max_grad_norm);
                adam_step_count += 1;
                policy.adam_step(current_lr, batch_size, 0.9, 0.999, 1e-8, 5e-4, adam_step_count);
            }
        }

        // ================================================================
        // MAPPO Critic + QMIX Mixer update (using cached h2/local_values)
        // ================================================================
        {
            critic.zero_grad();
            mixer.zero_grad();
            let mut critic_samples = 0usize;
            let mut mixer_samples = 0usize;

            // Batched critic update using cached h2 (no re-forwarding).
            let critic_in_dim = 2 * hidden;
            let n_total = rollout.len();
            if n_total > 0 {
                // Build per-env mean_pools from cached h2.
                let mut env_mean_pools: Vec<Vec<f32>> = vec![vec![0.0f32; hidden]; config.n_envs];
                let mut env_counts: Vec<usize> = vec![0; config.n_envs];
                for (i, &env_idx) in rollout_env_indices.iter().enumerate() {
                    for j in 0..hidden { env_mean_pools[env_idx][j] += cached_h2[i][j]; }
                    env_counts[env_idx] += 1;
                }
                for env_idx in 0..config.n_envs {
                    if env_counts[env_idx] > 0 {
                        let inv = 1.0 / env_counts[env_idx] as f32;
                        for j in 0..hidden { env_mean_pools[env_idx][j] *= inv; }
                    }
                }

                // Build flat critic input: concat(h2_i, mean_pool_env) for all transitions.
                let mut c_input = vec![0.0f32; n_total * critic_in_dim];
                for (i, &env_idx) in rollout_env_indices.iter().enumerate() {
                    let dst = i * critic_in_dim;
                    c_input[dst..dst + hidden].copy_from_slice(&cached_h2[i]);
                    c_input[dst + hidden..dst + critic_in_dim].copy_from_slice(&env_mean_pools[env_idx]);
                }

                // Batched forward through critic.
                let ch1 = matmul_bias_relu(&c_input, &critic.fc1.weights, &critic.fc1.biases,
                    n_total, critic.fc1.out_dim, critic_in_dim, true);
                let ch2 = matmul_bias_relu(&ch1, &critic.fc2.weights, &critic.fc2.biases,
                    n_total, critic.fc2.out_dim, critic.fc2.in_dim, true);
                let c_vals = matmul_bias_relu(&ch2, &critic.value_head.weights, &critic.value_head.biases,
                    n_total, 1, critic.value_head.in_dim, false);

                // Subsample for critic backward (full rollout is too large).
                // Use at most batch_size samples for gradient computation.
                let critic_batch = n_total.min(config.batch_size);
                let stride = n_total / critic_batch.max(1);
                for k in 0..critic_batch {
                    let i = (k * stride) % n_total;
                    let d = MAPPO_VF_COEF * 2.0 * (c_vals[i] - returns[i]);
                    critic.forward(&cached_h2[i], &env_mean_pools[rollout_env_indices[i]]);
                    critic.backward(d);
                }
                critic_samples = critic_batch;
            }

            // QMIX update: group by env, use cached local values.
            let mut step_groups: HashMap<usize, Vec<usize>> = HashMap::new();
            for (i, &env_idx) in rollout_env_indices.iter().enumerate() {
                step_groups.entry(env_idx).or_default().push(i);
            }
            for (&env_idx, group) in &step_groups {
                if group.len() < 2 { continue; }
                let n_agents = group.len().min(MAX_AGENTS);
                let local_vals: Vec<f32> = group.iter().take(n_agents)
                    .map(|&gi| cached_local_values[gi]).collect();
                let team_return: f32 = group.iter().map(|&gi| returns[gi]).sum::<f32>() / group.len() as f32;

                // Global state from cached h2 mean pool.
                let mut global_state = vec![0.0f32; hidden + GLOBAL_OBS_FEATURES];
                for &gi in group.iter().take(n_agents) {
                    for j in 0..hidden { global_state[j] += cached_h2[gi][j]; }
                }
                let inv = 1.0 / n_agents as f32;
                for j in 0..hidden { global_state[j] *= inv; }

                let team_val = mixer.forward(&local_vals, &global_state, n_agents);
                let d_mixer = QMIX_COEF * 2.0 * (team_val - team_return);
                mixer.backward(d_mixer);
                mixer_samples += 1;
            }

            if critic_samples > 0 {
                critic.clip_grad_norm(config.max_grad_norm);
                critic.adam_step(current_lr, critic_samples, 0.9, 0.999, 1e-8, 5e-4, adam_step_count);
            }
            if mixer_samples > 0 {
                mixer.clip_grad_norm(config.max_grad_norm);
                mixer.adam_step(current_lr, mixer_samples, 0.9, 0.999, 1e-8, 5e-4, adam_step_count);
            }
        }

        update_count += 1;

        // Save to checkpoint pool periodically.
        if update_count % CHECKPOINT_SAVE_FREQ as u64 == 0 {
            if let Ok(json) = serde_json::to_string(&policy) {
                checkpoint_pool.push(json);
            }
        }

        // Update opponent: sample from checkpoint pool or use current policy.
        if update_count % config.opponent_update_freq as u64 == 0 {
            if !checkpoint_pool.is_empty() {
                let seed = total_steps.wrapping_mul(997).wrapping_add(update_count * 31);
                if let Some(json) = checkpoint_pool.sample(seed) {
                    if let Ok(mut loaded) = serde_json::from_str::<PolicyNetV2>(json) {
                        loaded.init_grads();
                        opponent = loaded;
                        println!("  Opponent sampled from pool ({} checkpoints) at update {}", checkpoint_pool.len(), update_count);
                    } else {
                        opponent = policy.clone();
                        println!("  Opponent updated (pool parse failed) at update {}", update_count);
                    }
                } else {
                    opponent = policy.clone();
                }
            } else {
                opponent = policy.clone();
                println!("  Opponent updated (pool empty) at update {}", update_count);
            }
        }

        // Log progress.
        if !ep_rewards.is_empty() && update_count % 5 == 0 {
            let recent: Vec<f32> = ep_rewards.iter().rev().take(100).copied().collect();
            let mean_reward: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
            let elapsed = start_time.elapsed().as_secs();
            let fps = total_steps as f64 / elapsed.max(1) as f64;
            println!(
                "Update {}: steps={}, eps={}, mean_r={:.2}, trans={}, fps={:.0}, {}s, {}v{}",
                update_count, total_steps, ep_rewards.len(), mean_reward, n, fps, elapsed, curr_drones, curr_drones,
            );
        }

        // Evaluate.
        if update_count % config.eval_freq as u64 == 0 {
            let eval_seed = total_steps * 997 + update_count * 31;
            let (win_rate, mean_reward, mean_length) = evaluate(&policy, &ego_normalizer, &entity_normalizer, &config, eval_seed);
            println!("  Eval: win_A={:.1}%, reward={:.2}, len={:.0}",
                win_rate * 100.0, mean_reward, mean_length);

            let ckpt_path = format!("{}/checkpoint_{}.json", config.output_dir, update_count);
            policy.save(&ckpt_path).ok();

            if win_rate > best_win_rate {
                best_win_rate = win_rate;
                policy.save(&best_model_path).ok();
                println!("  New best model (win={:.1}%)", win_rate * 100.0);
            }
        }
    }

    policy.save(&final_model_path).ok();
    println!("\nTraining complete. Steps: {}, Best win: {:.1}%", total_steps, best_win_rate * 100.0);
}

fn parse_args() -> TrainConfig {
    let mut config = TrainConfig::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--timesteps" => { i += 1; config.total_timesteps = args[i].parse().unwrap(); }
            "--n-envs" => { i += 1; config.n_envs = args[i].parse().unwrap(); }
            "--drones" => { i += 1; config.drones_per_side = args[i].parse().unwrap(); }
            "--targets" => { i += 1; config.targets_per_side = args[i].parse().unwrap(); }
            "--max-ticks" => { i += 1; config.max_ticks = args[i].parse().unwrap(); }
            "--lr" => { i += 1; config.lr = args[i].parse().unwrap(); }
            "--output-dir" => { i += 1; config.output_dir = args[i].clone(); }
            "--eval-freq" => { i += 1; config.eval_freq = args[i].parse().unwrap(); }
            "--hidden" => { i += 1; config.hidden_size = args[i].parse().unwrap(); }
            "--n-steps" => { i += 1; config.n_steps = args[i].parse().unwrap(); }
            "--batch-size" => { i += 1; config.batch_size = args[i].parse().unwrap(); }
            "--n-epochs" => { i += 1; config.n_epochs = args[i].parse().unwrap(); }
            "--eval-episodes" => { i += 1; config.eval_episodes = args[i].parse().unwrap(); }
            "--opponent-update-freq" => { i += 1; config.opponent_update_freq = args[i].parse().unwrap(); }
            "--help" | "-h" => {
                println!("Usage: train [OPTIONS]");
                println!("  --timesteps N             Total training steps (default 50000000)");
                println!("  --n-envs N                Parallel environments (default 1024)");
                println!("  --drones N                Drones per side (default 24)");
                println!("  --targets N               Targets per side (default 6)");
                println!("  --max-ticks N             Max ticks per episode (default 6000)");
                println!("  --lr F                    Learning rate (default 1e-4)");
                println!("  --hidden N                Hidden layer size (default 256)");
                println!("  --n-steps N               Steps per rollout (default 512)");
                println!("  --batch-size N            Mini-batch size (default 4096)");
                println!("  --n-epochs N              PPO epochs per update (default 4)");
                println!("  --output-dir DIR          Output directory (default results)");
                println!("  --eval-freq N             Eval frequency (default 50)");
                println!("  --eval-episodes N         Eval episodes (default 50)");
                println!("  --opponent-update-freq N  Opponent update freq (default 30)");
                std::process::exit(0);
            }
            other => { eprintln!("Unknown argument: {}", other); std::process::exit(1); }
        }
        i += 1;
    }
    config
}
