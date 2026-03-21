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
use rl_train::ppo::Transition;
use rl_train::ppo::backward::PolicyNetV2PPOExt;
use rl_train::normalize::{RunningMeanStd, ValueNormalizer};
use rl_train::checkpoint_pool::CheckpointPool;
use rl_train::curriculum::Curriculum;
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
    #[allow(dead_code)]
    vf_coef: f32, // Unused: local_value_head disabled; centralized critic handles value prediction
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
    resume: Option<String>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            total_timesteps: 50_000_000,
            n_steps: 256,
            n_epochs: 4,
            batch_size: 8192,
            lr: 5e-4,
            gamma: 0.99,
            lam: 0.97,
            clip_range: 0.2,
            ent_coef_start: 0.03,
            ent_coef_end: 0.005,
            vf_coef: 0.5,
            max_grad_norm: 0.5,
            n_envs: 1024,
            hidden_size: 256,
            drones_per_side: 24,
            targets_per_side: 6,
            max_ticks: 10000,
            eval_freq: 10,
            eval_episodes: 50,
            opponent_update_freq: 30,
            output_dir: "results".to_string(),
            resume: None,
        }
    }
}

fn make_sim_config(tc: &TrainConfig, world_size: f32) -> SimConfig {
    // Dynamically set max_ticks so a drone can traverse the diagonal 1x.
    // Gives ~177 decision steps at stage 0 — enough for 4v4 combat.
    let speed_multiplier = 2.0f32;
    let distance_per_tick = 20.0 * 0.05 * speed_multiplier; // 2.0 m/tick
    let diagonal = world_size * std::f32::consts::SQRT_2;
    let dynamic_max_ticks = (diagonal / distance_per_tick).ceil() as u32;
    // Use the larger of the user-specified max_ticks and the dynamic value.
    let max_ticks = tc.max_ticks.max(dynamic_max_ticks);

    SimConfig {
        drones_per_side: tc.drones_per_side,
        targets_per_side: tc.targets_per_side,
        world_size,
        max_ticks,
        speed_multiplier,
        seed: 0,
        randomize_opponent: false,
        ..Default::default()
    }
}

/// Normalize a DroneObsV2 using the ego running normalizer.
///
/// Entity features are already normalized in obs_encoding (dx/w, dy/w, dist/diag,
/// vx/max_v, vy/max_v, heading/pi) so they pass through unchanged.
fn normalize_obs(obs: &DroneObsV2, ego_norm: &RunningMeanStd, _ent_norm: &RunningMeanStd) -> DroneObsV2 {
    let ego = ego_norm.normalize(&obs.ego);
    DroneObsV2 { ego, entities: obs.entities.clone(), n_entities: obs.n_entities }
}

/// Evaluate RL policy vs doctrine opponents (not self-play).
/// Runs half episodes against aggressive doctrine, half against defensive.
/// This measures absolute strength rather than relative self-play win rate.
fn evaluate(
    policy: &PolicyNetV2,
    ego_norm: &RunningMeanStd,
    ent_norm: &RunningMeanStd,
    config: &TrainConfig,
    stage: &rl_train::curriculum::CurriculumStage,
    base_seed: u64,
) -> (f32, f32, f32) {
    let results: Vec<(u32, f32, u64)> = (0..config.eval_episodes)
        .into_par_iter()
        .map(|ep| {
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed + ep as u64);
            let mut sc = make_sim_config(config, stage.world_size);
            sc.drones_per_side = stage.drones_per_side;
            sc.targets_per_side = stage.targets_per_side;
            // Alternate aggressive/defensive doctrine opponents.
            sc.randomize_opponent = true;
            let mut env = SimRunner::new(sc);

            // Reset for RL (group A) vs doctrine (group B).
            let ((mut obs_a, mut ids_a), (_obs_b, _ids_b)) =
                env.reset_selfplay_with_seed(10000 + ep as u64);
            let mut ep_reward = 0.0f32;
            let mut ep_len = 0u64;
            let mut win = 0u32;

            loop {
                let norm_a: Vec<DroneObsV2> = obs_a.iter().map(|o| normalize_obs(o, ego_norm, ent_norm)).collect();
                let egos_a: Vec<&[f32]> = norm_a.iter().map(|o| o.ego.as_slice()).collect();
                let ents_a: Vec<&[f32]> = norm_a.iter().map(|o| o.entities.as_slice()).collect();
                let nents_a: Vec<usize> = norm_a.iter().map(|o| o.n_entities).collect();
                // Use raw entities for action masking (consistent with training).
                let raw_ents_a: Vec<&[f32]> = obs_a.iter().map(|o| o.entities.as_slice()).collect();
                let (results_a, _) = policy.act_batch_with_h2_masked(&egos_a, &ents_a, &nents_a, Some(&raw_ents_a), &mut rng);
                let actions_a: Vec<(usize, u32)> = ids_a.iter()
                    .zip(results_a.iter())
                    .map(|(&id, &(act, _, _))| (id, act))
                    .collect();

                // Step with doctrine opponent (Group B handled by engine).
                let result = env.step_rl_vs_doctrine_v2(&actions_a);
                ep_reward += result.reward;
                ep_len += 1;

                if result.game_result == GameResult::AWins { win = 1; }
                if result.terminated || result.truncated { break; }

                obs_a = result.obs_a;
                ids_a = result.drone_ids_a;
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

/// Opponent type for each environment.
#[derive(Clone, Copy, PartialEq)]
enum OpponentType {
    /// Both groups RL-controlled (self-play).
    SelfPlay,
    /// Group B uses built-in doctrine (aggressive or defensive, randomized on reset).
    Doctrine,
}

/// Per-env state for self-play rollout collection.
struct EnvState {
    obs_a: Vec<DroneObsV2>,
    drone_ids_a: Vec<usize>,
    obs_b: Vec<DroneObsV2>,
    drone_ids_b: Vec<usize>,
    opponent_type: OpponentType,
}

/// Per-env action mapping into the flat batch.
struct EnvActionSlice {
    a_start: usize,
    a_count: usize,
    b_start: usize,
    b_count: usize,
}

/// Fraction of envs that use doctrine opponents (rest use self-play).
/// Pure doctrine in early stages; introduces self-play after agent can beat doctrine.
fn doctrine_fraction(curriculum_stage: usize) -> f32 {
    if curriculum_stage <= 1 {
        1.0  // Pure doctrine until agent beats it
    } else {
        0.5  // Mix doctrine + self-play in later stages
    }
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
    let mut policy = if let Some(ref path) = config.resume {
        println!("Resuming from: {}", path);
        PolicyNetV2::load(path).expect("Failed to load resume model")
    } else {
        PolicyNetV2::new(EGO_DIM, ENTITY_DIM, ACT_DIM, EMBED_DIM, config.hidden_size, &mut rng)
    };
    let mut opponent = policy.clone();

    // MAPPO centralized critic + QMIX mixer.
    let global_state_dim = config.hidden_size + GLOBAL_OBS_FEATURES;
    let mut critic = CentralizedCritic::new(config.hidden_size, &mut rng);
    let mut mixer = QMIXMixer::new(global_state_dim, MAX_AGENTS, &mut rng);

    // Running observation normalizers (Welford's online algorithm).
    let mut ego_normalizer = RunningMeanStd::new(EGO_DIM);
    // PopArt-style value normalization for stable critic learning.
    let mut value_normalizer = ValueNormalizer::new();
    // Skip normalizing categorical/binary entity dims: type_flag (6), alive_flag (7), is_current_target (9).
    // Note: entity normalization is disabled (features are pre-normalized in obs_encoding),
    // but we keep the normalizer for checkpoint compatibility and logging.
    let entity_normalizer = RunningMeanStd::new_with_skip(ENTITY_DIM, vec![6, 7, 9]);

    // Checkpoint pool for diverse self-play opponents.
    let mut checkpoint_pool = CheckpointPool::new(CHECKPOINT_POOL_SIZE);

    // Win-rate-based curriculum.
    let mut curriculum = Curriculum::new(config.drones_per_side, config.targets_per_side);
    let init_stage = curriculum.current().clone();

    let mut envs: Vec<SimRunner> = (0..config.n_envs)
        .map(|i| {
            let mut sc = make_sim_config(&config, init_stage.world_size);
            sc.drones_per_side = init_stage.drones_per_side;
            sc.targets_per_side = init_stage.targets_per_side;
            sc.seed = i as u64;
            SimRunner::new(sc)
        })
        .collect();

    let mut env_states: Vec<EnvState> = envs.iter().enumerate().map(|(i, e)| {
        let (obs_a, drone_ids_a) = e.observe_multi_group_v2(0);
        let (obs_b, drone_ids_b) = e.observe_multi_group_v2(1);
        let doc_frac = doctrine_fraction(curriculum.current_stage);
        let opponent_type = if (i as f32 / config.n_envs as f32) < doc_frac {
            OpponentType::Doctrine
        } else {
            OpponentType::SelfPlay
        };
        EnvState { obs_a, drone_ids_a, obs_b, drone_ids_b, opponent_type }
    }).collect();

    let mut total_steps = 0u64;
    let mut update_count = 0u64;
    let mut adam_step_count = 0usize;
    let mut critic_step_count = 0usize;
    let mut mixer_step_count = 0usize;
    let mut best_win_rate = 0.0f32;
    let mut ep_rewards: Vec<f32> = Vec::new();
    let mut ep_reward_accum = vec![0.0f32; config.n_envs];

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
    println!("  Win-rate curriculum: 4v4 → {}v{} (advance at >{:.0}% win rate, {} consecutive evals)",
        config.drones_per_side, config.drones_per_side,
        curriculum.advance_threshold * 100.0, curriculum.required_consecutive);
    println!("  CPUs: {}, Rayon: {}, Accelerate: {}", num_cpus, n_threads, cfg!(target_os = "macos"));
    println!();

    while config.total_timesteps == 0 || total_steps < config.total_timesteps {
        curriculum.tick_update();

        // ================================================================
        // Collect rollout
        // ================================================================
        let curr_stage = curriculum.current().clone();
        let est_transitions = config.n_steps * config.n_envs * curr_stage.drones_per_side as usize;
        let mut rollout: Vec<Transition> = Vec::with_capacity(est_transitions);
        let mut rollout_env_indices: Vec<usize> = Vec::with_capacity(est_transitions);

        // Buffer for ego normalizer update.
        let mut raw_ego_batch: Vec<f32> = Vec::with_capacity(est_transitions * EGO_DIM);

        // Cached h2 (flat) and local_value per transition (avoids re-forwarding in critic/mixer).
        let mut cached_h2_flat: Vec<f32> = Vec::with_capacity(est_transitions * config.hidden_size);
        let mut cached_local_values: Vec<f32> = Vec::with_capacity(est_transitions);
        // Per-(env, step) mean_pool cached at rollout time (for correct critic input).
        // Indexed by (step * n_envs + env_idx) to avoid per-transition duplication.
        let mut step_mean_pools: Vec<Vec<f32>> = Vec::new();
        let mut rollout_step_indices: Vec<usize> = Vec::with_capacity(est_transitions);
        // Team-only rewards for critic training (separate from mixed rewards used for PPO).
        let mut cached_team_rewards: Vec<f32> = Vec::with_capacity(est_transitions);

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
            // Collect raw ego obs for normalizer update.
            for env_st in &env_states {
                for obs in &env_st.obs_a {
                    raw_ego_batch.extend_from_slice(&obs.ego);
                }
            }

            // Build flat buffers directly from env_states.
            // Only ego features need normalization; entities are pre-normalized in obs_encoding.
            ego_refs_a.clear(); ent_refs_a.clear(); nent_a.clear();
            ego_refs_b.clear(); ent_refs_b.clear(); nent_b.clear();
            env_slices.clear(); flat_ids_a.clear(); flat_ids_b.clear();

            for env_st in &env_states {
                let a_start = ego_refs_a.len();
                for obs in &env_st.obs_a {
                    ego_refs_a.push(ego_normalizer.normalize(&obs.ego));
                    ent_refs_a.push(obs.entities.clone());
                    nent_a.push(obs.n_entities);
                }
                flat_ids_a.extend_from_slice(&env_st.drone_ids_a);

                let b_start = ego_refs_b.len();
                for obs in &env_st.obs_b {
                    ego_refs_b.push(ego_normalizer.normalize(&obs.ego));
                    ent_refs_b.push(obs.entities.clone());
                    nent_b.push(obs.n_entities);
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
            // Entity features are pre-normalized — same data used for both network input and action masking.
            let (batch_a, h2_flat_a) = policy.act_batch_with_h2_masked(&egos_a_sl, &ents_a_sl, &nent_a, Some(&ents_a_sl), &mut rng);

            let egos_b_sl: Vec<&[f32]> = ego_refs_b.iter().map(|e| e.as_slice()).collect();
            let ents_b_sl: Vec<&[f32]> = ent_refs_b.iter().map(|e| e.as_slice()).collect();
            let (batch_b, _) = opponent.act_batch_with_h2_masked(&egos_b_sl, &ents_b_sl, &nent_b, Some(&ents_b_sl), &mut rng);

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

            // Cache per-(env, step) mean_pools for critic update.
            step_mean_pools.extend(mean_pools.iter().cloned());

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
            let mut centralized_values = matmul_bias_relu(&c_h2, &critic.value_head.weights, &critic.value_head.biases,
                total_a, 1, critic.value_head.in_dim, false);
            // Denormalize critic outputs for GAE (critic learns in normalized space).
            value_normalizer.denormalize_batch(&mut centralized_values);

            // --- 3. Step envs in parallel (self-play or doctrine) ---
            let per_env_results: Vec<_> = envs.par_iter_mut()
                .zip(env_slices.par_iter())
                .zip(env_states.par_iter())
                .map(|((env, slice), state)| {
                    let actions_a: Vec<(usize, u32)> = (0..slice.a_count)
                        .map(|i| (flat_ids_a[slice.a_start + i], batch_a[slice.a_start + i].0))
                        .collect();
                    match state.opponent_type {
                        OpponentType::Doctrine => {
                            env.step_rl_vs_doctrine_v2(&actions_a)
                        }
                        OpponentType::SelfPlay => {
                            let actions_b: Vec<(usize, u32)> = (0..slice.b_count)
                                .map(|i| (flat_ids_b[slice.b_start + i], batch_b[slice.b_start + i].0))
                                .collect();
                            env.step_multi_selfplay(&actions_a, &actions_b)
                        }
                    }
                })
                .collect();

            // --- 4. Collect transitions + handle resets ---
            for env_idx in 0..config.n_envs {
                let result = &per_env_results[env_idx];
                let slice = &env_slices[env_idx];
                ep_reward_accum[env_idx] += result.reward;
                let terminated = result.terminated;
                let truncated = result.truncated;
                // Standard MAPPO: each agent receives the full team reward.
                // The centralized critic handles credit assignment.
                let per_drone_team_reward = result.reward;

                for i in 0..slice.a_count {
                    let batch_idx = slice.a_start + i;
                    let (action, log_prob, local_value) = batch_a[batch_idx];
                    let drone_id = flat_ids_a[batch_idx];
                    let individual_reward = result.individual_rewards_a.get(&drone_id).copied().unwrap_or(0.0);
                    let drone_died = result.drone_deaths_a.contains(&drone_id);
                    let value = centralized_values[batch_idx];

                    // Cache h2 (flat), local_value for critic/mixer update.
                    let h2_start = batch_idx * hidden;
                    cached_h2_flat.extend_from_slice(&h2_flat_a[h2_start..h2_start + hidden]);
                    cached_local_values.push(local_value);
                    cached_team_rewards.push(per_drone_team_reward);

                    rollout.push(Transition {
                        ego_obs: ego_refs_a[batch_idx].clone(),
                        entity_obs: ent_refs_a[batch_idx].clone(),
                        raw_entity_obs: ent_refs_a[batch_idx].clone(),
                        n_entities: nent_a[batch_idx],
                        action,
                        reward: per_drone_team_reward + individual_reward,
                        value,
                        log_prob,
                        done: terminated,
                        truncated,
                        drone_id,
                        drone_died,
                        // Dead drone: bootstrap with centralized critic value (team continues).
                        team_value_at_death: if drone_died { centralized_values[batch_idx] } else { 0.0 },
                    });
                    rollout_env_indices.push(env_idx);
                    rollout_step_indices.push(_step);
                }

                if terminated || truncated {
                    ep_rewards.push(ep_reward_accum[env_idx]);
                    ep_reward_accum[env_idx] = 0.0;
                    let new_seed = total_steps + env_idx as u64;

                    // Per-env curriculum config (blended during transitions).
                    let stage = curriculum.config_for_env(new_seed);

                    // Reassign opponent type: doctrine fraction decays over training.
                    let doc_frac = doctrine_fraction(curriculum.current_stage);
                    let new_opponent = if (new_seed % 1000) < (doc_frac * 1000.0) as u64 {
                        OpponentType::Doctrine
                    } else {
                        OpponentType::SelfPlay
                    };

                    let mut sc = make_sim_config(&config, stage.world_size);
                    sc.drones_per_side = stage.drones_per_side;
                    sc.targets_per_side = stage.targets_per_side;
                    sc.seed = new_seed;
                    sc.randomize_opponent = new_opponent == OpponentType::Doctrine;
                    let env_cfg = envs[env_idx].config();
                    if env_cfg.drones_per_side != sc.drones_per_side || env_cfg.targets_per_side != sc.targets_per_side || env_cfg.world_size != sc.world_size || sc.randomize_opponent != env_cfg.randomize_opponent {
                        envs[env_idx] = SimRunner::new(sc);
                    }
                    let ((obs_a, drone_ids_a), (obs_b, drone_ids_b)) =
                        envs[env_idx].reset_selfplay_with_seed(new_seed);
                    env_states[env_idx] = EnvState {
                        obs_a, drone_ids_a, obs_b, drone_ids_b,
                        opponent_type: new_opponent,
                    };
                } else {
                    let old_opponent = env_states[env_idx].opponent_type;
                    env_states[env_idx] = EnvState {
                        obs_a: result.obs_a.clone(),
                        drone_ids_a: result.drone_ids_a.clone(),
                        obs_b: result.obs_b.clone(),
                        drone_ids_b: result.drone_ids_b.clone(),
                        opponent_type: old_opponent,
                    };
                }
            }
            // Count total env transitions per rollout step (standard PPO convention).
            total_steps += config.n_envs as u64;
        }

        if rollout.is_empty() { continue; }

        // Update ego running normalizer with this rollout's raw observations.
        // Entity normalizer update is skipped: entity features are pre-normalized in obs_encoding.
        let ego_batch_size = raw_ego_batch.len() / EGO_DIM;
        if ego_batch_size > 0 {
            ego_normalizer.update_batch(&raw_ego_batch, ego_batch_size);
        }
        raw_ego_batch.clear();

        // Compute last values for GAE using local_value (consistent baseline).
        let hidden = config.hidden_size;
        // Per-(env, drone) last values for per-drone GAE.
        let mut last_values_map: HashMap<(usize, usize), f32> = HashMap::new();
        {
            let mut boot_egos: Vec<Vec<f32>> = Vec::new();
            let mut boot_ents: Vec<Vec<f32>> = Vec::new();
            let mut boot_nents: Vec<usize> = Vec::new();
            let mut boot_keys: Vec<(usize, usize)> = Vec::new(); // (env_idx, drone_id)
            for (env_idx, env_st) in env_states.iter().enumerate() {
                for (obs_i, obs) in env_st.obs_a.iter().enumerate() {
                    let norm = normalize_obs(obs, &ego_normalizer, &entity_normalizer);
                    boot_egos.push(norm.ego);
                    boot_ents.push(norm.entities);
                    boot_nents.push(norm.n_entities);
                    let drone_id = env_st.drone_ids_a.get(obs_i).copied().unwrap_or(0);
                    boot_keys.push((env_idx, drone_id));
                }
            }
            if !boot_egos.is_empty() {
                let e_sl: Vec<&[f32]> = boot_egos.iter().map(|e| e.as_slice()).collect();
                let ent_sl: Vec<&[f32]> = boot_ents.iter().map(|e| e.as_slice()).collect();
                let (_results, h2_boot) = policy.act_batch_with_h2_masked(&e_sl, &ent_sl, &boot_nents, Some(&ent_sl), &mut rng);

                // Compute per-env mean pools from bootstrap h2 values.
                let boot_total = boot_egos.len();
                let mut boot_env_offsets: Vec<(usize, usize)> = Vec::new(); // (start, count) per env
                {
                    let mut offset = 0usize;
                    for env_st in env_states.iter() {
                        let count = env_st.obs_a.len();
                        boot_env_offsets.push((offset, count));
                        offset += count;
                    }
                }

                let boot_mean_pools: Vec<Vec<f32>> = boot_env_offsets.iter().map(|&(start, count)| {
                    if count == 0 { return vec![0.0f32; hidden]; }
                    let mut pool = vec![0.0f32; hidden];
                    for i in 0..count {
                        let idx = start + i;
                        for j in 0..hidden { pool[j] += h2_boot[idx * hidden + j]; }
                    }
                    let inv = 1.0 / count as f32;
                    for j in 0..hidden { pool[j] *= inv; }
                    pool
                }).collect();

                // Build critic input for bootstrap values.
                let critic_in_dim = 2 * hidden;
                let mut boot_critic_input = vec![0.0f32; boot_total * critic_in_dim];
                for (env_idx, &(start, count)) in boot_env_offsets.iter().enumerate() {
                    for i in 0..count {
                        let idx = start + i;
                        let dst = idx * critic_in_dim;
                        boot_critic_input[dst..dst + hidden].copy_from_slice(
                            &h2_boot[idx * hidden..(idx + 1) * hidden]);
                        boot_critic_input[dst + hidden..dst + critic_in_dim].copy_from_slice(
                            &boot_mean_pools[env_idx]);
                    }
                }

                // Batched critic forward for bootstrap values.
                let bc_h1 = matmul_bias_relu(&boot_critic_input, &critic.fc1.weights, &critic.fc1.biases,
                    boot_total, critic.fc1.out_dim, critic_in_dim, true);
                let bc_h2 = matmul_bias_relu(&bc_h1, &critic.fc2.weights, &critic.fc2.biases,
                    boot_total, critic.fc2.out_dim, critic.fc2.in_dim, true);
                let mut boot_centralized_values = matmul_bias_relu(&bc_h2, &critic.value_head.weights, &critic.value_head.biases,
                    boot_total, 1, critic.value_head.in_dim, false);
                // Denormalize bootstrap values.
                value_normalizer.denormalize_batch(&mut boot_centralized_values);

                for (i, &(env_idx, drone_id)) in boot_keys.iter().enumerate() {
                    last_values_map.insert((env_idx, drone_id), boot_centralized_values[i]);
                }
            }
        }

        // Split rollout by (env, drone) for per-drone GAE.
        let mut drone_rollouts: HashMap<(usize, usize), Vec<(usize, &Transition)>> = HashMap::new();
        for (i, (trans, &env_idx)) in rollout.iter().zip(rollout_env_indices.iter()).enumerate() {
            drone_rollouts.entry((env_idx, trans.drone_id)).or_default().push((i, trans));
        }
        let mut advantages = vec![0.0f32; rollout.len()];
        let mut returns = vec![0.0f32; rollout.len()];
        for (&(env_idx, drone_id), group) in &drone_rollouts {
            if group.is_empty() { continue; }
            let last_val = last_values_map.get(&(env_idx, drone_id)).copied().unwrap_or(0.0);
            // Compute GAE directly from references to avoid cloning transitions.
            let n = group.len();
            let mut gae = 0.0f32;
            for j in (0..n).rev() {
                let t = group[j].1;
                let next_value = if j + 1 < n { group[j + 1].1.value } else { last_val };
                let next_non_terminal = if t.done && !t.truncated {
                    0.0
                } else if t.truncated {
                    1.0
                } else if t.drone_died {
                    gae = 0.0;
                    let bootstrap = t.team_value_at_death;
                    let delta = t.reward + config.gamma * bootstrap - t.value;
                    advantages[group[j].0] = delta;
                    returns[group[j].0] = delta + t.value;
                    continue;
                } else {
                    1.0
                };
                let delta = t.reward + config.gamma * next_value * next_non_terminal - t.value;
                gae = delta + config.gamma * config.lam * next_non_terminal * gae;
                advantages[group[j].0] = gae;
                returns[group[j].0] = gae + t.value;
            }
        }

        // Compute team-only returns for critic training (separate from mixed PPO returns).
        let mut team_returns = vec![0.0f32; rollout.len()];
        for ((_env_idx, _drone_id), group) in &drone_rollouts {
            // Simple discounted return of team-only rewards within each drone trajectory.
            // Reset at drone_died boundaries (drone's trajectory ends, team continues without it).
            let mut g = 0.0f32;
            for &(global_idx, _) in group.iter().rev() {
                let t = &rollout[global_idx];
                if t.drone_died && !t.done {
                    g = 0.0; // Drone's trajectory ends here
                }
                let done_flag = if t.done && !t.truncated { 0.0 } else { 1.0 };
                g = cached_team_rewards[global_idx] + config.gamma * done_flag * g;
                team_returns[global_idx] = g;
            }
        }

        // Update value normalizer with this rollout's returns.
        value_normalizer.update(&returns);

        // Normalize advantages.
        let n = rollout.len();
        let adv_mean: f32 = advantages.iter().sum::<f32>() / n as f32;
        let adv_var: f32 = advantages.iter().map(|a| (a - adv_mean).powi(2)).sum::<f32>() / n as f32;
        let adv_std = adv_var.sqrt().max(1e-8);
        for a in &mut advantages { *a = (*a - adv_mean) / adv_std; }

        // Anneal LR and entropy.
        // Use update_count for progress when --timesteps 0 (infinite mode).
        let progress = if config.total_timesteps == 0 {
            (update_count as f32 / 500.0).min(1.0) // Anneal over ~500 updates
        } else {
            (total_steps as f32 / config.total_timesteps as f32).min(1.0)
        };
        let current_lr = config.lr * (1.0 - progress).max(0.1); // Floor at 10% of initial LR
        // Cosine decay for entropy.
        let current_ent_coef = config.ent_coef_end + 0.5 * (config.ent_coef_start - config.ent_coef_end) * (1.0 + (std::f32::consts::PI * progress).cos());

        // ================================================================
        // MAPPO Critic + QMIX Mixer update BEFORE PPO (h2 is still fresh).
        // ================================================================
        {
            critic.zero_grad();
            mixer.zero_grad();
            let mut critic_samples = 0usize;
            let mut mixer_samples = 0usize;

            // Batched critic update using cached h2 and per-step mean_pools.
            let critic_in_dim = 2 * hidden;
            let n_total = rollout.len();
            if n_total > 0 {
                // Build flat critic input using per-(step, env) cached mean_pools.
                let mut c_input = vec![0.0f32; n_total * critic_in_dim];
                for i in 0..n_total {
                    let dst = i * critic_in_dim;
                    c_input[dst..dst + hidden].copy_from_slice(&cached_h2_flat[i * hidden..(i + 1) * hidden]);
                    let pool_idx = rollout_step_indices[i] * config.n_envs + rollout_env_indices[i];
                    c_input[dst + hidden..dst + critic_in_dim].copy_from_slice(&step_mean_pools[pool_idx]);
                }

                // Batched forward through critic.
                let ch1 = matmul_bias_relu(&c_input, &critic.fc1.weights, &critic.fc1.biases,
                    n_total, critic.fc1.out_dim, critic_in_dim, true);
                let ch2 = matmul_bias_relu(&ch1, &critic.fc2.weights, &critic.fc2.biases,
                    n_total, critic.fc2.out_dim, critic.fc2.in_dim, true);
                let c_vals = matmul_bias_relu(&ch2, &critic.value_head.weights, &critic.value_head.biases,
                    n_total, 1, critic.value_head.in_dim, false);

                // Strided subsample for critic backward.
                let critic_budget = (config.batch_size * 8).min(n_total);
                let stride = n_total / critic_budget.max(1);
                let mut critic_count = 0usize;
                for k in 0..critic_budget {
                    let i = (k * stride).min(n_total - 1);
                    // Normalize target to match critic's output space.
                    let normalized_target = value_normalizer.normalize(team_returns[i]);
                    let d = MAPPO_VF_COEF * 2.0 * (c_vals[i] - normalized_target);
                    let pool_idx = rollout_step_indices[i] * config.n_envs + rollout_env_indices[i];
                    critic.forward(&cached_h2_flat[i * hidden..(i + 1) * hidden], &step_mean_pools[pool_idx]);
                    critic.backward(d);
                    critic_count += 1;
                }
                critic_samples = critic_count;
            }

            // QMIX update: group by (env, step) for per-timestep credit assignment.
            let mut step_groups: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
            for (i, (&env_idx, &step_idx)) in rollout_env_indices.iter().zip(rollout_step_indices.iter()).enumerate() {
                step_groups.entry((env_idx, step_idx)).or_default().push(i);
            }
            for (_env_idx, group) in &step_groups {
                if group.len() < 2 { continue; }
                let n_agents = group.len().min(MAX_AGENTS);
                let local_vals: Vec<f32> = group.iter().take(n_agents)
                    .map(|&gi| cached_local_values[gi]).collect();
                let team_return: f32 = group.iter().map(|&gi| team_returns[gi]).sum::<f32>() / group.len() as f32;

                let mut global_state = vec![0.0f32; hidden + GLOBAL_OBS_FEATURES];
                for &gi in group.iter().take(n_agents) {
                    for j in 0..hidden { global_state[j] += cached_h2_flat[gi * hidden + j]; }
                }
                let inv = 1.0 / n_agents as f32;
                for j in 0..hidden { global_state[j] *= inv; }

                let sample_ego = &rollout[group[0]].ego_obs;
                if sample_ego.len() >= 25 {
                    for k in 0..GLOBAL_OBS_FEATURES.min(8) {
                        global_state[hidden + k] = sample_ego[17 + k];
                    }
                }

                let team_val = mixer.forward(&local_vals, &global_state, n_agents);
                let d_mixer = QMIX_COEF * 2.0 * (team_val - team_return);
                mixer.backward(d_mixer);
                mixer_samples += 1;
            }

            if critic_samples > 0 {
                critic.clip_grad_norm(config.max_grad_norm);
                critic_step_count += 1;
                critic.adam_step(current_lr, critic_samples, 0.9, 0.999, 1e-8, 5e-4, critic_step_count);
            }
            if mixer_samples > 0 {
                mixer.clip_grad_norm(config.max_grad_norm);
                mixer_step_count += 1;
                mixer.adam_step(current_lr, mixer_samples, 0.9, 0.999, 1e-8, 5e-4, mixer_step_count);
            }
        }

        // ================================================================
        // PPO update (after critic/QMIX so h2 was fresh for them).
        // ================================================================
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
                                &t.entity_obs,
                                t.action,
                                advantages[idx],
                                returns[idx],
                                t.log_prob,
                                config.clip_range,
                                0.0, // vf_coef=0: local_value_head disabled; centralized critic handles value prediction
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
                "Update {}: steps={}, eps={}, mean_r={:.2}, trans={}, fps={:.0}, {}s, {}",
                update_count, total_steps, ep_rewards.len(), mean_reward, n, fps, elapsed, curriculum.stage_label(),
            );
        }

        // Evaluate.
        if update_count % config.eval_freq as u64 == 0 {
            let eval_seed = total_steps * 997 + update_count * 31;
            let (win_rate, mean_reward, mean_length) = evaluate(&policy, &ego_normalizer, &entity_normalizer, &config, curriculum.current(), eval_seed);
            println!("  Eval: win_A={:.1}%, reward={:.2}, len={:.0}, stage={}",
                win_rate * 100.0, mean_reward, mean_length, curriculum.stage_label());

            // Win-rate-based curriculum advancement.
            let stage_change = curriculum.report_eval(win_rate);
            if stage_change == Some(true) {
                let _new_stage = curriculum.current();
                println!("  >>> Curriculum advanced to {} (win_rate={:.1}% exceeded {:.0}% threshold)",
                    curriculum.stage_label(), win_rate * 100.0, curriculum.advance_threshold * 100.0);
                // Recreate all envs at new stage.
                for i in 0..config.n_envs {
                    let stage = curriculum.config_for_env(i as u64);
                    let doc_frac = doctrine_fraction(curriculum.current_stage);
                    let opp = if (i as f32 / config.n_envs as f32) < doc_frac {
                        OpponentType::Doctrine
                    } else {
                        OpponentType::SelfPlay
                    };
                    let mut sc = make_sim_config(&config, stage.world_size);
                    sc.drones_per_side = stage.drones_per_side;
                    sc.targets_per_side = stage.targets_per_side;
                    sc.seed = total_steps + i as u64;
                    sc.randomize_opponent = opp == OpponentType::Doctrine;
                    envs[i] = SimRunner::new(sc);
                    let (obs_a, drone_ids_a) = envs[i].observe_multi_group_v2(0);
                    let (obs_b, drone_ids_b) = envs[i].observe_multi_group_v2(1);
                    env_states[i] = EnvState { obs_a, drone_ids_a, obs_b, drone_ids_b, opponent_type: opp };
                    ep_reward_accum[i] = 0.0;
                }
                // Reset best_win_rate for the new stage.
                best_win_rate = 0.0;
            } else if stage_change == Some(false) {
                println!("  <<< Curriculum demoted to {} (win_rate={:.1}% below {:.0}% threshold)",
                    curriculum.stage_label(), win_rate * 100.0, 30.0);
                // Recreate all envs at demoted stage.
                for i in 0..config.n_envs {
                    let stage = curriculum.config_for_env(i as u64);
                    let mut sc = make_sim_config(&config, stage.world_size);
                    sc.drones_per_side = stage.drones_per_side;
                    sc.targets_per_side = stage.targets_per_side;
                    sc.seed = total_steps + i as u64;
                    sc.randomize_opponent = true;
                    envs[i] = SimRunner::new(sc);
                    let (obs_a, drone_ids_a) = envs[i].observe_multi_group_v2(0);
                    let (obs_b, drone_ids_b) = envs[i].observe_multi_group_v2(1);
                    env_states[i] = EnvState { obs_a, drone_ids_a, obs_b, drone_ids_b, opponent_type: OpponentType::Doctrine };
                    ep_reward_accum[i] = 0.0;
                }
                best_win_rate = 0.0;
            }

            let ckpt_path = format!("{}/checkpoint_{}.json", config.output_dir, update_count);
            policy.save(&ckpt_path).ok();
            save_normalizers(&ckpt_path, &ego_normalizer, &entity_normalizer);

            if win_rate > best_win_rate {
                best_win_rate = win_rate;
                policy.save(&best_model_path).ok();
                save_normalizers(&best_model_path, &ego_normalizer, &entity_normalizer);
                println!("  New best model (win={:.1}%)", win_rate * 100.0);
            }
        }
    }

    policy.save(&final_model_path).ok();
    save_normalizers(&final_model_path, &ego_normalizer, &entity_normalizer);
    println!("\nTraining complete. Steps: {}, Best win: {:.1}%", total_steps, best_win_rate * 100.0);
}

/// Save normalizer stats alongside a model file.
/// Creates a `_normalizers.json` companion file.
fn save_normalizers(model_path: &str, ego_norm: &RunningMeanStd, ent_norm: &RunningMeanStd) {
    let norm_path = model_path.replace(".json", "_normalizers.json");
    let data = serde_json::json!({
        "ego": { "mean": ego_norm.mean, "var": ego_norm.var, "skip_indices": ego_norm.skip_indices },
        "entity": { "mean": ent_norm.mean, "var": ent_norm.var, "skip_indices": ent_norm.skip_indices },
    });
    if let Ok(json) = serde_json::to_string(&data) {
        std::fs::write(&norm_path, json).ok();
    }
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
            "--resume" => { i += 1; config.resume = Some(args[i].clone()); }
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
