//! Multi-agent self-play PPO training for per-drone RL.
//!
//! Performance optimizations:
//! - Batched inference: all drone observations flattened into a matrix,
//!   single cblas_sgemm call per layer (Accelerate on Apple Silicon).
//! - Rayon: environment stepping + PPO backward parallelized across cores.
//! - mimalloc: thread-local allocator eliminates malloc contention.
//! - Self-play with lagged opponent policy.
//! - Curriculum learning: 4v4 → 8v8 → 16v16 → 24v24.

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use drone_lib::sim_runner::{GameResult, SimConfig, SimRunner, SelfPlayStepResult, OBS_DIM, ACT_DIM};
use rl_train::mlp::{compute_gae, PolicyNet, Transition, PolicyNetPPOExt};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::time::Instant;


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

/// Run self-play evaluation with episodes parallelized across rayon threads.
fn evaluate(policy: &PolicyNet, config: &TrainConfig, base_seed: u64) -> (f32, f32, f32) {
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
                let refs_a: Vec<&[f32]> = obs_a.iter().map(|o| o.as_slice()).collect();
                let results_a = policy.act_batch(&refs_a, &mut rng);
                let actions_a: Vec<(usize, u32)> = ids_a.iter()
                    .zip(results_a.iter())
                    .map(|(&id, &(act, _, _))| (id, act))
                    .collect();

                let refs_b: Vec<&[f32]> = obs_b.iter().map(|o| o.as_slice()).collect();
                let results_b = policy.act_batch(&refs_b, &mut rng);
                let actions_b: Vec<(usize, u32)> = ids_b.iter()
                    .zip(results_b.iter())
                    .map(|(&id, &(act, _, _))| (id, act))
                    .collect();

                let result = env.step_multi_selfplay(&actions_a, &actions_b);
                ep_reward += result.reward;
                ep_len += 1;

                if result.game_result == GameResult::AWins {
                    win = 1;
                }

                if result.terminated || result.truncated {
                    break;
                }

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
    obs_a: Vec<[f32; OBS_DIM]>,
    drone_ids_a: Vec<usize>,
    obs_b: Vec<[f32; OBS_DIM]>,
    drone_ids_b: Vec<usize>,
}

/// Curriculum stage definition.
struct CurriculumStage {
    threshold: f32,
    drones_per_side: u32,
    targets_per_side: u32,
}

fn curriculum_config(progress: f32, final_drones: u32, final_targets: u32) -> (u32, u32) {
    let stages = [
        CurriculumStage { threshold: 0.0,  drones_per_side: 4,  targets_per_side: 2 },
        CurriculumStage { threshold: 0.15, drones_per_side: 8,  targets_per_side: 3 },
        CurriculumStage { threshold: 0.35, drones_per_side: 16, targets_per_side: 4 },
        CurriculumStage { threshold: 0.55, drones_per_side: final_drones, targets_per_side: final_targets },
    ];

    let mut drones = stages[0].drones_per_side;
    let mut targets = stages[0].targets_per_side;
    for stage in &stages {
        if progress >= stage.threshold {
            drones = stage.drones_per_side;
            targets = stage.targets_per_side;
        }
    }
    (drones, targets)
}

/// Per-env action mapping: which action indices (into the flat batch result) belong to each env.
struct EnvActionSlice {
    /// Offset into the flat batch results for Group A.
    a_start: usize,
    a_count: usize,
    /// Offset into the flat batch results for Group B.
    b_start: usize,
    b_count: usize,
}

fn main() {
    let config = parse_args();

    // Configure rayon thread pool to use all available cores.
    let num_cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .expect("Failed to configure rayon thread pool");

    std::fs::create_dir_all(&config.output_dir).ok();
    let best_model_path = format!("{}/best_model.json", config.output_dir);
    let final_model_path = format!("{}/final_model.json", config.output_dir);

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut policy = PolicyNet::new(OBS_DIM, ACT_DIM, config.hidden_size, &mut rng);
    let mut opponent = policy.clone();

    // Curriculum init.
    let (init_drones, init_targets) = curriculum_config(0.0, config.drones_per_side, config.targets_per_side);

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
        let (obs_a, drone_ids_a) = e.observe_multi_group(0);
        let (obs_b, drone_ids_b) = e.observe_multi_group(1);
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

    println!("Starting self-play multi-agent PPO training");
    println!("  Obs dim: {}, Act dim: {}, Hidden: {}", OBS_DIM, ACT_DIM, config.hidden_size);
    println!("  Total timesteps: {}", config.total_timesteps);
    println!("  Envs: {}, Steps/rollout: {}, Epochs: {}, Batch: {}",
        config.n_envs, config.n_steps, config.n_epochs, config.batch_size);
    println!("  LR: {} (annealed to 0), Gamma: {}, Clip: {}", config.lr, config.gamma, config.clip_range);
    println!("  Entropy coef: {} → {}, Max grad norm: {}", config.ent_coef_start, config.ent_coef_end, config.max_grad_norm);
    println!("  Self-play opponent update every {} PPO updates", config.opponent_update_freq);
    println!("  Final drones/side: {}, Final targets/side: {}", config.drones_per_side, config.targets_per_side);
    println!("  Curriculum: {}v{}/{}t → {}v{}/{}t",
        init_drones, init_drones, init_targets,
        config.drones_per_side, config.drones_per_side, config.targets_per_side);
    println!("  CPUs: {}, Rayon threads: {}, Accelerate: {}", num_cpus, n_threads, cfg!(target_os = "macos"));
    println!();

    while total_steps < config.total_timesteps {
        // Check curriculum stage.
        let progress = (total_steps as f32 / config.total_timesteps as f32).min(1.0);
        let (new_drones, new_targets) = curriculum_config(progress, config.drones_per_side, config.targets_per_side);
        if new_drones != curr_drones || new_targets != curr_targets {
            println!("  Curriculum: {}v{}/{}t → {}v{}/{}t at {:.1}% progress",
                curr_drones, curr_drones, curr_targets,
                new_drones, new_drones, new_targets,
                progress * 100.0);
            curr_drones = new_drones;
            curr_targets = new_targets;
            for i in 0..config.n_envs {
                let mut sc = make_sim_config(&config);
                sc.drones_per_side = curr_drones;
                sc.targets_per_side = curr_targets;
                sc.seed = total_steps + i as u64;
                envs[i] = SimRunner::new(sc);
                let (obs_a, drone_ids_a) = envs[i].observe_multi_group(0);
                let (obs_b, drone_ids_b) = envs[i].observe_multi_group(1);
                env_states[i] = EnvState { obs_a, drone_ids_a, obs_b, drone_ids_b };
                ep_reward_accum[i] = 0.0;
            }
        }

        // ================================================================
        // Collect rollout with batched inference + parallel env stepping
        // ================================================================
        // Pre-allocate rollout capacity: n_steps * n_envs * ~drones_per_side.
        let est_transitions = config.n_steps * config.n_envs * curr_drones as usize;
        let mut rollout: Vec<Transition> = Vec::with_capacity(est_transitions);
        let mut rollout_env_indices: Vec<usize> = Vec::with_capacity(est_transitions);

        // Pre-allocate reusable buffers for the rollout loop.
        let max_drones = curr_drones as usize * 2; // both groups
        let mut flat_obs_a: Vec<f32> = Vec::with_capacity(config.n_envs * max_drones * OBS_DIM);
        let mut flat_obs_b: Vec<f32> = Vec::with_capacity(config.n_envs * max_drones * OBS_DIM);
        let mut env_slices: Vec<EnvActionSlice> = Vec::with_capacity(config.n_envs);
        // Flat ID buffers: store all drone IDs contiguously, indexed by env_slices.
        let mut flat_ids_a: Vec<usize> = Vec::with_capacity(config.n_envs * max_drones);
        let mut flat_ids_b: Vec<usize> = Vec::with_capacity(config.n_envs * max_drones);

        for _step in 0..config.n_steps {
            // --- 1. Flatten obs + IDs into pre-allocated buffers (no heap alloc) ---
            flat_obs_a.clear();
            flat_obs_b.clear();
            flat_ids_a.clear();
            flat_ids_b.clear();
            env_slices.clear();

            for env_st in &env_states {
                let a_start = flat_obs_a.len() / OBS_DIM;
                for obs in &env_st.obs_a {
                    flat_obs_a.extend_from_slice(obs);
                }
                flat_ids_a.extend_from_slice(&env_st.drone_ids_a);
                let b_start = flat_obs_b.len() / OBS_DIM;
                for obs in &env_st.obs_b {
                    flat_obs_b.extend_from_slice(obs);
                }
                flat_ids_b.extend_from_slice(&env_st.drone_ids_b);
                env_slices.push(EnvActionSlice {
                    a_start,
                    a_count: env_st.obs_a.len(),
                    b_start,
                    b_count: env_st.obs_b.len(),
                });
            }

            // --- 2. Batched forward pass (Accelerate-optimized on macOS) ---
            let total_a = flat_obs_a.len() / OBS_DIM;
            let total_b = flat_obs_b.len() / OBS_DIM;
            let refs_a: Vec<&[f32]> = (0..total_a).map(|i| &flat_obs_a[i * OBS_DIM..(i + 1) * OBS_DIM]).collect();
            let refs_b: Vec<&[f32]> = (0..total_b).map(|i| &flat_obs_b[i * OBS_DIM..(i + 1) * OBS_DIM]).collect();
            let batch_a = policy.act_batch(&refs_a, &mut rng);
            let batch_b = opponent.act_batch(&refs_b, &mut rng);

            // --- 3. Build per-env actions + step in parallel ---
            let per_env_results: Vec<(Vec<(usize, u32)>, Vec<(usize, u32)>, SelfPlayStepResult)> =
                envs.par_iter_mut()
                    .zip(env_slices.par_iter())
                    .map(|(env, slice)| {
                        let actions_a: Vec<(usize, u32)> = (0..slice.a_count)
                            .map(|i| (flat_ids_a[slice.a_start + i], batch_a[slice.a_start + i].0))
                            .collect();
                        let actions_b: Vec<(usize, u32)> = (0..slice.b_count)
                            .map(|i| (flat_ids_b[slice.b_start + i], batch_b[slice.b_start + i].0))
                            .collect();
                        let result = env.step_multi_selfplay(&actions_a, &actions_b);
                        (actions_a, actions_b, result)
                    })
                    .collect();

            // --- 4. Collect transitions + handle resets ---
            for env_idx in 0..config.n_envs {
                let (_, _, ref result) = per_env_results[env_idx];
                let slice = &env_slices[env_idx];

                ep_reward_accum[env_idx] += result.reward;
                let done = result.terminated || result.truncated;
                let per_drone_reward = result.reward;

                for i in 0..slice.a_count {
                    let batch_idx = slice.a_start + i;
                    let (action, log_prob, value) = batch_a[batch_idx];
                    let mut obs = [0.0f32; OBS_DIM];
                    obs.copy_from_slice(&flat_obs_a[batch_idx * OBS_DIM..(batch_idx + 1) * OBS_DIM]);
                    rollout.push(Transition {
                        obs,
                        action,
                        reward: per_drone_reward,
                        value,
                        log_prob,
                        done,
                    });
                    rollout_env_indices.push(env_idx);
                }

                if done {
                    ep_rewards.push(ep_reward_accum[env_idx]);
                    ep_reward_accum[env_idx] = 0.0;

                    let new_seed = total_steps + env_idx as u64;
                    let env_cfg = envs[env_idx].config();
                    if env_cfg.drones_per_side != curr_drones || env_cfg.targets_per_side != curr_targets {
                        let mut sc = make_sim_config(&config);
                        sc.drones_per_side = curr_drones;
                        sc.targets_per_side = curr_targets;
                        sc.seed = new_seed;
                        envs[env_idx] = SimRunner::new(sc);
                        let (obs_a, drone_ids_a) = envs[env_idx].observe_multi_group(0);
                        let (obs_b, drone_ids_b) = envs[env_idx].observe_multi_group(1);
                        env_states[env_idx] = EnvState { obs_a, drone_ids_a, obs_b, drone_ids_b };
                    } else {
                        let ((obs_a, drone_ids_a), (obs_b, drone_ids_b)) =
                            envs[env_idx].reset_selfplay_with_seed(new_seed);
                        env_states[env_idx] = EnvState { obs_a, drone_ids_a, obs_b, drone_ids_b };
                    }
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

        if rollout.is_empty() {
            continue;
        }

        // Compute last values for GAE via batched inference.
        let mut last_values = vec![0.0f32; config.n_envs];
        {
            let mut bootstrap_obs: Vec<&[f32]> = Vec::new();
            let mut bootstrap_env_map: Vec<(usize, usize)> = Vec::new(); // (env_idx, count_for_env)
            for (env_idx, env_st) in env_states.iter().enumerate() {
                if env_st.obs_a.is_empty() { continue; }
                let start = bootstrap_obs.len();
                for obs in &env_st.obs_a {
                    bootstrap_obs.push(obs.as_slice());
                }
                bootstrap_env_map.push((env_idx, bootstrap_obs.len() - start));
            }
            if !bootstrap_obs.is_empty() {
                let bootstrap_results = policy.act_batch(&bootstrap_obs, &mut rng);
                let mut offset = 0;
                for &(env_idx, count) in &bootstrap_env_map {
                    let mut sum = 0.0f32;
                    for i in 0..count {
                        sum += bootstrap_results[offset + i].2; // value
                    }
                    last_values[env_idx] = sum / count as f32;
                    offset += count;
                }
            }
        }

        // Split rollout by environment for GAE computation.
        let mut env_rollouts: Vec<Vec<(usize, &Transition)>> = vec![Vec::new(); config.n_envs];
        for (i, (trans, &env_idx)) in rollout.iter().zip(rollout_env_indices.iter()).enumerate() {
            env_rollouts[env_idx].push((i, trans));
        }

        let mut advantages = vec![0.0f32; rollout.len()];
        let mut returns = vec![0.0f32; rollout.len()];

        for env_idx in 0..config.n_envs {
            let env_trans: Vec<Transition> = env_rollouts[env_idx]
                .iter()
                .map(|(_, t)| (*t).clone())
                .collect();
            if env_trans.is_empty() { continue; }

            let (adv, ret) = compute_gae(&env_trans, last_values[env_idx], config.gamma, config.lam);

            for (j, &(global_idx, _)) in env_rollouts[env_idx].iter().enumerate() {
                advantages[global_idx] = adv[j];
                returns[global_idx] = ret[j];
            }
        }

        // Normalize advantages.
        let adv_mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let adv_var: f32 = advantages.iter().map(|a| (a - adv_mean).powi(2)).sum::<f32>()
            / advantages.len() as f32;
        let adv_std = adv_var.sqrt().max(1e-8);
        for a in &mut advantages {
            *a = (*a - adv_mean) / adv_std;
        }

        // Anneal LR and entropy coefficient.
        let progress = (total_steps as f32 / config.total_timesteps as f32).min(1.0);
        let current_lr = config.lr * (1.0 - progress);
        let current_ent_coef = config.ent_coef_start + (config.ent_coef_end - config.ent_coef_start) * progress;

        // PPO update (parallel backward with pre-allocated thread-local policies).
        let n = rollout.len();
        let chunk_size = (config.batch_size + n_threads - 1) / n_threads;
        let mut thread_policies: Vec<PolicyNet> = (0..n_threads)
            .map(|_| policy.clone())
            .collect();

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

                // Copy current weights to thread-local policies (no allocation).
                for tp in &mut thread_policies[..n_chunks] {
                    tp.copy_weights_from(&policy);
                }

                // Parallel backward: each thread uses its pre-allocated policy.
                thread_policies[..n_chunks]
                    .par_iter_mut()
                    .zip(chunks.into_par_iter())
                    .for_each(|(local, chunk)| {
                        local.zero_grad();
                        for &idx in chunk {
                            let t = &rollout[idx];
                            local.backward_ppo(
                                &t.obs,
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

                // Reduce gradients to main policy.
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

        // Update opponent policy periodically.
        if update_count % config.opponent_update_freq as u64 == 0 {
            opponent = policy.clone();
            println!("  Opponent updated at update {}", update_count);
        }

        // Log progress.
        if !ep_rewards.is_empty() && update_count % 5 == 0 {
            let recent: Vec<f32> = ep_rewards.iter().rev().take(100).copied().collect();
            let mean_reward: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
            let elapsed = start_time.elapsed().as_secs();
            let fps = total_steps as f64 / elapsed.max(1) as f64;
            println!(
                "Update {}: steps={}, episodes={}, mean_reward={:.2}, transitions={}, fps={:.0}, elapsed={}s, curriculum={}v{}",
                update_count, total_steps, ep_rewards.len(), mean_reward, n, fps, elapsed, curr_drones, curr_drones,
            );
        }

        // Evaluate.
        if update_count % config.eval_freq as u64 == 0 {
            let eval_seed = total_steps * 997 + update_count * 31;
            let (win_rate, mean_reward, mean_length) = evaluate(&policy, &config, eval_seed);
            println!("  Eval (self-play): win_rate_A={:.1}%, mean_reward={:.2}, mean_length={:.0}",
                win_rate * 100.0, mean_reward, mean_length);

            let ckpt_path = format!("{}/checkpoint_{}.json", config.output_dir, update_count);
            policy.save(&ckpt_path).ok();

            if win_rate > best_win_rate {
                best_win_rate = win_rate;
                policy.save(&best_model_path).ok();
                println!("  New best model saved (win_rate={:.1}%)", win_rate * 100.0);
            }
        }
    }

    policy.save(&final_model_path).ok();
    println!("\nTraining complete.");
    println!("  Total steps: {}", total_steps);
    println!("  Best win rate: {:.1}%", best_win_rate * 100.0);
    println!("  Final model: {}", final_model_path);
    println!("  Best model: {}", best_model_path);
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
                println!("  --n-envs N                Parallel environments (default 256)");
                println!("  --drones N                Drones per side (default 24)");
                println!("  --targets N               Targets per side (default 6)");
                println!("  --max-ticks N             Max ticks per episode (default 10000)");
                println!("  --lr F                    Learning rate (default 3e-4, annealed to 0)");
                println!("  --hidden N                Hidden layer size (default 128)");
                println!("  --n-steps N               Steps per rollout (default 256)");
                println!("  --batch-size N            Mini-batch size (default 256)");
                println!("  --n-epochs N              PPO epochs per update (default 10)");
                println!("  --output-dir DIR          Output directory (default results)");
                println!("  --eval-freq N             Eval frequency in updates (default 50)");
                println!("  --eval-episodes N         Eval episodes per eval (default 20)");
                println!("  --opponent-update-freq N  Update opponent every N PPO updates (default 10)");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    config
}
