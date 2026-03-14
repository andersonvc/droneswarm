//! Evaluate a trained multi-agent PPO model's win rate against a fixed opponent.

use drone_lib::sim_runner::{GameResult, SimConfig, SimRunner};
use rl_train::mlp::PolicyNet;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: evaluate <model.json> [--episodes N] [--drones N] [--targets N]");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut n_episodes = 100usize;
    let mut drones_per_side = 24u32;
    let mut targets_per_side = 6u32;
    let mut max_ticks = 10000u32;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--episodes" => { i += 1; n_episodes = args[i].parse().unwrap(); }
            "--drones" => { i += 1; drones_per_side = args[i].parse().unwrap(); }
            "--targets" => { i += 1; targets_per_side = args[i].parse().unwrap(); }
            "--max-ticks" => { i += 1; max_ticks = args[i].parse().unwrap(); }
            _ => { eprintln!("Unknown argument: {}", args[i]); std::process::exit(1); }
        }
        i += 1;
    }

    let mut policy = PolicyNet::load(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {}", e);
        std::process::exit(1);
    });

    let mut rng = ChaCha8Rng::seed_from_u64(12345);

    let sim_config = SimConfig {
        drones_per_side,
        targets_per_side,
        max_ticks,
        ..Default::default()
    };

    let mut wins = 0u32;
    let mut losses = 0u32;
    let mut draws = 0u32;
    let mut total_reward = 0.0f64;
    let mut total_length = 0u64;

    for ep in 0..n_episodes {
        let mut env = SimRunner::new(sim_config.clone());
        let (mut obs_list, mut drone_ids) = env.reset_multi_with_seed(10000 + ep as u64);
        let mut ep_reward = 0.0f32;
        let mut ep_len = 0u64;
        let mut done = false;
        let mut result = GameResult::InProgress;

        while !done {
            let mut actions: Vec<(usize, u32)> = Vec::with_capacity(obs_list.len());
            for (obs, &drone_id) in obs_list.iter().zip(drone_ids.iter()) {
                let (action, _, _) = policy.act(obs, &mut rng);
                actions.push((drone_id, action));
            }

            let step = env.step_multi(&actions);
            ep_reward += step.reward;
            ep_len += 1;
            done = step.terminated || step.truncated;
            result = step.game_result;
            obs_list = step.observations;
            drone_ids = step.drone_ids;
        }

        match result {
            GameResult::AWins => wins += 1,
            GameResult::BWins => losses += 1,
            _ => draws += 1,
        }
        total_reward += ep_reward as f64;
        total_length += ep_len;
    }

    let win_rate = wins as f32 / n_episodes as f32 * 100.0;
    let loss_rate = losses as f32 / n_episodes as f32 * 100.0;
    let draw_rate = draws as f32 / n_episodes as f32 * 100.0;
    let mean_reward = total_reward / n_episodes as f64;
    let mean_length = total_length as f64 / n_episodes as f64;

    println!();
    println!("{}", "=".repeat(50));
    println!("Evaluation Results ({} episodes)", n_episodes);
    println!("{}", "=".repeat(50));
    println!("Win rate:     {:.1}% ({}/{})", win_rate, wins, n_episodes);
    println!("Loss rate:    {:.1}% ({}/{})", loss_rate, losses, n_episodes);
    println!("Draw rate:    {:.1}% ({}/{})", draw_rate, draws, n_episodes);
    println!("Mean reward:  {:.2}", mean_reward);
    println!("Mean length:  {:.0}", mean_length);
    println!("{}", "=".repeat(50));
}
