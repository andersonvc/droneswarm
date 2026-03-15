//! Generalized Advantage Estimation (GAE) for PPO.

use drone_lib::sim_runner::OBS_DIM;

/// A single rollout transition with inline observation (no heap allocation).
#[derive(Clone, Copy)]
pub struct Transition {
    pub obs: [f32; OBS_DIM],
    pub action: u32,
    pub reward: f32,
    pub value: f32,
    pub log_prob: f32,
    pub done: bool,
}

/// Compute GAE advantages and returns.
pub fn compute_gae(
    transitions: &[Transition],
    last_value: f32,
    gamma: f32,
    lam: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = transitions.len();
    let mut advantages = vec![0.0f32; n];
    let mut returns = vec![0.0f32; n];
    let mut gae = 0.0f32;

    for i in (0..n).rev() {
        let next_value = if i + 1 < n { transitions[i + 1].value } else { last_value };
        let next_non_terminal = if transitions[i].done { 0.0 } else { 1.0 };

        let delta = transitions[i].reward + gamma * next_value * next_non_terminal - transitions[i].value;
        gae = delta + gamma * lam * next_non_terminal * gae;
        advantages[i] = gae;
        returns[i] = gae + transitions[i].value;
    }

    (advantages, returns)
}
