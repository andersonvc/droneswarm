//! Generalized Advantage Estimation (GAE) for PPO.

use drone_lib::sim_runner::{EGO_DIM, ENTITY_DIM, MAX_ENTITIES, OBS_DIM_V2};

/// A single rollout transition with V2 structured observations.
#[derive(Clone)]
pub struct Transition {
    /// Ego features (EGO_DIM floats).
    pub ego_obs: Vec<f32>,
    /// Entity tokens flattened (n_entities * ENTITY_DIM floats).
    pub entity_obs: Vec<f32>,
    /// Number of real entities in entity_obs.
    pub n_entities: usize,
    pub action: u32,
    /// Combined team_reward + individual_reward.
    pub reward: f32,
    pub value: f32,
    pub log_prob: f32,
    pub done: bool,
    /// Drone died but episode continues (for death bootstrapping).
    pub drone_died: bool,
    /// Bootstrap value when drone dies mid-episode.
    pub team_value_at_death: f32,
}

impl Transition {
    /// Flatten ego + padded entities into a single vector for MLP compatibility.
    /// Returns a vector of OBS_DIM_V2 = EGO_DIM + MAX_ENTITIES * ENTITY_DIM elements.
    pub fn flat_obs(&self) -> Vec<f32> {
        let mut flat = vec![0.0f32; OBS_DIM_V2];
        flat[..EGO_DIM].copy_from_slice(&self.ego_obs);
        let ent_len = self.entity_obs.len().min(MAX_ENTITIES * ENTITY_DIM);
        flat[EGO_DIM..EGO_DIM + ent_len].copy_from_slice(&self.entity_obs[..ent_len]);
        flat
    }
}

/// Compute GAE advantages and returns with death bootstrapping.
///
/// When `drone_died && !done`, uses `team_value_at_death` as the bootstrap
/// value instead of 0, so the drone's trajectory doesn't get penalized for
/// the team continuing without it.
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
        let next_value = if i + 1 < n {
            transitions[i + 1].value
        } else {
            last_value
        };

        // Determine the effective next value for bootstrapping.
        let next_non_terminal = if transitions[i].done {
            0.0
        } else if transitions[i].drone_died {
            // Drone died but episode continues: bootstrap with team value.
            // Reset GAE accumulator since this drone's trajectory ends here.
            gae = 0.0;
            let bootstrap = transitions[i].team_value_at_death;
            let delta = transitions[i].reward + gamma * bootstrap - transitions[i].value;
            advantages[i] = delta;
            returns[i] = delta + transitions[i].value;
            continue;
        } else {
            1.0
        };

        let delta =
            transitions[i].reward + gamma * next_value * next_non_terminal - transitions[i].value;
        gae = delta + gamma * lam * next_non_terminal * gae;
        advantages[i] = gae;
        returns[i] = gae + transitions[i].value;
    }

    (advantages, returns)
}
