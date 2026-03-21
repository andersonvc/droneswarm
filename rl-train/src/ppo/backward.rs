//! PPO backward pass implementation.

use crate::network::policy::PolicyNet;
use drone_lib::game::action_mask::{compute_action_mask, apply_mask_to_logits};
use crate::network::policy_v2::PolicyNetV2;

/// Softmax over a slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum == 0.0 || sum.is_nan() {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }
    exp.iter().map(|&e| e / sum).collect()
}

/// Extension trait to add PPO-specific backward pass to PolicyNet.
pub trait PolicyNetPPOExt {
    /// Backward pass for PPO loss on a single sample.
    /// Accumulates gradients (caller should call sgd_step after full batch).
    fn backward_ppo(
        &mut self,
        obs: &[f32],
        action: u32,
        advantage: f32,
        ret: f32,
        old_log_prob: f32,
        clip_range: f32,
        vf_coef: f32,
        ent_coef: f32,
    );
}

impl PolicyNetPPOExt for PolicyNet {
    fn backward_ppo(
        &mut self,
        obs: &[f32],
        action: u32,
        advantage: f32,
        ret: f32,
        old_log_prob: f32,
        clip_range: f32,
        vf_coef: f32,
        ent_coef: f32,
    ) {
        // Forward.
        let (logits, value) = self.forward(obs);
        let probs = softmax(&logits);
        let new_log_prob = (probs[action as usize] + 1e-8).ln();

        // Policy gradient.
        let ratio = (new_log_prob - old_log_prob).exp().clamp(0.05, 20.0);
        let clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range);
        let surr1 = ratio * advantage;
        let surr2 = clipped_ratio * advantage;
        let _policy_loss = -surr1.min(surr2);

        // Value loss.
        let _value_loss = (value - ret).powi(2);

        // Entropy bonus.
        let _entropy: f32 = probs.iter()
            .map(|&p| if p > 1e-8 { -p * p.ln() } else { 0.0 })
            .sum();

        // Total loss gradient: d(policy_loss + vf_coef * value_loss - ent_coef * entropy) / d(logits, value)

        // -- Gradient of policy loss w.r.t. logits --
        // PPO clips the gradient when ratio moves outside [1-eps, 1+eps]
        // AND the unclipped surrogate is not the minimum.
        let ratio_in_range = ratio > 1.0 - clip_range && ratio < 1.0 + clip_range;
        let unclipped_is_min = surr1 <= surr2;
        // Gradient flows through ratio when: unclipped is min, OR ratio is in range.
        let d_policy_d_logprob = if unclipped_is_min || ratio_in_range {
            -advantage * ratio
        } else {
            0.0 // clipped out, no gradient
        };

        // Softmax Jacobian: d(log_prob_a)/d(logit_j) = (1 if j==a) - prob_j
        let act_dim = logits.len();
        let mut d_logits = vec![0.0f32; act_dim];
        for j in 0..act_dim {
            let indicator = if j == action as usize { 1.0 } else { 0.0 };
            // Policy gradient contribution.
            d_logits[j] += d_policy_d_logprob * (indicator - probs[j]);

            // Entropy gradient: d(-sum p*log(p))/d(logit_j)
            // = -sum_k d(p_k * log(p_k))/d(logit_j)
            // = -sum_k (log(p_k) + 1) * p_k * (1_{k==j} - p_j)
            let mut d_ent = 0.0;
            for k in 0..act_dim {
                let indicator_k = if k == j { 1.0 } else { 0.0 };
                if probs[k] > 1e-8 {
                    d_ent += -(probs[k].ln() + 1.0) * probs[k] * (indicator_k - probs[j]);
                }
            }
            d_logits[j] -= ent_coef * d_ent;
        }

        // -- Gradient of value loss w.r.t. value output --
        let d_value = vf_coef * 2.0 * (value - ret);

        // Backward through network.
        // Actor head backward.
        let grad_h3_actor = self.actor_head.backward(&d_logits);

        // Critic head backward.
        let grad_h3_critic = self.critic_head.backward(&[d_value]);

        // Sum gradients from both heads.
        let h3_dim = grad_h3_actor.len();
        let mut grad_h3: Vec<f32> = vec![0.0; h3_dim];
        for i in 0..h3_dim {
            grad_h3[i] = grad_h3_actor[i] + grad_h3_critic[i];
        }

        // fc3 backward.
        let grad_h2 = self.fc3.backward(&grad_h3);

        // fc2 backward.
        let grad_h1 = self.fc2.backward(&grad_h2);

        // fc1 backward.
        let _grad_input = self.fc1.backward(&grad_h1);
    }
}

/// Extension trait to add PPO-specific backward pass to PolicyNetV2.
pub trait PolicyNetV2PPOExt {
    /// Backward pass for PPO loss on a single sample with entity-attention.
    /// `raw_entities` is unnormalized entity data for action masking.
    fn backward_ppo_v2(
        &mut self,
        ego: &[f32],
        entities: &[f32],
        n_entities: usize,
        raw_entities: &[f32],
        action: u32,
        advantage: f32,
        ret: f32,
        old_log_prob: f32,
        clip_range: f32,
        vf_coef: f32,
        ent_coef: f32,
    );
}

impl PolicyNetV2PPOExt for PolicyNetV2 {
    fn backward_ppo_v2(
        &mut self,
        ego: &[f32],
        entities: &[f32],
        n_entities: usize,
        raw_entities: &[f32],
        action: u32,
        advantage: f32,
        ret: f32,
        old_log_prob: f32,
        clip_range: f32,
        vf_coef: f32,
        ent_coef: f32,
    ) {
        // Forward pass (caches activations in each layer).
        let (mut logits, value, _h2) = self.forward(ego, entities, n_entities);

        // Apply action mask using raw (unnormalized) entities.
        let mask = compute_action_mask(raw_entities, n_entities);
        apply_mask_to_logits(&mut logits, &mask);

        let probs = softmax(&logits);
        let new_log_prob = (probs[action as usize] + 1e-8).ln();

        // Policy gradient.
        let ratio = (new_log_prob - old_log_prob).exp().clamp(0.05, 20.0);
        let clipped_ratio = ratio.clamp(1.0 - clip_range, 1.0 + clip_range);
        let surr1 = ratio * advantage;
        let surr2 = clipped_ratio * advantage;

        // PPO clips gradient when ratio is outside range AND unclipped is not the min.
        let ratio_in_range = ratio > 1.0 - clip_range && ratio < 1.0 + clip_range;
        let unclipped_is_min = surr1 <= surr2;
        let d_policy_d_logprob = if unclipped_is_min || ratio_in_range {
            -advantage * ratio
        } else {
            0.0
        };

        // Gradient of PPO loss w.r.t. logits (policy + entropy).
        let act_dim = logits.len();
        let mut d_logits = vec![0.0f32; act_dim];
        for j in 0..act_dim {
            let indicator = if j == action as usize { 1.0 } else { 0.0 };
            d_logits[j] += d_policy_d_logprob * (indicator - probs[j]);

            let mut d_ent = 0.0;
            for k in 0..act_dim {
                let indicator_k = if k == j { 1.0 } else { 0.0 };
                if probs[k] > 1e-8 {
                    d_ent += -(probs[k].ln() + 1.0) * probs[k] * (indicator_k - probs[j]);
                }
            }
            d_logits[j] -= ent_coef * d_ent;
        }

        // Gradient of value loss w.r.t. value output.
        let d_value = vf_coef * 2.0 * (value - ret);

        // Backward through the full network via backward_input.
        let (_grad_ego, _grad_entities) = self.backward_input(&d_logits, d_value, n_entities);
    }
}