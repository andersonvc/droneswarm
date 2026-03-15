//! MAPPO Centralized Critic.
//!
//! Sees per-drone hidden state + mean pool of all drones in the env.
//! Produces a centralized value estimate per drone.

use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::network::layer::DenseLayer;

/// MAPPO Centralized Critic.
///
/// Input: concat(per_drone_h2, mean_pool(all_drones_h2)) → 2*hidden_dim
/// Output: centralized value estimate (scalar)
#[derive(Clone, Serialize, Deserialize)]
pub struct CentralizedCritic {
    pub fc1: DenseLayer,       // 2*hidden → hidden, ReLU
    pub fc2: DenseLayer,       // hidden → hidden, ReLU
    pub value_head: DenseLayer, // hidden → 1, linear
    pub hidden_dim: usize,
}

impl CentralizedCritic {
    pub fn new(hidden_dim: usize, rng: &mut impl Rng) -> Self {
        CentralizedCritic {
            fc1: DenseLayer::new(2 * hidden_dim, hidden_dim, true, rng),
            fc2: DenseLayer::new(hidden_dim, hidden_dim, true, rng),
            value_head: DenseLayer::new_with_gain(hidden_dim, 1, false, 1.0, rng),
            hidden_dim,
        }
    }

    /// Forward pass. Returns centralized value estimate.
    ///
    /// * `h2` - Per-drone hidden representation [hidden_dim]
    /// * `mean_pool_h2` - Mean of all drones' h2 in the env [hidden_dim]
    pub fn forward(&mut self, h2: &[f32], mean_pool_h2: &[f32]) -> f32 {
        let mut input = Vec::with_capacity(2 * self.hidden_dim);
        input.extend_from_slice(h2);
        input.extend_from_slice(mean_pool_h2);

        let h = self.fc1.forward(&input);
        let h = self.fc2.forward(&h);
        let v = self.value_head.forward(&h);
        v[0]
    }

    /// Backward pass. Returns gradient w.r.t. concat(h2, mean_pool_h2).
    ///
    /// `grad_value` is the gradient of the loss w.r.t. the value output.
    pub fn backward(&mut self, grad_value: f32) -> Vec<f32> {
        let grad_h = self.value_head.backward(&[grad_value]);
        let grad_h = self.fc2.backward(&grad_h);
        self.fc1.backward(&grad_h)
    }

    pub fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
        self.value_head.zero_grad();
    }

    pub fn add_grads_from(&mut self, other: &CentralizedCritic) {
        self.fc1.add_grads_from(&other.fc1);
        self.fc2.add_grads_from(&other.fc2);
        self.value_head.add_grads_from(&other.value_head);
    }

    pub fn copy_weights_from(&mut self, other: &CentralizedCritic) {
        self.fc1.copy_weights_from(&other.fc1);
        self.fc2.copy_weights_from(&other.fc2);
        self.value_head.copy_weights_from(&other.value_head);
    }

    pub fn adam_step(&mut self, lr: f32, batch_size: usize, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32, t: usize) {
        self.fc1.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.fc2.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.value_head.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
    }

    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let norm = self.grad_norm();
        if norm > max_norm {
            let scale = max_norm / norm;
            for layer in [&mut self.fc1, &mut self.fc2, &mut self.value_head] {
                for g in &mut layer.grad_weights { *g *= scale; }
                for g in &mut layer.grad_biases { *g *= scale; }
            }
        }
        norm
    }

    pub fn grad_norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        for layer in [&self.fc1, &self.fc2, &self.value_head] {
            for &g in &layer.grad_weights { sum_sq += g * g; }
            for &g in &layer.grad_biases { sum_sq += g * g; }
        }
        sum_sq.sqrt()
    }

    pub fn init_grads(&mut self) {
        self.fc1.init_grads();
        self.fc2.init_grads();
        self.value_head.init_grads();
    }
}
