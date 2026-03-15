//! Actor-Critic policy network for reinforcement learning.

use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::network::layer::{DenseLayer, matmul_bias_relu};

/// Softmax over a slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

/// Actor-Critic MLP policy network.
#[derive(Clone, Serialize, Deserialize)]
pub struct PolicyNet {
    pub fc1: DenseLayer,
    pub fc2: DenseLayer,
    pub fc3: DenseLayer,
    pub actor_head: DenseLayer,
    pub critic_head: DenseLayer,
}

impl PolicyNet {
    pub fn new(obs_dim: usize, act_dim: usize, hidden: usize, rng: &mut impl Rng) -> Self {
        PolicyNet {
            fc1: DenseLayer::new(obs_dim, hidden, true, rng),         // gain=sqrt(2) for ReLU
            fc2: DenseLayer::new(hidden, hidden, true, rng),          // gain=sqrt(2) for ReLU
            fc3: DenseLayer::new(hidden, hidden, true, rng),          // gain=sqrt(2) for ReLU
            actor_head: DenseLayer::new_with_gain(hidden, act_dim, false, 0.01, rng), // small gain → near-uniform initial policy
            critic_head: DenseLayer::new_with_gain(hidden, 1, false, 1.0, rng),       // gain=1.0 for value head
        }
    }

    /// Forward pass. Returns (action_logits, value).
    pub fn forward(&mut self, obs: &[f32]) -> (Vec<f32>, f32) {
        let h1 = self.fc1.forward(obs);
        let h2 = self.fc2.forward(&h1);
        let h3 = self.fc3.forward(&h2);
        let logits = self.actor_head.forward(&h3);
        let value = self.critic_head.forward(&h3);
        (logits, value[0])
    }

    /// Sample action from policy. Returns (action, log_prob, value).
    pub fn act(&mut self, obs: &[f32], rng: &mut impl Rng) -> (u32, f32, f32) {
        let (logits, value) = self.forward(obs);
        let probs = softmax(&logits);

        // Sample from categorical.
        let u: f32 = rng.gen();
        let mut cum = 0.0;
        let mut action = 0u32;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if u <= cum {
                action = i as u32;
                break;
            }
        }

        let log_prob = (probs[action as usize] + 1e-8).ln();
        (action, log_prob, value)
    }

    /// Parallel batched forward pass: splits observations across rayon threads,
    /// each running independent matmuls. Returns (logits_flat, values_flat).
    /// logits_flat has length batch_size * act_dim, values_flat has length batch_size.
    fn forward_batch_par(&self, obs_batch: &[&[f32]]) -> (Vec<f32>, Vec<f32>) {
        let batch_size = obs_batch.len();
        let n_threads = rayon::current_num_threads();
        let chunk_size = (batch_size + n_threads - 1) / n_threads;
        let act_dim = self.actor_head.out_dim;

        let chunks: Vec<(Vec<f32>, Vec<f32>)> = obs_batch
            .par_chunks(chunk_size)
            .map(|chunk| {
                let m = chunk.len();
                let obs_dim = self.fc1.in_dim;
                let mut input = Vec::with_capacity(m * obs_dim);
                for obs in chunk {
                    input.extend_from_slice(obs);
                }
                let h1 = matmul_bias_relu(&input, &self.fc1.weights, &self.fc1.biases,
                    m, self.fc1.out_dim, self.fc1.in_dim, true);
                let h2 = matmul_bias_relu(&h1, &self.fc2.weights, &self.fc2.biases,
                    m, self.fc2.out_dim, self.fc2.in_dim, true);
                let h3 = matmul_bias_relu(&h2, &self.fc3.weights, &self.fc3.biases,
                    m, self.fc3.out_dim, self.fc3.in_dim, true);
                let logits = matmul_bias_relu(&h3, &self.actor_head.weights, &self.actor_head.biases,
                    m, self.actor_head.out_dim, self.actor_head.in_dim, false);
                let values = matmul_bias_relu(&h3, &self.critic_head.weights, &self.critic_head.biases,
                    m, 1, self.critic_head.in_dim, false);
                (logits, values)
            })
            .collect();

        let mut all_logits = Vec::with_capacity(batch_size * act_dim);
        let mut all_values = Vec::with_capacity(batch_size);
        for (logits, values) in chunks {
            all_logits.extend_from_slice(&logits);
            all_values.extend_from_slice(&values);
        }
        (all_logits, all_values)
    }

    /// Batched inference: parallel forward pass + sequential action sampling.
    /// Uses rayon to split observations across threads for the matmul,
    /// then samples actions sequentially (fast, just RNG).
    ///
    /// Returns one (action, log_prob, value) per observation.
    pub fn act_batch(&self, obs_batch: &[&[f32]], rng: &mut impl Rng) -> Vec<(u32, f32, f32)> {
        let batch_size = obs_batch.len();
        if batch_size == 0 {
            return vec![];
        }

        // Parallel forward pass across rayon threads.
        let (logits, values) = self.forward_batch_par(obs_batch);

        // Sequential action sampling (fast — just softmax + RNG per obs).
        let act_dim = self.actor_head.out_dim;
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let logit_slice = &logits[i * act_dim..(i + 1) * act_dim];
            let probs = softmax(logit_slice);

            let u: f32 = rng.gen();
            let mut cum = 0.0;
            let mut action = 0u32;
            for (j, &p) in probs.iter().enumerate() {
                cum += p;
                if u <= cum {
                    action = j as u32;
                    break;
                }
            }

            let log_prob = (probs[action as usize] + 1e-8).ln();
            results.push((action, log_prob, values[i]));
        }

        results
    }

    /// Compute the global L2 norm of all accumulated gradients.
    pub fn grad_norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        for layer in [&self.fc1, &self.fc2, &self.fc3, &self.actor_head, &self.critic_head] {
            for &g in &layer.grad_weights {
                sum_sq += g * g;
            }
            for &g in &layer.grad_biases {
                sum_sq += g * g;
            }
        }
        sum_sq.sqrt()
    }

    /// Clip accumulated gradients by global L2 norm. Returns the original norm.
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let norm = self.grad_norm();
        if norm > max_norm {
            let scale = max_norm / norm;
            for layer in [&mut self.fc1, &mut self.fc2, &mut self.fc3, &mut self.actor_head, &mut self.critic_head] {
                for g in &mut layer.grad_weights {
                    *g *= scale;
                }
                for g in &mut layer.grad_biases {
                    *g *= scale;
                }
            }
        }
        norm
    }

    /// Apply SGD step to all layers.
    pub fn sgd_step(&mut self, lr: f32, batch_size: usize) {
        self.fc1.sgd_step(lr, batch_size);
        self.fc2.sgd_step(lr, batch_size);
        self.fc3.sgd_step(lr, batch_size);
        self.actor_head.sgd_step(lr, batch_size);
        self.critic_head.sgd_step(lr, batch_size);
    }

    /// Apply AdamW step to all layers.
    pub fn adam_step(&mut self, lr: f32, batch_size: usize, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32, t: usize) {
        self.fc1.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.fc2.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.fc3.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.actor_head.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.critic_head.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
    }

    /// Zero all gradients.
    pub fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
        self.fc3.zero_grad();
        self.actor_head.zero_grad();
        self.critic_head.zero_grad();
    }

    /// Add accumulated gradients from another policy (for parallel backward reduction).
    pub fn add_grads_from(&mut self, other: &PolicyNet) {
        self.fc1.add_grads_from(&other.fc1);
        self.fc2.add_grads_from(&other.fc2);
        self.fc3.add_grads_from(&other.fc3);
        self.actor_head.add_grads_from(&other.actor_head);
        self.critic_head.add_grads_from(&other.critic_head);
    }

    /// Copy weights from another policy into this one (no allocation).
    pub fn copy_weights_from(&mut self, other: &PolicyNet) {
        self.fc1.copy_weights_from(&other.fc1);
        self.fc2.copy_weights_from(&other.fc2);
        self.fc3.copy_weights_from(&other.fc3);
        self.actor_head.copy_weights_from(&other.actor_head);
        self.critic_head.copy_weights_from(&other.critic_head);
    }

    /// Initialize gradient buffers (needed after deserialization).
    pub fn init_grads(&mut self) {
        self.fc1.init_grads();
        self.fc2.init_grads();
        self.fc3.init_grads();
        self.actor_head.init_grads();
        self.critic_head.init_grads();
    }

    /// Save model to JSON file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)
    }

    /// Load model from JSON file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let mut net: Self = serde_json::from_str(&json)?;
        net.init_grads();
        Ok(net)
    }
}