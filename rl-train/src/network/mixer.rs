//! QMIX Value Decomposition Mixer.
//!
//! Uses hypernetworks to generate monotonic mixing weights from global state.
//! Combines per-drone local values into a team value while preserving
//! Individual-Global-Max (IGM) property: argmax of team value = argmax of locals.

use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::network::layer::DenseLayer;

const QMIX_HIDDEN: usize = 32;

/// ELU activation: x if x >= 0, else exp(x) - 1.
fn elu(x: f32) -> f32 {
    if x >= 0.0 { x } else { x.exp() - 1.0 }
}

/// Soft sign for abs() gradient: avoids dead gradient at zero.
/// Returns sign(x) but clamps to 1.0 when |x| < epsilon.
fn soft_sign(x: f32) -> f32 {
    if x.abs() < 1e-6 { 1.0 } else { x.signum() }
}

/// ELU derivative: 1 if x >= 0, else exp(x).
fn elu_deriv(x: f32) -> f32 {
    if x >= 0.0 { 1.0 } else { x.exp() }
}

/// QMIX hypernetwork mixer for value decomposition.
///
/// Takes per-drone local values and a global state vector,
/// produces a monotonic team value via learned mixing weights.
///
/// Architecture:
///   w1 = |hyper_w1(global_state)| → [max_agents, qmix_hidden]
///   b1 = hyper_b1(global_state)   → [qmix_hidden]
///   hidden = ELU(w1^T @ local_values + b1)
///   w2 = |hyper_w2(global_state)| → [qmix_hidden]
///   b2 = hyper_b2(global_state)   → [1]
///   V_team = w2 . hidden + b2
#[derive(Clone, Serialize, Deserialize)]
pub struct QMIXMixer {
    pub hyper_w1: DenseLayer, // global_state_dim → max_agents * qmix_hidden (linear)
    pub hyper_b1: DenseLayer, // global_state_dim → qmix_hidden (linear)
    pub hyper_w2: DenseLayer, // global_state_dim → qmix_hidden (linear)
    pub hyper_b2: DenseLayer, // global_state_dim → 1 (linear)
    pub max_agents: usize,
    pub global_state_dim: usize,

    // Cached for backward.
    #[serde(skip)]
    cache_local_values: Vec<f32>,     // [max_agents]
    #[serde(skip)]
    cache_w1: Vec<f32>,               // [max_agents * QMIX_HIDDEN] (after abs)
    #[serde(skip)]
    cache_w1_raw: Vec<f32>,           // [max_agents * QMIX_HIDDEN] (before abs)
    #[serde(skip)]
    cache_b1: Vec<f32>,               // [QMIX_HIDDEN]
    #[serde(skip)]
    cache_hidden_pre: Vec<f32>,       // [QMIX_HIDDEN] (before ELU)
    #[serde(skip)]
    cache_hidden: Vec<f32>,           // [QMIX_HIDDEN] (after ELU)
    #[serde(skip)]
    cache_w2: Vec<f32>,               // [QMIX_HIDDEN] (after abs)
    #[serde(skip)]
    cache_w2_raw: Vec<f32>,           // [QMIX_HIDDEN] (before abs)
    #[serde(skip)]
    cache_n_agents: usize,
}

impl QMIXMixer {
    /// Create a new QMIX mixer.
    ///
    /// * `global_state_dim` - Dimension of global state (hidden_dim + global_obs_features)
    /// * `max_agents` - Maximum number of agents (for padding)
    pub fn new(global_state_dim: usize, max_agents: usize, rng: &mut impl Rng) -> Self {
        QMIXMixer {
            hyper_w1: DenseLayer::new_with_gain(global_state_dim, max_agents * QMIX_HIDDEN, false, 0.1, rng),
            hyper_b1: DenseLayer::new_with_gain(global_state_dim, QMIX_HIDDEN, false, 0.1, rng),
            hyper_w2: DenseLayer::new_with_gain(global_state_dim, QMIX_HIDDEN, false, 0.1, rng),
            hyper_b2: DenseLayer::new_with_gain(global_state_dim, 1, false, 0.1, rng),
            max_agents,
            global_state_dim,
            cache_local_values: Vec::new(),
            cache_w1: Vec::new(),
            cache_w1_raw: Vec::new(),
            cache_b1: Vec::new(),
            cache_hidden_pre: Vec::new(),
            cache_hidden: Vec::new(),
            cache_w2: Vec::new(),
            cache_w2_raw: Vec::new(),
            cache_n_agents: 0,
        }
    }

    /// Forward pass. Returns team value.
    ///
    /// * `local_values` - Per-agent local values [n_agents]
    /// * `global_state` - Global state vector [global_state_dim]
    /// * `n_agents` - Number of actual agents (rest are padded zeros)
    pub fn forward(&mut self, local_values: &[f32], global_state: &[f32], n_agents: usize) -> f32 {
        let n = self.max_agents;
        let qh = QMIX_HIDDEN;

        // Pad local values to max_agents.
        let mut padded = vec![0.0f32; n];
        let actual = n_agents.min(n);
        padded[..actual].copy_from_slice(&local_values[..actual]);
        self.cache_local_values = padded.clone();
        self.cache_n_agents = n_agents;

        // Generate mixing weights from global state via hypernetworks.
        let w1_raw = self.hyper_w1.forward(global_state); // [n * qh]
        let b1 = self.hyper_b1.forward(global_state);     // [qh]
        let w2_raw = self.hyper_w2.forward(global_state);  // [qh]
        let b2 = self.hyper_b2.forward(global_state);      // [1]

        // Monotonicity: abs(weights).
        let w1: Vec<f32> = w1_raw.iter().map(|x| x.abs()).collect();
        let w2: Vec<f32> = w2_raw.iter().map(|x| x.abs()).collect();

        // hidden = ELU(padded @ w1_matrix + b1)
        // w1 is [n, qh] row-major: w1[i * qh + j] is weight from agent i to hidden j.
        let mut hidden_pre = vec![0.0f32; qh];
        for j in 0..qh {
            let mut sum = b1[j];
            for i in 0..n {
                sum += padded[i] * w1[i * qh + j];
            }
            hidden_pre[j] = sum;
        }
        let hidden: Vec<f32> = hidden_pre.iter().map(|&x| elu(x)).collect();

        // V_team = w2 . hidden + b2
        let mut v_team = b2[0];
        for j in 0..qh {
            v_team += w2[j] * hidden[j];
        }

        // Cache for backward.
        self.cache_w1_raw = w1_raw;
        self.cache_w1 = w1;
        self.cache_b1 = b1;
        self.cache_hidden_pre = hidden_pre;
        self.cache_hidden = hidden;
        self.cache_w2_raw = w2_raw;
        self.cache_w2 = w2;

        v_team
    }

    /// Backward pass. Returns gradient w.r.t. local_values [n_agents].
    ///
    /// `grad_v_team` is the gradient of the loss w.r.t. team value output.
    /// Accumulates gradients in hyper_w1, hyper_b1, hyper_w2, hyper_b2.
    pub fn backward(&mut self, grad_v_team: f32) -> Vec<f32> {
        let n = self.max_agents;
        let qh = QMIX_HIDDEN;

        // d_v_team/d_hidden[j] = w2[j]
        // d_v_team/d_w2[j] = hidden[j]
        // d_v_team/d_b2 = 1

        // Gradient w.r.t. b2.
        let d_b2 = vec![grad_v_team];
        self.hyper_b2.backward(&d_b2);

        // Gradient w.r.t. w2_raw (through abs).
        let mut d_w2_raw = vec![0.0f32; qh];
        for j in 0..qh {
            let d_w2_j = grad_v_team * self.cache_hidden[j];
            // d|x|/dx = sign(x), with soft_sign to avoid zero gradient at x=0.
            d_w2_raw[j] = d_w2_j * soft_sign(self.cache_w2_raw[j]);
        }
        self.hyper_w2.backward(&d_w2_raw);

        // Gradient w.r.t. hidden.
        let mut d_hidden = vec![0.0f32; qh];
        for j in 0..qh {
            d_hidden[j] = grad_v_team * self.cache_w2[j];
        }

        // Through ELU: d_hidden_pre[j] = d_hidden[j] * elu_deriv(hidden_pre[j])
        let mut d_hidden_pre = vec![0.0f32; qh];
        for j in 0..qh {
            d_hidden_pre[j] = d_hidden[j] * elu_deriv(self.cache_hidden_pre[j]);
        }

        // Gradient w.r.t. b1.
        self.hyper_b1.backward(&d_hidden_pre);

        // Gradient w.r.t. w1_raw (through abs).
        // d_hidden_pre[j]/d_w1[i*qh+j] = padded[i]
        let mut d_w1_raw = vec![0.0f32; n * qh];
        for i in 0..n {
            for j in 0..qh {
                let d_w1_ij = d_hidden_pre[j] * self.cache_local_values[i];
                d_w1_raw[i * qh + j] = d_w1_ij * soft_sign(self.cache_w1_raw[i * qh + j]);
            }
        }
        self.hyper_w1.backward(&d_w1_raw);

        // Gradient w.r.t. local_values (padded).
        // d_hidden_pre[j]/d_padded[i] = w1[i*qh+j]
        let mut d_padded = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..qh {
                d_padded[i] += d_hidden_pre[j] * self.cache_w1[i * qh + j];
            }
        }

        // Return only the actual agents' gradients.
        let actual = self.cache_n_agents.min(n);
        d_padded[..actual].to_vec()
    }

    pub fn zero_grad(&mut self) {
        self.hyper_w1.zero_grad();
        self.hyper_b1.zero_grad();
        self.hyper_w2.zero_grad();
        self.hyper_b2.zero_grad();
    }

    pub fn add_grads_from(&mut self, other: &QMIXMixer) {
        self.hyper_w1.add_grads_from(&other.hyper_w1);
        self.hyper_b1.add_grads_from(&other.hyper_b1);
        self.hyper_w2.add_grads_from(&other.hyper_w2);
        self.hyper_b2.add_grads_from(&other.hyper_b2);
    }

    pub fn copy_weights_from(&mut self, other: &QMIXMixer) {
        self.hyper_w1.copy_weights_from(&other.hyper_w1);
        self.hyper_b1.copy_weights_from(&other.hyper_b1);
        self.hyper_w2.copy_weights_from(&other.hyper_w2);
        self.hyper_b2.copy_weights_from(&other.hyper_b2);
    }

    pub fn adam_step(&mut self, lr: f32, batch_size: usize, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32, t: usize) {
        self.hyper_w1.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.hyper_b1.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.hyper_w2.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.hyper_b2.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
    }

    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let norm = self.grad_norm();
        if norm > max_norm {
            let scale = max_norm / norm;
            for layer in [&mut self.hyper_w1, &mut self.hyper_b1, &mut self.hyper_w2, &mut self.hyper_b2] {
                for g in &mut layer.grad_weights { *g *= scale; }
                for g in &mut layer.grad_biases { *g *= scale; }
            }
        }
        norm
    }

    pub fn grad_norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        for layer in [&self.hyper_w1, &self.hyper_b1, &self.hyper_w2, &self.hyper_b2] {
            for &g in &layer.grad_weights { sum_sq += g * g; }
            for &g in &layer.grad_biases { sum_sq += g * g; }
        }
        sum_sq.sqrt()
    }

    pub fn init_grads(&mut self) {
        self.hyper_w1.init_grads();
        self.hyper_b1.init_grads();
        self.hyper_w2.init_grads();
        self.hyper_b2.init_grads();
    }
}
