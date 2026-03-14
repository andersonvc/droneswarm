//! Hand-rolled MLP for the tiny policy network (64 → 128 → 128 → 14+1).
//!
//! No framework needed for ~21K parameters. Supports forward pass,
//! backpropagation, SGD weight updates, and batched inference.
//!
//! On macOS/Apple Silicon, batched matmuls use the Accelerate framework
//! (cblas_sgemm) for SIMD-optimized linear algebra.

use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Accelerate framework FFI (macOS only)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    /// C = alpha * op(A) * op(B) + beta * C
    /// Row-major: Order=101, NoTrans=111, Trans=112
    fn cblas_sgemm(
        order: i32, trans_a: i32, trans_b: i32,
        m: i32, n: i32, k: i32,
        alpha: f32, a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32, c: *mut f32, ldc: i32,
    );
}

/// Batched matrix multiply: C = A * W^T + bias (with optional ReLU).
///
/// A: (m × k) row-major — batch of input vectors
/// W: (n × k) row-major — weight matrix (n = out_dim, k = in_dim)
/// bias: (n,)
/// Returns C: (m × n) row-major — batch of output vectors
fn matmul_bias_relu(a: &[f32], w: &[f32], bias: &[f32], m: usize, n: usize, k: usize, relu: bool) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    #[cfg(target_os = "macos")]
    {
        // C = A * W^T using Accelerate's cblas_sgemm.
        unsafe {
            cblas_sgemm(
                101, 111, 112,                    // RowMajor, NoTrans, Trans
                m as i32, n as i32, k as i32,
                1.0, a.as_ptr(), k as i32,        // A: (m × k)
                w.as_ptr(), k as i32,             // W: (n × k), transposed → (k × n)
                0.0, c.as_mut_ptr(), n as i32,    // C: (m × n)
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        // Fallback: manual matmul (auto-vectorized by LLVM on most platforms).
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            for j in 0..n {
                let w_row = &w[j * k..(j + 1) * k];
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a_row[kk] * w_row[kk];
                }
                c[i * n + j] = sum;
            }
        }
    }

    // Add bias and optional ReLU.
    for i in 0..m {
        let row = &mut c[i * n..(i + 1) * n];
        for j in 0..n {
            row[j] += bias[j];
            if relu && row[j] < 0.0 {
                row[j] = 0.0;
            }
        }
    }

    c
}

/// A dense (fully connected) layer with optional ReLU activation.
#[derive(Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    pub weights: Vec<f32>,  // [out_dim * in_dim], row-major
    pub biases: Vec<f32>,   // [out_dim]
    pub in_dim: usize,
    pub out_dim: usize,
    pub relu: bool,

    // Gradient accumulators.
    #[serde(skip)]
    pub grad_weights: Vec<f32>,
    #[serde(skip)]
    pub grad_biases: Vec<f32>,

    // Cache for backprop.
    #[serde(skip)]
    pub last_input: Vec<f32>,
    #[serde(skip)]
    pub last_pre_activation: Vec<f32>,
}

impl DenseLayer {
    pub fn new(in_dim: usize, out_dim: usize, relu: bool, rng: &mut impl Rng) -> Self {
        Self::new_with_gain(in_dim, out_dim, relu, if relu { 2.0f32.sqrt() } else { 1.0 }, rng)
    }

    /// Create a layer with orthogonal initialization at a specific gain.
    pub fn new_with_gain(in_dim: usize, out_dim: usize, relu: bool, gain: f32, rng: &mut impl Rng) -> Self {
        let weights = orthogonal_init(in_dim, out_dim, gain, rng);
        let biases = vec![0.0; out_dim];

        DenseLayer {
            weights,
            biases,
            in_dim,
            out_dim,
            relu,
            grad_weights: vec![0.0; out_dim * in_dim],
            grad_biases: vec![0.0; out_dim],
            last_input: Vec::new(),
            last_pre_activation: Vec::new(),
        }
    }

    /// Forward pass for a single sample. Caches for backprop.
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.in_dim);
        self.last_input = input.to_vec();

        let mut output = vec![0.0; self.out_dim];
        for o in 0..self.out_dim {
            let mut sum = self.biases[o];
            let row_start = o * self.in_dim;
            for i in 0..self.in_dim {
                sum += self.weights[row_start + i] * input[i];
            }
            output[o] = sum;
        }

        self.last_pre_activation = output.clone();

        if self.relu {
            for v in &mut output {
                *v = v.max(0.0);
            }
        }
        output
    }

    /// Backward pass. Returns gradient w.r.t. input, accumulates weight/bias grads.
    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        debug_assert_eq!(grad_output.len(), self.out_dim);

        // Apply ReLU derivative.
        let mut effective_grad = grad_output.to_vec();
        if self.relu {
            for (i, g) in effective_grad.iter_mut().enumerate() {
                if self.last_pre_activation[i] <= 0.0 {
                    *g = 0.0;
                }
            }
        }

        // Accumulate weight and bias gradients.
        for o in 0..self.out_dim {
            self.grad_biases[o] += effective_grad[o];
            let row_start = o * self.in_dim;
            for i in 0..self.in_dim {
                self.grad_weights[row_start + i] += effective_grad[o] * self.last_input[i];
            }
        }

        // Compute gradient w.r.t. input.
        let mut grad_input = vec![0.0; self.in_dim];
        for i in 0..self.in_dim {
            for o in 0..self.out_dim {
                grad_input[i] += self.weights[o * self.in_dim + i] * effective_grad[o];
            }
        }
        grad_input
    }

    /// Apply SGD update and zero gradients.
    pub fn sgd_step(&mut self, lr: f32, batch_size: usize) {
        let scale = lr / batch_size as f32;
        for (w, g) in self.weights.iter_mut().zip(self.grad_weights.iter_mut()) {
            *w -= scale * *g;
            *g = 0.0;
        }
        for (b, g) in self.biases.iter_mut().zip(self.grad_biases.iter_mut()) {
            *b -= scale * *g;
            *g = 0.0;
        }
    }

    /// Zero gradient accumulators.
    pub fn zero_grad(&mut self) {
        self.grad_weights.fill(0.0);
        self.grad_biases.fill(0.0);
    }

    /// Add gradients from another layer (for parallel gradient reduction).
    pub fn add_grads_from(&mut self, other: &DenseLayer) {
        for (g, og) in self.grad_weights.iter_mut().zip(other.grad_weights.iter()) {
            *g += *og;
        }
        for (g, og) in self.grad_biases.iter_mut().zip(other.grad_biases.iter()) {
            *g += *og;
        }
    }

    /// Initialize gradient accumulators (needed after deserialization).
    pub fn init_grads(&mut self) {
        self.grad_weights = vec![0.0; self.out_dim * self.in_dim];
        self.grad_biases = vec![0.0; self.out_dim];
    }
}

/// Actor-Critic MLP policy network.
#[derive(Clone, Serialize, Deserialize)]
pub struct PolicyNet {
    pub fc1: DenseLayer,
    pub fc2: DenseLayer,
    pub actor_head: DenseLayer,
    pub critic_head: DenseLayer,
}

impl PolicyNet {
    pub fn new(obs_dim: usize, act_dim: usize, hidden: usize, rng: &mut impl Rng) -> Self {
        PolicyNet {
            fc1: DenseLayer::new(obs_dim, hidden, true, rng),         // gain=sqrt(2) for ReLU
            fc2: DenseLayer::new(hidden, hidden, true, rng),          // gain=sqrt(2) for ReLU
            actor_head: DenseLayer::new_with_gain(hidden, act_dim, false, 0.01, rng), // small gain → near-uniform initial policy
            critic_head: DenseLayer::new_with_gain(hidden, 1, false, 1.0, rng),       // gain=1.0 for value head
        }
    }

    /// Forward pass. Returns (action_logits, value).
    pub fn forward(&mut self, obs: &[f32]) -> (Vec<f32>, f32) {
        let h1 = self.fc1.forward(obs);
        let h2 = self.fc2.forward(&h1);
        let logits = self.actor_head.forward(&h2);
        let value = self.critic_head.forward(&h2);
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

    /// Batched inference: forward pass for many observations at once.
    /// Uses Accelerate (cblas_sgemm) on macOS for SIMD-optimized matmul.
    /// Does NOT cache activations (inference only, not for backprop).
    ///
    /// Returns one (action, log_prob, value) per observation.
    pub fn act_batch(&self, obs_batch: &[&[f32]], rng: &mut impl Rng) -> Vec<(u32, f32, f32)> {
        let batch_size = obs_batch.len();
        if batch_size == 0 {
            return vec![];
        }

        // Flatten obs into contiguous row-major matrix (batch_size × obs_dim).
        let obs_dim = self.fc1.in_dim;
        let mut input = Vec::with_capacity(batch_size * obs_dim);
        for obs in obs_batch {
            debug_assert_eq!(obs.len(), obs_dim);
            input.extend_from_slice(obs);
        }

        // Forward: input → fc1 (ReLU) → fc2 (ReLU) → actor_head, critic_head
        let h1 = matmul_bias_relu(&input, &self.fc1.weights, &self.fc1.biases,
            batch_size, self.fc1.out_dim, self.fc1.in_dim, true);
        let h2 = matmul_bias_relu(&h1, &self.fc2.weights, &self.fc2.biases,
            batch_size, self.fc2.out_dim, self.fc2.in_dim, true);
        let logits = matmul_bias_relu(&h2, &self.actor_head.weights, &self.actor_head.biases,
            batch_size, self.actor_head.out_dim, self.actor_head.in_dim, false);
        let values = matmul_bias_relu(&h2, &self.critic_head.weights, &self.critic_head.biases,
            batch_size, 1, self.critic_head.in_dim, false);

        // Sample actions per observation.
        let act_dim = self.actor_head.out_dim;
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let logit_slice = &logits[i * act_dim..(i + 1) * act_dim];
            let probs = softmax(logit_slice);

            // Categorical sample.
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
            let value = values[i];
            results.push((action, log_prob, value));
        }

        results
    }

    /// Backward pass for PPO loss on a single sample.
    /// Accumulates gradients (caller should call sgd_step after full batch).
    pub fn backward_ppo(
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
        let ratio = (new_log_prob - old_log_prob).exp();
        let surr1 = ratio * advantage;
        let surr2 = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * advantage;
        let _policy_loss = -surr1.min(surr2);

        // Value loss.
        let _value_loss = (value - ret).powi(2);

        // Entropy bonus.
        let _entropy: f32 = probs.iter()
            .map(|&p| if p > 1e-8 { -p * p.ln() } else { 0.0 })
            .sum();

        // Total loss gradient: d(policy_loss + vf_coef * value_loss - ent_coef * entropy) / d(logits, value)

        // -- Gradient of policy loss w.r.t. logits --
        // d(policy_loss)/d(logits) via d(policy_loss)/d(probs) * d(probs)/d(logits)
        // d(policy_loss)/d(log_prob_action) = -(ratio clipped gradient)
        let clipped = ratio > 1.0 - clip_range && ratio < 1.0 + clip_range;
        let use_surr1 = surr1 <= surr2;
        let d_policy_d_logprob = if use_surr1 || clipped {
            -advantage * ratio
        } else {
            0.0 // clipped, no gradient
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
        let grad_h2_actor = self.actor_head.backward(&d_logits);

        // Critic head backward.
        let grad_h2_critic = self.critic_head.backward(&[d_value]);

        // Sum gradients from both heads.
        let h2_dim = grad_h2_actor.len();
        let mut grad_h2: Vec<f32> = vec![0.0; h2_dim];
        for i in 0..h2_dim {
            grad_h2[i] = grad_h2_actor[i] + grad_h2_critic[i];
        }

        // fc2 backward.
        let grad_h1 = self.fc2.backward(&grad_h2);

        // fc1 backward.
        let _grad_input = self.fc1.backward(&grad_h1);
    }

    /// Compute the global L2 norm of all accumulated gradients.
    pub fn grad_norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        for layer in [&self.fc1, &self.fc2, &self.actor_head, &self.critic_head] {
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
            for layer in [&mut self.fc1, &mut self.fc2, &mut self.actor_head, &mut self.critic_head] {
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
        self.actor_head.sgd_step(lr, batch_size);
        self.critic_head.sgd_step(lr, batch_size);
    }

    /// Zero all gradients.
    pub fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
        self.actor_head.zero_grad();
        self.critic_head.zero_grad();
    }

    /// Add accumulated gradients from another policy (for parallel backward reduction).
    pub fn add_grads_from(&mut self, other: &PolicyNet) {
        self.fc1.add_grads_from(&other.fc1);
        self.fc2.add_grads_from(&other.fc2);
        self.actor_head.add_grads_from(&other.actor_head);
        self.critic_head.add_grads_from(&other.critic_head);
    }

    /// Initialize gradient buffers (needed after deserialization).
    pub fn init_grads(&mut self) {
        self.fc1.init_grads();
        self.fc2.init_grads();
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

/// Orthogonal weight initialization (simplified QR via Gram-Schmidt).
///
/// Generates an (out_dim × in_dim) weight matrix where rows are orthonormal
/// (scaled by `gain`). This is the standard initialization for PPO networks.
fn orthogonal_init(in_dim: usize, out_dim: usize, gain: f32, rng: &mut impl Rng) -> Vec<f32> {
    let rows = out_dim;
    let cols = in_dim;
    let n = rows.max(cols);

    // Generate random matrix.
    let mut a: Vec<f32> = (0..n * cols).map(|_| rng.gen_range(-1.0..1.0f32)).collect();

    // Gram-Schmidt orthogonalization on the first `rows` row vectors.
    for i in 0..rows.min(n) {
        // Subtract projections onto previous rows.
        for j in 0..i {
            let mut dot = 0.0f32;
            for k in 0..cols {
                dot += a[i * cols + k] * a[j * cols + k];
            }
            for k in 0..cols {
                a[i * cols + k] -= dot * a[j * cols + k];
            }
        }
        // Normalize.
        let norm: f32 = (0..cols).map(|k| a[i * cols + k] * a[i * cols + k]).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for k in 0..cols {
                a[i * cols + k] /= norm;
            }
        }
    }

    // Extract the first `rows` rows and scale by gain.
    let mut weights = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for k in 0..cols {
            weights[i * cols + k] = a[i * cols + k] * gain;
        }
    }
    weights
}

/// Softmax over a slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

/// A single rollout transition.
#[derive(Clone)]
pub struct Transition {
    pub obs: Vec<f32>,
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
