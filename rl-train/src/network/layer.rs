//! Dense (fully connected) layer implementation with optional ReLU activation.

use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::network::init::orthogonal_init;

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
pub fn matmul_bias_relu(a: &[f32], w: &[f32], bias: &[f32], m: usize, n: usize, k: usize, relu: bool) -> Vec<f32> {
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

    // Adam optimizer state.
    #[serde(skip)]
    pub adam_m_weights: Vec<f32>,  // First moment
    #[serde(skip)]
    pub adam_v_weights: Vec<f32>,  // Second moment
    #[serde(skip)]
    pub adam_m_biases: Vec<f32>,
    #[serde(skip)]
    pub adam_v_biases: Vec<f32>,

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
            adam_m_weights: vec![0.0; out_dim * in_dim],
            adam_v_weights: vec![0.0; out_dim * in_dim],
            adam_m_biases: vec![0.0; out_dim],
            adam_v_biases: vec![0.0; out_dim],
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

    /// Apply AdamW update (decoupled weight decay) and zero gradients.
    ///
    /// Weight decay is applied directly to weights, not through the gradient.
    /// This prevents decay from being diluted by Adam's moment estimates.
    /// Biases are not decayed (standard practice).
    pub fn adam_step(&mut self, lr: f32, batch_size: usize, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32, t: usize) {
        let scale = 1.0 / batch_size as f32;
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);

        for i in 0..self.weights.len() {
            let g = self.grad_weights[i] * scale;
            self.adam_m_weights[i] = beta1 * self.adam_m_weights[i] + (1.0 - beta1) * g;
            self.adam_v_weights[i] = beta2 * self.adam_v_weights[i] + (1.0 - beta2) * g * g;
            let m_hat = self.adam_m_weights[i] / bc1;
            let v_hat = self.adam_v_weights[i] / bc2;
            // AdamW: decay weights directly, not through gradient
            self.weights[i] -= lr * (m_hat / (v_hat.sqrt() + epsilon) + weight_decay * self.weights[i]);
        }
        for i in 0..self.biases.len() {
            let g = self.grad_biases[i] * scale;
            self.adam_m_biases[i] = beta1 * self.adam_m_biases[i] + (1.0 - beta1) * g;
            self.adam_v_biases[i] = beta2 * self.adam_v_biases[i] + (1.0 - beta2) * g * g;
            let m_hat = self.adam_m_biases[i] / bc1;
            let v_hat = self.adam_v_biases[i] / bc2;
            // No weight decay on biases
            self.biases[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }
        self.grad_weights.fill(0.0);
        self.grad_biases.fill(0.0);
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

    /// Copy weights and biases from another layer (no allocation if sizes match).
    pub fn copy_weights_from(&mut self, other: &DenseLayer) {
        self.weights.copy_from_slice(&other.weights);
        self.biases.copy_from_slice(&other.biases);
    }

    /// Initialize gradient accumulators (needed after deserialization).
    pub fn init_grads(&mut self) {
        let n_w = self.out_dim * self.in_dim;
        self.grad_weights = vec![0.0; n_w];
        self.grad_biases = vec![0.0; self.out_dim];
        self.adam_m_weights = vec![0.0; n_w];
        self.adam_v_weights = vec![0.0; n_w];
        self.adam_m_biases = vec![0.0; self.out_dim];
        self.adam_v_biases = vec![0.0; self.out_dim];
    }
}