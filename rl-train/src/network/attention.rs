//! Multi-head self-attention module for entity tokens.

use rand::Rng;
use serde::{Deserialize, Serialize};
use crate::network::layer::DenseLayer;

// ---------------------------------------------------------------------------
// Layer Normalization
// ---------------------------------------------------------------------------

/// Layer normalization: `y = gamma * (x - mean) / (std + eps) + beta`.
///
/// Normalizes over the `dim` dimension. Each call to `forward` caches the
/// input, normalized output, and standard deviation for a single entity.
/// Use `forward_batch` to process multiple entities and cache all of them
/// for the corresponding `backward_batch`.
#[derive(Clone, Serialize, Deserialize)]
pub struct LayerNorm {
    pub gamma: Vec<f32>,   // scale, [dim], init 1.0
    pub beta: Vec<f32>,    // bias, [dim], init 0.0
    pub dim: usize,

    // Gradient accumulators.
    #[serde(skip)]
    pub grad_gamma: Vec<f32>,
    #[serde(skip)]
    pub grad_beta: Vec<f32>,

    // Adam optimizer state.
    #[serde(skip)]
    pub m_gamma: Vec<f32>,
    #[serde(skip)]
    pub v_gamma: Vec<f32>,
    #[serde(skip)]
    pub m_beta: Vec<f32>,
    #[serde(skip)]
    pub v_beta: Vec<f32>,

    // Per-entity caches for backward (populated by forward_batch).
    #[serde(skip)]
    cache_inputs: Vec<Vec<f32>>,   // [n_entities][dim]
    #[serde(skip)]
    cache_normed: Vec<Vec<f32>>,   // [n_entities][dim]
    #[serde(skip)]
    cache_stds: Vec<f32>,          // [n_entities]
}

const LN_EPS: f32 = 1e-5;

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        LayerNorm {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            dim,
            grad_gamma: vec![0.0; dim],
            grad_beta: vec![0.0; dim],
            m_gamma: vec![0.0; dim],
            v_gamma: vec![0.0; dim],
            m_beta: vec![0.0; dim],
            v_beta: vec![0.0; dim],
            cache_inputs: Vec::new(),
            cache_normed: Vec::new(),
            cache_stds: Vec::new(),
        }
    }

    /// Forward pass for a single vector of length `dim`.
    /// Returns the normalized output. Does NOT cache (use `forward_batch` for training).
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.dim);
        let n = self.dim as f32;
        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let std = (var + LN_EPS).sqrt();
        let mut out = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let normed = (x[i] - mean) / std;
            out.push(self.gamma[i] * normed + self.beta[i]);
        }
        out
    }

    /// Forward pass for `n_entities` vectors, each of length `dim`.
    /// Caches all inputs, normalized values, and stds for `backward_batch`.
    ///
    /// `input`: [n_entities * dim] row-major
    /// Returns: [n_entities * dim] row-major
    pub fn forward_batch(&mut self, input: &[f32], n_entities: usize) -> Vec<f32> {
        let d = self.dim;
        debug_assert_eq!(input.len(), n_entities * d);
        let n = d as f32;

        self.cache_inputs.clear();
        self.cache_normed.clear();
        self.cache_stds.clear();

        let mut output = Vec::with_capacity(n_entities * d);
        for ent in 0..n_entities {
            let x = &input[ent * d..(ent + 1) * d];
            let mean: f32 = x.iter().sum::<f32>() / n;
            let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
            let std = (var + LN_EPS).sqrt();

            let mut normed = Vec::with_capacity(d);
            for i in 0..d {
                let xn = (x[i] - mean) / std;
                normed.push(xn);
                output.push(self.gamma[i] * xn + self.beta[i]);
            }

            self.cache_inputs.push(x.to_vec());
            self.cache_normed.push(normed);
            self.cache_stds.push(std);
        }
        output
    }

    /// Backward pass for `n_entities` vectors (must follow a `forward_batch` call).
    ///
    /// `grad_output`: [n_entities * dim] row-major
    /// Returns: [n_entities * dim] gradient w.r.t. input, row-major
    pub fn backward_batch(&mut self, grad_output: &[f32], n_entities: usize) -> Vec<f32> {
        let d = self.dim;
        let n = d as f32;
        debug_assert_eq!(grad_output.len(), n_entities * d);
        debug_assert_eq!(self.cache_inputs.len(), n_entities);

        let mut grad_input = vec![0.0f32; n_entities * d];

        for ent in 0..n_entities {
            let go = &grad_output[ent * d..(ent + 1) * d];
            let x = &self.cache_inputs[ent];
            let normed = &self.cache_normed[ent];
            let std = self.cache_stds[ent];
            let mean: f32 = x.iter().sum::<f32>() / n;

            // Accumulate parameter gradients.
            for i in 0..d {
                self.grad_gamma[i] += go[i] * normed[i];
                self.grad_beta[i] += go[i];
            }

            // d_normed = grad_output * gamma
            let mut d_normed = vec![0.0f32; d];
            for i in 0..d {
                d_normed[i] = go[i] * self.gamma[i];
            }

            // d_var = sum(d_normed * (x - mean) * -0.5 * (var + eps)^(-3/2))
            let inv_std_cubed = 1.0 / (std * std * std);
            let mut d_var = 0.0f32;
            for i in 0..d {
                d_var += d_normed[i] * (x[i] - mean) * -0.5 * inv_std_cubed;
            }

            // d_mean = sum(d_normed * -1/std) + d_var * sum(-2*(x-mean)) / N
            let mut d_mean = 0.0f32;
            let mut sum_x_minus_mean = 0.0f32;
            for i in 0..d {
                d_mean += d_normed[i] * (-1.0 / std);
                sum_x_minus_mean += -2.0 * (x[i] - mean);
            }
            d_mean += d_var * sum_x_minus_mean / n;

            // d_input[i] = d_normed[i] / std + d_var * 2*(x[i]-mean)/N + d_mean/N
            for i in 0..d {
                grad_input[ent * d + i] =
                    d_normed[i] / std
                    + d_var * 2.0 * (x[i] - mean) / n
                    + d_mean / n;
            }
        }

        grad_input
    }

    /// Zero gradient accumulators.
    pub fn zero_grad(&mut self) {
        self.grad_gamma.fill(0.0);
        self.grad_beta.fill(0.0);
    }

    /// Add accumulated gradients from another LayerNorm instance.
    pub fn add_grads_from(&mut self, other: &LayerNorm) {
        for (g, og) in self.grad_gamma.iter_mut().zip(other.grad_gamma.iter()) {
            *g += *og;
        }
        for (g, og) in self.grad_beta.iter_mut().zip(other.grad_beta.iter()) {
            *g += *og;
        }
    }

    /// Copy weights from another LayerNorm instance.
    pub fn copy_weights_from(&mut self, other: &LayerNorm) {
        self.gamma.copy_from_slice(&other.gamma);
        self.beta.copy_from_slice(&other.beta);
    }

    /// Initialize gradient and Adam state buffers (needed after deserialization).
    pub fn init_grads(&mut self) {
        self.grad_gamma = vec![0.0; self.dim];
        self.grad_beta = vec![0.0; self.dim];
        self.m_gamma = vec![0.0; self.dim];
        self.v_gamma = vec![0.0; self.dim];
        self.m_beta = vec![0.0; self.dim];
        self.v_beta = vec![0.0; self.dim];
    }

    /// Apply AdamW step to gamma and beta parameters.
    pub fn adam_step(&mut self, lr: f32, batch_size: usize, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32, t: usize) {
        let scale = 1.0 / batch_size as f32;
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);

        // Update gamma (with weight decay).
        for i in 0..self.dim {
            let g = self.grad_gamma[i] * scale;
            self.m_gamma[i] = beta1 * self.m_gamma[i] + (1.0 - beta1) * g;
            self.v_gamma[i] = beta2 * self.v_gamma[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m_gamma[i] / bc1;
            let v_hat = self.v_gamma[i] / bc2;
            self.gamma[i] -= lr * (m_hat / (v_hat.sqrt() + epsilon) + weight_decay * self.gamma[i]);
        }

        // Update beta (no weight decay on biases).
        for i in 0..self.dim {
            let g = self.grad_beta[i] * scale;
            self.m_beta[i] = beta1 * self.m_beta[i] + (1.0 - beta1) * g;
            self.v_beta[i] = beta2 * self.v_beta[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m_beta[i] / bc1;
            let v_hat = self.v_beta[i] / bc2;
            self.beta[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }

        self.grad_gamma.fill(0.0);
        self.grad_beta.fill(0.0);
    }

    /// Compute squared L2 norm of accumulated gradients.
    pub fn grad_norm_sq(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        for &g in &self.grad_gamma {
            sum_sq += g * g;
        }
        for &g in &self.grad_beta {
            sum_sq += g * g;
        }
        sum_sq
    }

    /// Scale all accumulated gradients by a factor.
    pub fn scale_grads(&mut self, scale: f32) {
        for g in &mut self.grad_gamma {
            *g *= scale;
        }
        for g in &mut self.grad_beta {
            *g *= scale;
        }
    }
}

/// Multi-head self-attention over a variable number of entity embeddings.
///
/// Projections (Q, K, V, O) are implemented as DenseLayers (relu=false).
/// Attention matmuls (Q@K^T, attn@V) are done manually per-head since they
/// operate on per-sample, variable-length sequences.
#[derive(Clone, Serialize, Deserialize)]
pub struct MultiHeadAttention {
    pub w_q: DenseLayer,  // embed_dim -> embed_dim, linear
    pub w_k: DenseLayer,  // embed_dim -> embed_dim, linear
    pub w_v: DenseLayer,  // embed_dim -> embed_dim, linear
    pub w_o: DenseLayer,  // embed_dim -> embed_dim, linear
    pub layer_norm: LayerNorm,  // post-residual layer normalization
    pub n_heads: usize,
    pub head_dim: usize,  // embed_dim / n_heads
    pub embed_dim: usize,

    // Cached values for backward pass.
    #[serde(skip)]
    cache_q: Vec<f32>,          // [n_entities, embed_dim]
    #[serde(skip)]
    cache_k: Vec<f32>,          // [n_entities, embed_dim]
    #[serde(skip)]
    cache_v: Vec<f32>,          // [n_entities, embed_dim]
    #[serde(skip)]
    cache_attn_weights: Vec<f32>, // [n_heads, n_entities, n_entities]
    #[serde(skip)]
    cache_n_entities: usize,
    // Per-entity input caches to fix DenseLayer cache corruption.
    // w_q/w_k/w_v all receive the same entity embedding, so one cache suffices.
    #[serde(skip)]
    cache_qkv_inputs: Vec<Vec<f32>>,  // per-entity inputs for Q/K/V projection backward
    #[serde(skip)]
    cache_wo_inputs: Vec<Vec<f32>>,   // per-entity inputs for W_O projection backward
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, n_heads: usize, rng: &mut impl Rng) -> Self {
        assert!(embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads");
        let head_dim = embed_dim / n_heads;
        MultiHeadAttention {
            w_q: DenseLayer::new_with_gain(embed_dim, embed_dim, false, 1.0, rng),
            w_k: DenseLayer::new_with_gain(embed_dim, embed_dim, false, 1.0, rng),
            w_v: DenseLayer::new_with_gain(embed_dim, embed_dim, false, 1.0, rng),
            w_o: DenseLayer::new_with_gain(embed_dim, embed_dim, false, 1.0, rng),
            layer_norm: LayerNorm::new(embed_dim),
            n_heads,
            head_dim,
            embed_dim,
            cache_q: Vec::new(),
            cache_k: Vec::new(),
            cache_v: Vec::new(),
            cache_attn_weights: Vec::new(),
            cache_n_entities: 0,
            cache_qkv_inputs: Vec::new(),
            cache_wo_inputs: Vec::new(),
        }
    }

    /// Forward pass through multi-head self-attention.
    ///
    /// `entity_embeds`: [n_entities, embed_dim] row-major
    /// `n_entities`: number of real (non-padding) entities
    ///
    /// Returns: [n_entities, embed_dim] row-major
    pub fn forward(&mut self, entity_embeds: &[f32], n_entities: usize) -> Vec<f32> {
        let e = self.embed_dim;
        let h = self.n_heads;
        let d = self.head_dim;
        debug_assert_eq!(entity_embeds.len(), n_entities * e);

        self.cache_n_entities = n_entities;

        // Project Q, K, V through DenseLayers (per entity).
        // Cache per-entity inputs so we can restore them before backward.
        let mut q_all = Vec::with_capacity(n_entities * e);
        let mut k_all = Vec::with_capacity(n_entities * e);
        let mut v_all = Vec::with_capacity(n_entities * e);
        self.cache_qkv_inputs.clear();
        for i in 0..n_entities {
            let slice = &entity_embeds[i * e..(i + 1) * e];
            q_all.extend_from_slice(&self.w_q.forward(slice));
            k_all.extend_from_slice(&self.w_k.forward(slice));
            v_all.extend_from_slice(&self.w_v.forward(slice));
            self.cache_qkv_inputs.push(slice.to_vec());
        }

        // Cache Q, K, V for backward.
        self.cache_q = q_all.clone();
        self.cache_k = k_all.clone();
        self.cache_v = v_all.clone();

        // Attention per head.
        // Q, K, V are [n_entities, embed_dim] = [n_entities, n_heads * head_dim]
        // Reshape to [n_heads, n_entities, head_dim] (logically).
        let scale = 1.0 / (d as f32).sqrt();
        let mut attn_weights = vec![0.0f32; h * n_entities * n_entities];
        let mut head_outputs = vec![0.0f32; n_entities * e]; // [n_entities, embed_dim]

        for head in 0..h {
            // Extract Q_h, K_h for this head: head_dim slice per entity.
            // For entity i, head h: q_all[i*e + head*d .. i*e + head*d + d]

            // Compute scores: Q_h @ K_h^T / sqrt(d) -> [n_entities, n_entities]
            for i in 0..n_entities {
                for j in 0..n_entities {
                    let mut dot = 0.0f32;
                    for kk in 0..d {
                        dot += q_all[i * e + head * d + kk] * k_all[j * e + head * d + kk];
                    }
                    attn_weights[head * n_entities * n_entities + i * n_entities + j] = dot * scale;
                }
            }

            // Softmax per row.
            for i in 0..n_entities {
                let row_start = head * n_entities * n_entities + i * n_entities;
                let row = &mut attn_weights[row_start..row_start + n_entities];
                softmax_inplace(row);
            }

            // out_h = attn @ V_h -> [n_entities, head_dim]
            for i in 0..n_entities {
                for kk in 0..d {
                    let mut sum = 0.0f32;
                    for j in 0..n_entities {
                        let w = attn_weights[head * n_entities * n_entities + i * n_entities + j];
                        sum += w * v_all[j * e + head * d + kk];
                    }
                    // Write into head_outputs at [i, head*d + kk]
                    head_outputs[i * e + head * d + kk] = sum;
                }
            }
        }

        // Cache attention weights for backward.
        self.cache_attn_weights = attn_weights;

        // Project through W_O per entity.
        // Cache per-entity inputs so we can restore them before backward.
        let mut output = Vec::with_capacity(n_entities * e);
        self.cache_wo_inputs.clear();
        for i in 0..n_entities {
            let concat_slice = &head_outputs[i * e..(i + 1) * e];
            output.extend_from_slice(&self.w_o.forward(concat_slice));
            self.cache_wo_inputs.push(concat_slice.to_vec());
        }

        // Residual connection: output += entity_embeds (input).
        for i in 0..n_entities * e {
            output[i] += entity_embeds[i];
        }

        // Post-residual layer normalization (per entity).
        let output = self.layer_norm.forward_batch(&output, n_entities);

        output
    }

    /// Backward pass through multi-head self-attention.
    ///
    /// `grad_output`: [n_entities, embed_dim] gradient from downstream
    /// `n_entities`: number of real entities (must match forward call)
    ///
    /// Returns: [n_entities, embed_dim] gradient w.r.t. entity_embeds input
    pub fn backward(&mut self, grad_output: &[f32], n_entities: usize) -> Vec<f32> {
        let e = self.embed_dim;
        let h = self.n_heads;
        let d = self.head_dim;
        let scale = 1.0 / (d as f32).sqrt();
        debug_assert_eq!(grad_output.len(), n_entities * e);
        debug_assert_eq!(self.cache_n_entities, n_entities);

        // Backward through layer normalization.
        let grad_output_ln = self.layer_norm.backward_batch(grad_output, n_entities);
        let grad_output = &grad_output_ln;

        // Backward through W_O: one per entity.
        // Restore per-entity cached input before each backward call.
        // d_concat_heads: [n_entities, embed_dim]
        let mut d_concat = vec![0.0f32; n_entities * e];
        for i in (0..n_entities).rev() {
            self.w_o.last_input = self.cache_wo_inputs[i].clone();
            let g_slice = &grad_output[i * e..(i + 1) * e];
            let d_in = self.w_o.backward(g_slice);
            d_concat[i * e..(i + 1) * e].copy_from_slice(&d_in);
        }

        // d_concat is gradient w.r.t. the concatenated head outputs.
        // Split back to per-head: d_out_h for each head at [i, head*d..head*d+d].

        // Backward through attention per head.
        let mut d_q_all = vec![0.0f32; n_entities * e];
        let mut d_k_all = vec![0.0f32; n_entities * e];
        let mut d_v_all = vec![0.0f32; n_entities * e];

        for head in 0..h {
            // d_out_h: [n_entities, head_dim] = d_concat[:, head*d..(head+1)*d]

            // Backward through out_h = attn @ V_h:
            // d_attn_raw[i][j] = sum_k d_out_h[i][k] * V_h[j][k]
            // d_V_h[j][k] += sum_i attn[i][j] * d_out_h[i][k]
            let mut d_attn_raw = vec![0.0f32; n_entities * n_entities];
            for i in 0..n_entities {
                for j in 0..n_entities {
                    let mut dot = 0.0f32;
                    for kk in 0..d {
                        dot += d_concat[i * e + head * d + kk]
                            * self.cache_v[j * e + head * d + kk];
                    }
                    d_attn_raw[i * n_entities + j] = dot;
                }
            }
            for j in 0..n_entities {
                for kk in 0..d {
                    let mut sum = 0.0f32;
                    for i in 0..n_entities {
                        let w = self.cache_attn_weights[head * n_entities * n_entities + i * n_entities + j];
                        sum += w * d_concat[i * e + head * d + kk];
                    }
                    d_v_all[j * e + head * d + kk] += sum;
                }
            }

            // Backward through softmax: d_scores = softmax_backward(d_attn_raw, attn_weights)
            let mut d_scores = vec![0.0f32; n_entities * n_entities];
            for i in 0..n_entities {
                let row_start = head * n_entities * n_entities + i * n_entities;
                let attn_row = &self.cache_attn_weights[row_start..row_start + n_entities];
                let d_raw_row = &d_attn_raw[i * n_entities..(i + 1) * n_entities];
                // softmax backward: d_score_j = attn_j * (d_raw_j - sum_k(attn_k * d_raw_k))
                let dot: f32 = attn_row.iter().zip(d_raw_row.iter()).map(|(&a, &d)| a * d).sum();
                for j in 0..n_entities {
                    d_scores[i * n_entities + j] = attn_row[j] * (d_raw_row[j] - dot);
                }
            }

            // Scale backward: d_scores already includes post-softmax grad;
            // scores were Q@K^T * scale, so d_scores_pre = d_scores * scale.
            // Backward through Q_h @ K_h^T:
            // d_Q_h[i][k] += sum_j d_scores_pre[i][j] * K_h[j][k]
            // d_K_h[j][k] += sum_i d_scores_pre[i][j] * Q_h[i][k]
            for i in 0..n_entities {
                for kk in 0..d {
                    let mut sum_q = 0.0f32;
                    for j in 0..n_entities {
                        sum_q += d_scores[i * n_entities + j] * scale
                            * self.cache_k[j * e + head * d + kk];
                    }
                    d_q_all[i * e + head * d + kk] += sum_q;
                }
            }
            for j in 0..n_entities {
                for kk in 0..d {
                    let mut sum_k = 0.0f32;
                    for i in 0..n_entities {
                        sum_k += d_scores[i * n_entities + j] * scale
                            * self.cache_q[i * e + head * d + kk];
                    }
                    d_k_all[j * e + head * d + kk] += sum_k;
                }
            }
        }

        // Backward through W_Q, W_K, W_V projections (per entity, reverse order).
        // Restore per-entity cached input before each backward call.
        let mut d_entity_embeds = vec![0.0f32; n_entities * e];
        for i in (0..n_entities).rev() {
            self.w_q.last_input = self.cache_qkv_inputs[i].clone();
            self.w_k.last_input = self.cache_qkv_inputs[i].clone();
            self.w_v.last_input = self.cache_qkv_inputs[i].clone();
            let d_q_slice = &d_q_all[i * e..(i + 1) * e];
            let d_k_slice = &d_k_all[i * e..(i + 1) * e];
            let d_v_slice = &d_v_all[i * e..(i + 1) * e];
            let d_in_q = self.w_q.backward(d_q_slice);
            let d_in_k = self.w_k.backward(d_k_slice);
            let d_in_v = self.w_v.backward(d_v_slice);
            for j in 0..e {
                d_entity_embeds[i * e + j] += d_in_q[j] + d_in_k[j] + d_in_v[j];
            }
        }

        // Residual connection gradient: pass through directly.
        for i in 0..n_entities * e {
            d_entity_embeds[i] += grad_output[i];
        }

        d_entity_embeds
    }

    /// Zero all gradient accumulators.
    pub fn zero_grad(&mut self) {
        self.w_q.zero_grad();
        self.w_k.zero_grad();
        self.w_v.zero_grad();
        self.w_o.zero_grad();
        self.layer_norm.zero_grad();
    }

    /// Add accumulated gradients from another attention module.
    pub fn add_grads_from(&mut self, other: &MultiHeadAttention) {
        self.w_q.add_grads_from(&other.w_q);
        self.w_k.add_grads_from(&other.w_k);
        self.w_v.add_grads_from(&other.w_v);
        self.w_o.add_grads_from(&other.w_o);
        self.layer_norm.add_grads_from(&other.layer_norm);
    }

    /// Copy weights from another attention module.
    pub fn copy_weights_from(&mut self, other: &MultiHeadAttention) {
        self.w_q.copy_weights_from(&other.w_q);
        self.w_k.copy_weights_from(&other.w_k);
        self.w_v.copy_weights_from(&other.w_v);
        self.w_o.copy_weights_from(&other.w_o);
        self.layer_norm.copy_weights_from(&other.layer_norm);
    }

    /// Initialize gradient buffers (needed after deserialization).
    pub fn init_grads(&mut self) {
        self.w_q.init_grads();
        self.w_k.init_grads();
        self.w_v.init_grads();
        self.w_o.init_grads();
        self.layer_norm.init_grads();
    }

    /// Apply AdamW step to all projection layers.
    pub fn adam_step(&mut self, lr: f32, batch_size: usize, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32, t: usize) {
        self.w_q.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.w_k.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.w_v.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.w_o.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.layer_norm.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
    }

    /// Compute L2 norm of all accumulated gradients.
    pub fn grad_norm_sq(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        for layer in [&self.w_q, &self.w_k, &self.w_v, &self.w_o] {
            for &g in &layer.grad_weights {
                sum_sq += g * g;
            }
            for &g in &layer.grad_biases {
                sum_sq += g * g;
            }
        }
        sum_sq += self.layer_norm.grad_norm_sq();
        sum_sq
    }

    /// Scale all accumulated gradients by a factor.
    pub fn scale_grads(&mut self, scale: f32) {
        for layer in [&mut self.w_q, &mut self.w_k, &mut self.w_v, &mut self.w_o] {
            for g in &mut layer.grad_weights {
                *g *= scale;
            }
            for g in &mut layer.grad_biases {
                *g *= scale;
            }
        }
        self.layer_norm.scale_grads(scale);
    }
}

/// In-place softmax over a mutable slice.
/// Guards against NaN when all inputs are -inf (produces uniform distribution).
fn softmax_inplace(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum == 0.0 || sum.is_nan() {
        let uniform = 1.0 / row.len() as f32;
        for v in row.iter_mut() {
            *v = uniform;
        }
    } else {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}
