//! Entity-attention Actor-Critic policy network (V2).
//!
//! Replaces the flat MLP with:
//! 1. Separate ego and entity encoders
//! 2. Multi-head self-attention over entity embeddings
//! 3. Mean pooling of attended entities
//! 4. Trunk MLP with actor and local-value heads

use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::network::layer::{DenseLayer, matmul_bias_relu};
use crate::network::attention::MultiHeadAttention;

/// Softmax over a slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

/// Entity-attention Actor-Critic policy network.
///
/// Architecture:
///   ego -> ego_encoder -> ego_embed [embed_dim]
///   entities -> entity_encoder -> entity_embeds [n_entities, embed_dim]
///   entity_embeds -> attn -> attn_out [n_entities, embed_dim]
///   attn_out -> mean_pool -> pooled [embed_dim]
///   concat(ego_embed, pooled) -> fc1 -> fc2 -> actor_head / local_value_head
#[derive(Clone, Serialize, Deserialize)]
pub struct PolicyNetV2 {
    pub ego_encoder: DenseLayer,      // EGO_DIM -> embed_dim, ReLU
    pub entity_encoder: DenseLayer,   // ENTITY_DIM -> embed_dim, ReLU
    pub attn: MultiHeadAttention,     // embed_dim -> embed_dim, n_heads
    pub fc1: DenseLayer,              // 2*embed_dim -> hidden, ReLU
    pub fc2: DenseLayer,              // hidden -> hidden, ReLU
    pub actor_head: DenseLayer,       // hidden -> ACT_DIM, linear
    pub local_value_head: DenseLayer, // hidden -> 1, linear
    pub embed_dim: usize,
}

impl PolicyNetV2 {
    /// Create a new entity-attention policy network.
    ///
    /// * `ego_dim` - Ego feature size (EGO_DIM = 25)
    /// * `entity_dim` - Per-entity feature size (ENTITY_DIM = 8)
    /// * `act_dim` - Action space size (ACT_DIM = 19)
    /// * `embed_dim` - Embedding dimension (64)
    /// * `hidden` - Hidden layer dimension (256)
    /// * `rng` - Random number generator for initialization
    pub fn new(
        ego_dim: usize,
        entity_dim: usize,
        act_dim: usize,
        embed_dim: usize,
        hidden: usize,
        rng: &mut impl Rng,
    ) -> Self {
        PolicyNetV2 {
            ego_encoder: DenseLayer::new(ego_dim, embed_dim, true, rng),         // gain=sqrt(2) for ReLU
            entity_encoder: DenseLayer::new(entity_dim, embed_dim, true, rng),   // gain=sqrt(2) for ReLU
            attn: MultiHeadAttention::new(embed_dim, 4, rng),                    // 4 heads
            fc1: DenseLayer::new(2 * embed_dim, hidden, true, rng),              // gain=sqrt(2) for ReLU
            fc2: DenseLayer::new(hidden, hidden, true, rng),                     // gain=sqrt(2) for ReLU
            actor_head: DenseLayer::new_with_gain(hidden, act_dim, false, 0.01, rng), // small gain -> near-uniform
            local_value_head: DenseLayer::new_with_gain(hidden, 1, false, 1.0, rng),  // gain=1.0
            embed_dim,
        }
    }

    /// Forward pass. Returns (action_logits, local_value, h2_hidden).
    ///
    /// * `ego` - Ego features [EGO_DIM]
    /// * `entities` - Entity features [n_entities * ENTITY_DIM], row-major
    /// * `n_entities` - Number of real entities
    ///
    /// `h2_hidden` is the 256-dim trunk representation for future MAPPO/QMIX (Phase 4).
    pub fn forward(&mut self, ego: &[f32], entities: &[f32], n_entities: usize) -> (Vec<f32>, f32, Vec<f32>) {
        let e = self.embed_dim;
        let entity_dim = self.entity_encoder.in_dim;

        // 1. Encode ego.
        let ego_embed = self.ego_encoder.forward(ego);

        // 2. Encode each entity.
        let mut entity_embeds = Vec::with_capacity(n_entities * e);
        for i in 0..n_entities {
            let ent_slice = &entities[i * entity_dim..(i + 1) * entity_dim];
            entity_embeds.extend_from_slice(&self.entity_encoder.forward(ent_slice));
        }

        // 3. Self-attention over entity embeddings.
        let attn_out = if n_entities > 0 {
            self.attn.forward(&entity_embeds, n_entities)
        } else {
            Vec::new()
        };

        // 4. Mean pool attended entities.
        let pooled = if n_entities > 0 {
            let mut pool = vec![0.0f32; e];
            let inv_n = 1.0 / n_entities as f32;
            for i in 0..n_entities {
                for j in 0..e {
                    pool[j] += attn_out[i * e + j] * inv_n;
                }
            }
            pool
        } else {
            vec![0.0f32; e]
        };

        // 5. Concatenate ego_embed and pooled.
        let mut trunk_input = Vec::with_capacity(2 * e);
        trunk_input.extend_from_slice(&ego_embed);
        trunk_input.extend_from_slice(&pooled);

        // 6-7. Trunk MLP.
        let h1 = self.fc1.forward(&trunk_input);
        let h2 = self.fc2.forward(&h1);

        // 8-9. Heads.
        let logits = self.actor_head.forward(&h2);
        let value = self.local_value_head.forward(&h2);

        (logits, value[0], h2)
    }

    /// Sample action from policy. Returns (action, log_prob, value).
    pub fn act(&mut self, ego: &[f32], entities: &[f32], n_entities: usize, rng: &mut impl Rng) -> (u32, f32, f32) {
        let (logits, value, _h2) = self.forward(ego, entities, n_entities);
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

    /// Batched inference with partial batching optimization.
    ///
    /// Batches ego encoding and entity encoding via single matmul calls,
    /// runs per-sample attention (variable-length), then batches the trunk.
    ///
    /// Returns one (action, log_prob, value) per sample.
    pub fn act_batch(
        &self,
        egos: &[&[f32]],
        entities_list: &[&[f32]],
        n_entities_list: &[usize],
        rng: &mut impl Rng,
    ) -> Vec<(u32, f32, f32)> {
        let (results, _h2) = self.act_batch_with_h2(egos, entities_list, n_entities_list, rng);
        results
    }

    /// Like act_batch but also returns h2 hidden representations (flat).
    ///
    /// Returns (actions_results, h2_flat) where h2_flat is [batch_size * hidden_dim].
    /// Used by MAPPO centralized critic and QMIX mixer during rollout collection.
    pub fn act_batch_with_h2(
        &self,
        egos: &[&[f32]],
        entities_list: &[&[f32]],
        n_entities_list: &[usize],
        rng: &mut impl Rng,
    ) -> (Vec<(u32, f32, f32)>, Vec<f32>) {
        let batch_size = egos.len();
        if batch_size == 0 {
            return (vec![], vec![]);
        }
        let e = self.embed_dim;
        let act_dim = self.actor_head.out_dim;
        let hidden = self.fc2.out_dim;

        // Split the full forward pass across rayon threads.
        // Each thread handles a chunk of samples end-to-end:
        // ego encode → entity encode → attention → pool → trunk → heads.
        let n_threads = rayon::current_num_threads();
        let chunk_size = (batch_size + n_threads - 1) / n_threads;

        let chunks: Vec<(usize, usize)> = (0..batch_size)
            .step_by(chunk_size)
            .map(|s| (s, (s + chunk_size).min(batch_size)))
            .collect();

        let chunk_results: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = chunks
            .par_iter()
            .map(|&(start, end)| {
                let m = end - start;
                let ego_dim = self.ego_encoder.in_dim;
                let entity_dim = self.entity_encoder.in_dim;

                let mut ego_flat = Vec::with_capacity(m * ego_dim);
                for i in start..end { ego_flat.extend_from_slice(egos[i]); }
                let ego_embeds = matmul_bias_relu(
                    &ego_flat, &self.ego_encoder.weights, &self.ego_encoder.biases,
                    m, e, ego_dim, true,
                );

                let total_ent: usize = n_entities_list[start..end].iter().sum();
                let mut ent_flat = Vec::with_capacity(total_ent * entity_dim);
                for i in start..end {
                    ent_flat.extend_from_slice(&entities_list[i][..n_entities_list[i] * entity_dim]);
                }
                let all_ent_embeds = if total_ent > 0 {
                    matmul_bias_relu(
                        &ent_flat, &self.entity_encoder.weights, &self.entity_encoder.biases,
                        total_ent, e, entity_dim, true,
                    )
                } else { Vec::new() };

                let mut trunk_inputs = Vec::with_capacity(m * 2 * e);
                let mut ent_off = 0usize;
                for local_i in 0..m {
                    let n_ent = n_entities_list[start + local_i];
                    let ego_embed = &ego_embeds[local_i * e..(local_i + 1) * e];
                    let pooled = if n_ent > 0 {
                        let ent_embeds = &all_ent_embeds[ent_off * e..(ent_off + n_ent) * e];
                        let mut attn_clone = self.attn.clone();
                        let attn_out = attn_clone.forward(ent_embeds, n_ent);
                        let inv_n = 1.0 / n_ent as f32;
                        let mut pool = vec![0.0f32; e];
                        for j in 0..n_ent { for k in 0..e { pool[k] += attn_out[j * e + k] * inv_n; } }
                        pool
                    } else { vec![0.0f32; e] };
                    trunk_inputs.extend_from_slice(ego_embed);
                    trunk_inputs.extend_from_slice(&pooled);
                    ent_off += n_ent;
                }

                let trunk_dim = 2 * e;
                let h1 = matmul_bias_relu(&trunk_inputs, &self.fc1.weights, &self.fc1.biases, m, self.fc1.out_dim, trunk_dim, true);
                let h2 = matmul_bias_relu(&h1, &self.fc2.weights, &self.fc2.biases, m, self.fc2.out_dim, self.fc2.in_dim, true);
                let logits = matmul_bias_relu(&h2, &self.actor_head.weights, &self.actor_head.biases, m, act_dim, self.actor_head.in_dim, false);
                let values = matmul_bias_relu(&h2, &self.local_value_head.weights, &self.local_value_head.biases, m, 1, self.local_value_head.in_dim, false);
                (logits, values, h2)
            })
            .collect();

        // Gather from all chunks.
        let mut logits_flat = Vec::with_capacity(batch_size * act_dim);
        let mut values_flat = Vec::with_capacity(batch_size);
        let mut h2_flat = Vec::with_capacity(batch_size * hidden);
        for (l, v, h) in &chunk_results {
            logits_flat.extend_from_slice(l);
            values_flat.extend_from_slice(v);
            h2_flat.extend_from_slice(h);
        }

        // --- Sequential action sampling ---
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let logit_slice = &logits_flat[i * act_dim..(i + 1) * act_dim];
            let probs = softmax(logit_slice);
            let u: f32 = rng.gen();
            let mut cum = 0.0;
            let mut action = 0u32;
            for (j, &p) in probs.iter().enumerate() {
                cum += p;
                if u <= cum { action = j as u32; break; }
            }
            let log_prob = (probs[action as usize] + 1e-8).ln();
            results.push((action, log_prob, values_flat[i]));
        }

        (results, h2_flat)
    }

    /// Backward pass through the full network. Returns gradient w.r.t. input
    /// (not typically needed, but required for completeness).
    ///
    /// `grad_logits` - gradient from actor head [ACT_DIM]
    /// `grad_value` - gradient from value head [1]
    pub fn backward_input(
        &mut self,
        grad_logits: &[f32],
        grad_value: f32,
        n_entities: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let e = self.embed_dim;

        // Backward through heads.
        let grad_h2_actor = self.actor_head.backward(grad_logits);
        let grad_h2_value = self.local_value_head.backward(&[grad_value]);

        // Sum gradients from both heads.
        let hidden = grad_h2_actor.len();
        let mut grad_h2 = vec![0.0f32; hidden];
        for i in 0..hidden {
            grad_h2[i] = grad_h2_actor[i] + grad_h2_value[i];
        }

        // fc2 backward.
        let grad_h1 = self.fc2.backward(&grad_h2);

        // fc1 backward.
        let grad_trunk = self.fc1.backward(&grad_h1);

        // Split trunk gradient into ego_embed and pooled parts.
        let grad_ego_embed = &grad_trunk[..e];
        let grad_pooled = &grad_trunk[e..2 * e];

        // Backward through mean pool: distribute 1/n_entities to each entity.
        let grad_ego = self.ego_encoder.backward(grad_ego_embed);

        let mut grad_entities = Vec::new();
        if n_entities > 0 {
            let inv_n = 1.0 / n_entities as f32;
            let mut grad_attn_out = vec![0.0f32; n_entities * e];
            for i in 0..n_entities {
                for j in 0..e {
                    grad_attn_out[i * e + j] = grad_pooled[j] * inv_n;
                }
            }

            // Backward through attention.
            let grad_entity_embeds = self.attn.backward(&grad_attn_out, n_entities);

            // Backward through entity encoder (per entity, reverse order).
            let entity_dim = self.entity_encoder.in_dim;
            grad_entities = vec![0.0f32; n_entities * entity_dim];
            for i in (0..n_entities).rev() {
                let g_slice = &grad_entity_embeds[i * e..(i + 1) * e];
                let d_in = self.entity_encoder.backward(g_slice);
                grad_entities[i * entity_dim..(i + 1) * entity_dim].copy_from_slice(&d_in);
            }
        }

        (grad_ego, grad_entities)
    }

    /// Compute the global L2 norm of all accumulated gradients.
    pub fn grad_norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        for layer in [
            &self.ego_encoder, &self.entity_encoder,
            &self.fc1, &self.fc2, &self.actor_head, &self.local_value_head,
        ] {
            for &g in &layer.grad_weights {
                sum_sq += g * g;
            }
            for &g in &layer.grad_biases {
                sum_sq += g * g;
            }
        }
        sum_sq += self.attn.grad_norm_sq();
        sum_sq.sqrt()
    }

    /// Clip accumulated gradients by global L2 norm. Returns the original norm.
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let norm = self.grad_norm();
        if norm > max_norm {
            let scale = max_norm / norm;
            for layer in [
                &mut self.ego_encoder, &mut self.entity_encoder,
                &mut self.fc1, &mut self.fc2, &mut self.actor_head, &mut self.local_value_head,
            ] {
                for g in &mut layer.grad_weights {
                    *g *= scale;
                }
                for g in &mut layer.grad_biases {
                    *g *= scale;
                }
            }
            self.attn.scale_grads(scale);
        }
        norm
    }

    /// Apply AdamW step to all layers.
    pub fn adam_step(&mut self, lr: f32, batch_size: usize, beta1: f32, beta2: f32, epsilon: f32, weight_decay: f32, t: usize) {
        self.ego_encoder.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.entity_encoder.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.attn.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.fc1.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.fc2.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.actor_head.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
        self.local_value_head.adam_step(lr, batch_size, beta1, beta2, epsilon, weight_decay, t);
    }

    /// Zero all gradients.
    pub fn zero_grad(&mut self) {
        self.ego_encoder.zero_grad();
        self.entity_encoder.zero_grad();
        self.attn.zero_grad();
        self.fc1.zero_grad();
        self.fc2.zero_grad();
        self.actor_head.zero_grad();
        self.local_value_head.zero_grad();
    }

    /// Add accumulated gradients from another policy (for parallel gradient reduction).
    pub fn add_grads_from(&mut self, other: &PolicyNetV2) {
        self.ego_encoder.add_grads_from(&other.ego_encoder);
        self.entity_encoder.add_grads_from(&other.entity_encoder);
        self.attn.add_grads_from(&other.attn);
        self.fc1.add_grads_from(&other.fc1);
        self.fc2.add_grads_from(&other.fc2);
        self.actor_head.add_grads_from(&other.actor_head);
        self.local_value_head.add_grads_from(&other.local_value_head);
    }

    /// Copy weights from another policy into this one (no allocation).
    pub fn copy_weights_from(&mut self, other: &PolicyNetV2) {
        self.ego_encoder.copy_weights_from(&other.ego_encoder);
        self.entity_encoder.copy_weights_from(&other.entity_encoder);
        self.attn.copy_weights_from(&other.attn);
        self.fc1.copy_weights_from(&other.fc1);
        self.fc2.copy_weights_from(&other.fc2);
        self.actor_head.copy_weights_from(&other.actor_head);
        self.local_value_head.copy_weights_from(&other.local_value_head);
    }

    /// Initialize gradient buffers (needed after deserialization).
    pub fn init_grads(&mut self) {
        self.ego_encoder.init_grads();
        self.entity_encoder.init_grads();
        self.attn.init_grads();
        self.fc1.init_grads();
        self.fc2.init_grads();
        self.actor_head.init_grads();
        self.local_value_head.init_grads();
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
