//! Inference-only MLP for running trained policy networks.
//!
//! Deserializes the same JSON format as rl-train's PolicyNet
//! but includes only the forward pass — no gradients, no backprop.
//! Designed to run in WASM with zero overhead.

use serde::Deserialize;

/// A dense layer (inference only).
#[derive(Clone, Deserialize)]
pub struct DenseLayer {
    pub weights: Vec<f32>, // [out_dim * in_dim], row-major
    pub biases: Vec<f32>,  // [out_dim]
    pub in_dim: usize,
    pub out_dim: usize,
    pub relu: bool,
}

impl DenseLayer {
    /// Forward pass for a single sample.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.in_dim);
        let mut output = vec![0.0; self.out_dim];
        for o in 0..self.out_dim {
            let mut sum = self.biases[o];
            let row_start = o * self.in_dim;
            for i in 0..self.in_dim {
                sum += self.weights[row_start + i] * input[i];
            }
            if self.relu {
                sum = sum.max(0.0);
            }
            output[o] = sum;
        }
        output
    }
}

/// Inference-only actor-critic policy network.
///
/// Architecture: fc1(relu) → fc2(relu) → fc3(relu) → actor_head(linear)
/// Only the actor head is used for action selection.
#[derive(Clone, Deserialize)]
pub struct InferenceNet {
    pub fc1: DenseLayer,
    pub fc2: DenseLayer,
    pub fc3: DenseLayer,
    pub actor_head: DenseLayer,
    // critic_head is deserialized but unused for inference
    pub critic_head: DenseLayer,
}

/// Observation normalizer for inference (loaded from trained model).
///
/// Stores the running mean and variance computed during training.
/// Applied to observations before feeding them to the network.
#[derive(Clone, Deserialize)]
pub struct ObsNormalizer {
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
}

impl ObsNormalizer {
    /// Normalize a single observation: `(x - mean) / sqrt(var + 1e-8)`, clipped to [-10, 10].
    pub fn normalize(&self, input: &[f32]) -> Vec<f32> {
        input
            .iter()
            .zip(self.mean.iter().zip(self.var.iter()))
            .map(|(&x, (&m, &v))| ((x - m) / (v + 1e-8).sqrt()).clamp(-10.0, 10.0))
            .collect()
    }
}

// ============================================================================
// V2 Inference: Entity-Attention Policy Network
// ============================================================================

/// Inference-only multi-head self-attention.
///
/// Mirrors training MultiHeadAttention without gradient caches.
#[derive(Clone, Deserialize)]
pub struct InferenceAttention {
    pub w_q: DenseLayer,
    pub w_k: DenseLayer,
    pub w_v: DenseLayer,
    pub w_o: DenseLayer,
    pub n_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,
}

impl InferenceAttention {
    /// Forward pass: self-attention over entity embeddings.
    ///
    /// * `entity_embeds` - [n_entities, embed_dim] row-major
    /// * `n_entities` - number of real entities
    ///
    /// Returns [n_entities, embed_dim] attended embeddings.
    pub fn forward(&self, entity_embeds: &[f32], n_entities: usize) -> Vec<f32> {
        if n_entities == 0 {
            return Vec::new();
        }
        let e = self.embed_dim;
        let h = self.n_heads;
        let d = self.head_dim;

        // Project Q, K, V for each entity.
        let mut q = Vec::with_capacity(n_entities * e);
        let mut k = Vec::with_capacity(n_entities * e);
        let mut v = Vec::with_capacity(n_entities * e);
        for i in 0..n_entities {
            let ent = &entity_embeds[i * e..(i + 1) * e];
            q.extend_from_slice(&self.w_q.forward(ent));
            k.extend_from_slice(&self.w_k.forward(ent));
            v.extend_from_slice(&self.w_v.forward(ent));
        }

        // Per-head attention.
        let scale = 1.0 / (d as f32).sqrt();
        let mut concat_out = vec![0.0f32; n_entities * e];

        for head in 0..h {
            let offset = head * d;

            // Compute attention scores: Q_h @ K_h^T / sqrt(d)
            let mut scores = vec![0.0f32; n_entities * n_entities];
            for i in 0..n_entities {
                for j in 0..n_entities {
                    let mut dot = 0.0f32;
                    for kk in 0..d {
                        dot += q[i * e + offset + kk] * k[j * e + offset + kk];
                    }
                    scores[i * n_entities + j] = dot * scale;
                }
            }

            // Softmax per row.
            for i in 0..n_entities {
                let row = &mut scores[i * n_entities..(i + 1) * n_entities];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max).exp();
                    sum += *v;
                }
                for v in row.iter_mut() {
                    *v /= sum;
                }
            }

            // Weighted sum: attn @ V_h
            for i in 0..n_entities {
                for kk in 0..d {
                    let mut sum = 0.0f32;
                    for j in 0..n_entities {
                        sum += scores[i * n_entities + j] * v[j * e + offset + kk];
                    }
                    concat_out[i * e + offset + kk] = sum;
                }
            }
        }

        // Output projection.
        let mut output = Vec::with_capacity(n_entities * e);
        for i in 0..n_entities {
            let row = &concat_out[i * e..(i + 1) * e];
            output.extend_from_slice(&self.w_o.forward(row));
        }
        output
    }
}

/// Inference-only entity-attention policy network (V2).
///
/// Mirrors training PolicyNetV2 without gradients.
/// Architecture:
///   ego → ego_encoder → ego_embed
///   entities → entity_encoder → attention → mean_pool → pooled
///   concat(ego_embed, pooled) → fc1 → fc2 → actor_head → logits
#[derive(Clone, Deserialize)]
pub struct InferenceNetV2 {
    pub ego_encoder: DenseLayer,
    pub entity_encoder: DenseLayer,
    pub attn: InferenceAttention,
    pub fc1: DenseLayer,
    pub fc2: DenseLayer,
    pub actor_head: DenseLayer,
    /// Deserialized but unused for inference.
    pub local_value_head: DenseLayer,
    pub embed_dim: usize,
    /// Optional normalizers (loaded separately, not from model JSON).
    #[serde(skip)]
    pub ego_normalizer: Option<ObsNormalizer>,
    #[serde(skip)]
    pub entity_normalizer: Option<ObsNormalizer>,
}

impl InferenceNetV2 {
    /// Load from JSON string (same format as rl-train's PolicyNetV2::save).
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("Failed to parse V2 model JSON: {}", e))
    }

    /// Forward pass. Returns action logits.
    pub fn forward(&self, ego: &[f32], entities: &[f32], n_entities: usize) -> Vec<f32> {
        let e = self.embed_dim;
        let entity_dim = self.entity_encoder.in_dim;

        // Optionally normalize inputs.
        let ego_norm;
        let ego_input = if let Some(ref norm) = self.ego_normalizer {
            ego_norm = norm.normalize(ego);
            &ego_norm
        } else {
            ego
        };

        // 1. Encode ego.
        let ego_embed = self.ego_encoder.forward(ego_input);

        // 2. Encode entities.
        let mut entity_embeds = Vec::with_capacity(n_entities * e);
        for i in 0..n_entities {
            let ent = &entities[i * entity_dim..(i + 1) * entity_dim];
            let ent_input = if let Some(ref norm) = self.entity_normalizer {
                norm.normalize(ent)
            } else {
                ent.to_vec()
            };
            entity_embeds.extend_from_slice(&self.entity_encoder.forward(&ent_input));
        }

        // 3. Self-attention.
        let attn_out = if n_entities > 0 {
            self.attn.forward(&entity_embeds, n_entities)
        } else {
            Vec::new()
        };

        // 4. Mean pool.
        let pooled = if n_entities > 0 {
            let inv_n = 1.0 / n_entities as f32;
            let mut pool = vec![0.0f32; e];
            for i in 0..n_entities {
                for j in 0..e {
                    pool[j] += attn_out[i * e + j] * inv_n;
                }
            }
            pool
        } else {
            vec![0.0f32; e]
        };

        // 5. Trunk.
        let mut trunk_input = Vec::with_capacity(2 * e);
        trunk_input.extend_from_slice(&ego_embed);
        trunk_input.extend_from_slice(&pooled);
        let h1 = self.fc1.forward(&trunk_input);
        let h2 = self.fc2.forward(&h1);
        self.actor_head.forward(&h2)
    }

    /// Select the best action (argmax of logits).
    pub fn act(&self, ego: &[f32], entities: &[f32], n_entities: usize) -> u32 {
        let logits = self.forward(ego, entities, n_entities);
        let mut best = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best = i as u32;
            }
        }
        best
    }
}

impl InferenceNet {
    /// Load from JSON string (same format as rl-train's PolicyNet::save).
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("Failed to parse model JSON: {}", e))
    }

    /// Forward pass through the actor network. Returns action logits.
    pub fn forward(&self, obs: &[f32]) -> Vec<f32> {
        let h1 = self.fc1.forward(obs);
        let h2 = self.fc2.forward(&h1);
        let h3 = self.fc3.forward(&h2);
        self.actor_head.forward(&h3)
    }

    /// Select the best action (argmax of softmax probabilities).
    pub fn act(&self, obs: &[f32]) -> u32 {
        let logits = self.forward(obs);
        // Argmax — equivalent to argmax of softmax since softmax is monotonic.
        let mut best_action = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_action = i as u32;
            }
        }
        best_action
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_layer_forward() {
        let layer = DenseLayer {
            weights: vec![1.0, 0.0, 0.0, 1.0],
            biases: vec![0.0, 0.0],
            in_dim: 2,
            out_dim: 2,
            relu: false,
        };
        let out = layer.forward(&[3.0, 5.0]);
        assert_eq!(out, vec![3.0, 5.0]); // identity
    }

    #[test]
    fn test_dense_layer_relu() {
        let layer = DenseLayer {
            weights: vec![1.0, -1.0],
            biases: vec![0.0],
            in_dim: 2,
            out_dim: 1,
            relu: true,
        };
        assert_eq!(layer.forward(&[1.0, 2.0]), vec![0.0]); // 1 - 2 = -1, relu → 0
        assert_eq!(layer.forward(&[3.0, 1.0]), vec![2.0]); // 3 - 1 = 2, relu → 2
    }

    #[test]
    fn test_argmax_action() {
        // Construct a trivial net where fc1, fc2, and fc3 are identity-like
        // and actor_head maps directly
        let net = InferenceNet {
            fc1: DenseLayer {
                weights: vec![1.0, 0.0, 0.0, 1.0],
                biases: vec![0.0, 0.0],
                in_dim: 2,
                out_dim: 2,
                relu: true,
            },
            fc2: DenseLayer {
                weights: vec![1.0, 0.0, 0.0, 1.0],
                biases: vec![0.0, 0.0],
                in_dim: 2,
                out_dim: 2,
                relu: true,
            },
            fc3: DenseLayer {
                weights: vec![1.0, 0.0, 0.0, 1.0],
                biases: vec![0.0, 0.0],
                in_dim: 2,
                out_dim: 2,
                relu: true,
            },
            actor_head: DenseLayer {
                weights: vec![1.0, 0.0, 0.0, 1.0, -1.0, -1.0],
                biases: vec![0.0, 0.0, 0.0],
                in_dim: 2,
                out_dim: 3,
                relu: false,
            },
            critic_head: DenseLayer {
                weights: vec![0.0, 0.0],
                biases: vec![0.0],
                in_dim: 2,
                out_dim: 1,
                relu: false,
            },
        };
        // Input [3.0, 1.0]:
        // fc1: [3.0, 1.0] → fc2: [3.0, 1.0] → fc3: [3.0, 1.0]
        // actor: [3*1+1*0, 3*0+1*1, 3*-1+1*-1] = [3.0, 1.0, -4.0]
        // argmax = 0
        assert_eq!(net.act(&[3.0, 1.0]), 0);
        // Input [1.0, 3.0]:
        // actor: [1.0, 3.0, -4.0]
        // argmax = 1
        assert_eq!(net.act(&[1.0, 3.0]), 1);
    }
}
