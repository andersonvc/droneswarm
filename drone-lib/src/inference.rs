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
/// Architecture: fc1(relu) → fc2(relu) → actor_head(linear)
/// Only the actor head is used for action selection.
#[derive(Clone, Deserialize)]
pub struct InferenceNet {
    pub fc1: DenseLayer,
    pub fc2: DenseLayer,
    pub actor_head: DenseLayer,
    // critic_head is deserialized but unused for inference
    pub critic_head: DenseLayer,
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
        self.actor_head.forward(&h2)
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
        // Construct a trivial net where fc1 and fc2 are identity-like
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
        // fc1: [3.0, 1.0] → fc2: [3.0, 1.0]
        // actor: [3*1+1*0, 3*0+1*1, 3*-1+1*-1] = [3.0, 1.0, -4.0]
        // argmax = 0
        assert_eq!(net.act(&[3.0, 1.0]), 0);
        // Input [1.0, 3.0]:
        // actor: [1.0, 3.0, -4.0]
        // argmax = 1
        assert_eq!(net.act(&[1.0, 3.0]), 1);
    }
}
