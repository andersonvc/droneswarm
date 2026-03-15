//! Integration tests for MultiHeadAttention forward, backward, and gradient checks.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl_train::network::MultiHeadAttention;

const EMBED_DIM: usize = 64;
const N_HEADS: usize = 4;
const EPS: f32 = 1e-4;

#[test]
fn test_attention_forward_shape() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut attn = MultiHeadAttention::new(EMBED_DIM, N_HEADS, &mut rng);

    let n_entities = 5;
    let input = vec![0.1f32; n_entities * EMBED_DIM];
    let output = attn.forward(&input, n_entities);

    assert_eq!(
        output.len(),
        n_entities * EMBED_DIM,
        "Output should have shape [n_entities * embed_dim] = {} elements, got {}",
        n_entities * EMBED_DIM,
        output.len()
    );
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Output element {} should be finite, got {}",
            i, v
        );
    }
}

#[test]
fn test_attention_single_entity() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut attn = MultiHeadAttention::new(EMBED_DIM, N_HEADS, &mut rng);

    // With a single entity, self-attention weight is always 1.0 (softmax of single element).
    // So the output is a linear transform of the input: W_O @ (W_V @ input).
    let input = vec![0.5f32; EMBED_DIM];
    let output = attn.forward(&input, 1);

    assert_eq!(
        output.len(),
        EMBED_DIM,
        "Single entity output should have {} elements",
        EMBED_DIM
    );
    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Single entity output element {} should be finite, got {}",
            i, v
        );
    }

    // Verify it is deterministic: same input, same output.
    let mut attn2 = attn.clone();
    let output2 = attn2.forward(&input, 1);
    for (i, (&v1, &v2)) in output.iter().zip(output2.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < EPS,
            "Single entity output should be deterministic: element {} differs ({} vs {})",
            i, v1, v2
        );
    }
}

#[test]
fn test_attention_backward_shape() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut attn = MultiHeadAttention::new(EMBED_DIM, N_HEADS, &mut rng);

    let n_entities = 4;
    let input = vec![0.3f32; n_entities * EMBED_DIM];
    let output = attn.forward(&input, n_entities);

    let grad_output = vec![1.0f32; output.len()];
    let grad_input = attn.backward(&grad_output, n_entities);

    assert_eq!(
        grad_input.len(),
        n_entities * EMBED_DIM,
        "Backward grad_input should have shape [n_entities * embed_dim] = {} elements, got {}",
        n_entities * EMBED_DIM,
        grad_input.len()
    );
    for (i, &g) in grad_input.iter().enumerate() {
        assert!(
            g.is_finite(),
            "Gradient element {} should be finite, got {}",
            i, g
        );
    }
}

#[test]
fn test_attention_gradient_numerical() {
    // Use small dimensions for numerical gradient check.
    let small_embed = 8;
    let small_heads = 2;
    let n_entities = 3;
    let eps = 1e-4f32;

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let base_attn = MultiHeadAttention::new(small_embed, small_heads, &mut rng);

    let input = vec![0.2f32; n_entities * small_embed];

    // Compute analytical gradients via forward + backward.
    let mut attn_fwd = base_attn.clone();
    let output = attn_fwd.forward(&input, n_entities);
    let grad_output = vec![1.0f32; output.len()]; // d(loss)/d(output) = 1 when loss = sum(output)
    attn_fwd.backward(&grad_output, n_entities);

    // Analytical gradients for w_q weights.
    let analytical_grads: Vec<f32> = attn_fwd.w_q.grad_weights.clone();

    // Check numerical gradients for first 10 w_q weights.
    let n_check = 10.min(base_attn.w_q.weights.len());
    for idx in 0..n_check {
        // f(x + eps)
        let mut attn_plus = base_attn.clone();
        attn_plus.w_q.weights[idx] += eps;
        let out_plus = attn_plus.forward(&input, n_entities);
        let loss_plus: f32 = out_plus.iter().sum();

        // f(x - eps)
        let mut attn_minus = base_attn.clone();
        attn_minus.w_q.weights[idx] -= eps;
        let out_minus = attn_minus.forward(&input, n_entities);
        let loss_minus: f32 = out_minus.iter().sum();

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let analytical = analytical_grads[idx];

        // Relative error check, with tolerance for small gradients.
        let abs_diff = (numerical - analytical).abs();
        let denom = numerical.abs().max(analytical.abs()).max(1e-7);
        let rel_error = abs_diff / denom;

        assert!(
            rel_error < 0.05,
            "w_q weight[{}]: numerical={:.6}, analytical={:.6}, rel_error={:.4} exceeds 5% tolerance",
            idx, numerical, analytical, rel_error
        );
    }
}
