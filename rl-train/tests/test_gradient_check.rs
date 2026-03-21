//! Numerical gradient verification tests using finite differences.
//!
//! These tests compare analytical (backprop) gradients against numerical gradients
//! computed via central finite differences: (f(w+eps) - f(w-eps)) / (2*eps).
//! This catches bugs where backward passes compute incorrect gradients,
//! especially for shared-weight layers called multiple times (e.g., entity_encoder).

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl_train::network::layer::DenseLayer;
use rl_train::network::attention::MultiHeadAttention;
use rl_train::network::policy_v2::PolicyNetV2;

// ---------------------------------------------------------------------------
// Test 1: DenseLayer with relu=true
// ---------------------------------------------------------------------------

#[test]
fn test_dense_layer_relu_gradient() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let in_dim = 4;
    let out_dim = 8;
    let eps = 1e-4f32;
    let rtol = 1e-1f32;

    let base_layer = DenseLayer::new(in_dim, out_dim, true, &mut rng);
    let input = vec![0.5, -0.3, 0.8, -0.1];

    // Forward + backward on a clone to get analytical gradients.
    let mut layer_fwd = base_layer.clone();
    layer_fwd.zero_grad();
    let _output = layer_fwd.forward(&input);
    let grad_output = vec![1.0f32; out_dim]; // loss = sum(output)
    let _grad_input = layer_fwd.backward(&grad_output);

    let analytical_weight_grads = layer_fwd.grad_weights.clone();
    let analytical_bias_grads = layer_fwd.grad_biases.clone();

    // Check weight gradients numerically.
    let mut n_failed = 0usize;
    let mut max_rel_err = 0.0f32;
    let n_weights = base_layer.weights.len();
    for i in 0..n_weights {
        let mut l_plus = base_layer.clone();
        l_plus.weights[i] += eps;
        let loss_plus: f32 = l_plus.forward(&input).iter().sum();

        let mut l_minus = base_layer.clone();
        l_minus.weights[i] -= eps;
        let loss_minus: f32 = l_minus.forward(&input).iter().sum();

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let analytical = analytical_weight_grads[i];
        let rel_err = if analytical.abs() > 1e-7 || numerical.abs() > 1e-7 {
            (analytical - numerical).abs() / (analytical.abs() + numerical.abs() + 1e-8)
        } else {
            0.0
        };
        if rel_err > max_rel_err { max_rel_err = rel_err; }
        if rel_err > rtol { n_failed += 1; }
    }
    assert_eq!(
        n_failed, 0,
        "DenseLayer(relu) weight gradient check: {}/{} failed, max_rel_err={:.6}",
        n_failed, n_weights, max_rel_err
    );

    // Check bias gradients numerically.
    n_failed = 0;
    max_rel_err = 0.0;
    let n_biases = base_layer.biases.len();
    for i in 0..n_biases {
        let mut l_plus = base_layer.clone();
        l_plus.biases[i] += eps;
        let loss_plus: f32 = l_plus.forward(&input).iter().sum();

        let mut l_minus = base_layer.clone();
        l_minus.biases[i] -= eps;
        let loss_minus: f32 = l_minus.forward(&input).iter().sum();

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let analytical = analytical_bias_grads[i];
        let rel_err = if analytical.abs() > 1e-7 || numerical.abs() > 1e-7 {
            (analytical - numerical).abs() / (analytical.abs() + numerical.abs() + 1e-8)
        } else {
            0.0
        };
        if rel_err > max_rel_err { max_rel_err = rel_err; }
        if rel_err > rtol { n_failed += 1; }
    }
    assert_eq!(
        n_failed, 0,
        "DenseLayer(relu) bias gradient check: {}/{} failed, max_rel_err={:.6}",
        n_failed, n_biases, max_rel_err
    );
}

// ---------------------------------------------------------------------------
// Test 2: Entity encoder with n_entities > 1 (catches P0-2 class bugs)
// ---------------------------------------------------------------------------

#[test]
fn test_entity_encoder_gradient_multi_entity() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let ego_dim = 4;
    let entity_dim = 3;
    let act_dim = 5;
    let embed_dim = 8;
    let hidden = 16;
    let n_entities = 3;
    // Max pooling creates piecewise-linear gradients; use larger eps and tolerance
    // to avoid hitting switching boundaries where analytical vs numerical disagree.
    let eps = 1e-3f32;
    let rtol = 2e-1f32;

    let base_net = PolicyNetV2::new(ego_dim, entity_dim, act_dim, embed_dim, hidden, &mut rng);
    let ego = vec![0.5, -0.2, 0.3, 0.1];
    let entities = vec![
        0.1, -0.4, 0.7,   // entity 0
        -0.3, 0.6, -0.2,  // entity 1
        0.8, 0.1, -0.5,   // entity 2
    ];

    // Forward + backward on a clone to get analytical gradients.
    // Keep base_net pristine for numerical perturbation clones.
    let mut net_fwd = base_net.clone();
    net_fwd.zero_grad();
    let (_logits, _value, _h2) = net_fwd.forward(&ego, &entities, n_entities);
    let grad_logits = vec![1.0f32; act_dim];
    net_fwd.backward_input(&grad_logits, 0.0, n_entities);

    let analytical_grads = net_fwd.entity_encoder.grad_weights.clone();

    // Numerically check entity_encoder weight gradients.
    // Clone from base_net (clean state) for each perturbation.
    let mut n_failed = 0usize;
    let mut max_rel_err = 0.0f32;
    let n_weights = base_net.entity_encoder.weights.len();
    for i in 0..n_weights {
        let mut n_plus = base_net.clone();
        n_plus.entity_encoder.weights[i] += eps;
        let (logits_plus, _, _) = n_plus.forward(&ego, &entities, n_entities);
        let loss_plus: f32 = logits_plus.iter().sum();

        let mut n_minus = base_net.clone();
        n_minus.entity_encoder.weights[i] -= eps;
        let (logits_minus, _, _) = n_minus.forward(&ego, &entities, n_entities);
        let loss_minus: f32 = logits_minus.iter().sum();

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let analytical = analytical_grads[i];
        let rel_err = if analytical.abs() > 1e-7 || numerical.abs() > 1e-7 {
            (analytical - numerical).abs() / (analytical.abs() + numerical.abs() + 1e-8)
        } else {
            0.0
        };
        if rel_err > max_rel_err { max_rel_err = rel_err; }
        if rel_err > rtol { n_failed += 1; }
    }
    assert_eq!(
        n_failed, 0,
        "Entity encoder weight gradient check ({} entities): {}/{} failed, max_rel_err={:.6}",
        n_entities, n_failed, n_weights, max_rel_err
    );
}

// ---------------------------------------------------------------------------
// Test 3: MultiHeadAttention with n_entities > 1
// ---------------------------------------------------------------------------

#[test]
fn test_multi_head_attention_gradient() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let embed_dim = 8;
    let n_heads = 2;
    let n_entities = 3;
    // Attention weight gradients flow through softmax + LayerNorm, need larger eps and tolerance for f32.
    let eps = 1e-2f32;
    let rtol = 3e-1f32;

    let base_attn = MultiHeadAttention::new(embed_dim, n_heads, &mut rng);

    // Create distinct entity embeddings.
    let entity_embeds: Vec<f32> = (0..n_entities * embed_dim)
        .map(|i| (i as f32 * 0.17 + 0.3).sin() * 0.5)
        .collect();

    // Use a weighted sum as loss to avoid degenerate softmax gradients.
    // With uniform grad_output, softmax backward zeros out Q/K gradients.
    let loss_weights: Vec<f32> = (0..n_entities * embed_dim)
        .map(|i| (i as f32 * 0.31 + 0.1).cos())
        .collect();

    // Forward + backward to get analytical gradients.
    // loss = dot(loss_weights, output), so grad_output = loss_weights.
    let mut attn_fwd = base_attn.clone();
    attn_fwd.zero_grad();
    let _output = attn_fwd.forward(&entity_embeds, n_entities);
    let _grad_input = attn_fwd.backward(&loss_weights, n_entities);

    let analytical_wq = attn_fwd.w_q.grad_weights.clone();
    let analytical_wk = attn_fwd.w_k.grad_weights.clone();
    let analytical_wv = attn_fwd.w_v.grad_weights.clone();
    let analytical_wo = attn_fwd.w_o.grad_weights.clone();

    // Check each projection's weight gradients numerically.
    // Clone from base_attn (clean state) for each perturbation.
    for (name, analytical, get_set) in [
        ("w_q", &analytical_wq, 0u8),
        ("w_k", &analytical_wk, 1u8),
        ("w_v", &analytical_wv, 2u8),
        ("w_o", &analytical_wo, 3u8),
    ] {
        let mut n_failed = 0usize;
        let mut max_rel_err = 0.0f32;
        let n_weights = analytical.len();
        for i in 0..n_weights {
            // Perturb +eps on a clean clone.
            let mut a_plus = base_attn.clone();
            match get_set {
                0 => a_plus.w_q.weights[i] += eps,
                1 => a_plus.w_k.weights[i] += eps,
                2 => a_plus.w_v.weights[i] += eps,
                _ => a_plus.w_o.weights[i] += eps,
            }
            let out_plus = a_plus.forward(&entity_embeds, n_entities);
            let loss_plus: f32 = out_plus.iter().zip(loss_weights.iter()).map(|(o, w)| o * w).sum();

            // Perturb -eps on a clean clone.
            let mut a_minus = base_attn.clone();
            match get_set {
                0 => a_minus.w_q.weights[i] -= eps,
                1 => a_minus.w_k.weights[i] -= eps,
                2 => a_minus.w_v.weights[i] -= eps,
                _ => a_minus.w_o.weights[i] -= eps,
            }
            let out_minus = a_minus.forward(&entity_embeds, n_entities);
            let loss_minus: f32 = out_minus.iter().zip(loss_weights.iter()).map(|(o, w)| o * w).sum();

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let anal = analytical[i];
            let rel_err = if anal.abs() > 1e-7 || numerical.abs() > 1e-7 {
                (anal - numerical).abs() / (anal.abs() + numerical.abs() + 1e-8)
            } else {
                0.0
            };
            if rel_err > max_rel_err { max_rel_err = rel_err; }
            if rel_err > rtol { n_failed += 1; }
        }
        assert_eq!(
            n_failed, 0,
            "MHA {} weight gradient check: {}/{} failed, max_rel_err={:.6}",
            name, n_failed, n_weights, max_rel_err
        );
    }
}

#[test]
fn test_layer_norm_gradient() {
    use rl_train::network::attention::LayerNorm;
    let dim = 8;
    let n_entities = 3;
    let eps = 1e-4f32;
    let rtol = 1e-1f32;

    let mut ln = LayerNorm::new(dim);
    let input: Vec<f32> = (0..n_entities * dim)
        .map(|i| ((i as f32 * 0.17 + 0.3).sin() * 0.5))
        .collect();

    // Use weighted sum as loss (sum(output) is ~constant with LayerNorm since it normalizes to mean=0).
    let loss_weights: Vec<f32> = (0..n_entities * dim)
        .map(|i| (i as f32 * 0.31 + 0.1).cos())
        .collect();
    let compute_loss = |out: &[f32]| -> f32 {
        out.iter().zip(loss_weights.iter()).map(|(o, w)| o * w).sum()
    };

    // Forward + backward
    ln.zero_grad();
    let output = ln.forward_batch(&input, n_entities);
    // grad_output = d(loss)/d(output) = loss_weights
    let d_input = ln.backward_batch(&loss_weights, n_entities);

    // Check input gradients numerically
    let mut n_failed = 0usize;
    let mut max_rel_err = 0.0f32;
    for i in 0..input.len() {
        let mut inp_plus = input.clone();
        inp_plus[i] += eps;
        let out_plus = compute_loss(&ln.forward_batch(&inp_plus, n_entities));
        let mut inp_minus = input.clone();
        inp_minus[i] -= eps;
        let out_minus = compute_loss(&ln.forward_batch(&inp_minus, n_entities));

        let numerical = (out_plus - out_minus) / (2.0 * eps);
        let analytical = d_input[i];
        let rel_err = (analytical - numerical).abs() / (analytical.abs() + numerical.abs() + 1e-8);
        if rel_err > max_rel_err { max_rel_err = rel_err; }
        if rel_err > rtol { n_failed += 1; }
    }
    assert_eq!(n_failed, 0,
        "LayerNorm input gradient check: {}/{} failed, max_rel_err={:.6}",
        n_failed, input.len(), max_rel_err);
}
