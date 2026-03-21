//! Tests for PolicyNet forward pass, save/load, and initialization.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl_train::mlp::{orthogonal_init, PolicyNet};

const OBS_DIM: usize = 64;
const ACT_DIM: usize = 14;
const HIDDEN: usize = 128;

#[test]
fn test_forward_output_shape() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut net = PolicyNet::new(OBS_DIM, ACT_DIM, HIDDEN, &mut rng);

    let obs = vec![0.1f32; OBS_DIM];
    let (logits, value) = net.forward(&obs);

    assert_eq!(logits.len(), ACT_DIM, "Action logits should have length {}", ACT_DIM);
    assert!(!value.is_nan(), "Value should not be NaN");
}

#[test]
fn test_save_load_roundtrip() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let net1 = PolicyNet::new(OBS_DIM, ACT_DIM, HIDDEN, &mut rng);

    // Save to temporary file
    let temp_path = "/tmp/test_model.json";
    net1.save(temp_path).expect("Save should succeed");

    // Load from file
    let mut net2 = PolicyNet::load(temp_path).expect("Load should succeed");

    // Test that outputs are identical
    let obs = vec![0.5f32; OBS_DIM];
    let mut net1_clone = net1.clone();
    let (logits1, value1) = net1_clone.forward(&obs);
    let (logits2, value2) = net2.forward(&obs);

    assert_eq!(logits1.len(), logits2.len());
    for (l1, l2) in logits1.iter().zip(logits2.iter()) {
        assert!((l1 - l2).abs() < 1e-6, "Logits should be nearly identical");
    }
    assert!((value1 - value2).abs() < 1e-6, "Values should be nearly identical");

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_orthogonal_init_norms() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let net = PolicyNet::new(OBS_DIM, ACT_DIM, HIDDEN, &mut rng);

    // Check that weight matrices have reasonable norms
    let fc1_weights = &net.fc1.weights;
    let fc2_weights = &net.fc2.weights;
    let actor_weights = &net.actor_head.weights;
    let critic_weights = &net.critic_head.weights;

    // Calculate average squared weight magnitude for each layer
    let fc1_avg_sq: f32 = fc1_weights.iter().map(|w| w * w).sum::<f32>() / fc1_weights.len() as f32;
    let fc2_avg_sq: f32 = fc2_weights.iter().map(|w| w * w).sum::<f32>() / fc2_weights.len() as f32;
    let actor_avg_sq: f32 = actor_weights.iter().map(|w| w * w).sum::<f32>() / actor_weights.len() as f32;
    let critic_avg_sq: f32 = critic_weights.iter().map(|w| w * w).sum::<f32>() / critic_weights.len() as f32;

    // Orthogonal init: avg_sq ≈ gain^2 / max(in_dim, out_dim).
    // fc1: 64 -> 128 (tall), fc2: 128 -> 128 (square).
    let expected_fc1_avg_sq = 2.0 / OBS_DIM.max(HIDDEN) as f32;
    let expected_fc2_avg_sq = 2.0 / HIDDEN as f32;

    assert!(fc1_avg_sq > 0.3 * expected_fc1_avg_sq && fc1_avg_sq < 3.0 * expected_fc1_avg_sq,
            "fc1 avg_sq = {}, expected ~{}", fc1_avg_sq, expected_fc1_avg_sq);
    assert!(fc2_avg_sq > 0.3 * expected_fc2_avg_sq && fc2_avg_sq < 3.0 * expected_fc2_avg_sq,
            "fc2 avg_sq = {}, expected ~{}", fc2_avg_sq, expected_fc2_avg_sq);

    // Actor head uses small gain (0.01), so should have much smaller weights
    assert!(actor_avg_sq < 0.01, "Actor head should have small weights");

    // Critic head uses gain=1.0, so avg_sq ≈ 1.0 / in_dim
    let expected_critic_avg_sq = 1.0 / HIDDEN as f32;
    assert!(critic_avg_sq > 0.1 * expected_critic_avg_sq && critic_avg_sq < 3.0 * expected_critic_avg_sq,
            "Critic head should have reasonable magnitude");
}

/// Verify orthogonal init for out_dim > in_dim (e.g., entity_encoder 8->64).
/// Previously, rows beyond in_dim were all-zero (dead neurons).
#[test]
fn test_orthogonal_init_tall_matrix() {
    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let in_dim = 8;
    let out_dim = 64;
    let gain = 1.41421356f32; // sqrt(2)

    let weights = orthogonal_init(in_dim, out_dim, gain, &mut rng);
    assert_eq!(weights.len(), out_dim * in_dim);

    // 1. No rows are all-zero (the primary bug: rows beyond in_dim were dead).
    for row in 0..out_dim {
        let row_norm_sq: f32 = (0..in_dim)
            .map(|k| weights[row * in_dim + k] * weights[row * in_dim + k])
            .sum();
        assert!(
            row_norm_sq > 1e-6,
            "Row {} is dead (norm^2 = {})",
            row,
            row_norm_sq
        );
    }

    // 2. Columns are orthogonal. For the tall case, columns of the weight matrix
    //    correspond to rows of the pre-transpose orthonormalized matrix, so they
    //    should be orthogonal with norm = gain.
    for i in 0..in_dim {
        // Column norm should equal gain.
        let col_norm: f32 = (0..out_dim)
            .map(|r| weights[r * in_dim + i] * weights[r * in_dim + i])
            .sum::<f32>()
            .sqrt();
        assert!(
            (col_norm - gain).abs() < 0.01,
            "Column {} norm = {}, expected ~{}",
            i,
            col_norm,
            gain
        );

        // Pairwise dot products between different columns should be ~0.
        for j in (i + 1)..in_dim {
            let dot: f32 = (0..out_dim)
                .map(|r| weights[r * in_dim + i] * weights[r * in_dim + j])
                .sum();
            assert!(
                dot.abs() < 0.01,
                "Columns {} and {} have dot product {}, expected ~0",
                i,
                j,
                dot
            );
        }
    }

    // 3. Row norms should be approximately gain * sqrt(in_dim / out_dim) by
    //    symmetry. Just verify they are all nonzero and reasonably similar.
    let expected_row_norm = gain * (in_dim as f32 / out_dim as f32).sqrt();
    let mut min_row_norm = f32::MAX;
    let mut max_row_norm = 0.0f32;
    for row in 0..out_dim {
        let norm: f32 = (0..in_dim)
            .map(|k| weights[row * in_dim + k] * weights[row * in_dim + k])
            .sum::<f32>()
            .sqrt();
        min_row_norm = min_row_norm.min(norm);
        max_row_norm = max_row_norm.max(norm);
    }
    // Row norms won't be exactly equal but should be in a reasonable range.
    assert!(
        min_row_norm > 0.1 * expected_row_norm,
        "Minimum row norm {} is too small (expected ~{})",
        min_row_norm,
        expected_row_norm
    );
    assert!(
        max_row_norm < 5.0 * expected_row_norm,
        "Maximum row norm {} is too large (expected ~{})",
        max_row_norm,
        expected_row_norm
    );
}