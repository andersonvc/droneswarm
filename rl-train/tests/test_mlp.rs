//! Tests for PolicyNet forward pass, save/load, and initialization.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rl_train::mlp::PolicyNet;

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

    // ReLU layers should have gain ≈ sqrt(2) ≈ 1.41, so avg_sq ≈ 2.0 / in_dim
    let expected_fc1_avg_sq = 2.0 / OBS_DIM as f32;
    let expected_fc2_avg_sq = 2.0 / HIDDEN as f32;

    assert!(fc1_avg_sq > 0.5 * expected_fc1_avg_sq && fc1_avg_sq < 2.0 * expected_fc1_avg_sq,
            "fc1 weights should have reasonable magnitude");
    assert!(fc2_avg_sq > 0.5 * expected_fc2_avg_sq && fc2_avg_sq < 2.0 * expected_fc2_avg_sq,
            "fc2 weights should have reasonable magnitude");

    // Actor head uses small gain (0.01), so should have much smaller weights
    assert!(actor_avg_sq < 0.01, "Actor head should have small weights");

    // Critic head uses gain=1.0, so avg_sq ≈ 1.0 / in_dim
    let expected_critic_avg_sq = 1.0 / HIDDEN as f32;
    assert!(critic_avg_sq > 0.1 * expected_critic_avg_sq && critic_avg_sq < 3.0 * expected_critic_avg_sq,
            "Critic head should have reasonable magnitude");
}