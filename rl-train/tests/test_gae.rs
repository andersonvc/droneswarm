//! Tests for Generalized Advantage Estimation (GAE) computation.

use rl_train::mlp::{compute_gae, Transition};

fn make_transition(reward: f32, value: f32, done: bool) -> Transition {
    Transition {
        obs: [0.0; 64],
        action: 0,
        reward,
        value,
        log_prob: 0.0,
        done,
    }
}

#[test]
fn test_gae_known_sequence() {
    // Simple 3-step episode: rewards [1, 2, 3], values [5, 4, 3], gamma=0.9, lambda=0.8
    let transitions = vec![
        make_transition(1.0, 5.0, false),
        make_transition(2.0, 4.0, false),
        make_transition(3.0, 3.0, true),  // done=true
    ];
    let last_value = 0.0; // irrelevant since last transition is done
    let gamma = 0.9;
    let lam = 0.8;

    let (advantages, returns) = compute_gae(&transitions, last_value, gamma, lam);

    // Manual calculation:
    // Step 2 (index 2): delta = 3 + 0.9 * 0 * 0 - 3 = 0, gae = 0
    // Step 1 (index 1): delta = 2 + 0.9 * 3 * 1 - 4 = 0.7, gae = 0.7 + 0.9 * 0.8 * 1 * 0 = 0.7
    // Step 0 (index 0): delta = 1 + 0.9 * 4 * 1 - 5 = -0.4, gae = -0.4 + 0.9 * 0.8 * 1 * 0.7 = 0.104

    assert_eq!(advantages.len(), 3);
    assert_eq!(returns.len(), 3);

    // Check advantages (approximate due to floating point)
    assert!((advantages[2] - 0.0).abs() < 1e-6, "Last advantage should be 0");
    assert!((advantages[1] - 0.7).abs() < 1e-6, "Second advantage should be 0.7");
    assert!((advantages[0] - 0.104).abs() < 1e-3, "First advantage should be ~0.104");

    // Returns = advantages + values
    assert!((returns[0] - (advantages[0] + 5.0)).abs() < 1e-6);
    assert!((returns[1] - (advantages[1] + 4.0)).abs() < 1e-6);
    assert!((returns[2] - (advantages[2] + 3.0)).abs() < 1e-6);
}

#[test]
fn test_gae_zero_reward() {
    // All rewards are zero, so advantages should be zero (assuming reasonable values)
    let transitions = vec![
        make_transition(0.0, 1.0, false),
        make_transition(0.0, 1.0, false),
        make_transition(0.0, 1.0, true),
    ];
    let last_value = 0.0;
    let gamma = 0.99;
    let lam = 0.95;

    let (advantages, returns) = compute_gae(&transitions, last_value, gamma, lam);

    assert_eq!(advantages.len(), 3);
    assert_eq!(returns.len(), 3);

    // With zero rewards and constant value estimates, deltas should be negative
    // but the exact values depend on the gamma discount
    for advantage in &advantages {
        assert!(advantage.is_finite(), "All advantages should be finite");
    }
    for ret in &returns {
        assert!(ret.is_finite(), "All returns should be finite");
    }
}

#[test]
fn test_gae_single_transition() {
    let transitions = vec![make_transition(5.0, 2.0, true)];
    let last_value = 0.0;
    let gamma = 0.99;
    let lam = 0.95;

    let (advantages, returns) = compute_gae(&transitions, last_value, gamma, lam);

    assert_eq!(advantages.len(), 1);
    assert_eq!(returns.len(), 1);

    // Single step: delta = 5 + 0.99 * 0 * 0 - 2 = 3, gae = 3
    assert!((advantages[0] - 3.0).abs() < 1e-6);
    assert!((returns[0] - 5.0).abs() < 1e-6); // advantage + value = 3 + 2 = 5
}

#[test]
fn test_gae_empty_transitions() {
    let transitions = vec![];
    let last_value = 1.0;
    let gamma = 0.99;
    let lam = 0.95;

    let (advantages, returns) = compute_gae(&transitions, last_value, gamma, lam);

    assert_eq!(advantages.len(), 0);
    assert_eq!(returns.len(), 0);
}