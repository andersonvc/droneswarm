//! Tests for Generalized Advantage Estimation (GAE) computation.

use rl_train::mlp::{compute_gae, Transition};

fn make_transition(reward: f32, value: f32, done: bool) -> Transition {
    Transition {
        ego_obs: vec![0.0; 25],
        entity_obs: vec![],
        raw_entity_obs: vec![],
        n_entities: 0,
        action: 0,
        reward,
        value,
        log_prob: 0.0,
        done,
        truncated: false,
        drone_id: 0,
        drone_died: false,
        team_value_at_death: 0.0,
    }
}

fn make_transition_ext(reward: f32, value: f32, done: bool, truncated: bool, drone_id: usize) -> Transition {
    Transition {
        ego_obs: vec![0.0; 25],
        entity_obs: vec![],
        raw_entity_obs: vec![],
        n_entities: 0,
        action: 0,
        reward,
        value,
        log_prob: 0.0,
        done,
        truncated,
        drone_id,
        drone_died: false,
        team_value_at_death: 0.0,
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

#[test]
fn test_gae_truncated_bootstraps_with_value() {
    // Truncated episode should bootstrap with V(s_last) instead of 0.
    // Compare: a terminal (done=true, truncated=false) vs truncated (done=true, truncated=true).
    let gamma = 0.99;
    let lam = 0.95;
    let last_value = 10.0;

    // Terminal episode: bootstraps with 0.
    let terminal = vec![make_transition(1.0, 5.0, true)];
    let (adv_term, _) = compute_gae(&terminal, last_value, gamma, lam);

    // Truncated episode: should bootstrap with last_value (10.0).
    let truncated = vec![make_transition_ext(1.0, 5.0, true, true, 0)];
    let (adv_trunc, _) = compute_gae(&truncated, last_value, gamma, lam);

    // Terminal: delta = 1 + 0.99 * 0 - 5 = -4, adv = -4
    assert!((adv_term[0] - (-4.0)).abs() < 1e-6, "Terminal should bootstrap with 0");

    // Truncated: next_non_terminal = 1.0, next_value = last_value = 10.0
    // delta = 1 + 0.99 * 10.0 * 1.0 - 5 = 5.9, adv = 5.9
    assert!((adv_trunc[0] - 5.9).abs() < 1e-6, "Truncated should bootstrap with V(s), got {}", adv_trunc[0]);
}

#[test]
fn test_gae_truncated_multi_step() {
    // 3-step episode ending in truncation. GAE should NOT reset at truncated boundary.
    let gamma = 0.9;
    let lam = 0.8;
    let last_value = 2.0;

    let transitions = vec![
        make_transition_ext(1.0, 5.0, false, false, 0),
        make_transition_ext(2.0, 4.0, false, false, 0),
        make_transition_ext(3.0, 3.0, true, true, 0),  // truncated, not terminal
    ];

    let (advantages, returns) = compute_gae(&transitions, last_value, gamma, lam);

    // Step 2 (truncated): next_non_terminal = 1.0, next_value = last_value = 2.0
    // delta = 3 + 0.9 * 2.0 * 1.0 - 3 = 1.8, gae = 1.8
    assert!((advantages[2] - 1.8).abs() < 1e-6, "Truncated step adv should be 1.8, got {}", advantages[2]);

    // Step 1: next_value = transitions[2].value = 3.0, next_non_terminal = 1.0
    // delta = 2 + 0.9 * 3.0 * 1.0 - 4 = 0.7
    // gae = 0.7 + 0.9 * 0.8 * 1.0 * 1.8 = 0.7 + 1.296 = 1.996
    assert!((advantages[1] - 1.996).abs() < 1e-3, "Step 1 adv should be ~1.996, got {}", advantages[1]);

    // Returns = advantages + values
    for i in 0..3 {
        assert!((returns[i] - (advantages[i] + transitions[i].value)).abs() < 1e-6);
    }
}

#[test]
fn test_gae_per_drone_independent_advantages() {
    // Two drones interleaved in the same sequence should get independent advantages
    // when computed separately per drone_id.
    let gamma = 0.9;
    let lam = 0.8;

    // Drone 0: simple trajectory [r=1, v=5] -> [r=2, v=4, done]
    let drone0_transitions = vec![
        make_transition_ext(1.0, 5.0, false, false, 0),
        make_transition_ext(2.0, 4.0, true, false, 0),
    ];
    let (adv_drone0, _) = compute_gae(&drone0_transitions, 0.0, gamma, lam);

    // Drone 1: different trajectory [r=10, v=1] -> [r=20, v=2, done]
    let drone1_transitions = vec![
        make_transition_ext(10.0, 1.0, false, false, 1),
        make_transition_ext(20.0, 2.0, true, false, 1),
    ];
    let (adv_drone1, _) = compute_gae(&drone1_transitions, 0.0, gamma, lam);

    // Drone 0 advantages:
    // Step 1: delta = 2 + 0.9*0*0 - 4 = -2, gae = -2
    // Step 0: delta = 1 + 0.9*4*1 - 5 = -0.4, gae = -0.4 + 0.9*0.8*1*(-2) = -0.4 - 1.44 = -1.84
    assert!((adv_drone0[1] - (-2.0)).abs() < 1e-6, "Drone0 step1 got {}", adv_drone0[1]);
    assert!((adv_drone0[0] - (-1.84)).abs() < 1e-3, "Drone0 step0 got {}", adv_drone0[0]);

    // Drone 1 advantages:
    // Step 1: delta = 20 + 0.9*0*0 - 2 = 18, gae = 18
    // Step 0: delta = 10 + 0.9*2*1 - 1 = 10.8, gae = 10.8 + 0.9*0.8*1*18 = 10.8 + 12.96 = 23.76
    assert!((adv_drone1[1] - 18.0).abs() < 1e-6, "Drone1 step1 got {}", adv_drone1[1]);
    assert!((adv_drone1[0] - 23.76).abs() < 1e-3, "Drone1 step0 got {}", adv_drone1[0]);

    // Key check: the advantages should be very different between the two drones.
    // If they were interleaved and computed together, drone 0's GAE at step 0 would be
    // corrupted by drone 1's high-reward transitions and vice versa.
    assert!((adv_drone0[0] - adv_drone1[0]).abs() > 10.0,
        "Per-drone advantages must differ significantly: drone0={}, drone1={}",
        adv_drone0[0], adv_drone1[0]);
}