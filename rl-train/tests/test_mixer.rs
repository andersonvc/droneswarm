//! Integration tests for QMIXMixer forward, monotonicity, and backward gradient checks.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl_train::network::QMIXMixer;

const GLOBAL_STATE_DIM: usize = 264;
const MAX_AGENTS: usize = 24;
const EPS: f32 = 1e-4;

#[test]
fn test_qmix_forward_produces_scalar() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut mixer = QMIXMixer::new(GLOBAL_STATE_DIM, MAX_AGENTS, &mut rng);

    let n_agents = 4;
    let local_values = vec![1.0f32, 0.5, -0.3, 2.0];
    let global_state: Vec<f32> = (0..GLOBAL_STATE_DIM)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    let v_team = mixer.forward(&local_values, &global_state, n_agents);

    assert!(
        v_team.is_finite(),
        "Team value should be finite, got {}", v_team
    );
}

#[test]
fn test_qmix_monotonicity() {
    // QMIX key property: increasing a single agent's local value should
    // increase (or maintain) the team value, because mixing weights are abs(hypernetwork output).
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n_agents = 3;
    let global_state: Vec<f32> = (0..GLOBAL_STATE_DIM)
        .map(|i| (i as f32 * 0.02).cos())
        .collect();

    // Test monotonicity for each agent position.
    for agent_idx in 0..n_agents {
        let mut mixer = QMIXMixer::new(GLOBAL_STATE_DIM, MAX_AGENTS, &mut rng);

        let base_values = vec![1.0f32, 2.0, 3.0];
        let v_team_base = mixer.forward(&base_values, &global_state, n_agents);

        let mut increased_values = base_values.clone();
        increased_values[agent_idx] += 0.5;
        let mut mixer2 = mixer.clone();
        let v_team_increased = mixer2.forward(&increased_values, &global_state, n_agents);

        assert!(
            v_team_increased >= v_team_base - EPS,
            "QMIX monotonicity violated for agent {}: increasing local value from {:.4} to {:.4} \
             decreased team value from {:.6} to {:.6}",
            agent_idx,
            base_values[agent_idx],
            increased_values[agent_idx],
            v_team_base,
            v_team_increased
        );
    }
}

#[test]
fn test_qmix_backward_shape() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut mixer = QMIXMixer::new(GLOBAL_STATE_DIM, MAX_AGENTS, &mut rng);

    let n_agents = 5;
    let local_values: Vec<f32> = (0..n_agents).map(|i| i as f32 * 0.5).collect();
    let global_state: Vec<f32> = vec![0.1f32; GLOBAL_STATE_DIM];

    let _v_team = mixer.forward(&local_values, &global_state, n_agents);
    let grad_local = mixer.backward(1.0);

    assert_eq!(
        grad_local.len(), n_agents,
        "Backward grad should have length n_agents={}, got {}", n_agents, grad_local.len()
    );
    for (i, &g) in grad_local.iter().enumerate() {
        assert!(
            g.is_finite(),
            "Gradient for agent {} should be finite, got {}", i, g
        );
    }
}

#[test]
fn test_qmix_backward_gradient_numerical() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let base_mixer = QMIXMixer::new(GLOBAL_STATE_DIM, MAX_AGENTS, &mut rng);

    let n_agents = 3;
    let local_values = vec![1.0f32, -0.5, 2.0];
    let global_state: Vec<f32> = (0..GLOBAL_STATE_DIM)
        .map(|i| (i as f32 * 0.03).sin())
        .collect();
    let eps = 1e-4f32;

    // Compute analytical gradients.
    let mut mixer_ana = base_mixer.clone();
    let _v_team = mixer_ana.forward(&local_values, &global_state, n_agents);
    let analytical_grads = mixer_ana.backward(1.0);

    // Numerical gradient check for each local value.
    for idx in 0..n_agents {
        let mut vals_plus = local_values.clone();
        vals_plus[idx] += eps;
        let mut mixer_plus = base_mixer.clone();
        let v_plus = mixer_plus.forward(&vals_plus, &global_state, n_agents);

        let mut vals_minus = local_values.clone();
        vals_minus[idx] -= eps;
        let mut mixer_minus = base_mixer.clone();
        let v_minus = mixer_minus.forward(&vals_minus, &global_state, n_agents);

        let numerical = (v_plus - v_minus) / (2.0 * eps);
        let analytical = analytical_grads[idx];

        let abs_diff = (numerical - analytical).abs();
        let denom = numerical.abs().max(analytical.abs()).max(1e-7);
        let rel_error = abs_diff / denom;

        assert!(
            rel_error < 0.05,
            "local_values[{}]: numerical={:.6}, analytical={:.6}, rel_error={:.4} exceeds 5% tolerance",
            idx, numerical, analytical, rel_error
        );
    }
}
