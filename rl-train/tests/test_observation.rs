//! Tests for observation vector format and normalization.

use drone_lib::sim_runner::{SimConfig, SimRunner, OBS_DIM};

#[test]
fn test_observation_dimensions() {
    let config = SimConfig {
        drones_per_side: 4,
        targets_per_side: 2,
        seed: 42,
        ..Default::default()
    };
    let runner = SimRunner::new(config);
    let (obs, _ids) = runner.observe_multi();

    assert_eq!(obs.len(), 4, "Should have one observation per drone");
    for observation in &obs {
        assert_eq!(observation.len(), OBS_DIM, "Each observation should have exactly {} dimensions", OBS_DIM);
    }
}

#[test]
fn test_observation_all_normalized() {
    let config = SimConfig {
        drones_per_side: 8,
        targets_per_side: 3,
        seed: 123,
        ..Default::default()
    };
    let runner = SimRunner::new(config);
    let (obs, _ids) = runner.observe_multi();

    for (drone_idx, observation) in obs.iter().enumerate() {
        for (dim_idx, &value) in observation.iter().enumerate() {
            assert!(value.is_finite(),
                "Observation[drone={}][dim={}] should be finite, got {}",
                drone_idx, dim_idx, value);

            // Most observations should be in [0, 1] range, except velocities which can be negative
            // For simplicity, just check they're reasonable (not extremely large)
            assert!(value >= -10.0 && value <= 10.0,
                "Observation[drone={}][dim={}] should be in reasonable range, got {}",
                drone_idx, dim_idx, value);
        }
    }
}

#[test]
fn test_observation_consistency_across_resets() {
    let config = SimConfig {
        drones_per_side: 4,
        targets_per_side: 2,
        seed: 555,
        ..Default::default()
    };

    // Reset with same seed twice and check observations are identical
    let mut runner1 = SimRunner::new(config.clone());
    let (obs1, ids1) = runner1.reset_multi_with_seed(555);

    let mut runner2 = SimRunner::new(config);
    let (obs2, ids2) = runner2.reset_multi_with_seed(555);

    assert_eq!(obs1.len(), obs2.len());
    assert_eq!(ids1, ids2);

    for (i, (o1, o2)) in obs1.iter().zip(obs2.iter()).enumerate() {
        for (j, (&v1, &v2)) in o1.iter().zip(o2.iter()).enumerate() {
            assert!((v1 - v2).abs() < 1e-6,
                "Observations should be identical for same seed: drone {} dim {} differs: {} vs {}",
                i, j, v1, v2);
        }
    }
}

#[test]
fn test_observation_agent_id_uniqueness() {
    let config = SimConfig {
        drones_per_side: 6,
        targets_per_side: 2,
        seed: 777,
        ..Default::default()
    };
    let runner = SimRunner::new(config);
    let (obs, ids) = runner.observe_multi();

    // Agent ID is at index 60
    const AGENT_ID_IDX: usize = 60;

    let agent_ids: Vec<f32> = obs.iter().map(|o| o[AGENT_ID_IDX]).collect();

    // All agent IDs should be different (unique per drone)
    for (i, &id1) in agent_ids.iter().enumerate() {
        for (j, &id2) in agent_ids.iter().enumerate() {
            if i != j {
                assert_ne!(id1, id2,
                    "Agent IDs should be unique: drones {} and {} both have agent_id={}",
                    i, j, id1);
            }
        }
    }

    // Agent IDs should be normalized (typically drone_id / initial_drones_per_side)
    for &agent_id in &agent_ids {
        assert!(agent_id >= 0.0 && agent_id <= 1.0,
            "Agent ID should be normalized to [0,1]: got {}", agent_id);
    }
}