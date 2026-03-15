//! Integration tests for V2 observation encoding (ego + entity tokens).

use drone_lib::sim_runner::{SimConfig, SimRunner, EGO_DIM, ENTITY_DIM};
use drone_lib::game::obs_layout::{entity_idx, entity_type};

fn make_config() -> SimConfig {
    SimConfig {
        drones_per_side: 4,
        targets_per_side: 2,
        seed: 42,
        ..Default::default()
    }
}

#[test]
fn test_obs_v2_ego_dimensions() {
    let runner = SimRunner::new(make_config());
    let (obs, _ids) = runner.observe_multi_group_v2(0);

    assert!(!obs.is_empty(), "Should have at least one observation");
    for (i, ob) in obs.iter().enumerate() {
        assert_eq!(
            ob.ego.len(), EGO_DIM,
            "Drone {} ego should have exactly {} elements, got {}",
            i, EGO_DIM, ob.ego.len()
        );
    }
}

#[test]
fn test_obs_v2_entity_dimensions() {
    let runner = SimRunner::new(make_config());
    let (obs, _ids) = runner.observe_multi_group_v2(0);

    for (i, ob) in obs.iter().enumerate() {
        assert_eq!(
            ob.entities.len(),
            ob.n_entities * ENTITY_DIM,
            "Drone {} entity data length {} should equal n_entities ({}) * ENTITY_DIM ({})",
            i, ob.entities.len(), ob.n_entities, ENTITY_DIM
        );
        // Length must be a multiple of ENTITY_DIM.
        assert_eq!(
            ob.entities.len() % ENTITY_DIM, 0,
            "Drone {} entity data length {} should be a multiple of ENTITY_DIM={}",
            i, ob.entities.len(), ENTITY_DIM
        );
    }
}

#[test]
fn test_obs_v2_consistency() {
    let config = make_config();

    let runner1 = SimRunner::new(config.clone());
    let (obs1, ids1) = runner1.observe_multi_group_v2(0);

    let runner2 = SimRunner::new(config);
    let (obs2, ids2) = runner2.observe_multi_group_v2(0);

    assert_eq!(ids1, ids2, "Drone IDs should be identical for same seed");
    assert_eq!(obs1.len(), obs2.len(), "Observation count should match");

    for (i, (o1, o2)) in obs1.iter().zip(obs2.iter()).enumerate() {
        assert_eq!(o1.n_entities, o2.n_entities, "Drone {} n_entities should match", i);
        for (j, (&v1, &v2)) in o1.ego.iter().zip(o2.ego.iter()).enumerate() {
            assert!(
                (v1 - v2).abs() < 1e-6,
                "Drone {} ego[{}] should be deterministic: {} vs {}",
                i, j, v1, v2
            );
        }
        for (j, (&v1, &v2)) in o1.entities.iter().zip(o2.entities.iter()).enumerate() {
            assert!(
                (v1 - v2).abs() < 1e-6,
                "Drone {} entity[{}] should be deterministic: {} vs {}",
                i, j, v1, v2
            );
        }
    }
}

#[test]
fn test_obs_v2_entity_types() {
    let runner = SimRunner::new(make_config());
    let (obs, _ids) = runner.observe_multi_group_v2(0);

    let valid_types = [
        entity_type::ENEMY_DRONE,
        entity_type::FRIENDLY_DRONE,
        entity_type::ENEMY_TARGET,
        entity_type::FRIENDLY_TARGET,
    ];

    for (drone_i, ob) in obs.iter().enumerate() {
        for ent in 0..ob.n_entities {
            let type_flag = ob.entities[ent * ENTITY_DIM + entity_idx::TYPE_FLAG];
            assert!(
                valid_types.contains(&type_flag),
                "Drone {} entity {} type_flag={} should be one of {:?}",
                drone_i, ent, type_flag, valid_types
            );
        }
    }
}

#[test]
fn test_obs_v2_alive_flags() {
    let runner = SimRunner::new(make_config());
    let (obs, _ids) = runner.observe_multi_group_v2(0);

    for (drone_i, ob) in obs.iter().enumerate() {
        for ent in 0..ob.n_entities {
            let alive_flag = ob.entities[ent * ENTITY_DIM + entity_idx::ALIVE_FLAG];
            assert!(
                (alive_flag - 1.0).abs() < 1e-6,
                "Drone {} entity {} alive_flag should be 1.0 for real entities, got {}",
                drone_i, ent, alive_flag
            );
        }
    }
}

#[test]
fn test_obs_v2_all_finite() {
    let runner = SimRunner::new(make_config());
    let (obs, _ids) = runner.observe_multi_group_v2(0);

    for (drone_i, ob) in obs.iter().enumerate() {
        for (j, &v) in ob.ego.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Drone {} ego[{}] should be finite, got {}", drone_i, j, v
            );
        }
        for (j, &v) in ob.entities.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Drone {} entity_data[{}] should be finite, got {}", drone_i, j, v
            );
        }
    }
}
