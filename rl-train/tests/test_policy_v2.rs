//! Integration tests for PolicyNetV2 forward, act, batch, save/load, and backward.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rl_train::network::PolicyNetV2;
use rl_train::ppo::PolicyNetV2PPOExt;

const EGO_DIM: usize = 25;
const ENTITY_DIM: usize = 10;
const ACT_DIM: usize = 13;
const EMBED_DIM: usize = 64;
const HIDDEN: usize = 256;

#[test]
fn test_policy_v2_forward_shapes() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut net = PolicyNetV2::new(EGO_DIM, ENTITY_DIM, ACT_DIM, EMBED_DIM, HIDDEN, &mut rng);

    let ego = vec![0.1f32; EGO_DIM];
    let n_entities = 3;
    let entities = vec![0.2f32; n_entities * ENTITY_DIM];

    let (logits, value, h2) = net.forward(&ego, &entities, n_entities);

    assert_eq!(
        logits.len(), ACT_DIM,
        "Logits should have {} elements, got {}", ACT_DIM, logits.len()
    );
    assert_eq!(
        h2.len(), HIDDEN,
        "h2 hidden should have {} elements, got {}", HIDDEN, h2.len()
    );
    assert!(value.is_finite(), "Value should be finite, got {}", value);
    for (i, &l) in logits.iter().enumerate() {
        assert!(l.is_finite(), "Logit {} should be finite, got {}", i, l);
    }
}

#[test]
fn test_policy_v2_forward_zero_entities() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut net = PolicyNetV2::new(EGO_DIM, ENTITY_DIM, ACT_DIM, EMBED_DIM, HIDDEN, &mut rng);

    let ego = vec![0.5f32; EGO_DIM];
    let entities: Vec<f32> = vec![];

    let (logits, value, h2) = net.forward(&ego, &entities, 0);

    assert_eq!(
        logits.len(), ACT_DIM,
        "Zero-entity forward should still produce {} logits", ACT_DIM
    );
    assert!(value.is_finite(), "Zero-entity value should be finite, got {}", value);
    assert_eq!(
        h2.len(), HIDDEN,
        "Zero-entity h2 should have {} elements", HIDDEN
    );
}

#[test]
fn test_policy_v2_act_valid_action() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut net = PolicyNetV2::new(EGO_DIM, ENTITY_DIM, ACT_DIM, EMBED_DIM, HIDDEN, &mut rng);

    let ego = vec![0.3f32; EGO_DIM];
    let entities = vec![0.1f32; 2 * ENTITY_DIM];

    let (action, log_prob, value) = net.act(&ego, &entities, 2, Some(&entities), &mut rng);

    assert!(
        (action as usize) < ACT_DIM,
        "Action {} should be in [0, {})", action, ACT_DIM
    );
    assert!(
        log_prob < 0.0,
        "log_prob should be negative (log of probability), got {}", log_prob
    );
    assert!(
        value.is_finite(),
        "Value from act() should be finite, got {}", value
    );
}

#[test]
fn test_policy_v2_act_batch() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let net = PolicyNetV2::new(EGO_DIM, ENTITY_DIM, ACT_DIM, EMBED_DIM, HIDDEN, &mut rng);

    let batch_size = 4;
    let ego_data: Vec<Vec<f32>> = (0..batch_size)
        .map(|i| vec![0.1 * (i + 1) as f32; EGO_DIM])
        .collect();
    let egos: Vec<&[f32]> = ego_data.iter().map(|e| e.as_slice()).collect();

    let entity_counts = [2, 0, 5, 1];
    let entity_data: Vec<Vec<f32>> = entity_counts
        .iter()
        .map(|&n| vec![0.2f32; n * ENTITY_DIM])
        .collect();
    let entities: Vec<&[f32]> = entity_data.iter().map(|e| e.as_slice()).collect();

    let results = net.act_batch(&egos, &entities, &entity_counts, &mut rng);

    assert_eq!(
        results.len(), batch_size,
        "act_batch should return {} results, got {}", batch_size, results.len()
    );
    for (i, &(action, log_prob, value)) in results.iter().enumerate() {
        assert!(
            (action as usize) < ACT_DIM,
            "Sample {} action {} should be in [0, {})", i, action, ACT_DIM
        );
        assert!(log_prob.is_finite(), "Sample {} log_prob should be finite", i);
        assert!(value.is_finite(), "Sample {} value should be finite", i);
    }
}

#[test]
fn test_policy_v2_act_batch_with_h2() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let net = PolicyNetV2::new(EGO_DIM, ENTITY_DIM, ACT_DIM, EMBED_DIM, HIDDEN, &mut rng);

    let batch_size = 3;
    let ego_data: Vec<Vec<f32>> = (0..batch_size)
        .map(|i| vec![0.1 * (i + 1) as f32; EGO_DIM])
        .collect();
    let egos: Vec<&[f32]> = ego_data.iter().map(|e| e.as_slice()).collect();

    let entity_counts = [2, 4, 1];
    let entity_data: Vec<Vec<f32>> = entity_counts
        .iter()
        .map(|&n| vec![0.3f32; n * ENTITY_DIM])
        .collect();
    let entities: Vec<&[f32]> = entity_data.iter().map(|e| e.as_slice()).collect();

    let (results, h2_flat) = net.act_batch_with_h2(&egos, &entities, &entity_counts, &mut rng);

    assert_eq!(
        results.len(), batch_size,
        "act_batch_with_h2 should return {} results", batch_size
    );
    assert_eq!(
        h2_flat.len(), batch_size * HIDDEN,
        "h2_flat should have batch_size * hidden = {} elements, got {}",
        batch_size * HIDDEN, h2_flat.len()
    );
    for &v in &h2_flat {
        assert!(v.is_finite(), "h2 element should be finite, got {}", v);
    }
}

#[test]
fn test_policy_v2_save_load_roundtrip() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let net = PolicyNetV2::new(EGO_DIM, ENTITY_DIM, ACT_DIM, EMBED_DIM, HIDDEN, &mut rng);

    let temp_path = "/tmp/test_policy_v2_roundtrip.json";
    net.save(temp_path).expect("Save should succeed");

    let mut loaded = PolicyNetV2::load(temp_path).expect("Load should succeed");

    // Forward both with same input and compare.
    let ego = vec![0.5f32; EGO_DIM];
    let entities = vec![0.3f32; 3 * ENTITY_DIM];
    let n_entities = 3;

    let mut net_clone = net.clone();
    let (logits1, value1, h2_1) = net_clone.forward(&ego, &entities, n_entities);
    let (logits2, value2, h2_2) = loaded.forward(&ego, &entities, n_entities);

    assert_eq!(logits1.len(), logits2.len());
    for (i, (&l1, &l2)) in logits1.iter().zip(logits2.iter()).enumerate() {
        assert!(
            (l1 - l2).abs() < 1e-6,
            "Logit {} should match after save/load: {} vs {}", i, l1, l2
        );
    }
    assert!(
        (value1 - value2).abs() < 1e-6,
        "Value should match after save/load: {} vs {}", value1, value2
    );
    for (i, (&a, &b)) in h2_1.iter().zip(h2_2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "h2 element {} should match after save/load: {} vs {}", i, a, b
        );
    }

    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_policy_v2_backward_runs() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut net = PolicyNetV2::new(EGO_DIM, ENTITY_DIM, ACT_DIM, EMBED_DIM, HIDDEN, &mut rng);

    let ego = vec![0.2f32; EGO_DIM];
    let entities = vec![0.1f32; 2 * ENTITY_DIM];

    net.zero_grad();
    net.backward_ppo_v2(
        &ego,
        &entities,
        2,            // n_entities
        &entities,    // raw_entities (unnormalized, same in test)
        5,            // action
        0.5,          // advantage
        1.0,          // return
        -2.0,         // old_log_prob
        0.2,          // clip_range
        0.5,          // vf_coef
        0.01,         // ent_coef
    );

    // Verify that at least some gradients are non-zero after backward.
    let grad_norm = net.grad_norm();
    assert!(
        grad_norm > 0.0,
        "Gradient norm should be positive after backward, got {}", grad_norm
    );
}

#[test]
fn test_inference_v2_matches_training() {
    use drone_lib::inference::InferenceNetV2;

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let net = PolicyNetV2::new(EGO_DIM, ENTITY_DIM, ACT_DIM, EMBED_DIM, HIDDEN, &mut rng);

    // Serialize to JSON and load as InferenceNetV2.
    let json = serde_json::to_string(&net).expect("Serialization should succeed");
    let inf_net = InferenceNetV2::from_json(&json).expect("InferenceNetV2 parse should succeed");

    // Forward both with the same input.
    let ego = vec![0.4f32; EGO_DIM];
    let entities = vec![0.15f32; 4 * ENTITY_DIM];
    let n_entities = 4;

    let mut net_clone = net.clone();
    let (logits_train, _value, _h2) = net_clone.forward(&ego, &entities, n_entities);
    let logits_inf = inf_net.forward(&ego, &entities, n_entities);

    assert_eq!(
        logits_train.len(), logits_inf.len(),
        "Training and inference logits should have same length"
    );
    for (i, (&lt, &li)) in logits_train.iter().zip(logits_inf.iter()).enumerate() {
        assert!(
            (lt - li).abs() < 1e-5,
            "Logit {} should match between training ({}) and inference ({})",
            i, lt, li
        );
    }
}
