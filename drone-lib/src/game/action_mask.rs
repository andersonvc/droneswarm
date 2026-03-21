//! Action masking for the 13-action V2 space.
//!
//! Computes which actions are valid given the current entity counts.
//! Invalid actions get -inf logits before softmax, preventing the agent
//! from wasting exploration on impossible actions.

use super::obs_layout::{entity_idx, entity_type, ACT_DIM, ENTITY_DIM};

/// Compute a boolean action mask from entity tokens.
///
/// Returns [ACT_DIM] where `true` = valid, `false` = masked.
/// At least one action (evade) is always valid.
pub fn compute_action_mask(entities: &[f32], n_entities: usize) -> [bool; ACT_DIM] {
    // Count entity types.
    let mut n_enemy_drones = 0usize;
    let mut n_enemy_targets = 0usize;
    let mut n_friendly_targets = 0usize;

    for i in 0..n_entities {
        let type_flag = entities[i * ENTITY_DIM + entity_idx::TYPE_FLAG];
        if (type_flag - entity_type::ENEMY_DRONE).abs() < 0.17 {
            n_enemy_drones += 1;
        } else if (type_flag - entity_type::ENEMY_TARGET).abs() < 0.17 {
            n_enemy_targets += 1;
        } else if (type_flag - entity_type::FRIENDLY_TARGET).abs() < 0.17 {
            n_friendly_targets += 1;
        }
    }

    let has_enemy_targets = n_enemy_targets > 0;
    let has_enemy_drones = n_enemy_drones > 0;
    let has_2_enemy_drones = n_enemy_drones >= 2;
    let has_friendly_targets = n_friendly_targets > 0;

    let mut mask = [false; ACT_DIM];

    // 0-2: Attack nearest/farthest/least-defended enemy target (direct)
    mask[0] = has_enemy_targets;
    mask[1] = has_enemy_targets;
    mask[2] = has_enemy_targets;

    // 3-5: Attack nearest/farthest/least-defended enemy target (evasive)
    mask[3] = has_enemy_targets;
    mask[4] = has_enemy_targets;
    mask[5] = has_enemy_targets;

    // 6: Intercept nearest enemy drone
    mask[6] = has_enemy_drones;

    // 7: Intercept 2nd nearest enemy drone
    mask[7] = has_2_enemy_drones;

    // 8: Intercept enemy cluster
    mask[8] = has_enemy_drones;

    // 9-10: Defend nearest friendly target
    mask[9] = has_friendly_targets;
    mask[10] = has_friendly_targets;

    // 11: Patrol perimeter (needs friendly targets for route)
    mask[11] = has_friendly_targets;

    // 12: Evade — always valid
    mask[12] = true;

    mask
}

/// Apply action mask to logits: set masked actions to -1e9 (before softmax).
pub fn apply_mask_to_logits(logits: &mut [f32], mask: &[bool]) {
    for (i, &valid) in mask.iter().enumerate() {
        if !valid {
            logits[i] = -1e9;
        }
    }
}
