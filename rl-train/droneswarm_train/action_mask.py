"""Batched action masking — replicates Rust action_mask.rs logic."""

import torch

from .constants import (
    ACT_DIM,
    ENEMY_DRONE,
    ENEMY_TARGET,
    FRIENDLY_TARGET,
    MAX_ENTITIES,
    TYPE_FLAG_IDX,
    TYPE_THRESHOLD,
)


def compute_action_mask_batch(
    entity_obs: torch.Tensor,
    n_entities: torch.Tensor,
) -> torch.Tensor:
    """Compute boolean action mask for a batch of observations.

    Args:
        entity_obs: [B, MAX_ENTITIES, ENTITY_DIM]
        n_entities: [B] int tensor

    Returns:
        [B, ACT_DIM] bool tensor where True = valid action.
    """
    B = entity_obs.shape[0]
    device = entity_obs.device

    indices = torch.arange(MAX_ENTITIES, device=device).unsqueeze(0)
    valid = indices < n_entities.unsqueeze(1)

    type_flags = entity_obs[:, :, TYPE_FLAG_IDX]

    is_enemy_drone = (type_flags - ENEMY_DRONE).abs() < TYPE_THRESHOLD
    is_enemy_target = (type_flags - ENEMY_TARGET).abs() < TYPE_THRESHOLD
    is_friendly_target = (type_flags - FRIENDLY_TARGET).abs() < TYPE_THRESHOLD

    is_enemy_drone = is_enemy_drone & valid
    is_enemy_target = is_enemy_target & valid
    is_friendly_target = is_friendly_target & valid

    n_enemy_drones = is_enemy_drone.sum(dim=1)
    n_enemy_targets = is_enemy_target.sum(dim=1)
    n_friendly_targets = is_friendly_target.sum(dim=1)

    has_enemy_targets = n_enemy_targets > 0
    has_enemy_drones = n_enemy_drones > 0
    has_2_enemy_drones = n_enemy_drones >= 2
    has_friendly_targets = n_friendly_targets > 0

    mask = torch.zeros(B, ACT_DIM, dtype=torch.bool, device=device)

    # 0-5: Attack enemy targets (direct and evasive)
    for i in range(6):
        mask[:, i] = has_enemy_targets

    # 6: Intercept nearest enemy drone
    mask[:, 6] = has_enemy_drones

    # 7: Intercept 2nd nearest enemy drone
    mask[:, 7] = has_2_enemy_drones

    # 8: Intercept enemy cluster
    mask[:, 8] = has_enemy_drones

    # 9-11: Defend/patrol friendly targets
    for i in range(9, 12):
        mask[:, i] = has_friendly_targets

    # 12: Evade — always valid
    mask[:, 12] = True

    return mask
