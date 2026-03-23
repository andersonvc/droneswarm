"""Rollout collection from vectorized environments."""

import numpy as np
import torch
import torch.nn.functional as F

import droneswarm_env

from .action_mask import compute_action_mask_batch
from .constants import MASK_FILL_VALUE
from .model import PolicyNetV2Torch
from .normalize import RunningMeanStd


def obs_to_tensors(
    obs: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert numpy observation dict to (ego, entities, n_entities) tensors."""
    ego = torch.from_numpy(np.asarray(obs["ego_obs"])).to(
        device, dtype=torch.float32
    )
    ent = torch.from_numpy(np.asarray(obs["entity_obs"])).to(
        device, dtype=torch.float32
    )
    n_ent = torch.from_numpy(np.asarray(obs["n_entities"])).to(
        device, dtype=torch.int32
    )
    return ego, ent, n_ent


def collect_rollout(
    env: droneswarm_env.VecSimRunner,
    model: PolicyNetV2Torch,
    ego_norm: RunningMeanStd,
    obs: dict,
    n_steps: int,
    device: torch.device,
) -> tuple[dict, dict]:
    """Collect rollout data from vectorized environment.

    Returns (rollout, next_obs) where rollout is a dict of concatenated
    tensors and next_obs is the observation dict for the next step.
    """
    storage = {
        "ego_obs": [],
        "entity_obs": [],
        "n_entities": [],
        "actions": [],
        "log_probs": [],
        "values": [],
        "rewards": [],
        "team_rewards": [],
        "dones": [],
        "truncated": [],
        "drone_died": [],
        "drone_ids": [],
        "env_indices": [],
    }

    for _ in range(n_steps):
        ego, ent, n_ent = obs_to_tensors(obs, device)

        ego_norm.update(ego)
        normed_ego = ego_norm.normalize(ego)

        with torch.no_grad():
            logits, values = model(normed_ego, ent, n_ent)
            masks = compute_action_mask_batch(ent, n_ent)
            logits = logits.masked_fill(~masks, MASK_FILL_VALUE)
            # NaN guard: if weights diverged, reset logits to uniform.
            if logits.isnan().any():
                print("[WARN] NaN in logits, resetting to uniform", flush=True)
                logits = torch.zeros_like(logits)
                logits = logits.masked_fill(~masks, MASK_FILL_VALUE)
            probs = F.softmax(logits, dim=-1)
            probs = probs.clamp(min=1e-8)
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        step_result = env.step(actions.cpu().numpy().astype(np.int32))

        # Store transition (ego_obs stored unnormalized for later re-norm).
        storage["ego_obs"].append(ego)
        storage["entity_obs"].append(ent)
        storage["n_entities"].append(n_ent)
        storage["actions"].append(actions)
        storage["log_probs"].append(log_probs)
        storage["values"].append(values)
        storage["rewards"].append(
            torch.from_numpy(
                np.asarray(step_result["rewards"], dtype=np.float32)
            ).to(device)
        )
        storage["team_rewards"].append(
            torch.from_numpy(
                np.asarray(step_result["team_rewards"], dtype=np.float32)
            ).to(device)
        )
        storage["dones"].append(
            torch.from_numpy(
                np.asarray(step_result["dones"], dtype=np.float32)
            ).to(device)
        )
        storage["truncated"].append(
            torch.from_numpy(
                np.asarray(step_result["truncated"], dtype=np.float32)
            ).to(device)
        )
        storage["drone_died"].append(
            torch.from_numpy(
                np.asarray(step_result["drone_died"], dtype=np.float32)
            ).to(device)
        )
        storage["drone_ids"].append(
            torch.from_numpy(
                np.asarray(obs["drone_ids"], dtype=np.int32)
            ).to(device)
        )
        storage["env_indices"].append(
            torch.from_numpy(
                np.asarray(obs["env_indices"], dtype=np.int32)
            ).to(device)
        )

        obs = step_result

    rollout = {k: torch.cat(v, dim=0) for k, v in storage.items()}
    return rollout, obs
