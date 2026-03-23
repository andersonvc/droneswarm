"""Policy evaluation against doctrine opponents."""

import numpy as np
import torch
import torch.nn.functional as F

import droneswarm_env

from .action_mask import compute_action_mask_batch
from .constants import MASK_FILL_VALUE, SPEED_MULTIPLIER
from .curriculum import compute_max_ticks
from .model import PolicyNetV2Torch
from .normalize import RunningMeanStd
from .rollout import obs_to_tensors


def evaluate(
    model: PolicyNetV2Torch,
    ego_norm: RunningMeanStd,
    device: torch.device,
    stage: dict,
    n_episodes: int = 50,
    n_eval_envs: int = 50,
) -> tuple[float, float]:
    """Evaluate policy against doctrine opponents.

    Creates a separate VecSimRunner for evaluation to avoid disturbing
    training environments.

    Returns (win_rate, mean_reward).
    """
    n_eval_envs = min(n_eval_envs, n_episodes)
    max_ticks = compute_max_ticks(stage["world_size"])

    eval_env = droneswarm_env.VecSimRunner(
        n_envs=n_eval_envs,
        drones_per_side=stage["drones"],
        targets_per_side=stage["targets"],
        world_size=stage["world_size"],
        max_ticks=max_ticks,
        speed_multiplier=SPEED_MULTIPLIER,
        skip_orca=True,
    )

    model.eval()
    wins = 0
    total_reward = 0.0
    episodes_done = 0

    ep_rewards = [0.0] * n_eval_envs

    obs = eval_env.reset()

    while episodes_done < n_episodes:
        ego, ent, n_ent = obs_to_tensors(obs, device)
        normed_ego = ego_norm.normalize(ego)

        with torch.no_grad():
            logits, _ = model(normed_ego, ent, n_ent)
            masks = compute_action_mask_batch(ent, n_ent)
            logits = logits.masked_fill(~masks, MASK_FILL_VALUE)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample()

        step_result = eval_env.step(actions.cpu().numpy().astype(np.int32))

        prev_env_indices = np.asarray(obs["env_indices"])
        prev_rewards = np.asarray(step_result["rewards"])
        prev_dones = np.asarray(step_result["dones"])

        for env_idx in range(n_eval_envs):
            drone_mask = prev_env_indices == env_idx
            if not drone_mask.any():
                continue

            env_reward = prev_rewards[drone_mask].mean()
            ep_rewards[env_idx] += float(env_reward)

            if prev_dones[drone_mask].any():
                total_reward += ep_rewards[env_idx]
                if ep_rewards[env_idx] > 0:
                    wins += 1
                episodes_done += 1
                ep_rewards[env_idx] = 0.0

                if episodes_done >= n_episodes:
                    break

        obs = step_result

    model.train()

    win_rate = wins / max(episodes_done, 1)
    mean_reward = total_reward / max(episodes_done, 1)
    return win_rate, mean_reward
