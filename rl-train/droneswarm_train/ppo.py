"""PPO update and GAE computation."""

import time

import numpy as np
import torch
import torch.nn.functional as F

from .action_mask import compute_action_mask_batch
from .constants import MASK_FILL_VALUE
from .model import PolicyNetV2Torch


def ppo_update(
    model: PolicyNetV2Torch,
    optimizer: torch.optim.AdamW,
    rollout: dict,
    device: torch.device,
    n_epochs: int = 3,
    batch_size: int = 4096,
    clip_range: float = 0.2,
    ent_coef: float = 0.03,
    max_grad_norm: float = 0.5,
) -> dict:
    """Run PPO update epochs on the model. Returns per-epoch stats."""
    # Trim entity padding to max actual entity count.
    max_actual_ent = int(rollout["n_entities"].max().item())
    max_actual_ent = max(max_actual_ent, 1)
    entity_obs_trimmed = rollout["entity_obs"][:, :max_actual_ent, :]

    # Precompute action masks on CPU before moving to device.
    action_masks = compute_action_mask_batch(
        rollout["entity_obs"], rollout["n_entities"]
    )

    # Move data to device.
    ego_obs = rollout["ego_obs"].to(device)
    entity_obs = entity_obs_trimmed.to(device)
    n_entities = rollout["n_entities"].to(device)
    actions = rollout["actions"].to(device)
    old_log_probs = rollout["log_probs"].to(device)
    advantages = rollout["advantages"].to(device)
    action_masks = action_masks.to(device)

    N = ego_obs.shape[0]

    stats = {
        "policy_loss": [],
        "entropy": [],
        "clip_fraction": [],
        "epoch_time": [],
    }

    for epoch in range(n_epochs):
        t0 = time.monotonic()
        perm = torch.randperm(N, device=device)

        epoch_policy_loss = 0.0
        epoch_entropy = 0.0
        epoch_clip_frac = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]

            mb_ego = ego_obs[idx]
            mb_ent = entity_obs[idx]
            mb_n_ent = n_entities[idx]
            mb_actions = actions[idx]
            mb_old_lp = old_log_probs[idx]
            mb_adv = advantages[idx]
            mb_mask = action_masks[idx]

            logits, _ = model(mb_ego, mb_ent, mb_n_ent)
            logits = logits.clamp(-50.0, 50.0)  # MPS stability
            logits = logits.masked_fill(~mb_mask, MASK_FILL_VALUE)

            probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
            log_probs = torch.log(probs)
            new_log_prob = log_probs.gather(
                1, mb_actions.unsqueeze(1)
            ).squeeze(1)

            ratio = (new_log_prob - mb_old_lp).exp().clamp(0.05, 20.0)
            surr1 = ratio * mb_adv
            surr2 = (
                ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * mb_adv
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = -(probs * log_probs).sum(dim=-1).mean()

            loss = policy_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            # Skip step if gradients are NaN (prevents weight corruption).
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                print("[WARN] NaN gradients, skipping optimizer step", flush=True)

            with torch.no_grad():
                clip_frac = (
                    ((ratio - 1.0).abs() > clip_range).float().mean().item()
                )

            epoch_policy_loss += policy_loss.item()
            epoch_entropy += entropy.item()
            epoch_clip_frac += clip_frac
            n_batches += 1

        dt = time.monotonic() - t0

        avg_ploss = epoch_policy_loss / max(n_batches, 1)
        avg_ent = epoch_entropy / max(n_batches, 1)
        avg_clip = epoch_clip_frac / max(n_batches, 1)

        stats["policy_loss"].append(avg_ploss)
        stats["entropy"].append(avg_ent)
        stats["clip_fraction"].append(avg_clip)
        stats["epoch_time"].append(dt)

    return stats


def compute_gae(
    rollout: dict,
    last_values: dict,
    gamma: float,
    lam: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(env, drone) GAE with death bootstrapping.

    Handles three cases:
      - True terminal (done && !truncated): bootstrap with 0
      - Truncated (done && truncated): bootstrap with V(s')
      - Drone died (!done): reset GAE, bootstrap with current value
    """
    N = rollout["rewards"].shape[0]
    adv_np = np.zeros(N, dtype=np.float32)
    ret_np = np.zeros(N, dtype=np.float32)

    env_ids = rollout["env_indices"].cpu().numpy()
    drone_ids = rollout["drone_ids"].cpu().numpy()
    rewards = rollout["rewards"].cpu().numpy()
    values = rollout["values"].cpu().numpy()
    dones = rollout["dones"].cpu().numpy()
    truncated = rollout["truncated"].cpu().numpy()
    drone_died = rollout["drone_died"].cpu().numpy()

    # Group transitions by (env_idx, drone_id).
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(N):
        key = (int(env_ids[i]), int(drone_ids[i]))
        groups.setdefault(key, []).append(i)

    for (env_idx, drone_id), indices in groups.items():
        last_val = last_values.get((env_idx, drone_id), 0.0)
        gae = 0.0

        for j in reversed(range(len(indices))):
            idx = indices[j]
            r = float(rewards[idx])
            v = float(values[idx])
            d = float(dones[idx])
            tr = float(truncated[idx])
            died = float(drone_died[idx])

            if j + 1 < len(indices):
                next_val = float(values[indices[j + 1]])
            else:
                next_val = last_val

            if d > 0.5 and tr < 0.5:
                # True terminal: bootstrap with 0.
                next_non_terminal = 0.0
            elif died > 0.5 and d < 0.5:
                # Drone died but episode continues: reset GAE.
                gae = 0.0
                bootstrap = v
                delta = r + gamma * bootstrap - v
                adv_np[idx] = delta
                ret_np[idx] = delta + v
                continue
            elif tr > 0.5:
                next_non_terminal = 1.0
            else:
                next_non_terminal = 1.0

            delta = r + gamma * next_val * next_non_terminal - v
            gae = delta + gamma * lam * next_non_terminal * gae
            adv_np[idx] = gae
            ret_np[idx] = gae + v

    advantages = torch.from_numpy(adv_np).to(device)
    returns = torch.from_numpy(ret_np).to(device)
    return advantages, returns
