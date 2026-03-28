"""PPO update with IMPALA-style V-trace off-policy correction and GAE computation."""

import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from .action_mask import compute_action_mask_batch
from .constants import MASK_FILL_VALUE
from .model import PolicyNetV2Torch


class ReplayBuffer:
    """Stores recent rollouts for off-policy reuse with V-trace correction."""

    def __init__(self, max_size: int = 3):
        self.buffer: deque[dict] = deque(maxlen=max_size)

    def push(self, rollout: dict) -> None:
        """Store a rollout on CPU to save GPU memory."""
        cpu_rollout = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                       for k, v in rollout.items()}
        self.buffer.append(cpu_rollout)

    def get_all(self) -> list[dict]:
        """Return all stored rollouts (newest first)."""
        return list(reversed(self.buffer))

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedMemory:
    """Stores high-value transitions for prioritized replay during PPO.

    Keeps transitions with large |advantage| or |reward| — the moments
    where the agent was most surprised or where rare events (kills, wins)
    occurred. These are mixed into PPO mini-batches to amplify learning
    from critical moments.
    """

    def __init__(self, max_size: int = 50_000):
        self.max_size = max_size
        self.ego_obs: list[torch.Tensor] = []
        self.entity_obs: list[torch.Tensor] = []
        self.n_entities: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.advantages: list[torch.Tensor] = []
        self._size = 0

    def store(self, rollout: dict, advantages: torch.Tensor,
              adv_threshold: float = 1.5):
        """Extract and store high-value transitions from a rollout.

        Selects transitions where |advantage| > adv_threshold * std(advantages).
        """
        adv = advantages.cpu()
        adv_std = adv.std().item()
        adv_cutoff = adv_threshold * max(adv_std, 0.1)

        # Select high-value indices.
        indices = (adv.abs() > adv_cutoff).nonzero(as_tuple=True)[0]

        if len(indices) == 0:
            return

        # Store on CPU.
        self.ego_obs.append(rollout["ego_obs"][indices].cpu())
        self.entity_obs.append(rollout["entity_obs"][indices].cpu())
        self.n_entities.append(rollout["n_entities"][indices].cpu())
        self.actions.append(rollout["actions"][indices].cpu())
        self.log_probs.append(rollout["log_probs"][indices].cpu())
        self.advantages.append(adv[indices])

        self._size += len(indices)

        # Trim if over capacity — drop oldest.
        if self._size > self.max_size:
            self._compact()

    def _compact(self):
        """Concatenate and trim to max_size, keeping newest."""
        if not self.ego_obs:
            return
        ego = torch.cat(self.ego_obs)
        ent = torch.cat(self.entity_obs)
        n_ent = torch.cat(self.n_entities)
        act = torch.cat(self.actions)
        lp = torch.cat(self.log_probs)
        adv = torch.cat(self.advantages)

        # Keep the last max_size entries.
        if len(ego) > self.max_size:
            ego = ego[-self.max_size:]
            ent = ent[-self.max_size:]
            n_ent = n_ent[-self.max_size:]
            act = act[-self.max_size:]
            lp = lp[-self.max_size:]
            adv = adv[-self.max_size:]

        self.ego_obs = [ego]
        self.entity_obs = [ent]
        self.n_entities = [n_ent]
        self.actions = [act]
        self.log_probs = [lp]
        self.advantages = [adv]
        self._size = len(ego)

    def sample(self, n: int, device: torch.device) -> dict | None:
        """Sample n transitions, move to device. Returns None if empty."""
        if self._size == 0:
            return None

        self._compact()
        ego = self.ego_obs[0]
        total = len(ego)
        n = min(n, total)

        idx = torch.randint(0, total, (n,))

        return {
            "ego_obs": ego[idx].to(device),
            "entity_obs": self.entity_obs[0][idx].to(device),
            "n_entities": self.n_entities[0][idx].to(device),
            "actions": self.actions[0][idx].to(device),
            "old_log_probs": self.log_probs[0][idx].to(device),
            "advantages": self.advantages[0][idx].to(device),
        }

    def __len__(self) -> int:
        return self._size


def _ppo_step(
    model: PolicyNetV2Torch,
    optimizer: torch.optim.AdamW,
    ego_obs: torch.Tensor,
    entity_obs: torch.Tensor,
    n_entities: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_masks: torch.Tensor,
    is_off_policy: bool,
    batch_size: int,
    clip_range: float,
    ent_coef: float,
    max_grad_norm: float,
    vtrace_rho_max: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run one epoch of PPO over the given data. Returns per-batch stats."""
    N = ego_obs.shape[0]
    perm = torch.randperm(N, device=device)

    total_ploss = 0.0
    total_ent = 0.0
    total_clip = 0.0
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

        # MPS stability.
        logits = logits.float().clamp(-20.0, 20.0)
        logits = logits.masked_fill(~mb_mask, -1e4)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        new_log_prob = log_probs.gather(1, mb_actions.unsqueeze(1)).squeeze(1)

        ratio = (new_log_prob - mb_old_lp).exp().clamp(0.05, 20.0)

        if is_off_policy:
            # V-trace: clip importance weights to reduce variance from stale data.
            # rho = min(rho_max, pi/mu) — truncated importance sampling.
            rho = ratio.clamp(max=vtrace_rho_max).detach()
            surr1 = rho * mb_adv
            surr2 = rho.clamp(1.0 - clip_range, 1.0 + clip_range) * mb_adv
            # Scale loss by mean(rho) to normalize for the effective sample count.
            policy_loss = -torch.min(surr1, surr2).mean()
        else:
            # Standard PPO clipping for on-policy data.
            surr1 = ratio * mb_adv
            surr2 = ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = policy_loss - ent_coef * entropy

        if not torch.isfinite(loss):
            print("[WARN] NaN loss, skipping batch", flush=True)
            n_batches += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if torch.isfinite(grad_norm):
            optimizer.step()
        else:
            print("[WARN] NaN gradients, skipping optimizer step", flush=True)

        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > clip_range).float().mean().item()

        # Early stop: if >30% of samples are clipped, the policy has drifted
        # too far — further updates on this data will be harmful.
        if clip_frac > 0.3:
            return {
                "policy_loss": policy_loss.item(),
                "entropy": entropy.item(),
                "clip_fraction": clip_frac,
                "early_stopped": True,
            }

        total_ploss += policy_loss.item()
        total_ent += entropy.item()
        total_clip += clip_frac
        n_batches += 1

    return {
        "policy_loss": total_ploss / max(n_batches, 1),
        "entropy": total_ent / max(n_batches, 1),
        "clip_fraction": total_clip / max(n_batches, 1),
    }


def _prepare_rollout(rollout: dict, device: torch.device):
    """Move rollout to device and trim entity padding."""
    max_ent = max(int(rollout["n_entities"].max().item()), 1)
    ent_trimmed = rollout["entity_obs"][:, :max_ent, :]

    action_masks = compute_action_mask_batch(
        rollout["entity_obs"], rollout["n_entities"]
    )

    return {
        "ego_obs": rollout["ego_obs"].to(device),
        "entity_obs": ent_trimmed.to(device),
        "n_entities": rollout["n_entities"].to(device),
        "actions": rollout["actions"].to(device),
        "old_log_probs": rollout["log_probs"].to(device),
        "advantages": rollout["advantages"].to(device),
        "action_masks": action_masks.to(device),
    }


def ppo_update(
    model: PolicyNetV2Torch,
    optimizer: torch.optim.AdamW,
    rollout: dict,
    device: torch.device,
    n_epochs: int = 10,
    batch_size: int = 12288,
    clip_range: float = 0.2,
    ent_coef: float = 0.03,
    max_grad_norm: float = 0.5,
    replay_buffer: ReplayBuffer | None = None,
    priority_memory: PrioritizedMemory | None = None,
) -> dict:
    """Run PPO update with optional IMPALA-style off-policy replay.

    The current rollout is trained on-policy (standard PPO clipping).
    Past rollouts from the replay buffer are trained with V-trace
    importance weight clipping for off-policy correction.

    Returns per-epoch stats.
    """
    # Prepare current (on-policy) rollout.
    current = _prepare_rollout(rollout, device)

    # Prepare off-policy rollouts from replay buffer.
    off_policy_data = []
    if replay_buffer is not None and len(replay_buffer) > 0:
        for old_rollout in replay_buffer.get_all():
            off_policy_data.append(_prepare_rollout(old_rollout, device))

    stats = {
        "policy_loss": [],
        "entropy": [],
        "clip_fraction": [],
        "epoch_time": [],
    }

    for epoch in range(n_epochs):
        t0 = time.monotonic()

        # Train on current (on-policy) rollout.
        epoch_stats = _ppo_step(
            model, optimizer,
            current["ego_obs"], current["entity_obs"], current["n_entities"],
            current["actions"], current["old_log_probs"],
            current["advantages"], current["action_masks"],
            is_off_policy=False,
            batch_size=batch_size,
            clip_range=clip_range,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            device=device,
        )

        # Prioritized replay: mix in high-value transitions from past rollouts.
        if priority_memory is not None and len(priority_memory) > 0:
            # Sample 20% of batch_size from priority memory.
            n_priority = max(batch_size // 5, 256)
            priority_batch = priority_memory.sample(n_priority, device)
            if priority_batch is not None:
                masks = compute_action_mask_batch(
                    priority_batch["entity_obs"].cpu(),
                    priority_batch["n_entities"].cpu(),
                ).to(device)
                # Trim entities.
                max_ent = max(int(priority_batch["n_entities"].max().item()), 1)
                _ppo_step(
                    model, optimizer,
                    priority_batch["ego_obs"],
                    priority_batch["entity_obs"][:, :max_ent, :],
                    priority_batch["n_entities"],
                    priority_batch["actions"],
                    priority_batch["old_log_probs"],
                    priority_batch["advantages"],
                    masks,
                    is_off_policy=True,
                    batch_size=n_priority,
                    clip_range=clip_range,
                    ent_coef=ent_coef * 0.5,
                    max_grad_norm=max_grad_norm,
                    vtrace_rho_max=0.5,
                    device=device,
                )

        # Early stop: if clip fraction is too high, policy has drifted —
        # stop all remaining epochs to prevent collapse.
        if epoch_stats.get("early_stopped", False):
            print(f"  [PPO] Early stop at epoch {epoch+1}/{n_epochs} "
                  f"(clip_frac={epoch_stats['clip_fraction']:.3f})", flush=True)
            stats["policy_loss"].append(epoch_stats["policy_loss"])
            stats["entropy"].append(epoch_stats["entropy"])
            stats["clip_fraction"].append(epoch_stats["clip_fraction"])
            stats["epoch_time"].append(time.monotonic() - t0)
            break

        # Train on off-policy replayed rollouts (V-trace corrected).
        for i, old_data in enumerate(off_policy_data):
            rho_max = 1.0 / (i + 2)
            off_stats = _ppo_step(
                model, optimizer,
                old_data["ego_obs"], old_data["entity_obs"], old_data["n_entities"],
                old_data["actions"], old_data["old_log_probs"],
                old_data["advantages"], old_data["action_masks"],
                is_off_policy=True,
                batch_size=batch_size,
                clip_range=clip_range,
                ent_coef=ent_coef * 0.5,
                max_grad_norm=max_grad_norm,
                vtrace_rho_max=rho_max,
                device=device,
            )
            if off_stats.get("early_stopped", False):
                break

        dt = time.monotonic() - t0
        stats["policy_loss"].append(epoch_stats["policy_loss"])
        stats["entropy"].append(epoch_stats["entropy"])
        stats["clip_fraction"].append(epoch_stats["clip_fraction"])
        stats["epoch_time"].append(dt)

    # Push current rollout to replay buffer for future off-policy use.
    if replay_buffer is not None:
        replay_buffer.push(rollout)

    # Store high-value transitions in priority memory.
    if priority_memory is not None:
        priority_memory.store(rollout, rollout["advantages"])

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
