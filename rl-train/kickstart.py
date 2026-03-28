#!/usr/bin/env python3
"""Kickstart RL policy by imitating defensive doctrine behavior.

Runs episodes with a simple heuristic policy (mimicking defensive doctrine),
records (observation, action) pairs, then trains the policy network supervised
with cross-entropy loss. The resulting weights are used to initialize RL training,
giving the agent a strong starting point instead of random exploration.

Usage:
    python rl-train/kickstart.py --n-episodes 5000 --device mps
    python rl-train/train.py --resume results/kickstart_model.json --start-stage 1 ...
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import droneswarm_env

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from droneswarm_train.model import PolicyNetV2Torch
from droneswarm_train.weights import save_model_weights
from droneswarm_train.constants import EGO_DIM, ENTITY_DIM, ACT_DIM, MAX_ENTITIES


def doctrine_heuristic(ego_obs, entity_obs, n_entities):
    """Compute doctrine-like actions for a batch of drones.

    Mimics defensive doctrine: ~50% of drones defend, ~50% attack.
    Uses the drone's relative_alive_index (ego[16]) to split roles:
      - Lower rank (closer to 0) → defend
      - Higher rank (closer to 1) → attack

    Returns: [N] int32 tensor of action indices.
    """
    N = ego_obs.shape[0]
    actions = np.zeros(N, dtype=np.int32)

    for i in range(N):
        ego = ego_obs[i]
        ents = entity_obs[i]
        n_ent = int(n_entities[i])

        # Drone's rank in its group (0=first, 1=last).
        rank = float(ego[16])  # RELATIVE_ALIVE_INDEX

        # Count entity types.
        n_enemy_targets = 0
        n_enemy_drones = 0
        n_friendly_targets = 0
        nearest_enemy_target_dist = float('inf')
        nearest_enemy_drone_dist = float('inf')

        for j in range(n_ent):
            ent = ents[j]
            type_flag = float(ent[6])
            dist = float(ent[2])  # normalized distance

            if abs(type_flag - 0.0) < 0.17:  # enemy drone
                n_enemy_drones += 1
                if dist < nearest_enemy_drone_dist:
                    nearest_enemy_drone_dist = dist
            elif abs(type_flag - 0.67) < 0.17:  # enemy target
                n_enemy_targets += 1
                if dist < nearest_enemy_target_dist:
                    nearest_enemy_target_dist = dist
            elif abs(type_flag - 1.0) < 0.17:  # friendly target
                n_friendly_targets += 1

        # Doctrine split: bottom 50% by rank defend, top 50% attack.
        is_defender = rank < 0.5

        if is_defender:
            # Defender behavior.
            if n_enemy_drones > 0 and nearest_enemy_drone_dist < 0.3:
                # Enemy drone nearby — intercept it.
                actions[i] = 6  # intercept nearest
            elif n_friendly_targets > 0:
                # Defend nearest friendly target.
                actions[i] = 9  # defend tight
            else:
                # No targets to defend — patrol.
                actions[i] = 11  # patrol
        else:
            # Attacker behavior.
            if n_enemy_targets > 0:
                # Attack nearest enemy target (direct).
                actions[i] = 0  # attack nearest direct
            elif n_enemy_drones > 0:
                # No targets left — intercept enemy drones.
                actions[i] = 6  # intercept nearest
            else:
                # Nothing to do — evade.
                actions[i] = 12  # evade

    return actions


def collect_doctrine_data(n_episodes, n_envs, stage, device):
    """Run episodes with doctrine heuristic and collect (obs, action) pairs."""
    speed_multiplier = 4.0
    distance_per_tick = 20.0 * 0.05 * speed_multiplier
    diagonal = stage["world_size"] * math.sqrt(2.0)
    max_ticks = int(math.ceil(diagonal / distance_per_tick))

    env = droneswarm_env.VecSimRunner(
        n_envs=n_envs,
        drones_per_side=stage["drones"],
        targets_per_side=stage["targets"],
        world_size=stage["world_size"],
        max_ticks=max(10000, max_ticks),
        speed_multiplier=speed_multiplier,
        skip_orca=True,
    )

    all_ego = []
    all_ent = []
    all_n_ent = []
    all_actions = []

    episodes_done = 0
    obs = env.reset()
    t0 = time.monotonic()

    while episodes_done < n_episodes:
        ego = obs["ego_obs"]       # [N, 25] numpy
        ent = obs["entity_obs"]    # [N, S, 10] numpy
        n_ent = obs["n_entities"]  # [N] numpy

        # Compute doctrine-like actions.
        actions = doctrine_heuristic(ego, ent, n_ent)

        # Store (obs, action) pairs.
        all_ego.append(ego.copy())
        all_ent.append(ent.copy())
        all_n_ent.append(n_ent.copy())
        all_actions.append(actions.copy())

        # Step environment.
        result = env.step(actions)
        obs = result

        # Count completed episodes.
        dones = result["dones"]
        episodes_done += int(dones.sum())

        if episodes_done % 500 == 0 and episodes_done > 0:
            elapsed = time.monotonic() - t0
            n_samples = sum(a.shape[0] for a in all_actions)
            print(f"  {episodes_done}/{n_episodes} episodes, "
                  f"{n_samples:,} samples, {elapsed:.0f}s", flush=True)

    # Concatenate all data.
    ego_all = np.concatenate(all_ego, axis=0)
    ent_all = np.concatenate(all_ent, axis=0)
    n_ent_all = np.concatenate(all_n_ent, axis=0)
    actions_all = np.concatenate(all_actions, axis=0)

    print(f"Collected {len(actions_all):,} samples from {episodes_done} episodes "
          f"in {time.monotonic() - t0:.0f}s", flush=True)

    return ego_all, ent_all, n_ent_all, actions_all


def train_supervised(model, ego, ent, n_ent, actions, device,
                     n_epochs=20, batch_size=8192, lr=1e-3):
    """Train model with cross-entropy loss on doctrine demonstrations."""
    N = len(actions)
    ego_t = torch.from_numpy(ego).to(device)
    ent_t = torch.from_numpy(ent).to(device)
    n_ent_t = torch.from_numpy(n_ent).to(device)
    actions_t = torch.from_numpy(actions).long().to(device)

    # Trim entity padding.
    max_ent = max(int(n_ent.max()), 1)
    ent_t = ent_t[:, :max_ent, :]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining supervised on {N:,} samples, {n_epochs} epochs, "
          f"batch_size={batch_size}", flush=True)

    for epoch in range(n_epochs):
        t0 = time.monotonic()
        perm = torch.randperm(N, device=device)
        total_loss = 0.0
        total_correct = 0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = perm[start:end]

            logits, _ = model(ego_t[idx], ent_t[idx], n_ent_t[idx])
            logits = logits.float().clamp(-20.0, 20.0)

            loss = F.cross_entropy(logits, actions_t[idx])

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == actions_t[idx]).sum().item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = total_correct / N * 100
        dt = time.monotonic() - t0

        print(f"  Epoch {epoch+1:2d}/{n_epochs}: loss={avg_loss:.4f}, "
              f"accuracy={accuracy:.1f}%, time={dt:.1f}s", flush=True)

    return model


def main():
    parser = argparse.ArgumentParser(description="Kickstart RL from doctrine imitation")
    parser.add_argument("--n-episodes", type=int, default=5000)
    parser.add_argument("--n-envs", type=int, default=256)
    parser.add_argument("--stage", type=int, default=1, help="0=4v4, 1=8v8, 2=16v16")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", default="results/kickstart_model.json")
    args = parser.parse_args()

    # Device.
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Stage config.
    stages = [
        {"drones": 4, "targets": 2, "world_size": 2500.0},
        {"drones": 8, "targets": 4, "world_size": 5000.0},
        {"drones": 16, "targets": 6, "world_size": 7500.0},
        {"drones": 24, "targets": 6, "world_size": 10000.0},
    ]
    stage = stages[args.stage]
    print(f"Stage {args.stage}: {stage['drones']}v{stage['drones']}, "
          f"{stage['targets']}t, {stage['world_size']}m world")

    # Collect doctrine demonstrations.
    print(f"\nCollecting {args.n_episodes} episodes of doctrine play...")
    ego, ent, n_ent, actions = collect_doctrine_data(
        args.n_episodes, args.n_envs, stage, device
    )

    # Print action distribution.
    unique, counts = np.unique(actions, return_counts=True)
    print("\nAction distribution:")
    action_names = [
        "atk_near_d", "atk_far_d", "atk_least_d",
        "atk_near_e", "atk_far_e", "atk_least_e",
        "int_near", "int_2nd", "int_cluster",
        "def_tight", "def_wide", "patrol", "evade"
    ]
    for a, c in zip(unique, counts):
        pct = c / len(actions) * 100
        name = action_names[a] if a < len(action_names) else f"unk_{a}"
        print(f"  {a:2d} ({name:12s}): {c:8d} ({pct:5.1f}%)")

    # Create and train model.
    model = PolicyNetV2Torch().to(device)
    model = train_supervised(
        model, ego, ent, n_ent, actions, device,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Save.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.cpu()
    save_model_weights(model, args.output)
    print(f"\nSaved kickstart model to {args.output}")
    print(f"Resume RL training with: python rl-train/train.py --resume {args.output} --start-stage {args.stage}")


if __name__ == "__main__":
    main()
