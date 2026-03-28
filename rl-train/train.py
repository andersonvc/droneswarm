#!/usr/bin/env python3
"""RL training loop for droneswarm.

Uses:
  - droneswarm_env.VecSimRunner (Rust PyO3 binding) for fast parallel simulation
  - PyTorch for PolicyNetV2 forward/backward + PPO update (GPU via MPS/CUDA)
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import droneswarm_env

from droneswarm_train.constants import EGO_DIM, ACT_DIM, SPEED_MULTIPLIER
from droneswarm_train.curriculum import Curriculum, compute_max_ticks
from droneswarm_train.evaluate import evaluate
from droneswarm_train.model import PolicyNetV2Torch
from droneswarm_train.normalize import RunningMeanStd, ValueNormalizer
from droneswarm_train.ppo import compute_gae, ppo_update, ReplayBuffer, PrioritizedMemory
from droneswarm_train.rollout import collect_rollout, obs_to_tensors
from droneswarm_train.weights import (
    load_model_weights,
    load_normalizers,
    save_model_weights,
    save_normalizers,
)


def build_optimizer(
    model: PolicyNetV2Torch, lr: float
) -> torch.optim.AdamW:
    """Build AdamW with per-parameter-group weight decay matching Rust."""
    decay_params = []
    no_decay_params = []

    for name in ["ego_encoder", "entity_encoder", "fc1", "fc2"]:
        layer = getattr(model, name)
        decay_params.extend(layer.parameters())

    for name in ["attn", "attn2", "actor_head", "local_value_head"]:
        layer = getattr(model, name)
        no_decay_params.extend(layer.parameters())

    param_groups = [
        {"params": decay_params, "weight_decay": 5e-4},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8
    )


def _reconfigure_envs(
    env: droneswarm_env.VecSimRunner,
    stage: dict,
    n_envs: int,
) -> None:
    """Reconfigure all environments for a new curriculum stage."""
    env.reconfigure(
        env_indices=list(range(n_envs)),
        drones_per_side=stage["drones"],
        targets_per_side=stage["targets"],
        world_size=stage["world_size"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="RL training loop for droneswarm."
    )
    parser.add_argument("--n-envs", type=int, default=0, help="Parallel envs (0=auto-scale per stage)")
    parser.add_argument("--n-steps", type=int, default=0, help="Steps per rollout (0=auto-scale per stage)")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=12288)
    parser.add_argument("--replay-size", type=int, default=0, help="Number of past rollouts for off-policy replay (0=disabled)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent-coef-start", type=float, default=0.03)
    parser.add_argument("--ent-coef-end", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--total-timesteps", type=int, default=50_000_000)
    parser.add_argument("--drones", type=int, default=24)
    parser.add_argument("--targets", type=int, default=6)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to model JSON to resume from",
    )
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--start-stage", type=int, default=0, help="Starting curriculum stage (0=4v4, 1=8v8, 2=16v16, 3=24v24)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="droneswarm", help="W&B project name")
    parser.add_argument("--wandb-name", default=None, help="W&B run name")
    args = parser.parse_args()

    # Select device.
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if args.device != "cpu":
            print(f"[warn] {args.device} not available, falling back to cpu")
        device = torch.device("cpu")
        # Use all CPU cores for PyTorch operations.
        n_cores = os.cpu_count() or 8
        torch.set_num_threads(n_cores)
        torch.set_num_interop_threads(n_cores)
    print(f"Device: {device} (threads: {torch.get_num_threads()})")

    # Initialize wandb.
    if args.wandb and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "n_envs": args.n_envs,
                "n_steps": args.n_steps,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "gamma": args.gamma,
                "lam": args.lam,
                "clip_range": args.clip,
                "ent_coef_start": args.ent_coef_start,
                "ent_coef_end": args.ent_coef_end,
                "max_grad_norm": args.max_grad_norm,
                "device": str(device),
                "start_stage": args.start_stage,
                "resume": args.resume,
            },
        )
    elif args.wandb and not HAS_WANDB:
        print("[warn] --wandb requested but wandb not installed. pip install wandb")

    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "best_model.json")

    # Curriculum.
    curriculum = Curriculum(
        final_drones=args.drones, final_targets=args.targets
    )
    if args.start_stage > 0:
        curriculum.current_stage = min(args.start_stage, len(curriculum.stages) - 1)
        print(f"Starting at curriculum stage {curriculum.current_stage}: {curriculum.stage_label()}")
    stage = curriculum.current()
    max_ticks = compute_max_ticks(stage["world_size"])

    # Auto-scale n_envs and n_steps per curriculum stage.
    n_envs = args.n_envs if args.n_envs > 0 else curriculum.compute_n_envs()
    n_steps = args.n_steps if args.n_steps > 0 else curriculum.compute_n_steps()
    print(f"  Auto-scaled: n_envs={n_envs}, n_steps={n_steps} "
          f"(stage {curriculum.current_stage}: {stage['drones']}v{stage['drones']}, "
          f"{stage['world_size']}m world)")

    # Create environments.
    env = droneswarm_env.VecSimRunner(
        n_envs=n_envs,
        drones_per_side=stage["drones"],
        targets_per_side=stage["targets"],
        world_size=stage["world_size"],
        max_ticks=max_ticks,
        speed_multiplier=SPEED_MULTIPLIER,
        skip_orca=True,
    )

    # Create model.
    model = PolicyNetV2Torch().to(device)
    if args.resume:
        load_model_weights(model, args.resume)
        print(f"Resumed from {args.resume}")
    model.train()

    optimizer = build_optimizer(model, lr=args.lr)
    replay_buffer = ReplayBuffer(max_size=args.replay_size)
    priority_memory = PrioritizedMemory(max_size=50_000)

    # Restore optimizer state if available alongside resume checkpoint.
    if args.resume:
        adam_path = args.resume.replace(".json", "_adam.pt")
        if os.path.exists(adam_path):
            state = torch.load(
                adam_path, map_location=device, weights_only=True
            )
            optimizer.load_state_dict(state)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
            print(f"Restored Adam state from {adam_path}")

    ego_norm = RunningMeanStd(EGO_DIM, device)
    value_norm = ValueNormalizer()

    # Restore normalizers if resuming.
    if args.resume:
        if load_normalizers(args.resume, ego_norm, value_norm):
            print("Restored normalizer state")

    # Reset envs.
    obs = env.reset()

    update_count = 0
    total_steps = 0
    best_win_rate = 0.0
    start_time = time.monotonic()

    # Auto-rollback state for collapse recovery.
    last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    last_good_optim = optimizer.state_dict()
    last_good_update = 0

    print(f"Starting PPO training")
    print(f"  Ego dim: {EGO_DIM}, Entity dim: {droneswarm_env.ENTITY_DIM}, Embed dim: 64")
    print(f"  Act dim: {ACT_DIM}, Hidden: 256, Attn heads: 4")
    print(f"  Total timesteps: {args.total_timesteps}")
    print(
        f"  Envs: {n_envs}, Steps/rollout: {n_steps}, "
        f"Epochs: {args.n_epochs}, Batch: {args.batch_size}"
    )
    print(
        f"  LR: {args.lr} (annealed), Gamma: {args.gamma}, "
        f"Clip: {args.clip}"
    )
    print(
        f"  Entropy coef: {args.ent_coef_start} -> {args.ent_coef_end}"
    )
    print(
        f"  Curriculum: 4v4 -> {args.drones}v{args.drones} "
        f"(advance at >{curriculum.advance_threshold*100:.0f}% win rate, "
        f"{curriculum.required_consecutive} consecutive evals)"
    )
    print()

    hot_config = {}

    while args.total_timesteps == 0 or total_steps < args.total_timesteps:
        # Hot-reload config from train_config.json every update.
        config_path = os.path.join(os.path.dirname(__file__), "..", "train_config.json")
        try:
            with open(config_path) as f:
                hot_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            hot_config = {}

        # Apply hot config overrides.
        effective_lr = hot_config.get("lr", args.lr)
        effective_clip = hot_config.get("clip_range", args.clip)
        effective_grad_norm = hot_config.get("max_grad_norm", args.max_grad_norm)
        effective_epochs = hot_config.get("n_epochs", args.n_epochs)
        effective_batch = hot_config.get("batch_size", args.batch_size)

        # Anneal LR and entropy coefficient.
        if args.total_timesteps == 0:
            progress = min(update_count / 500.0, 1.0)
        else:
            progress = min(total_steps / args.total_timesteps, 1.0)

        current_lr = effective_lr * max(1.0 - progress, 0.1)

        if "ent_coef" in hot_config:
            current_ent = hot_config["ent_coef"]
        else:
            current_ent = args.ent_coef_end + 0.5 * (
                args.ent_coef_start - args.ent_coef_end
            ) * (1 + math.cos(math.pi * progress))

        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # Collect rollout.
        t0 = time.monotonic()
        rollout, obs = collect_rollout(
            env, model, ego_norm, obs, n_steps, device
        )
        t_rollout = time.monotonic() - t0

        n_transitions = rollout["rewards"].shape[0]

        # Compute last values for GAE bootstrap.
        with torch.no_grad():
            ego, ent, n_ent = obs_to_tensors(obs, device)
            normed_ego = ego_norm.normalize(ego)
            _, last_vals = model(normed_ego, ent, n_ent)

        # Build last_values map keyed by (env_idx, drone_id).
        last_values = {}
        env_ids = np.asarray(obs["env_indices"])
        drone_ids_arr = np.asarray(obs["drone_ids"])
        for i in range(len(env_ids)):
            last_values[
                (int(env_ids[i]), int(drone_ids_arr[i]))
            ] = last_vals[i].item()

        # Compute GAE.
        advantages, returns_ = compute_gae(
            rollout, last_values, args.gamma, args.lam, device
        )

        # Update value normalizer.
        value_norm.update(returns_)

        # Normalize advantages.
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp(min=1e-8)
        advantages = (advantages - adv_mean) / adv_std

        # Build rollout dict for PPO update (re-normalize ego with current stats).
        t0 = time.monotonic()
        ppo_rollout = {
            "ego_obs": ego_norm.normalize(rollout["ego_obs"]),
            "entity_obs": rollout["entity_obs"],
            "n_entities": rollout["n_entities"],
            "actions": rollout["actions"],
            "log_probs": rollout["log_probs"],
            "advantages": advantages,
            "returns": returns_,
        }
        stats = ppo_update(
            model=model,
            optimizer=optimizer,
            rollout=ppo_rollout,
            device=device,
            n_epochs=effective_epochs,
            batch_size=effective_batch,
            clip_range=effective_clip,
            ent_coef=current_ent,
            max_grad_norm=effective_grad_norm,
            replay_buffer=replay_buffer,
            priority_memory=priority_memory,
        )
        t_ppo = time.monotonic() - t0

        update_count += 1
        total_steps = update_count * n_steps * n_envs

        # MPS stability: clamp weights and fix NaN every update.
        with torch.no_grad():
            nan_found = False
            for param in model.parameters():
                if param.isnan().any() or param.isinf().any():
                    nan_found = True
                    param.copy_(torch.where(
                        torch.isfinite(param), param, torch.zeros_like(param)
                    ))
                param.clamp_(-5.0, 5.0)
            if nan_found:
                print("[WARN] Sanitized NaN/Inf in model weights", flush=True)

        # Collapse detection and auto-rollback.
        last_ent_val = stats["entropy"][-1] if stats["entropy"] else 2.5
        last_clip_val = stats["clip_fraction"][-1] if stats["clip_fraction"] else 0.0

        if last_ent_val > 1.5 and last_clip_val < 0.2:
            # Policy is healthy — save as "last good" checkpoint.
            last_good_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            last_good_optim = optimizer.state_dict()
            last_good_update = update_count

        if last_ent_val < 0.8 or last_clip_val > 0.8:
            # Policy collapsed — rollback to last good checkpoint.
            if last_good_state is not None:
                print(
                    f"[ROLLBACK] Collapse detected (entropy={last_ent_val:.3f}, "
                    f"clip={last_clip_val:.3f}). Restoring update {last_good_update}.",
                    flush=True,
                )
                model.load_state_dict({k: v.to(device) for k, v in last_good_state.items()})
                optimizer.load_state_dict(last_good_optim)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

        # Log progress.
        if update_count % 5 == 0:
            mean_r = rollout["rewards"].mean().item()
            elapsed = time.monotonic() - start_time
            fps = total_steps / max(elapsed, 1.0)
            last_ploss = stats["policy_loss"][-1] if stats["policy_loss"] else 0
            last_ent = stats["entropy"][-1] if stats["entropy"] else 0
            last_clip = stats["clip_fraction"][-1] if stats["clip_fraction"] else 0
            print(
                f"Update {update_count}: steps={total_steps}, "
                f"mean_r={mean_r:.3f}, trans={n_transitions}, "
                f"rollout={t_rollout:.1f}s, ppo={t_ppo:.1f}s, "
                f"fps={fps:.0f}, lr={current_lr:.2e}, "
                f"ent_coef={current_ent:.4f}, "
                f"ploss={last_ploss:.4f}, entropy={last_ent:.3f}, "
                f"clip={last_clip:.3f}, "
                f"{elapsed:.0f}s, {curriculum.stage_label()}"
            )

            if args.wandb and HAS_WANDB:
                wandb.log({
                    "train/mean_reward": mean_r,
                    "train/policy_loss": last_ploss,
                    "train/entropy": last_ent,
                    "train/clip_fraction": last_clip,
                    "train/lr": current_lr,
                    "train/ent_coef": current_ent,
                    "train/fps": fps,
                    "train/transitions": n_transitions,
                    "train/rollout_time": t_rollout,
                    "train/ppo_time": t_ppo,
                    "train/total_steps": total_steps,
                    "curriculum/stage": curriculum.current_stage,
                }, step=update_count)

        # Evaluate.
        if update_count % args.eval_freq == 0:
            t0 = time.monotonic()
            win_rate, mean_reward = evaluate(
                model,
                ego_norm,
                device,
                curriculum.current(),
                n_episodes=args.eval_episodes,
            )
            t_eval = time.monotonic() - t0
            print(
                f"  Eval: win_rate={win_rate*100:.1f}%, "
                f"mean_r={mean_reward:.2f}, "
                f"stage={curriculum.stage_label()}, "
                f"time={t_eval:.1f}s"
            )

            if args.wandb and HAS_WANDB:
                wandb.log({
                    "eval/win_rate": win_rate,
                    "eval/mean_reward": mean_reward,
                    "eval/time": t_eval,
                    "curriculum/stage": curriculum.current_stage,
                }, step=update_count)

            # Win-rate-based curriculum.
            stage_change = curriculum.report_eval(win_rate)
            if stage_change == "advanced":
                # Recompute n_envs and n_steps for new stage.
                n_envs = args.n_envs if args.n_envs > 0 else curriculum.compute_n_envs()
                n_steps = args.n_steps if args.n_steps > 0 else curriculum.compute_n_steps()
                new_stage = curriculum.current()
                new_max_ticks = compute_max_ticks(new_stage["world_size"])
                print(
                    f"  >>> Curriculum advanced to "
                    f"{curriculum.stage_label()} "
                    f"(win_rate={win_rate*100:.1f}% exceeded "
                    f"{curriculum.advance_threshold*100:.0f}% threshold)"
                )
                print(
                    f"  Auto-scaled: n_envs={n_envs}, n_steps={n_steps}"
                )
                # Recreate environment with new scale.
                env = droneswarm_env.VecSimRunner(
                    n_envs=n_envs,
                    drones_per_side=new_stage["drones"],
                    targets_per_side=new_stage["targets"],
                    world_size=new_stage["world_size"],
                    max_ticks=new_max_ticks,
                    speed_multiplier=SPEED_MULTIPLIER,
                    skip_orca=True,
                )
                obs = env.reset()
                best_win_rate = 0.0

            # Save checkpoint.
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_{update_count}.json"
            )
            save_model_weights(model, ckpt_path)
            save_normalizers(ckpt_path, ego_norm, value_norm)

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                save_model_weights(model, best_model_path)
                save_normalizers(best_model_path, ego_norm, value_norm)
                print(
                    f"  New best model (win={win_rate*100:.1f}%)"
                )

        # Periodic checkpoint (between evals).
        elif update_count % 50 == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_{update_count}.json"
            )
            save_model_weights(model, ckpt_path)
            save_normalizers(ckpt_path, ego_norm, value_norm)
            # Save optimizer state for resumability.
            adam_path = ckpt_path.replace(".json", "_adam.pt")
            torch.save(optimizer.state_dict(), adam_path)

    # Save final model.
    final_path = os.path.join(args.output_dir, "final_model.json")
    save_model_weights(model, final_path)
    save_normalizers(final_path, ego_norm, value_norm)
    adam_path = final_path.replace(".json", "_adam.pt")
    torch.save(optimizer.state_dict(), adam_path)
    print(
        f"\nTraining complete. Steps: {total_steps}, "
        f"Best win: {best_win_rate*100:.1f}%"
    )

    if args.wandb and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
