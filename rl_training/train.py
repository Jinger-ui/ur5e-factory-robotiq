"""
RL Training Script — standalone entry-point for extern-controller mode.

Usage (from project root, with Webots running and controller set to <extern>):
    set WEBOTS_HOME=C:\Program Files\Webots
    python rl_training/train.py [--timesteps 100000] [--device cuda]

This script can also be imported by the embedded Webots controller.
"""

import argparse
import datetime
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="UR5e RL Grasp Training")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--checkpoint-freq", type=int, default=10_000)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--success-height", type=float, default=0.20)
    return p.parse_args()


def main():
    args = parse_args()

    webots_home = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")
    webots_python = os.path.join(webots_home, "lib", "controller", "python")
    if os.path.isdir(webots_python):
        sys.path.insert(0, webots_python)

    try:
        from controller import Supervisor
    except ImportError:
        sys.exit(
            "Cannot import Webots controller module.\n"
            "Set WEBOTS_HOME or run inside Webots."
        )

    from rl_training.ur5e_grasp_env import UR5eGraspEnv
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback

    supervisor = Supervisor()
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_dir = os.path.join(PROJECT_ROOT, "rl_training", "models")
    log_dir   = os.path.join(PROJECT_ROOT, "rl_training", "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRAIN] device={device}  timesteps={args.timesteps}")

    env_cfg = {
        "max_steps":      args.max_steps,
        "sub_steps":      4,
        "success_height": args.success_height,
        "randomize_ball": True,
        "rand_range":     0.05,
    }
    env = UR5eGraspEnv(supervisor, cfg=env_cfg)

    model = SAC(
        "MlpPolicy", env,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=1000,
        gamma=0.99, tau=0.005,
        verbose=1, device=device,
        tensorboard_log=log_dir,
    )

    hp = {
        "algorithm": "SAC", "device": device,
        "learning_rate": args.lr, "batch_size": args.batch_size,
        "buffer_size": args.buffer_size, "total_timesteps": args.timesteps,
        "gamma": 0.99, "tau": 0.005, "run_name": run_name,
        "env": env_cfg,
    }
    hp_path = os.path.join(PROJECT_ROOT, "rl_training", "data",
                           f"hyperparams_{run_name}.json")
    os.makedirs(os.path.dirname(hp_path), exist_ok=True)
    with open(hp_path, "w") as f:
        json.dump(hp, f, indent=2)

    ckpt_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq, save_path=model_dir,
        name_prefix=f"sac_ur5e_{run_name}",
    )
    model.learn(total_timesteps=args.timesteps, callback=ckpt_cb,
                tb_log_name=f"SAC_{run_name}")

    final = os.path.join(model_dir, f"model_SAC_{run_name}")
    model.save(final)
    print(f"[TRAIN] Done. Model → {final}.zip")

    ts = int(supervisor.getBasicTimeStep())
    while supervisor.step(ts) != -1:
        pass


if __name__ == "__main__":
    main()
