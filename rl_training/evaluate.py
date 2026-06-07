"""
Evaluation script — load a trained model and compute success metrics.

Can be run standalone (extern-controller mode) or imported by the
embedded Webots controller.

Usage:
    python rl_training/evaluate.py --model rl_training/models/model_SAC_xxx.zip
"""

import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def evaluate_model(env, model, n_episodes=100, deterministic=True):
    """Run *n_episodes* and return list[dict] with per-episode stats."""
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        results.append({
            "episode":     ep + 1,
            "reward":      round(total_reward, 2),
            "steps":       info.get("steps", 0),
            "success":     info.get("success", False),
            "ball_height": round(info.get("ball_height", 0.0), 4),
        })
        tag = "OK" if info.get("success") else "--"
        print(f"  [{tag}] Episode {ep+1:3d}/{n_episodes}  "
              f"R={total_reward:+8.1f}  steps={info.get('steps',0):3d}  "
              f"ball_h={info.get('ball_height',0):.3f}")
    return results


def save_results(results, data_dir):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(data_dir, f"eval_results_{ts}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    successes = sum(1 for r in results if r["success"])
    n = len(results)
    rewards = [r["reward"] for r in results]
    steps   = [r["steps"]  for r in results]

    summary = {
        "timestamp":      ts,
        "n_episodes":     n,
        "success_rate":   round(successes / n, 4),
        "mean_reward":    round(float(np.mean(rewards)), 2),
        "std_reward":     round(float(np.std(rewards)), 2),
        "mean_steps":     round(float(np.mean(steps)), 1),
    }
    summary_path = os.path.join(data_dir, f"eval_summary_{ts}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Success rate : {successes}/{n} = {100*successes/n:.1f}%")
    print(f"  Mean reward  : {summary['mean_reward']}  (std={summary['std_reward']})")
    print(f"  Mean steps   : {summary['mean_steps']}")
    print(f"  CSV          : {csv_path}")
    print(f"  Summary      : {summary_path}")
    print(f"{'='*50}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .zip model")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=400)
    args = parser.parse_args()

    webots_home = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")
    webots_python = os.path.join(webots_home, "lib", "controller", "python")
    if os.path.isdir(webots_python):
        sys.path.insert(0, webots_python)

    from controller import Supervisor
    from rl_training.ur5e_grasp_env import UR5eGraspEnv
    from stable_baselines3 import SAC

    supervisor = Supervisor()
    env = UR5eGraspEnv(supervisor, cfg={
        "max_steps": args.max_steps,
        "randomize_ball": True,
    })
    model = SAC.load(args.model, env=env)

    results = evaluate_model(env, model, n_episodes=args.episodes)
    data_dir = os.path.join(PROJECT_ROOT, "rl_training", "data")
    os.makedirs(data_dir, exist_ok=True)
    save_results(results, data_dir)

    ts = int(supervisor.getBasicTimeStep())
    while supervisor.step(ts) != -1:
        pass


if __name__ == "__main__":
    main()
