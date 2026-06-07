"""
UR5e RL Controller — Webots entry-point for RL training / evaluation.
Runs inside the Webots controller process.
"""

import sys
import os
import json
import datetime

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

try:
    from controller import Supervisor
except ImportError:
    sys.exit("ERROR: Must be run from Webots (controller module not found).")

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def _ensure_dirs():
    for sub in ("models", "logs", "data"):
        d = os.path.join(PROJECT_ROOT, "rl_training", sub)
        os.makedirs(d, exist_ok=True)


def _detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            _log(f"GPU detected: {name}")
            return "cuda"
    except Exception:
        pass
    _log("Using CPU for training")
    return "cpu"


def _load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_hyperparams(hp, run_name):
    out = os.path.join(PROJECT_ROOT, "rl_training", "data",
                       f"hyperparams_{run_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(hp, f, indent=2)
    _log(f"Hyperparameters saved → {out}")

# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train(supervisor, cfg):
    from rl_training.ur5e_grasp_env import UR5eGraspEnv

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import (
            CheckpointCallback, BaseCallback,
        )
    except ImportError:
        _log("ERROR: stable-baselines3 not installed. "
             "Run: pip install stable-baselines3 gymnasium")
        return

    device    = _detect_device()
    algo      = cfg.get("algorithm", "SAC")
    lr        = cfg.get("learning_rate", 3e-4)
    bs        = cfg.get("batch_size", 256)
    buf       = cfg.get("buffer_size", 100_000)
    total     = cfg.get("total_timesteps", 50_000)
    ckpt_freq = cfg.get("checkpoint_freq", 10_000)
    gamma     = cfg.get("gamma", 0.99)
    tau       = cfg.get("tau", 0.005)
    ls        = cfg.get("learning_starts", 1_000)

    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_dir = os.path.join(PROJECT_ROOT, "rl_training", "models")
    log_dir   = os.path.join(PROJECT_ROOT, "rl_training", "logs")

    env_cfg = {
        "max_steps":      cfg.get("max_steps", 400),
        "sub_steps":      cfg.get("sub_steps", 4),
        "success_height": cfg.get("success_height", 0.20),
        "randomize_ball": cfg.get("randomize_ball", True),
        "rand_range":     cfg.get("rand_range", 0.05),
    }

    _log("Creating Gymnasium environment …")
    env = UR5eGraspEnv(supervisor, cfg=env_cfg)

    _log(f"Creating {algo} model  (device={device})")
    model = SAC(
        "MlpPolicy", env,
        learning_rate=lr,
        batch_size=bs,
        buffer_size=buf,
        learning_starts=ls,
        gamma=gamma,
        tau=tau,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
    )

    hp = {
        "algorithm": algo, "device": device,
        "learning_rate": lr, "batch_size": bs,
        "buffer_size": buf, "total_timesteps": total,
        "gamma": gamma, "tau": tau,
        "learning_starts": ls, "checkpoint_freq": ckpt_freq,
        "env": env_cfg, "run_name": run_name,
    }
    _save_hyperparams(hp, run_name)

    # --- Logging callback ---------------------------------------------------
    class _LogCB(BaseCallback):
        def __init__(self):
            super().__init__()
            self._ep_count = 0
            self._ep_reward = 0.0
            self._ep_steps = 0

        def _on_step(self):
            self._ep_reward += self.locals.get("rewards", [0.0])[0]
            self._ep_steps += 1
            dones = self.locals.get("dones", [False])
            if dones[0]:
                self._ep_count += 1
                info = self.locals.get("infos", [{}])[0]
                success = info.get("success", False)
                ball_h  = info.get("ball_height", 0.0)
                _log(f"  Episode {self._ep_count:4d} | "
                     f"R={self._ep_reward:+8.1f} | "
                     f"steps={self._ep_steps:3d} | "
                     f"ball_h={ball_h:.3f} | "
                     f"{'SUCCESS' if success else 'fail'}")
                self._ep_reward = 0.0
                self._ep_steps = 0
            return True

    ckpt_cb = CheckpointCallback(
        save_freq=ckpt_freq, save_path=model_dir,
        name_prefix=f"sac_ur5e_{run_name}",
    )

    _log(f"Starting training — {total} timesteps …")
    try:
        model.learn(
            total_timesteps=total,
            callback=[ckpt_cb, _LogCB()],
            tb_log_name=f"SAC_{run_name}",
        )
    except KeyboardInterrupt:
        _log("Training interrupted by user.")

    final_path = os.path.join(model_dir, f"model_SAC_{run_name}")
    model.save(final_path)
    _log(f"Final model saved → {final_path}.zip")
    return model

# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def evaluate(supervisor, cfg, model_path=None):
    from rl_training.ur5e_grasp_env import UR5eGraspEnv

    try:
        from stable_baselines3 import SAC
    except ImportError:
        _log("ERROR: stable-baselines3 not installed.")
        return

    if model_path is None:
        mdir = os.path.join(PROJECT_ROOT, "rl_training", "models")
        zips = sorted(
            [f for f in os.listdir(mdir) if f.startswith("model_SAC") and f.endswith(".zip")],
        )
        if not zips:
            _log("No trained model found in rl_training/models/")
            return
        model_path = os.path.join(mdir, zips[-1])
    _log(f"Loading model: {model_path}")

    env_cfg = {
        "max_steps":      cfg.get("max_steps", 400),
        "sub_steps":      cfg.get("sub_steps", 4),
        "success_height": cfg.get("success_height", 0.20),
        "randomize_ball": True,
        "rand_range":     0.05,
    }
    env = UR5eGraspEnv(supervisor, cfg=env_cfg)
    model = SAC.load(model_path, env=env)

    n_eval   = cfg.get("eval_episodes", 50)
    results  = []
    _log(f"Evaluating for {n_eval} episodes …")

    for ep in range(n_eval):
        obs, _ = env.reset()
        total_r = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            done = terminated or truncated
        results.append({
            "episode":    ep + 1,
            "reward":     total_r,
            "steps":      info["steps"],
            "success":    info["success"],
            "ball_height": info["ball_height"],
        })
        tag = "SUCCESS" if info["success"] else "fail"
        _log(f"  Eval {ep+1:3d}/{n_eval} | R={total_r:+8.1f} | {tag}")

    import csv
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(PROJECT_ROOT, "rl_training", "data",
                            f"eval_results_{ts}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode","reward","steps",
                                          "success","ball_height"])
        w.writeheader()
        w.writerows(results)

    successes = sum(1 for r in results if r["success"])
    avg_r = sum(r["reward"] for r in results) / max(len(results), 1)
    _log(f"\n  Success rate : {successes}/{n_eval} = "
         f"{100*successes/n_eval:.1f}%")
    _log(f"  Avg reward   : {avg_r:.1f}")
    _log(f"  Results saved → {csv_path}")

# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def main():
    _ensure_dirs()
    supervisor = Supervisor()
    cfg = _load_config()

    mode = cfg.get("mode", os.environ.get("RL_MODE", "train"))
    _log(f"=== UR5e RL Controller ===  mode={mode}")

    if mode == "train":
        model = train(supervisor, cfg)
        if model and cfg.get("eval_after_train", True):
            _log("\n--- Post-training evaluation ---")
            evaluate(supervisor, cfg)
    elif mode == "eval":
        model_path = cfg.get("model_path")
        evaluate(supervisor, cfg, model_path=model_path)
    else:
        _log(f"Unknown mode: {mode}")

    _log("Controller finished. Idling …")
    ts = int(supervisor.getBasicTimeStep())
    while supervisor.step(ts) != -1:
        pass


if __name__ == "__main__":
    main()
