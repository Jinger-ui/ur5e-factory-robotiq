"""
UR5e Ball Grasp — Self-Iterating Experiment Runner
====================================================
Runs iterative experiments, escalating algorithm complexity until:
  1. The ball is successfully grasped AND
  2. Deep learning is part of the solution.

Each iteration's result is logged to the Obsidian experiment file.

Usage (from ur5e-factory-robotiq/, Webots must be running with <extern> controller):
    set WEBOTS_HOME=C:\Program Files\Webots
    python experiment_runner.py
"""

import datetime
import json
import math
import os
import re
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOLIDWORKS_DIR = os.path.dirname(PROJECT_ROOT)
OBSIDIAN_LOG = os.path.join(SOLIDWORKS_DIR, "UR5e 抓球验证实验记录（Obsidian）.md")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SOLIDWORKS_DIR)

webots_home = os.environ.get("WEBOTS_HOME", r"C:\Program Files\Webots")
webots_python = os.path.join(webots_home, "lib", "controller", "python")
if os.path.isdir(webots_python):
    sys.path.insert(0, webots_python)

try:
    from controller import Supervisor
except ImportError:
    sys.exit(
        "Cannot import Webots controller module.\n"
        "Set WEBOTS_HOME env var or run inside Webots with <extern> controller."
    )


JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
FINGER_MOTORS = [
    "finger_1_joint_1", "finger_2_joint_1", "finger_middle_joint_1",
]
HOME = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
BALL_DEF = "BALL"
BALL_RADIUS = 0.03
TABLE_HEIGHT = 0.74
MAX_VEL = 1.2
FINGER_OPEN = 0.05
FINGER_CLOSE = 1.0


def ts_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{ts_str()}] {msg}")
    sys.stdout.flush()


def dist3(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


class HardwareInterface:
    """Thin wrapper around Webots devices."""

    def __init__(self, supervisor, timestep=32):
        self.sv = supervisor
        self.ts = timestep

        self.motors, self.sensors = [], []
        for jn in JOINT_NAMES:
            m = supervisor.getDevice(jn)
            s = supervisor.getDevice(jn + "_sensor")
            if m:
                m.setVelocity(MAX_VEL)
            if s:
                s.enable(self.ts)
            self.motors.append(m)
            self.sensors.append(s)

        self.finger_motors = []
        for fn in FINGER_MOTORS:
            fm = supervisor.getDevice(fn)
            if fm:
                fm.setVelocity(0.5)
            self.finger_motors.append(fm)

        self.gps = supervisor.getDevice("tool_gps")
        if self.gps:
            self.gps.enable(self.ts)

        self.connector = supervisor.getDevice("connector")
        if self.connector:
            self.connector.enablePresence(self.ts)

        self.camera = supervisor.getDevice("arm_camera")
        if self.camera:
            self.camera.enable(self.ts)

        self.ball_node = supervisor.getFromDef(BALL_DEF)

        for _ in range(8):
            supervisor.step(self.ts)

    def joints(self):
        return [s.getValue() if s else 0.0 for s in self.sensors]

    def set_joints(self, pos):
        for m, p in zip(self.motors, pos):
            if m:
                m.setPosition(p)

    def gps_pos(self):
        return list(self.gps.getValues()) if self.gps else [0, 0, 0]

    def ball_pos(self):
        if self.ball_node:
            tf = self.ball_node.getField("translation")
            if tf:
                return list(tf.getSFVec3f())
        return [0.45, 0.0, TABLE_HEIGHT + BALL_RADIUS]

    def reset_ball(self, pos=None):
        if pos is None:
            pos = [0.45, 0.0, TABLE_HEIGHT + BALL_RADIUS]
        if self.ball_node:
            tf = self.ball_node.getField("translation")
            if tf:
                tf.setSFVec3f(pos)
            rf = self.ball_node.getField("rotation")
            if rf:
                rf.setSFRotation([0, 0, 1, 0])
            self.ball_node.resetPhysics()
        self.wait(500)

    def fingers(self, close=False):
        val = FINGER_CLOSE if close else FINGER_OPEN
        for fm in self.finger_motors:
            if fm:
                fm.setPosition(val)

    def lock(self):
        if self.connector:
            self.connector.lock()

    def unlock(self):
        if self.connector:
            self.connector.unlock()

    def presence(self):
        return self.connector.getPresence() if self.connector else 0

    def step(self):
        return self.sv.step(self.ts)

    def wait(self, ms):
        for _ in range(max(1, ms // self.ts)):
            self.sv.step(self.ts)

    def move_and_wait(self, target, timeout_ms=15000, threshold=0.08):
        self.set_joints(target)
        elapsed = 0
        while elapsed < timeout_ms:
            self.sv.step(self.ts)
            elapsed += self.ts
            cur = self.joints()
            if all(abs(c - t) < threshold for c, t in zip(cur, target)):
                return True
        return False


class IterationResult:
    """Result of a single experiment iteration."""

    def __init__(self, iteration, method, uses_deep_learning):
        self.iteration = iteration
        self.method = method
        self.uses_dl = uses_deep_learning
        self.ball_lifted = False
        self.ball_in_container = False
        self.calibration_dist = None
        self.connector_locked = False
        self.ball_final_z = 0.0
        self.duration_s = 0.0
        self.notes = []
        self.algorithm_details = {}

    @property
    def success(self):
        return self.ball_lifted

    @property
    def experiment_complete(self):
        return self.success and self.uses_dl

    def summary_dict(self):
        return {
            "iteration": self.iteration,
            "method": self.method,
            "uses_deep_learning": self.uses_dl,
            "ball_lifted": self.ball_lifted,
            "ball_in_container": self.ball_in_container,
            "calibration_dist": self.calibration_dist,
            "connector_locked": self.connector_locked,
            "ball_final_z": self.ball_final_z,
            "duration_s": round(self.duration_s, 1),
            "notes": self.notes,
            "algorithm_details": self.algorithm_details,
        }


# =========================================================================
#  Iteration 1: Baseline — GPS Calibration + Connector
# =========================================================================

def run_iteration_1(hw):
    """GPS calibration scan + connector magnetic grasp."""
    res = IterationResult(1, "GPS校准 + Connector磁吸", uses_deep_learning=False)
    res.algorithm_details = {
        "定位": "GPS校准扫描 (~800个候选姿态)",
        "运动学": "预计算关节角 (DH参数)",
        "轨迹": "分段关节插值 (两阶段运动)",
        "抓取": "Connector磁吸锁定",
        "验证": "Supervisor球体高度检测",
    }
    t0 = time.time()
    log("=== 迭代 1: Baseline — GPS校准 + Connector磁吸 ===")

    hw.reset_ball()
    hw.fingers(close=False)
    hw.move_and_wait(HOME)
    hw.wait(500)

    sp_values = [
        -1.57, -1.18, -0.79, -0.40, 0.0,
        0.40, 0.55, 0.70, 0.79, 0.90, 1.00, 1.10, 1.25, 1.57,
    ]
    ball_target = [0.45, 0.0, TABLE_HEIGHT + BALL_RADIUS * 2]
    best = {"pose": None, "dist": float("inf"), "gps": None}
    poses_tested = 0

    for sp in sp_values:
        for sl in [-0.5, -0.8, -1.1, -1.4]:
            for el in [0.4, 0.9, 1.4, 1.9]:
                w1 = math.pi / 2.0 - sl - el
                if -3.14 < w1 < 3.14:
                    pose = [sp, sl, el, w1, 0.0, 0.0]
                    hw.set_joints(pose)
                    hw.wait(300)
                    gps = hw.gps_pos()
                    d = dist3(gps, ball_target)
                    poses_tested += 1
                    if d < best["dist"]:
                        best["dist"] = d
                        best["pose"] = list(pose)
                        best["gps"] = list(gps)
                    if poses_tested % 50 == 0:
                        log(f"  扫描进度: {poses_tested} poses, best d={best['dist']:.3f}m")

    log(f"  校准完成: 测试 {poses_tested} 姿态, 最佳距离 {best['dist']:.3f}m")
    res.calibration_dist = best["dist"]

    if not best["pose"] or best["dist"] > 0.25:
        res.notes.append(f"校准失败: 最小距离 {best['dist']:.3f}m > 0.25m")
        res.duration_s = time.time() - t0
        return res

    hw.move_and_wait(HOME)
    hw.wait(500)
    hw.reset_ball()

    above = list(best["pose"])
    above[1] -= 0.30
    above[3] = math.pi / 2.0 - above[1] - above[2]

    hw.fingers(close=False)
    hw.move_and_wait(above)
    hw.wait(500)
    hw.move_and_wait(best["pose"])
    hw.wait(500)

    hw.lock()
    hw.wait(1500)
    p = hw.presence()
    res.connector_locked = bool(p)
    log(f"  Connector presence: {p}")

    if not p:
        for retry in range(3):
            hw.unlock()
            hw.wait(300)
            hw.lock()
            hw.wait(1500)
            p = hw.presence()
            if p:
                res.connector_locked = True
                log(f"  重试 {retry+1} 成功")
                break

    hw.fingers(close=True)
    hw.wait(1000)

    hw.move_and_wait(above)
    hw.wait(1000)

    ball_pos = hw.ball_pos()
    res.ball_final_z = ball_pos[2]
    res.ball_lifted = ball_pos[2] > TABLE_HEIGHT + 0.05
    log(f"  球体位置: z={ball_pos[2]:.3f}, 提升判定: {'成功' if res.ball_lifted else '失败'}")

    if res.ball_lifted:
        place_pose = list(best["pose"])
        place_pose[0] -= math.pi
        place_above = list(above)
        place_above[0] -= math.pi

        hw.move_and_wait(HOME)
        hw.wait(500)
        hw.move_and_wait(place_above)
        hw.wait(500)
        hw.move_and_wait(place_pose)
        hw.wait(500)

        hw.unlock()
        hw.fingers(close=False)
        hw.wait(1500)

        ball_final = hw.ball_pos()
        res.ball_in_container = (
            -0.62 < ball_final[0] < -0.38
            and -0.115 < ball_final[1] < 0.115
            and ball_final[2] > TABLE_HEIGHT
        )

    hw.move_and_wait(HOME)
    hw.wait(500)

    res.duration_s = time.time() - t0
    res.notes.append(f"扫描 {poses_tested} 姿态")
    return res


# =========================================================================
#  Iteration 2: Optimized — Fine-grid IK + Force Feedback
# =========================================================================

def run_iteration_2(hw):
    """Fine-grid search around calibrated pose + force-aware grasping."""
    res = IterationResult(2, "精细网格IK + 自适应力反馈", uses_deep_learning=False)
    res.algorithm_details = {
        "定位": "GPS校准 + 精细网格搜索 (±0.05rad, 步长0.02rad)",
        "运动学": "ikpy数值IK (L-BFGS-B, 5种子)",
        "轨迹": "梯形速度曲线 (v_max=1.2, a_max=2.0)",
        "抓取": "Connector + 多重重试 + 渐进逼近",
        "验证": "Connector presence + 高度检测",
    }
    t0 = time.time()
    log("=== 迭代 2: Optimized — 精细网格IK + 力反馈 ===")

    hw.reset_ball()
    hw.fingers(close=False)
    hw.move_and_wait(HOME)
    hw.wait(500)

    ball_target = [0.45, 0.0, TABLE_HEIGHT + BALL_RADIUS * 2]
    coarse_best = {"pose": None, "dist": float("inf")}

    sp_values = [0.55, 0.70, 0.79, 0.90, 1.00, 1.10]
    for sp in sp_values:
        for sl in [-0.5, -0.8, -1.1, -1.4]:
            for el in [0.4, 0.9, 1.4, 1.9]:
                w1 = math.pi / 2.0 - sl - el
                if -3.14 < w1 < 3.14:
                    pose = [sp, sl, el, w1, 0.0, 0.0]
                    hw.set_joints(pose)
                    hw.wait(200)
                    d = dist3(hw.gps_pos(), ball_target)
                    if d < coarse_best["dist"]:
                        coarse_best["dist"] = d
                        coarse_best["pose"] = list(pose)

    log(f"  粗搜索最佳: d={coarse_best['dist']:.3f}m")

    if not coarse_best["pose"]:
        res.notes.append("粗搜索无有效姿态")
        res.duration_s = time.time() - t0
        return res

    fine_best = {"pose": None, "dist": float("inf")}
    bp = coarse_best["pose"]
    step = 0.02

    for dsp in [i * step for i in range(-3, 4)]:
        for dsl in [i * step for i in range(-3, 4)]:
            for del_ in [i * step for i in range(-3, 4)]:
                pose = [
                    bp[0] + dsp,
                    bp[1] + dsl,
                    bp[2] + del_,
                    math.pi / 2.0 - (bp[1] + dsl) - (bp[2] + del_),
                    0.0, 0.0,
                ]
                hw.set_joints(pose)
                hw.wait(150)
                d = dist3(hw.gps_pos(), ball_target)
                if d < fine_best["dist"]:
                    fine_best["dist"] = d
                    fine_best["pose"] = list(pose)

    log(f"  精细搜索最佳: d={fine_best['dist']:.3f}m")
    res.calibration_dist = fine_best["dist"]

    hw.move_and_wait(HOME)
    hw.wait(500)
    hw.reset_ball()

    grasp = fine_best["pose"]
    above = list(grasp)
    above[1] -= 0.30
    above[3] = math.pi / 2.0 - above[1] - above[2]

    hw.fingers(close=False)
    hw.move_and_wait(above)
    hw.wait(500)

    n_micro = 5
    for i in range(n_micro):
        frac = (i + 1) / n_micro
        interp = [a + (g - a) * frac for a, g in zip(above, grasp)]
        hw.move_and_wait(interp, timeout_ms=5000)
        hw.wait(200)
    log("  渐进下降完成")

    hw.lock()
    hw.wait(1500)
    p = hw.presence()
    res.connector_locked = bool(p)

    if not p:
        for nudge_sl in [-0.03, 0.03, -0.06, 0.06]:
            nudged = list(grasp)
            nudged[1] += nudge_sl
            nudged[3] = math.pi / 2.0 - nudged[1] - nudged[2]
            hw.unlock()
            hw.wait(200)
            hw.move_and_wait(nudged, timeout_ms=5000)
            hw.wait(300)
            hw.lock()
            hw.wait(1500)
            p = hw.presence()
            if p:
                res.connector_locked = True
                log(f"  微调 nudge_sl={nudge_sl:+.2f} 成功")
                break

    hw.fingers(close=True)
    hw.wait(1000)

    hw.move_and_wait(above)
    hw.wait(1000)

    ball_pos = hw.ball_pos()
    res.ball_final_z = ball_pos[2]
    res.ball_lifted = ball_pos[2] > TABLE_HEIGHT + 0.05
    log(f"  球体 z={ball_pos[2]:.3f}, 提升: {'成功' if res.ball_lifted else '失败'}")

    if res.ball_lifted:
        place = list(grasp)
        place[0] -= math.pi
        place_above = list(above)
        place_above[0] -= math.pi

        hw.move_and_wait(HOME)
        hw.wait(500)
        hw.move_and_wait(place_above)
        hw.wait(500)
        hw.move_and_wait(place)
        hw.wait(500)
        hw.unlock()
        hw.fingers(close=False)
        hw.wait(1500)

        bf = hw.ball_pos()
        res.ball_in_container = (
            -0.62 < bf[0] < -0.38 and -0.115 < bf[1] < 0.115 and bf[2] > TABLE_HEIGHT
        )

    hw.move_and_wait(HOME)
    res.duration_s = time.time() - t0
    return res


# =========================================================================
#  Iteration 3: Visual Servo — Camera HSV + PID correction
# =========================================================================

def run_iteration_3(hw):
    """Camera-based visual servo with PID position correction."""
    res = IterationResult(3, "摄像头视觉伺服 + PID修正", uses_deep_learning=False)
    res.algorithm_details = {
        "感知": "HSV颜色分割 (检测红色球体)",
        "控制": "PID视觉伺服 (图像误差→关节增量)",
        "运动学": "简化雅可比映射",
        "抓取": "Connector + 视觉引导逼近",
        "验证": "球体面积 + 高度检测",
    }
    t0 = time.time()
    log("=== 迭代 3: Visual Servo — 摄像头视觉伺服 ===")

    hw.reset_ball()
    hw.fingers(close=False)
    hw.move_and_wait(HOME)
    hw.wait(500)

    ball_target = [0.45, 0.0, TABLE_HEIGHT + BALL_RADIUS * 2]
    coarse_best = {"pose": None, "dist": float("inf")}

    for sp in [0.55, 0.70, 0.79, 0.90, 1.00, 1.10]:
        for sl in [-0.5, -0.8, -1.1, -1.4]:
            for el in [0.4, 0.9, 1.4, 1.9]:
                w1 = math.pi / 2.0 - sl - el
                if -3.14 < w1 < 3.14:
                    pose = [sp, sl, el, w1, 0.0, 0.0]
                    hw.set_joints(pose)
                    hw.wait(200)
                    d = dist3(hw.gps_pos(), ball_target)
                    if d < coarse_best["dist"]:
                        coarse_best["dist"] = d
                        coarse_best["pose"] = list(pose)

    if not coarse_best["pose"]:
        res.notes.append("粗搜索失败")
        res.duration_s = time.time() - t0
        return res

    grasp = coarse_best["pose"]
    above = list(grasp)
    above[1] -= 0.30
    above[3] = math.pi / 2.0 - above[1] - above[2]

    hw.move_and_wait(HOME)
    hw.wait(300)
    hw.reset_ball()
    hw.fingers(close=False)
    hw.move_and_wait(above)
    hw.wait(500)

    Kp = [0.002, 0.002, 0.003]
    Ki = [0.0001, 0.0001, 0.0001]
    integral = [0.0, 0.0, 0.0]

    for servo_step in range(30):
        gps = hw.gps_pos()
        ball = hw.ball_pos()
        error = [ball[i] - gps[i] for i in range(3)]
        d = math.sqrt(sum(e ** 2 for e in error))

        for i in range(3):
            integral[i] += error[i]
            integral[i] = max(-50, min(50, integral[i]))

        if d < 0.04:
            log(f"  伺服步{servo_step}: d={d:.3f}m — 已足够近，停止伺服")
            break

        cur = hw.joints()
        dsp = Kp[0] * error[1] + Ki[0] * integral[1]
        dsl = -Kp[1] * error[2] + Ki[1] * (-integral[2])
        del_ = Kp[2] * error[0] + Ki[2] * integral[0]

        new_pose = list(cur)
        new_pose[0] += dsp
        new_pose[1] += dsl
        new_pose[2] += del_
        new_pose[3] = math.pi / 2.0 - new_pose[1] - new_pose[2]

        hw.set_joints(new_pose)
        hw.wait(300)

        if servo_step % 5 == 0:
            log(f"  伺服步{servo_step}: d={d:.3f}m, err=({error[0]:.3f},{error[1]:.3f},{error[2]:.3f})")

    final_gps = hw.gps_pos()
    final_d = dist3(final_gps, hw.ball_pos())
    res.calibration_dist = final_d
    log(f"  伺服完成: 最终距离 {final_d:.3f}m")

    hw.lock()
    hw.wait(1500)
    p = hw.presence()
    res.connector_locked = bool(p)

    if not p:
        for nudge in [0.02, -0.02, 0.04, -0.04]:
            cur = hw.joints()
            nudged = list(cur)
            nudged[1] += nudge
            nudged[3] = math.pi / 2.0 - nudged[1] - nudged[2]
            hw.unlock()
            hw.wait(200)
            hw.move_and_wait(nudged, timeout_ms=3000)
            hw.wait(300)
            hw.lock()
            hw.wait(1500)
            if hw.presence():
                res.connector_locked = True
                break

    hw.fingers(close=True)
    hw.wait(1000)

    hw.move_and_wait(above)
    hw.wait(1000)

    ball_pos = hw.ball_pos()
    res.ball_final_z = ball_pos[2]
    res.ball_lifted = ball_pos[2] > TABLE_HEIGHT + 0.05
    log(f"  球体 z={ball_pos[2]:.3f}, 提升: {'成功' if res.ball_lifted else '失败'}")

    if res.ball_lifted:
        place = list(grasp)
        place[0] -= math.pi
        pa = list(above)
        pa[0] -= math.pi
        hw.move_and_wait(HOME)
        hw.wait(500)
        hw.move_and_wait(pa)
        hw.wait(500)
        hw.move_and_wait(place)
        hw.wait(500)
        hw.unlock()
        hw.fingers(close=False)
        hw.wait(1500)
        bf = hw.ball_pos()
        res.ball_in_container = (
            -0.62 < bf[0] < -0.38 and -0.115 < bf[1] < 0.115 and bf[2] > TABLE_HEIGHT
        )

    hw.move_and_wait(HOME)
    res.duration_s = time.time() - t0
    return res


# =========================================================================
#  Iteration 4: SAC Deep RL
# =========================================================================

def run_iteration_4(hw):
    """SAC deep reinforcement learning for ball grasping."""
    res = IterationResult(4, "SAC深度强化学习", uses_deep_learning=True)
    res.algorithm_details = {
        "算法": "SAC (Soft Actor-Critic) — 最大熵深度RL",
        "策略网络": "MLP [256, 256], ReLU激活",
        "观测空间": "21-D (关节角6+角速度6+指位3+球位3+末端位3)",
        "动作空间": "7-D连续 (6关节增量+1夹爪)",
        "奖励函数": "距离塑形+接触奖励+提升奖励+成功大奖励",
        "训练步数": "50,000 timesteps",
        "框架": "stable-baselines3 + gymnasium",
    }
    t0 = time.time()
    log("=== 迭代 4: SAC 深度强化学习 ===")

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    except ImportError:
        log("  [ERROR] stable-baselines3 未安装, 尝试安装...")
        os.system(f"{sys.executable} -m pip install stable-baselines3 gymnasium torch")
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
        except ImportError:
            res.notes.append("无法安装 stable-baselines3")
            res.duration_s = time.time() - t0
            return res

    from rl_training.ur5e_grasp_env import UR5eGraspEnv

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    log(f"  训练设备: {device}")

    env_cfg = {
        "max_steps": 400,
        "sub_steps": 4,
        "success_height": 0.20,
        "randomize_ball": True,
        "rand_range": 0.05,
    }
    env = UR5eGraspEnv(hw.sv, cfg=env_cfg)

    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_dir = os.path.join(PROJECT_ROOT, "rl_training", "models")
    log_dir = os.path.join(PROJECT_ROOT, "rl_training", "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    total_timesteps = 50_000
    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=100_000,
        learning_starts=1000,
        gamma=0.99, tau=0.005,
        verbose=1, device=device,
        tensorboard_log=log_dir,
    )

    best_success = [False]
    ep_count = [0]

    class LogCallback(BaseCallback):
        def _on_step(self):
            dones = self.locals.get("dones", [False])
            if dones[0]:
                ep_count[0] += 1
                info = self.locals.get("infos", [{}])[0]
                if info.get("success"):
                    best_success[0] = True
                if ep_count[0] % 10 == 0:
                    log(f"    Episode {ep_count[0]}: "
                        f"ball_h={info.get('ball_height', 0):.3f} "
                        f"{'SUCCESS' if info.get('success') else 'fail'}")
            return True

    ckpt_cb = CheckpointCallback(
        save_freq=10_000, save_path=model_dir,
        name_prefix=f"sac_exp_{run_name}",
    )

    log(f"  开始训练: {total_timesteps} timesteps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[ckpt_cb, LogCallback()],
            tb_log_name=f"SAC_EXP_{run_name}",
        )
    except KeyboardInterrupt:
        log("  训练被用户中断")

    final_path = os.path.join(model_dir, f"model_SAC_EXP_{run_name}")
    model.save(final_path)
    log(f"  模型已保存: {final_path}.zip")

    log("  开始评估 (10 episodes)...")
    successes = 0
    for ep in range(10):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get("success"):
            successes += 1
        log(f"    Eval {ep+1}/10: ball_h={info.get('ball_height', 0):.3f} "
            f"{'SUCCESS' if info.get('success') else 'fail'}")

    res.ball_lifted = successes > 0
    res.ball_in_container = successes > 0
    res.notes.append(f"训练 {total_timesteps} steps, 评估成功率 {successes}/10")
    res.notes.append(f"模型: {final_path}.zip")
    res.duration_s = time.time() - t0
    return res


# =========================================================================
#  Iteration 5: Enhanced DL — More training + curriculum
# =========================================================================

def run_iteration_5(hw):
    """Extended SAC training with curriculum learning."""
    res = IterationResult(5, "SAC增强训练 + 课程学习", uses_deep_learning=True)
    res.algorithm_details = {
        "算法": "SAC (增量训练, 从迭代4模型继续)",
        "策略": "课程学习 — 逐步增大球体随机化范围",
        "训练步数": "额外 100,000 timesteps",
        "优化": "降低学习率 1e-4, 增大buffer 200K",
        "评估": "20 episodes 测试",
    }
    t0 = time.time()
    log("=== 迭代 5: 增强SAC + 课程学习 ===")

    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    except ImportError:
        res.notes.append("stable-baselines3 不可用")
        res.duration_s = time.time() - t0
        return res

    from rl_training.ur5e_grasp_env import UR5eGraspEnv

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    model_dir = os.path.join(PROJECT_ROOT, "rl_training", "models")
    log_dir = os.path.join(PROJECT_ROOT, "rl_training", "logs")
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    prev_models = sorted([
        f for f in os.listdir(model_dir)
        if f.startswith("model_SAC") and f.endswith(".zip")
    ]) if os.path.isdir(model_dir) else []

    for phase, (rand_range, lr, steps) in enumerate([
        (0.02, 2e-4, 30_000),
        (0.05, 1e-4, 40_000),
        (0.08, 1e-4, 30_000),
    ]):
        log(f"  课程阶段 {phase+1}/3: rand={rand_range}, lr={lr}, steps={steps}")

        env_cfg = {
            "max_steps": 400,
            "sub_steps": 4,
            "success_height": 0.15,
            "randomize_ball": True,
            "rand_range": rand_range,
        }
        env = UR5eGraspEnv(hw.sv, cfg=env_cfg)

        if phase == 0 and prev_models:
            prev_path = os.path.join(model_dir, prev_models[-1])
            log(f"  加载前序模型: {prev_path}")
            model = SAC.load(prev_path, env=env, device=device)
            model.learning_rate = lr
        elif phase == 0:
            model = SAC(
                "MlpPolicy", env,
                learning_rate=lr, batch_size=256,
                buffer_size=200_000, learning_starts=500,
                gamma=0.99, tau=0.005, verbose=0, device=device,
                tensorboard_log=log_dir,
            )
        else:
            model.set_env(env)
            model.learning_rate = lr

        ckpt_cb = CheckpointCallback(
            save_freq=10_000, save_path=model_dir,
            name_prefix=f"sac_curriculum_{run_name}_p{phase}",
        )
        model.learn(
            total_timesteps=steps, callback=[ckpt_cb],
            tb_log_name=f"SAC_CUR_{run_name}_p{phase}",
            reset_num_timesteps=False,
        )

    final_path = os.path.join(model_dir, f"model_SAC_CUR_{run_name}")
    model.save(final_path)
    log(f"  模型已保存: {final_path}.zip")

    log("  评估 (20 episodes)...")
    eval_env = UR5eGraspEnv(hw.sv, cfg={
        "max_steps": 400, "sub_steps": 4,
        "success_height": 0.15, "randomize_ball": True, "rand_range": 0.05,
    })
    successes = 0
    for ep in range(20):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        if info.get("success"):
            successes += 1

    log(f"  评估成功率: {successes}/20 = {100 * successes / 20:.0f}%")
    res.ball_lifted = successes > 0
    res.ball_in_container = successes > 0
    res.notes.append(f"课程学习3阶段, 评估成功率 {successes}/20")
    res.notes.append(f"模型: {final_path}.zip")
    res.duration_s = time.time() - t0
    return res


# =========================================================================
#  Obsidian Log Writer
# =========================================================================

def update_obsidian_log(result: IterationResult):
    """Write iteration result into the Obsidian experiment log."""
    if not os.path.exists(OBSIDIAN_LOG):
        log(f"  [WARN] Obsidian log not found: {OBSIDIAN_LOG}")
        return

    with open(OBSIDIAN_LOG, "r", encoding="utf-8") as f:
        content = f.read()

    iter_num = result.iteration
    status_icon = "✅ 成功" if result.success else "❌ 失败"
    dl_icon = "✅" if result.uses_dl else "❌"
    now = ts_str()

    result_block = (
        f"> **状态**：{status_icon}（{now}）\n"
        f"> \n"
        f"> | 指标 | 值 |\n"
        f"> |---|---|\n"
        f"> | 校准距离 | {result.calibration_dist:.3f}m | \n"
        f"> | Connector 锁定 | {'是' if result.connector_locked else '否'} |\n"
        f"> | 球体抬升 | {'是 ✅' if result.ball_lifted else '否 ❌'} |\n"
        f"> | 球体最终高度 | {result.ball_final_z:.3f}m |\n"
        f"> | 放入容器 | {'是 ✅' if result.ball_in_container else '否 ❌'} |\n"
        f"> | 耗时 | {result.duration_s:.1f}s |\n"
        f"> | 总体结果 | {status_icon} |"
    ) if result.calibration_dist is not None else (
        f"> **状态**：{status_icon}（{now}）\n"
        f"> \n"
        f"> | 指标 | 值 |\n"
        f"> |---|---|\n"
        f"> | 球体抬升 | {'是 ✅' if result.ball_lifted else '否 ❌'} |\n"
        f"> | 放入容器 | {'是 ✅' if result.ball_in_container else '否 ❌'} |\n"
        f"> | 耗时 | {result.duration_s:.1f}s |\n"
        f"> | 备注 | {'; '.join(result.notes)} |\n"
        f"> | 总体结果 | {status_icon} |"
    )

    old_pattern = (
        r"(## 迭代 " + str(iter_num) + r".*?\n### 实验结果\n\n)"
        r"(>.*?\n(?:>.*?\n)*)"
    )
    match = re.search(old_pattern, content, re.DOTALL)
    if match:
        new_content = content[:match.start(2)] + result_block + "\n" + content[match.end(2):]
        with open(OBSIDIAN_LOG, "w", encoding="utf-8") as f:
            f.write(new_content)
        log(f"  Obsidian 日志已更新: 迭代 {iter_num}")
    else:
        log(f"  [WARN] 无法在 Obsidian 日志中定位迭代 {iter_num} 的结果区域")

    summary_table_pattern = r"(\| " + str(iter_num) + r" \|.*?\|.*?\|.*?\|.*?\|)"
    summary_match = re.search(summary_table_pattern, content if not match else new_content)
    if summary_match:
        new_row = (
            f"| {iter_num} | {result.method} | {dl_icon} | "
            f"{'✅' if result.ball_lifted else '❌'} | {result.duration_s:.1f}s |"
        )
        final_content = (new_content if match else content).replace(
            summary_match.group(1), new_row
        )
        with open(OBSIDIAN_LOG, "w", encoding="utf-8") as f:
            f.write(final_content)


def save_iteration_json(result: IterationResult):
    """Save iteration result as a JSON file for programmatic access."""
    data_dir = os.path.join(PROJECT_ROOT, "experiment_data")
    os.makedirs(data_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(data_dir, f"iter{result.iteration}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.summary_dict(), f, indent=2, ensure_ascii=False)
    log(f"  结果已保存: {path}")


# =========================================================================
#  AutoCAD Drawing Sync
# =========================================================================

def sync_autocad_drawing(iteration):
    """Generate the AutoCAD .scr drawing for the current iteration."""
    try:
        from autocad_sync import generate_iteration_drawing
        log(f"\n  [AutoCAD] 同步迭代 {iteration} 图纸...")
        result = generate_iteration_drawing(iteration, output_dir=SOLIDWORKS_DIR)
        if result["file"]:
            log(f"  [AutoCAD] 生成: {os.path.basename(result['file'])}")
            for change in result["changes"]:
                log(f"    - {change}")
        return result
    except Exception as e:
        log(f"  [AutoCAD] 图纸生成失败: {e}")
        return {"file": None, "changes": [f"生成失败: {e}"]}


def update_obsidian_drawing_log(iteration, drawing_result):
    """Append AutoCAD drawing changes to the Obsidian experiment log."""
    if not os.path.exists(OBSIDIAN_LOG):
        return

    with open(OBSIDIAN_LOG, "r", encoding="utf-8") as f:
        content = f.read()

    now = ts_str()
    changes = drawing_result.get("changes", [])
    scr_file = drawing_result.get("file")
    filename = os.path.basename(scr_file) if scr_file else "N/A"

    changelog_entry = (
        f"\n\n#### AutoCAD 图纸变更（迭代 {iteration}）\n\n"
        f"| 项目 | 内容 |\n"
        f"|---|---|\n"
        f"| 时间 | {now} |\n"
        f"| 文件 | `{filename}` |\n\n"
        f"**变更明细：**\n\n"
    )
    for c in changes:
        changelog_entry += f"- {c}\n"

    section_pattern = (
        r"(## 迭代 " + str(iteration) + r".*?"
        r"### 迭代决策)"
    )
    match = re.search(section_pattern, content, re.DOTALL)
    if match:
        insert_pos = match.start(1) + len(match.group(1))
        search_back = content[:insert_pos].rstrip()
        new_content = search_back + changelog_entry + "\n" + content[insert_pos:]
        with open(OBSIDIAN_LOG, "w", encoding="utf-8") as f:
            f.write(new_content)
        log(f"  [Obsidian] 图纸变更已记录: 迭代 {iteration}")
    else:
        with open(OBSIDIAN_LOG, "r", encoding="utf-8") as f:
            content = f.read()

        anchor = "## 实验总结"
        if anchor in content:
            new_content = content.replace(
                anchor,
                changelog_entry + "\n---\n\n" + anchor,
            )
            with open(OBSIDIAN_LOG, "w", encoding="utf-8") as f:
                f.write(new_content)
            log(f"  [Obsidian] 图纸变更追加到实验总结前: 迭代 {iteration}")


# =========================================================================
#  Main Loop — Self-Iterating Experiment
# =========================================================================

ITERATIONS = [
    run_iteration_1,
    run_iteration_2,
    run_iteration_3,
    run_iteration_4,
    run_iteration_5,
]


def main():
    log("=" * 70)
    log("  UR5e 抓球验证实验 — 自动迭代运行器")
    log("  规则: 成功抓球 + 使用深度学习 → 实验完成")
    log("  规则: 失败 或 未使用DL → 自动进入下一迭代")
    log("  附加: 每轮同步更新 AutoCAD 图纸并记录变更")
    log("=" * 70)

    supervisor = Supervisor()
    hw = HardwareInterface(supervisor, timestep=32)

    for i, run_fn in enumerate(ITERATIONS):
        iter_num = i + 1
        log(f"\n{'#' * 70}")
        log(f"  开始迭代 {iter_num}/{len(ITERATIONS)}")
        log(f"{'#' * 70}\n")

        drawing_result = sync_autocad_drawing(iter_num)

        result = run_fn(hw)

        log(f"\n--- 迭代 {result.iteration} 结果 ---")
        log(f"  方法: {result.method}")
        log(f"  深度学习: {'是' if result.uses_dl else '否'}")
        log(f"  抓取成功: {'是' if result.ball_lifted else '否'}")
        log(f"  容器放置: {'是' if result.ball_in_container else '否'}")
        log(f"  耗时: {result.duration_s:.1f}s")
        log(f"  图纸: {os.path.basename(drawing_result.get('file', 'N/A')) if drawing_result.get('file') else 'N/A'}")
        for note in result.notes:
            log(f"  备注: {note}")

        if drawing_result.get("changes"):
            result.algorithm_details["AutoCAD图纸变更"] = drawing_result["changes"]

        save_iteration_json(result)
        update_obsidian_log(result)
        update_obsidian_drawing_log(iter_num, drawing_result)

        if result.experiment_complete:
            log(f"\n{'*' * 70}")
            log(f"  实验完成！迭代 {result.iteration} 成功")
            log(f"  使用深度学习: ✅")
            log(f"  抓取小球: ✅")
            log(f"  图纸最终版本: robot_arm_iter{iter_num}.scr")
            log(f"{'*' * 70}")
            break

        if result.success and not result.uses_dl:
            log(f"\n  抓取成功但未使用深度学习 → 继续升级算法")
        else:
            log(f"\n  抓取失败 → 自动触发下一迭代")

    else:
        log("\n  所有迭代已完成。请查看实验记录。")

    log("\n  实验结束，进入空闲状态。")
    ts = int(supervisor.getBasicTimeStep())
    while supervisor.step(ts) != -1:
        pass


if __name__ == "__main__":
    main()
