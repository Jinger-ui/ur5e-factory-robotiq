"""
UR5e + Robotiq 3F Gripper Ball Grasping — Gymnasium Environment
================================================================
Wraps a Webots Supervisor to expose the standard Gymnasium interface.
Observation (21-D): joint angles, joint velocities, finger positions,
                    ball position, end-effector position.
Action (7-D):      6 joint position increments  +  1 gripper open/close.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
_FINGER_MOTORS = [
    "finger_1_joint_1", "finger_2_joint_1", "finger_middle_joint_1",
]
_HOME = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]

_JOINT_LIMITS_LO = [-6.28, -6.28, -3.14, -6.28, -6.28, -6.28]
_JOINT_LIMITS_HI = [ 6.28,  6.28,  3.14,  6.28,  6.28,  6.28]

FINGER_OPEN  = 0.05
FINGER_CLOSE = 1.0


class UR5eGraspEnv(gym.Env):
    """Gymnasium environment for UR5e ball-grasping in Webots."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, supervisor, cfg=None):
        super().__init__()
        self.supervisor = supervisor
        self.ts = int(supervisor.getBasicTimeStep())
        self.cfg = cfg or {}

        self.max_steps      = self.cfg.get("max_steps", 400)
        self.sub_steps      = self.cfg.get("sub_steps", 4)
        self.table_height   = 0.74
        self.ball_init      = [0.45, 0.0, self.table_height + 0.03]
        self.success_height = self.cfg.get("success_height", 0.20)
        self.randomize_ball = self.cfg.get("randomize_ball", True)
        self.rand_range     = self.cfg.get("rand_range", 0.05)

        # -- spaces ----------------------------------------------------------
        act_lo = np.array([-0.1]*6 + [-1.0], dtype=np.float32)
        act_hi = np.array([ 0.1]*6 + [ 1.0], dtype=np.float32)
        self.action_space = spaces.Box(act_lo, act_hi, dtype=np.float32)

        obs_lo = np.full(21, -10.0, dtype=np.float32)
        obs_hi = np.full(21,  10.0, dtype=np.float32)
        self.observation_space = spaces.Box(obs_lo, obs_hi, dtype=np.float32)

        # -- devices ----------------------------------------------------------
        self.motors, self.sensors = [], []
        for jn in _JOINT_NAMES:
            m = supervisor.getDevice(jn)
            s = supervisor.getDevice(jn + "_sensor")
            if m:
                m.setVelocity(1.5)
            if s:
                s.enable(self.ts)
            self.motors.append(m)
            self.sensors.append(s)

        self.finger_motors = []
        self.finger_sensors = []
        for fn in _FINGER_MOTORS:
            fm = supervisor.getDevice(fn)
            fs = supervisor.getDevice(fn + "_sensor")
            if fm:
                fm.setVelocity(0.5)
            if fs:
                fs.enable(self.ts)
            self.finger_motors.append(fm)
            self.finger_sensors.append(fs)

        self.gps = supervisor.getDevice("tool_gps")
        if self.gps:
            self.gps.enable(self.ts)

        self.connector = supervisor.getDevice("connector")
        if self.connector:
            self.connector.enablePresence(self.ts)

        self.ball_node = supervisor.getFromDef("BALL")

        for _ in range(8):
            supervisor.step(self.ts)

        # -- episode state ----------------------------------------------------
        self._step_count   = 0
        self._prev_dist    = None
        self._prev_angles  = None
        self._gripper_locked = False

    # --------------------------------------------------------------------- #
    #  Observation                                                           #
    # --------------------------------------------------------------------- #
    def _get_obs(self):
        angles = np.array(
            [s.getValue() if s else 0.0 for s in self.sensors], dtype=np.float32
        )
        if self._prev_angles is not None:
            dt = self.ts * self.sub_steps / 1000.0
            velocities = (angles - self._prev_angles) / max(dt, 1e-6)
        else:
            velocities = np.zeros(6, dtype=np.float32)
        self._prev_angles = angles.copy()

        finger_pos = np.array(
            [fs.getValue() if fs else 0.0 for fs in self.finger_sensors],
            dtype=np.float32,
        )

        ball_pos = np.zeros(3, dtype=np.float32)
        if self.ball_node:
            tf = self.ball_node.getField("translation")
            if tf:
                ball_pos = np.array(tf.getSFVec3f(), dtype=np.float32)

        ee_pos = np.zeros(3, dtype=np.float32)
        if self.gps:
            ee_pos = np.array(self.gps.getValues(), dtype=np.float32)

        obs = np.concatenate([angles, velocities, finger_pos, ball_pos, ee_pos])
        return np.nan_to_num(obs, nan=0.0).astype(np.float32)

    # --------------------------------------------------------------------- #
    #  Reward                                                                #
    # --------------------------------------------------------------------- #
    def _compute_reward(self, obs, action):
        ee   = obs[18:21]
        ball = obs[15:18]
        dist = float(np.linalg.norm(ee - ball))
        ball_z = float(ball[2])

        r = 0.0

        # 1. distance reward  (always active)
        r -= dist * 2.0

        # 2. approach shaping
        if self._prev_dist is not None:
            r += (self._prev_dist - dist) * 30.0
        self._prev_dist = dist

        # 3. proximity bonuses
        if dist < 0.15:
            r += 2.0
        if dist < 0.08:
            r += 5.0
        if dist < 0.04:
            r += 10.0

        # 4. connector presence
        if self.connector and self.connector.getPresence():
            r += 20.0

        # 5. ball lifted above table
        height_above = ball_z - self.table_height
        if height_above > 0.03:
            r += height_above * 200.0

        # 6. success
        if height_above > self.success_height:
            r += 1000.0

        # 7. time penalty
        r -= 0.05

        # 8. action smoothness penalty
        if action is not None:
            r -= 0.01 * float(np.sum(np.abs(action[:6])))

        return r

    # --------------------------------------------------------------------- #
    #  Done / truncated                                                      #
    # --------------------------------------------------------------------- #
    def _check_terminated(self, obs):
        ball_z = float(obs[17])
        if ball_z - self.table_height > self.success_height:
            return True, True
        if ball_z < 0.1:
            return True, False
        return False, False

    def _check_truncated(self):
        return self._step_count >= self.max_steps

    # --------------------------------------------------------------------- #
    #  reset / step                                                          #
    # --------------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count   = 0
        self._prev_dist    = None
        self._prev_angles  = None
        self._gripper_locked = False

        if self.connector:
            self.connector.unlock()
        for fm in self.finger_motors:
            if fm:
                fm.setPosition(FINGER_OPEN)
        for m, p in zip(self.motors, _HOME):
            if m:
                m.setPosition(p)

        if self.ball_node:
            pos = list(self.ball_init)
            if self.randomize_ball and self.np_random is not None:
                pos[0] += float(self.np_random.uniform(-self.rand_range, self.rand_range))
                pos[1] += float(self.np_random.uniform(-self.rand_range, self.rand_range))
            tf = self.ball_node.getField("translation")
            if tf:
                tf.setSFVec3f(pos)
            rf = self.ball_node.getField("rotation")
            if rf:
                rf.setSFRotation([0, 0, 1, 0])
            self.ball_node.resetPhysics()

        for _ in range(30):
            self.supervisor.step(self.ts)

        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32)

        # apply joint increments
        cur = [s.getValue() if s else 0.0 for s in self.sensors]
        for i, (m, delta) in enumerate(zip(self.motors, action[:6])):
            if m:
                new_p = cur[i] + float(delta)
                new_p = max(_JOINT_LIMITS_LO[i], min(_JOINT_LIMITS_HI[i], new_p))
                m.setPosition(new_p)

        # gripper
        grip_val = float(action[6])
        if grip_val > 0.0:
            for fm in self.finger_motors:
                if fm:
                    fm.setPosition(FINGER_CLOSE)
            if self.connector and not self._gripper_locked:
                self.connector.lock()
                self._gripper_locked = True
        else:
            for fm in self.finger_motors:
                if fm:
                    fm.setPosition(FINGER_OPEN)
            if self.connector and self._gripper_locked:
                self.connector.unlock()
                self._gripper_locked = False

        for _ in range(self.sub_steps):
            if self.supervisor.step(self.ts) == -1:
                break

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        terminated, success = self._check_terminated(obs)
        truncated = self._check_truncated()

        info = {
            "success":     success,
            "steps":       self._step_count,
            "ball_height": float(obs[17]),
            "distance":    float(np.linalg.norm(obs[18:21] - obs[15:18])),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
