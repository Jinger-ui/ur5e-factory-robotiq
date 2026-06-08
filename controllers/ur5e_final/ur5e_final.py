"""
UR5e Final Factory Controller v11 — Tuned Color Sorting
========================================================
Based on proven v10, with safe timing reductions only.
Red ball -> LEFT (RED), Blue ball -> RIGHT (BLUE).
"""

import sys
import os
import math
import time as _time

try:
    from controller import Supervisor
except ImportError:
    sys.exit("Must be run from Webots.")

TIME_STEP = 32

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.log")
_log_file = open(LOG_PATH, "w", encoding="utf-8")
_orig_print = print


def print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    _log_file.write(" ".join(str(a) for a in args) + "\n")
    _log_file.flush()


JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
SENSOR_NAMES = [n + "_sensor" for n in JOINT_NAMES]

FINGER_MOTOR_NAMES = [
    "finger_1_joint_1",
    "finger_2_joint_1",
    "finger_middle_joint_1",
]
FINGER_OPEN = 0.05
FINGER_CLOSE = 1.0

HOME = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]

BALL_CENTER_Z = 0.77

BASE_GRASP = [0.790, -0.500, 1.900, 0.171, 0.0, 0.0]
BASE_ABOVE = [0.790, -0.800, 1.900, 0.471, 0.0, 0.0]
REF_PICK_XY = (0.45, 0.0)

MAX_VEL = 3.0
FINGER_VEL = 1.2

_sp_place_base = BASE_GRASP[0] - math.pi

RED_CONTAINER_Y = -0.10
BLUE_CONTAINER_Y = -0.30

_red_y_off = math.atan2(RED_CONTAINER_Y, 0.5)
_blue_y_off = math.atan2(BLUE_CONTAINER_Y, 0.5)

PLACE_POSES = {
    "RED": {
        "place":  [_sp_place_base - _red_y_off, -0.500, 1.900, 0.171, 0.0, 0.0],
        "above":  [_sp_place_base - _red_y_off, -0.800, 1.900, 0.471, 0.0, 0.0],
        "bounds": {"x_min": -0.615, "x_max": -0.385,
                   "y_min": -0.19,  "y_max": -0.01, "z_min": 0.73},
        "label":  "LEFT (RED)",
    },
    "BLUE": {
        "place":  [_sp_place_base - _blue_y_off, -0.500, 1.900, 0.171, 0.0, 0.0],
        "above":  [_sp_place_base - _blue_y_off, -0.800, 1.900, 0.471, 0.0, 0.0],
        "bounds": {"x_min": -0.615, "x_max": -0.385,
                   "y_min": -0.39,  "y_max": -0.21, "z_min": 0.73},
        "label":  "RIGHT (BLUE)",
    },
}

BALLS = [
    {"def": "BALL1", "name": "Red",  "start": [0.45,  0.08, BALL_CENTER_Z],
     "color_label": "RED"},
    {"def": "BALL2", "name": "Blue", "start": [0.45, -0.08, BALL_CENTER_Z],
     "color_label": "BLUE"},
]


def sp_offset_for(pick_y):
    ref_angle = math.atan2(REF_PICK_XY[1], REF_PICK_XY[0])
    tgt_angle = math.atan2(pick_y, 0.45)
    return tgt_angle - ref_angle


class UR5eFinalController:

    def __init__(self):
        self.robot = Supervisor()
        self.ts = TIME_STEP

        self.motors = []
        self.sensors = []
        for jn, sn in zip(JOINT_NAMES, SENSOR_NAMES):
            m = self.robot.getDevice(jn)
            if m:
                m.setVelocity(MAX_VEL)
            self.motors.append(m)
            s = self.robot.getDevice(sn)
            if s:
                s.enable(self.ts)
            self.sensors.append(s)

        self.finger_motors = []
        for fn in FINGER_MOTOR_NAMES:
            fm = self.robot.getDevice(fn)
            if fm:
                fm.setVelocity(FINGER_VEL)
            self.finger_motors.append(fm)

        self.gps = self.robot.getDevice("tool_gps")
        if self.gps:
            self.gps.enable(self.ts)

        self.connector = self.robot.getDevice("connector")
        if self.connector:
            self.connector.enablePresence(self.ts)

        for _ in range(4):
            self.robot.step(self.ts)

        print(f"[INIT] Vel={MAX_VEL} M=6 GPS=Y Conn=Y")

    # ---- motion ----

    def reset_ball(self, bi):
        node = self.robot.getFromDef(bi["def"])
        if node:
            node.getField("translation").setSFVec3f(list(bi["start"]))
            node.getField("rotation").setSFRotation([0, 0, 1, 0])
            node.resetPhysics()
        self.wait_ms(160)

    def get_joints(self):
        return [s.getValue() if s else 0.0 for s in self.sensors]

    def set_joints(self, pos):
        for m, p in zip(self.motors, pos):
            if m:
                m.setPosition(p)

    def gps_pos(self):
        return list(self.gps.getValues()) if self.gps else [0, 0, 0]

    def wait_ms(self, ms):
        for _ in range(max(1, ms // self.ts)):
            self.robot.step(self.ts)

    def wait_reach(self, target, timeout_ms=12000, threshold=0.15):
        elapsed = 0
        while elapsed < timeout_ms:
            self.robot.step(self.ts)
            elapsed += self.ts
            if all(abs((s.getValue() if s else 0.0) - t) < threshold
                   for s, t in zip(self.sensors, target)):
                return True
        return False

    def move(self, target, label="", settle_ms=300,
             timeout_ms=None, threshold=None):
        for m in self.motors:
            if m:
                m.setVelocity(MAX_VEL)
        self.set_joints(target)
        kw = {}
        if timeout_ms is not None:
            kw["timeout_ms"] = timeout_ms
        if threshold is not None:
            kw["threshold"] = threshold
        reached = self.wait_reach(target, **kw)
        self.wait_ms(settle_ms)
        gps = self.gps_pos()
        tag = "OK" if reached else "T/O"
        if label:
            print(f"  [{tag:3s}] {label}  ({gps[0]:.3f},{gps[1]:.3f},{gps[2]:.3f})")
        return gps, reached

    def move_seq(self, target, label="", settle_ms=300):
        for m in self.motors:
            if m:
                m.setVelocity(MAX_VEL)
        cur = self.get_joints()
        if abs(target[3] - cur[3]) > 0.5:
            phase1 = list(cur)
            phase1[3] = target[3]
            phase1[4] = target[4]
            phase1[5] = target[5]
            self.set_joints(phase1)
            self.wait_reach(phase1, timeout_ms=5000)
            self.wait_ms(64)
        self.set_joints(target)
        reached = self.wait_reach(target)
        self.wait_ms(settle_ms)
        gps = self.gps_pos()
        tag = "OK" if reached else "T/O"
        if label:
            print(f"  [{tag:3s}] {label}  ({gps[0]:.3f},{gps[1]:.3f},{gps[2]:.3f})")
        return gps, reached

    def fingers_open(self):
        for fm in self.finger_motors:
            if fm:
                fm.setPosition(FINGER_OPEN)

    def fingers_close(self):
        for fm in self.finger_motors:
            if fm:
                fm.setPosition(FINGER_CLOSE)

    def grab(self):
        if not self.connector:
            return False
        self.connector.lock()
        self.wait_ms(1000)
        if self.connector.getPresence():
            print("  >> ATTACHED")
            return True
        for r in range(3):
            self.connector.unlock()
            self.wait_ms(200)
            self.connector.lock()
            self.wait_ms(1000)
            if self.connector.getPresence():
                print(f"  >> R{r+1} ATTACHED")
                return True
        print("  >> FAIL")
        return False

    def release(self):
        if self.connector:
            self.connector.unlock()
            self.wait_ms(300)

    def get_ball_pos(self, bi):
        node = self.robot.getFromDef(bi["def"])
        if node:
            return list(node.getField("translation").getSFVec3f())
        return list(bi["start"])

    def pos_in_container(self, pos, color):
        b = PLACE_POSES[color]["bounds"]
        return (b["x_min"] < pos[0] < b["x_max"] and
                b["y_min"] < pos[1] < b["y_max"] and
                pos[2] > b["z_min"])

    # ---- pick-and-place (v10 flow, tighter timing) ----

    def pick_and_place(self, bi, idx, total):
        t0 = _time.time()
        color = bi["color_label"]
        sp_d = sp_offset_for(bi["start"][1])

        grasp = list(BASE_GRASP); grasp[0] += sp_d
        above = list(BASE_ABOVE); above[0] += sp_d

        pcfg = PLACE_POSES[color]
        lbl = pcfg["label"]
        print(f"\n  [{idx}/{total}] {bi['name']} -> {lbl}")

        # PICK
        self.move(HOME, "HOME", settle_ms=200)
        self.fingers_open()
        self.move_seq(above, "ABV", settle_ms=300)
        self.move(grasp, "GRP", settle_ms=800,
                  timeout_ms=18000, threshold=0.08)
        self.grab()
        self.fingers_close()
        self.wait_ms(500)

        # LIFT
        self.move(above, "LIFT", settle_ms=150)
        bz = self.get_ball_pos(bi)[2]
        lifted = bz > BALL_CENTER_Z + 0.05
        print(f"  lift={'Y' if lifted else 'N'} z={bz:.3f}")

        # TRANSIT -> PLACE
        self.move_seq(HOME, "TRNS", settle_ms=150)
        self.move_seq(pcfg["above"], "ABV_P", settle_ms=2000)
        self.move(pcfg["place"], "PLC", settle_ms=3000)

        self.release()
        self.fingers_open()
        self.wait_ms(1500)

        bf = self.get_ball_pos(bi)
        ok = self.pos_in_container(bf, color)
        dt = _time.time() - t0
        print(f"  => {lbl}:{'Y' if ok else 'N'} "
              f"({bf[0]:.3f},{bf[1]:.3f},{bf[2]:.3f}) [{dt:.1f}s]")
        return lifted, ok

    def run(self):
        t0 = _time.time()
        print("=" * 40)
        print(f"  UR5e v11  Vel={MAX_VEL}")
        print("  Red->LEFT  Blue->RIGHT")
        print("=" * 40)

        self.fingers_open()
        self.wait_ms(160)
        for b in BALLS:
            self.reset_ball(b)

        results = []
        for i, b in enumerate(BALLS, 1):
            l, p = self.pick_and_place(b, i, len(BALLS))
            results.append((b["name"], b["color_label"], l, p))

        self.move(HOME, "DONE", settle_ms=100)

        elapsed = _time.time() - t0
        print(f"\n{'=' * 40}")
        ok = all(l and p for _, _, l, p in results)
        for n, c, l, p in results:
            print(f"  [{'OK' if l and p else 'FL'}] {n} -> {PLACE_POSES[c]['label']}")
        print(f"\n  {'SUCCESS!' if ok else 'PARTIAL'}  {elapsed:.1f}s")
        print("=" * 40)

        while self.robot.step(self.ts) != -1:
            pass


if __name__ == "__main__":
    UR5eFinalController().run()
