"""
UR5e Final Factory Controller v10 — Fast & Reliable Color Sorting
=================================================================
Red ball -> LEFT (RED) container, Blue ball -> RIGHT (BLUE) container.
Optimized: higher joint velocity, shorter settle/wait times,
reliable HOME transit between pick and place.
"""

import sys
import os
import math
import time as _time

try:
    from controller import Supervisor
except ImportError:
    sys.exit("Must be run from Webots.")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

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

BALL_RADIUS = 0.03
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
    {"def": "BALL1", "name": "Red Ball",   "start": [0.45,  0.08, BALL_CENTER_Z],
     "hsv_low": (0, 100, 100), "hsv_high": (10, 255, 255), "color_label": "RED"},
    {"def": "BALL2", "name": "Blue Ball",  "start": [0.45, -0.08, BALL_CENTER_Z],
     "hsv_low": (100, 100, 80), "hsv_high": (130, 255, 255), "color_label": "BLUE"},
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

        self.camera = self.robot.getDevice("arm_camera")
        if self.camera:
            self.camera.enable(self.ts)
            if self.camera.hasRecognition():
                self.camera.recognitionEnable(self.ts)

        self.display = self.robot.getDevice("arm_display")

        for _ in range(4):
            self.robot.step(self.ts)

        print(f"[INIT] Motors={sum(1 for m in self.motors if m)}/6 "
              f"Fingers={sum(1 for f in self.finger_motors if f)}/3 "
              f"GPS={'OK' if self.gps else 'NO'} "
              f"Conn={'OK' if self.connector else 'NO'} "
              f"Cam={'OK' if self.camera else 'NO'} "
              f"CV2={'OK' if CV2_AVAILABLE else 'NO'} "
              f"MaxVel={MAX_VEL}")

    # ---- vision (lightweight) ----

    def scan_opencv(self, ball_info):
        if not CV2_AVAILABLE or not self.camera:
            return None
        w, h = self.camera.getWidth(), self.camera.getHeight()
        raw = self.camera.getImage()
        if not raw:
            return None
        img = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(ball_info["hsv_low"]),
                                np.array(ball_info["hsv_high"]))
        if ball_info["color_label"] == "RED":
            mask2 = cv2.inRange(hsv, np.array((170, 100, 100)),
                                     np.array((180, 255, 255)))
            mask = cv2.bitwise_or(mask, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 20:
            return None
        M = cv2.moments(largest)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else 0
        return {"cx": cx, "cy": cy, "area": area, "color": ball_info["color_label"]}

    # ---- motion primitives ----

    def reset_ball(self, ball_info):
        node = self.robot.getFromDef(ball_info["def"])
        if node:
            tf = node.getField("translation")
            if tf:
                tf.setSFVec3f(list(ball_info["start"]))
            rf = node.getField("rotation")
            if rf:
                rf.setSFRotation([0, 0, 1, 0])
            node.resetPhysics()
        self.wait_ms(200)

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
            cur = self.get_joints()
            if all(abs(c - t) < threshold for c, t in zip(cur, target)):
                return True
        return False

    def move(self, target, label="", settle_ms=500,
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
            print(f"  [{tag:3s}] {label}  ({gps[0]:.3f}, {gps[1]:.3f}, {gps[2]:.3f})")
        return gps, reached

    def move_seq(self, target, label="", settle_ms=500):
        """Move with wrist-first sequencing to avoid stalls on large rotations."""
        for m in self.motors:
            if m:
                m.setVelocity(MAX_VEL)
        cur = self.get_joints()
        wrist_diff = abs(target[3] - cur[3])
        if wrist_diff > 0.5:
            phase1 = list(cur)
            phase1[3] = target[3]
            phase1[4] = target[4]
            phase1[5] = target[5]
            self.set_joints(phase1)
            self.wait_reach(phase1, timeout_ms=6000)
            self.wait_ms(100)
        self.set_joints(target)
        reached = self.wait_reach(target)
        self.wait_ms(settle_ms)
        gps = self.gps_pos()
        tag = "OK" if reached else "T/O"
        if label:
            print(f"  [{tag:3s}] {label}  ({gps[0]:.3f}, {gps[1]:.3f}, {gps[2]:.3f})")
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
        self.wait_ms(800)
        if self.connector.getPresence():
            print("  >> ATTACHED")
            return True
        for retry in range(3):
            self.connector.unlock()
            self.wait_ms(200)
            self.connector.lock()
            self.wait_ms(800)
            if self.connector.getPresence():
                print(f"  >> Retry {retry+1} ATTACHED")
                return True
        print("  >> GRAB FAILED")
        return False

    def release(self):
        if self.connector:
            self.connector.unlock()
            self.wait_ms(300)

    def get_ball_pos(self, ball_info):
        node = self.robot.getFromDef(ball_info["def"])
        if node:
            tf = node.getField("translation")
            if tf:
                return list(tf.getSFVec3f())
        return list(ball_info["start"])

    def pos_in_container(self, pos, color_label):
        b = PLACE_POSES[color_label]["bounds"]
        return (b["x_min"] < pos[0] < b["x_max"] and
                b["y_min"] < pos[1] < b["y_max"] and
                pos[2] > b["z_min"])

    # ---- fast pick-and-place ----

    def pick_and_place(self, ball_info, idx, total):
        t0 = _time.time()
        name = ball_info["name"]
        color = ball_info["color_label"]
        pick_y = ball_info["start"][1]
        sp_delta = sp_offset_for(pick_y)

        grasp = list(BASE_GRASP); grasp[0] += sp_delta
        above = list(BASE_ABOVE); above[0] += sp_delta

        place_cfg = PLACE_POSES[color]
        place_angles = place_cfg["place"]
        above_place = place_cfg["above"]
        target_label = place_cfg["label"]

        print(f"\n--- [{idx}/{total}] {name} -> {target_label} ---")

        # PICK phase
        self.move(HOME, "HOME", settle_ms=300)
        self.fingers_open()
        self.move_seq(above, "ABOVE PICK", settle_ms=300)

        cv = self.scan_opencv(ball_info)
        if cv:
            print(f"  [CV] {cv['color']} center=({cv['cx']},{cv['cy']}) area={cv['area']:.0f}px")

        self.move(grasp, "GRASP", settle_ms=800,
                 timeout_ms=18000, threshold=0.08)
        grabbed = self.grab()
        self.fingers_close()
        self.wait_ms(600)

        self.move(above, "LIFT", settle_ms=200)
        ball_z = self.get_ball_pos(ball_info)[2]
        lifted = ball_z > BALL_CENTER_Z + 0.05
        print(f"  lifted={'YES' if lifted else 'NO'} z={ball_z:.3f}")

        # PLACE phase — via HOME for safe transit
        self.move_seq(HOME, "TRANSIT", settle_ms=200)
        self.move_seq(above_place, "ABOVE PLACE", settle_ms=2500)
        self.move(place_angles, "PLACE", settle_ms=3500)

        self.release()
        self.fingers_open()
        self.wait_ms(2000)

        ball_final = self.get_ball_pos(ball_info)
        ok = self.pos_in_container(ball_final, color)
        dt = _time.time() - t0
        print(f"  => {target_label}: {'YES' if ok else 'NO'} "
              f"pos=({ball_final[0]:.3f},{ball_final[1]:.3f},{ball_final[2]:.3f}) "
              f"[{dt:.1f}s]")

        return lifted, ok

    def run(self):
        t_total = _time.time()

        print()
        print("=" * 50)
        print("  UR5e v10 — Fast Color Sorting")
        print(f"  MaxVel={MAX_VEL} rad/s")
        print("  Red->LEFT  Blue->RIGHT")
        print("=" * 50)

        self.fingers_open()
        self.wait_ms(200)
        for b in BALLS:
            self.reset_ball(b)
            print(f"  {b['name']} reset")

        results = []
        for i, b in enumerate(BALLS, 1):
            lifted, placed = self.pick_and_place(b, i, len(BALLS))
            results.append((b["name"], b["color_label"], lifted, placed))

        # Return home after last ball
        self.move(HOME, "FINAL HOME", settle_ms=200)

        elapsed = _time.time() - t_total
        print(f"\n{'=' * 50}")
        all_ok = all(l and p for _, _, l, p in results)
        for name, color, lifted, placed in results:
            target = PLACE_POSES[color]["label"]
            s = "OK" if (lifted and placed) else "FAIL"
            print(f"  [{s}] {name} -> {target}")
        print(f"\n  {'SUCCESS!' if all_ok else 'PARTIAL FAIL'}  Total: {elapsed:.1f}s")
        print("=" * 50)

        while self.robot.step(self.ts) != -1:
            pass


if __name__ == "__main__":
    UR5eFinalController().run()
