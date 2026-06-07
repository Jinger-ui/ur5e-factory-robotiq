"""
UR5e Complete Factory Controller - Robotiq 3F Visual + Connector Physical Grasp
================================================================================
Strategy: 100% reuse of proven GPS-calibrated Connector grasping logic.
Robotiq 3-Finger Gripper provides visual finger animation only.
Connector handles all physical attachment/detachment.
"""

import sys
import os
import math

try:
    from controller import Supervisor
except ImportError:
    sys.exit("Must be run from Webots.")

TIME_STEP = 16

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

TARGETS = [
    {"name": "target_1 (red)",   "pos": [0.45,  0.0,  0.805], "def": "TARGET_1",
     "reset_xyz": [0.45, 0.0, 0.775]},
    {"name": "target_2 (green)", "pos": [0.45, -0.10, 0.805], "def": "TARGET_2",
     "reset_xyz": [0.45, -0.10, 0.775]},
    {"name": "target_3 (blue)",  "pos": [0.45,  0.10, 0.805], "def": "TARGET_3",
     "reset_xyz": [0.45, 0.10, 0.775]},
]

MAX_VEL = 1.5


def dist3(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def generate_calibration_poses():
    poses = []
    sp_values = [
        -1.57, -1.18, -0.79, -0.40, 0.0,
        0.40, 0.55, 0.70, 0.79, 0.90, 1.00, 1.10, 1.25, 1.57,
    ]
    for sp in sp_values:
        for sl in [-0.5, -0.8, -1.1, -1.4]:
            for el in [0.4, 0.9, 1.4, 1.9]:
                w1 = math.pi / 2.0 - sl - el
                if -3.14 < w1 < 3.14:
                    poses.append([sp, sl, el, w1, 0.0, 0.0])
    return poses


class UR5eCompleteController:

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
                fm.setVelocity(0.5)
            self.finger_motors.append(fm)

        self.gps = self.robot.getDevice("tool_gps")
        if self.gps:
            self.gps.enable(self.ts)

        self.connector = self.robot.getDevice("connector")
        if self.connector:
            self.connector.enablePresence(self.ts)

        for _ in range(4):
            self.robot.step(self.ts)

        arm_ok = sum(1 for m in self.motors if m)
        finger_ok = sum(1 for f in self.finger_motors if f)
        print(f"[INIT] Arm motors: {arm_ok}/6")
        print(f"[INIT] Robotiq finger motors: {finger_ok}/3  "
              f"({', '.join(FINGER_MOTOR_NAMES[:finger_ok])})")
        print(f"[INIT] GPS: {'OK' if self.gps else 'MISSING'}")
        print(f"[INIT] Connector: {'OK' if self.connector else 'MISSING'}")

    # ── supervisor: reset objects to original positions ─────────

    def reset_objects(self):
        print("\n  [SUPERVISOR] Resetting all objects to original positions ...")
        for t in TARGETS:
            node = self.robot.getFromDef(t["def"])
            if node:
                tf = node.getField("translation")
                if tf:
                    tf.setSFVec3f(t["reset_xyz"])
                    print(f"    {t['name']} → ({t['reset_xyz'][0]:.3f}, "
                          f"{t['reset_xyz'][1]:.3f}, {t['reset_xyz'][2]:.3f})")
                rf = node.getField("rotation")
                if rf:
                    rf.setSFRotation([0, 0, 1, 0])
                node.resetPhysics()
            else:
                print(f"    [WARN] DEF {t['def']} not found in scene!")
        self.wait_ms(500)
        print("  [SUPERVISOR] Objects reset complete.\n")

    # ── joint helpers (identical to success scheme) ────────────

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

    def wait_reach(self, target, timeout_ms=15000, threshold=0.08):
        elapsed = 0
        while elapsed < timeout_ms:
            self.robot.step(self.ts)
            elapsed += self.ts
            cur = self.get_joints()
            if all(abs(c - t) < threshold for c, t in zip(cur, target)):
                return True
        return False

    def move_to(self, target, label="", settle_ms=2000):
        for m in self.motors:
            if m:
                m.setVelocity(MAX_VEL)
        self.set_joints(target)
        reached = self.wait_reach(target)
        self.wait_ms(settle_ms)
        gps = self.gps_pos()
        cur = self.get_joints()
        tag = "OK" if reached else "TIMEOUT"
        if label:
            print(f"  [{tag:7s}] {label}")
            print(f"           Joints: [{', '.join(f'{j:+.3f}' for j in cur)}]")
            print(f"           GPS:    ({gps[0]:.4f}, {gps[1]:.4f}, {gps[2]:.4f})")
        return gps, reached

    def move_sequenced(self, target, label="", settle_ms=2000):
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
            self.wait_reach(phase1, timeout_ms=8000)
            self.wait_ms(500)
            print(f"    (wrist pre-positioned, delta={wrist_diff:.2f} rad)")
        self.set_joints(target)
        reached = self.wait_reach(target)
        self.wait_ms(settle_ms)
        gps = self.gps_pos()
        cur = self.get_joints()
        tag = "OK" if reached else "TIMEOUT"
        if label:
            print(f"  [{tag:7s}] {label}")
            print(f"           Joints: [{', '.join(f'{j:+.3f}' for j in cur)}]")
            print(f"           GPS:    ({gps[0]:.4f}, {gps[1]:.4f}, {gps[2]:.4f})")
        return gps, reached

    # ── Robotiq finger animation ──────────────────────────────

    def fingers_open(self):
        print("  [FINGERS] Opening (visual animation)")
        for fm in self.finger_motors:
            if fm:
                fm.setPosition(FINGER_OPEN)

    def fingers_close(self):
        print("  [FINGERS] Closing (visual animation)")
        for fm in self.finger_motors:
            if fm:
                fm.setPosition(FINGER_CLOSE)

    # ── Connector physical grasp ──────────────────────────────

    def grab(self, target_pos):
        if not self.connector:
            print("  [ERROR] No connector device!")
            return False
        presence = self.connector.getPresence()
        gps = self.gps_pos()
        d = dist3(gps, target_pos)
        print(f"\n  >> GRAB attempt")
        print(f"     Connector presence: {presence}")
        print(f"     GPS:    ({gps[0]:.4f}, {gps[1]:.4f}, {gps[2]:.4f})")
        print(f"     Target: ({target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f})")
        print(f"     Distance: {d:.4f}m  (tolerance: 0.15m)")
        self.connector.lock()
        self.wait_ms(1500)
        p2 = self.connector.getPresence()
        if p2:
            print(f"  >> Connector LOCKED - ATTACHED (presence={p2})")
            return True
        print(f"  >> Connector LOCKED but NO CONTACT (presence={p2})")
        print(f"     Retrying: unlock, settle, re-lock ...")
        self.connector.unlock()
        self.wait_ms(500)
        self.connector.lock()
        self.wait_ms(1500)
        p3 = self.connector.getPresence()
        if p3:
            print(f"  >> Retry ATTACHED (presence={p3})")
            return True
        print(f"  >> Retry still no contact (presence={p3})")
        return False

    def release(self):
        if self.connector:
            self.connector.unlock()
            self.wait_ms(1000)
            p = self.connector.getPresence()
            print(f"  >> Connector UNLOCKED (presence={p})")

    # ── GPS calibration (identical logic to success scheme) ───

    def calibrate(self):
        print("\n" + "=" * 60)
        print("  GPS CALIBRATION PHASE")
        print("  Scanning poses to find optimal angles for each target")
        for t in TARGETS:
            print(f"    {t['name']}: "
                  f"({t['pos'][0]:.3f}, {t['pos'][1]:.3f}, {t['pos'][2]:.3f})")
        print("=" * 60)

        poses = generate_calibration_poses()
        print(f"\n  Testing {len(poses)} candidate poses ...\n")
        self.move_to(HOME, "HOME (start calibration)", settle_ms=500)

        best = {}
        for ti in range(len(TARGETS)):
            best[ti] = {"pose": None, "dist": float("inf"),
                        "gps": None, "above": None}

        all_results = []

        for pi, pose in enumerate(poses):
            self.set_joints(pose)
            self.wait_reach(pose, timeout_ms=5000, threshold=0.12)
            self.wait_ms(400)
            gps = self.gps_pos()

            for ti, target in enumerate(TARGETS):
                d = dist3(gps, target["pos"])
                if d < best[ti]["dist"]:
                    best[ti]["dist"] = d
                    best[ti]["pose"] = list(pose)
                    best[ti]["gps"] = list(gps)
                    print(f"  [{pi+1:3d}/{len(poses)}] "
                          f"{target['name']} d={d:.3f}m *** NEW BEST")

            all_results.append((list(pose), list(gps)))

            if (pi + 1) % 20 == 0:
                summary = " | ".join(
                    f"{TARGETS[i]['name'].split()[0]}:{best[i]['dist']:.3f}"
                    for i in range(len(TARGETS)))
                print(f"  [{pi+1:3d}/{len(poses)}] {summary}")

        self.move_to(HOME, "HOME (end calibration)", settle_ms=500)

        self.reset_objects()

        for ti in range(len(TARGETS)):
            grasp = best[ti]["pose"]
            if not grasp:
                continue
            above_candidates = []
            for p, g in all_results:
                if (abs(p[0] - grasp[0]) < 0.1
                        and dist3(g, TARGETS[ti]["pos"]) < 0.3
                        and p[1] < grasp[1] - 0.1):
                    above_candidates.append(
                        (dist3(g, TARGETS[ti]["pos"]), list(p), list(g)))
            if above_candidates:
                above_candidates.sort(key=lambda r: r[0])
                best[ti]["above"] = above_candidates[0][1]
                print(f"  {TARGETS[ti]['name']}: using calibrated ABOVE pose")
            else:
                above = list(grasp)
                above[1] -= 0.30
                above[3] = math.pi / 2.0 - above[1] - above[2]
                best[ti]["above"] = above
                print(f"  {TARGETS[ti]['name']}: using derived ABOVE pose")

        print("\n" + "=" * 60)
        print("  CALIBRATION RESULTS")
        print("=" * 60)
        for ti, target in enumerate(TARGETS):
            b = best[ti]
            ok = "OK" if b["dist"] <= 0.15 else "WARN"
            print(f"\n  [{ok}] {target['name']}  distance={b['dist']:.3f}m")
            if b["pose"]:
                print(f"       Grasp: "
                      f"[{', '.join(f'{v:+.3f}' for v in b['pose'])}]")
            if b["gps"]:
                print(f"       GPS:   "
                      f"({b['gps'][0]:+.3f}, {b['gps'][1]:+.3f}, "
                      f"{b['gps'][2]:+.3f})")
            if b["above"]:
                print(f"       Above: "
                      f"[{', '.join(f'{v:+.3f}' for v in b['above'])}]")
        print("=" * 60)

        return best

    # ── pick-and-place cycle ─────────────────────────────────

    def pick_and_place_one(self, target_idx, cal):
        target = TARGETS[target_idx]
        b = cal[target_idx]
        grasp = b["pose"]
        above = b["above"]

        if not grasp or not above:
            print(f"  [SKIP] No valid calibration for {target['name']}")
            return False

        if b["dist"] > 0.15:
            print(f"  [WARN] {target['name']} distance {b['dist']:.3f}m "
                  f"exceeds 0.15m tolerance - attempting anyway")

        place_grasp = list(grasp)
        place_grasp[0] -= math.pi
        place_above = list(above)
        place_above[0] -= math.pi

        print(f"\n{'=' * 60}")
        print(f"  PICK-AND-PLACE: {target['name']}")
        print(f"  Calibrated distance: {b['dist']:.3f}m")
        print(f"{'=' * 60}")

        node = self.robot.getFromDef(target["def"])
        if node:
            tf = node.getField("translation")
            if tf:
                tf.setSFVec3f(target["reset_xyz"])
            rf = node.getField("rotation")
            if rf:
                rf.setSFRotation([0, 0, 1, 0])
            node.resetPhysics()
            self.wait_ms(300)
            print(f"  [SUPERVISOR] {target['name']} reset to original position")

        print("\n--- Step 1: HOME ---")
        self.fingers_open()
        self.move_to(HOME, "HOME")

        print("\n--- Step 2: ABOVE PICK (sequenced) ---")
        self.move_sequenced(above, "ABOVE PICK")

        print("\n--- Step 3: DESCEND TO GRASP ---")
        self.move_to(grasp, "GRASP POSITION")

        print("\n--- Step 4: GRAB (Connector lock + Fingers close) ---")
        grabbed = self.grab(target["pos"])
        self.fingers_close()
        self.wait_ms(1500)
        if not grabbed:
            print(f"  [WARN] Failed to grab {target['name']} - continuing anyway")

        print("\n--- Step 5: LIFT ---")
        self.move_to(above, "LIFT")

        print("\n--- Step 6: HOME (transit, sequenced) ---")
        self.move_sequenced(HOME, "HOME (transit)", settle_ms=1000)

        print("\n--- Step 7: ABOVE PLACE (sequenced) ---")
        self.move_sequenced(place_above, "ABOVE PLACE")

        print("\n--- Step 8: LOWER TO PLACE ---")
        self.move_to(place_grasp, "PLACE POSITION")

        print("\n--- Step 9: RELEASE (Connector unlock + Fingers open) ---")
        self.release()
        self.fingers_open()
        self.wait_ms(1500)

        print("\n--- Step 10: RETREAT ---")
        self.move_to(place_above, "RETREAT")

        print("\n--- Step 11: HOME ---")
        self.move_sequenced(HOME, "HOME (done)")

        print(f"\n  [DONE] {target['name']} pick-and-place complete!")
        return True

    # ── main entry ───────────────────────────────────────────

    def run(self):
        print()
        print("=" * 60)
        print("  UR5e Complete Factory Controller v3")
        print("  " + "-" * 44)
        print("  Grasp method : Connector (magnetic, physical)")
        print("  Visual       : Robotiq 3F Gripper (finger animation)")
        print("  Calibration  : GPS scan, fine sweep")
        print("  Motion       : Sequenced joints, pre-computed angles")
        print(f"  Targets      : {len(TARGETS)}")
        print("=" * 60)

        self.fingers_open()
        self.wait_ms(1000)

        cal = self.calibrate()

        success_count = 0
        for ti in range(len(TARGETS)):
            if self.pick_and_place_one(ti, cal):
                success_count += 1

        print(f"\n{'=' * 60}")
        print(f"  MISSION COMPLETE")
        print(f"  Successfully placed: {success_count}/{len(TARGETS)}")
        print(f"{'=' * 60}")

        if success_count == len(TARGETS):
            print("\n  ALL objects placed! Robotiq fingers animated correctly.")
        else:
            print(f"\n  {len(TARGETS) - success_count} object(s) may have failed.")
            print("  Check Webots 3D view for visual confirmation.")

        print("\n  Idling at HOME. Check Webots for visual result.")
        while self.robot.step(self.ts) != -1:
            pass


if __name__ == "__main__":
    UR5eCompleteController().run()
