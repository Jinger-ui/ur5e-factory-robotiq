"""
UR5e Final Factory Controller - Robotiq 3F Visual + Connector Ball Grasp
=========================================================================
Based on proven ur5e_complete_controller.py logic.
Single ball target, GPS calibration, Connector physical grasp,
Robotiq 3-Finger visual animation, pick from table → place into container.
"""

import sys
import os
import math

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

BALL_DEF = "BALL"
BALL_RADIUS = 0.03
BALL_CENTER_Z = 0.77
BALL_CONNECTOR_Z = BALL_CENTER_Z + BALL_RADIUS  # 0.80

BALL_TARGET = {
    "name": "ball (red)",
    "pos": [0.45, 0.0, BALL_CONNECTOR_Z],
    "def": BALL_DEF,
    "reset_xyz": [0.45, 0.0, BALL_CENTER_Z],
}

PLACE_POS = [-0.5, 0.0, 0.85]

MAX_VEL = 1.2


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

    def reset_ball(self):
        print("\n  [SUPERVISOR] Resetting ball to original position ...")
        node = self.robot.getFromDef(BALL_DEF)
        if node:
            tf = node.getField("translation")
            if tf:
                tf.setSFVec3f(BALL_TARGET["reset_xyz"])
                print(f"    ball → ({BALL_TARGET['reset_xyz'][0]:.3f}, "
                      f"{BALL_TARGET['reset_xyz'][1]:.3f}, "
                      f"{BALL_TARGET['reset_xyz'][2]:.3f})")
            rf = node.getField("rotation")
            if rf:
                rf.setSFRotation([0, 0, 1, 0])
            node.resetPhysics()
        else:
            print("    [WARN] DEF BALL not found in scene!")
        self.wait_ms(500)
        print("  [SUPERVISOR] Ball reset complete.\n")

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
        print(f"     Distance: {d:.4f}m  (tolerance: 0.20m)")

        self.connector.lock()
        self.wait_ms(1500)
        p2 = self.connector.getPresence()
        if p2:
            print(f"  >> Connector LOCKED - ATTACHED (presence={p2})")
            return True
        print(f"  >> Connector LOCKED but NO CONTACT (presence={p2})")

        for retry in range(3):
            print(f"     Retry {retry+1}: unlock, settle, re-lock ...")
            self.connector.unlock()
            self.wait_ms(500)
            self.connector.lock()
            self.wait_ms(1500)
            p3 = self.connector.getPresence()
            if p3:
                print(f"  >> Retry {retry+1} ATTACHED (presence={p3})")
                return True
            print(f"  >> Retry {retry+1} still no contact (presence={p3})")

        return False

    def release(self):
        if self.connector:
            self.connector.unlock()
            self.wait_ms(1000)
            p = self.connector.getPresence()
            print(f"  >> Connector UNLOCKED (presence={p})")

    def get_ball_world_pos(self):
        node = self.robot.getFromDef(BALL_DEF)
        if node:
            tf = node.getField("translation")
            if tf:
                return list(tf.getSFVec3f())
        return BALL_TARGET["reset_xyz"]

    def calibrate(self):
        print("\n" + "=" * 60)
        print("  GPS CALIBRATION PHASE")
        print("  Scanning poses to find optimal angles for ball")
        tp = BALL_TARGET["pos"]
        print(f"    Ball connector target: ({tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f})")
        print("=" * 60)

        poses = generate_calibration_poses()
        print(f"\n  Testing {len(poses)} candidate poses ...\n")
        self.move_to(HOME, "HOME (start calibration)", settle_ms=500)

        best = {"pose": None, "dist": float("inf"), "gps": None, "above": None}
        all_results = []

        for pi, pose in enumerate(poses):
            self.set_joints(pose)
            self.wait_reach(pose, timeout_ms=5000, threshold=0.12)
            self.wait_ms(300)
            gps = self.gps_pos()

            d = dist3(gps, BALL_TARGET["pos"])
            if d < best["dist"]:
                best["dist"] = d
                best["pose"] = list(pose)
                best["gps"] = list(gps)
                print(f"  [{pi+1:3d}/{len(poses)}] "
                      f"ball d={d:.3f}m *** NEW BEST")

            all_results.append((list(pose), list(gps)))

            if (pi + 1) % 30 == 0:
                print(f"  [{pi+1:3d}/{len(poses)}] best so far: {best['dist']:.3f}m")

        self.move_to(HOME, "HOME (end calibration)", settle_ms=500)
        self.reset_ball()

        grasp = best["pose"]
        if grasp:
            above_candidates = []
            for p, g in all_results:
                if (abs(p[0] - grasp[0]) < 0.1
                        and dist3(g, BALL_TARGET["pos"]) < 0.3
                        and p[1] < grasp[1] - 0.1):
                    above_candidates.append(
                        (dist3(g, BALL_TARGET["pos"]), list(p), list(g)))
            if above_candidates:
                above_candidates.sort(key=lambda r: r[0])
                best["above"] = above_candidates[0][1]
                print(f"  Ball: using calibrated ABOVE pose")
            else:
                above = list(grasp)
                above[1] -= 0.30
                above[3] = math.pi / 2.0 - above[1] - above[2]
                best["above"] = above
                print(f"  Ball: using derived ABOVE pose")

        print("\n" + "=" * 60)
        print("  CALIBRATION RESULTS")
        print("=" * 60)
        ok = "OK" if best["dist"] <= 0.20 else "WARN"
        print(f"\n  [{ok}] Ball  distance={best['dist']:.3f}m")
        if best["pose"]:
            print(f"       Grasp: [{', '.join(f'{v:+.3f}' for v in best['pose'])}]")
        if best["gps"]:
            print(f"       GPS:   ({best['gps'][0]:+.3f}, {best['gps'][1]:+.3f}, "
                  f"{best['gps'][2]:+.3f})")
        if best["above"]:
            print(f"       Above: [{', '.join(f'{v:+.3f}' for v in best['above'])}]")
        print("=" * 60)

        return best

    def pick_and_place_ball(self, cal):
        grasp = cal["pose"]
        above = cal["above"]

        if not grasp or not above:
            print("  [SKIP] No valid calibration for ball")
            return False

        if cal["dist"] > 0.20:
            print(f"  [WARN] Ball distance {cal['dist']:.3f}m exceeds 0.20m "
                  "tolerance - attempting anyway")

        place_grasp = list(grasp)
        place_grasp[0] -= math.pi
        place_above = list(above)
        place_above[0] -= math.pi

        print(f"\n{'=' * 60}")
        print(f"  PICK-AND-PLACE: Ball")
        print(f"  Calibrated distance: {cal['dist']:.3f}m")
        print(f"{'=' * 60}")

        self.reset_ball()

        real_pos = self.get_ball_world_pos()
        print(f"  [INFO] Ball actual world pos: "
              f"({real_pos[0]:.4f}, {real_pos[1]:.4f}, {real_pos[2]:.4f})")

        print("\n--- Step 1: HOME ---")
        self.fingers_open()
        self.move_to(HOME, "HOME")

        print("\n--- Step 2: ABOVE PICK (sequenced) ---")
        self.move_sequenced(above, "ABOVE PICK")

        print("\n--- Step 3: DESCEND TO GRASP ---")
        self.move_to(grasp, "GRASP POSITION")

        print("\n--- Step 4: GRAB (Connector lock + Fingers close) ---")
        ball_conn_pos = list(BALL_TARGET["pos"])
        real_pos = self.get_ball_world_pos()
        ball_conn_pos_real = [real_pos[0], real_pos[1], real_pos[2] + BALL_RADIUS]
        print(f"  [INFO] Ball connector world pos (actual): "
              f"({ball_conn_pos_real[0]:.4f}, {ball_conn_pos_real[1]:.4f}, "
              f"{ball_conn_pos_real[2]:.4f})")

        grabbed = self.grab(ball_conn_pos)
        self.fingers_close()
        self.wait_ms(1500)
        if not grabbed:
            print("  [WARN] Failed to grab ball on first approach")
            print("  [RETRY] Adjusting and retrying...")
            self.connector.unlock()
            self.wait_ms(300)
            self.connector.lock()
            self.wait_ms(1500)
            p = self.connector.getPresence()
            grabbed = bool(p)
            if grabbed:
                print(f"  >> Extra retry ATTACHED (presence={p})")
            else:
                print(f"  >> Extra retry still no contact (presence={p})")

        print("\n--- Step 5: LIFT ---")
        self.move_to(above, "LIFT")

        ball_after_lift = self.get_ball_world_pos()
        print(f"  [CHECK] Ball pos after lift: ({ball_after_lift[0]:.4f}, "
              f"{ball_after_lift[1]:.4f}, {ball_after_lift[2]:.4f})")
        lifted = ball_after_lift[2] > BALL_CENTER_Z + 0.05
        print(f"  [CHECK] Ball lifted? {'YES' if lifted else 'NO'} "
              f"(z={ball_after_lift[2]:.3f}, threshold={BALL_CENTER_Z + 0.05:.3f})")

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

        ball_final = self.get_ball_world_pos()
        print(f"  [CHECK] Ball final pos: ({ball_final[0]:.4f}, "
              f"{ball_final[1]:.4f}, {ball_final[2]:.4f})")

        in_container = (
            -0.62 < ball_final[0] < -0.38 and
            -0.115 < ball_final[1] < 0.115 and
            ball_final[2] > 0.74
        )
        print(f"  [CHECK] Ball in container? {'YES' if in_container else 'NO'}")

        print("\n--- Step 10: RETREAT ---")
        self.move_to(place_above, "RETREAT")

        print("\n--- Step 11: HOME ---")
        self.move_sequenced(HOME, "HOME (done)")

        success = lifted and in_container
        status = "SUCCESS" if success else "PARTIAL"
        print(f"\n  [{status}] Ball pick-and-place complete!")
        print(f"    Lifted: {lifted}")
        print(f"    In container: {in_container}")
        return success

    def run(self):
        print()
        print("=" * 60)
        print("  UR5e Final Factory Controller")
        print("  " + "-" * 44)
        print("  Grasp method : Connector (magnetic, physical)")
        print("  Visual       : Robotiq 3F Gripper (finger animation)")
        print("  Calibration  : GPS scan, fine sweep")
        print("  Motion       : Sequenced joints, pre-computed angles")
        print("  Target       : Red ball (sphere r=0.03m)")
        print("  Place        : Container on place table")
        print("=" * 60)

        self.fingers_open()
        self.wait_ms(500)

        cal = self.calibrate()
        success = self.pick_and_place_ball(cal)

        print(f"\n{'=' * 60}")
        print(f"  MISSION {'COMPLETE - SUCCESS' if success else 'FINISHED - CHECK RESULT'}")
        print(f"  Ball picked up and placed: {'YES' if success else 'NEEDS VERIFICATION'}")
        print(f"{'=' * 60}")

        if success:
            print("\n  Ball successfully picked up by Robotiq 3F gripper")
            print("  and placed into the container on the place table!")
        else:
            print("\n  Check Webots 3D view for visual confirmation.")

        print("\n  Idling at HOME. Check Webots for visual result.")
        while self.robot.step(self.ts) != -1:
            pass


if __name__ == "__main__":
    UR5eFinalController().run()
