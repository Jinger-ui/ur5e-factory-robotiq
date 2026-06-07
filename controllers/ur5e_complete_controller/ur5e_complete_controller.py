"""
UR5e Robotiq 3F Factory Controller - Ball Pick-and-Place
=========================================================
Robotiq 3-Finger Gripper visual animation + Connector physics grab.
GPS calibration finds optimal joint angles for the connector/GPS tip.
Proven strategy: fingers open before release for clean ball drop.
"""

import sys
import os
import math

try:
    from controller import Supervisor
except ImportError:
    sys.exit("Must be run from Webots.")

TIME_STEP = 16
MAX_VEL = 1.5

JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
SENSOR_NAMES = [n + "_sensor" for n in JOINT_NAMES]

FINGER_MOTORS = ["finger_1_joint_1", "finger_2_joint_1", "finger_middle_joint_1"]
FINGER_OPEN = 0.0
FINGER_CLOSE = 1.0
FINGER_VEL = 0.5

HOME = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]

BALL_RESET_POS = [0.45, 0.0, 0.775]
BALL_CONN_TARGET = [0.45, 0.0, 0.805]

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.log")
_log_file = open(LOG_PATH, "w", encoding="utf-8")
_orig_print = print


def print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    _log_file.write(" ".join(str(a) for a in args) + "\n")
    _log_file.flush()


def dist3(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def generate_calibration_poses(sp_center):
    poses = []
    sp_range = [sp_center + d for d in [-0.40, -0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25, 0.40]]
    for sp in sp_range:
        for sl in [-0.4, -0.6, -0.8, -1.0, -1.2, -1.4]:
            for el in [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]:
                w1 = math.pi / 2.0 - sl - el
                if -3.14 < w1 < 3.14:
                    poses.append([sp, sl, el, w1, 0.0, 0.0])
    return poses


class Controller:

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
        for fn in FINGER_MOTORS:
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

        self.fingers_open()
        for _ in range(4):
            self.robot.step(self.ts)

        arm_ok = sum(1 for m in self.motors if m)
        finger_ok = sum(1 for f in self.finger_motors if f)
        print(f"[INIT] Arm motors: {arm_ok}/6")
        print(f"[INIT] Finger motors: {finger_ok}/3")
        print(f"[INIT] GPS: {'OK' if self.gps else 'MISSING'}")
        print(f"[INIT] Connector: {'OK' if self.connector else 'MISSING'}")

    def fingers_open(self):
        for fm in self.finger_motors:
            if fm:
                fm.setPosition(FINGER_OPEN)

    def fingers_close(self):
        for fm in self.finger_motors:
            if fm:
                fm.setPosition(FINGER_CLOSE)

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
        tag = "OK" if reached else "TIMEOUT"
        if label:
            print(f"  [{tag:7s}] {label}  GPS: ({gps[0]:.4f}, {gps[1]:.4f}, {gps[2]:.4f})")
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
        self.set_joints(target)
        reached = self.wait_reach(target)
        self.wait_ms(settle_ms)
        gps = self.gps_pos()
        tag = "OK" if reached else "TIMEOUT"
        if label:
            print(f"  [{tag:7s}] {label}  GPS: ({gps[0]:.4f}, {gps[1]:.4f}, {gps[2]:.4f})")
        return gps, reached

    def get_ball_pos(self):
        node = self.robot.getFromDef("BALL")
        if node:
            return list(node.getPosition())
        return None

    def reset_ball(self):
        node = self.robot.getFromDef("BALL")
        if node:
            tf = node.getField("translation")
            tf.setSFVec3f(list(BALL_RESET_POS))
            rf = node.getField("rotation")
            rf.setSFRotation([0, 0, 1, 0])
            node.resetPhysics()
            self.wait_ms(500)
            print(f"  [RESET] Ball -> ({BALL_RESET_POS[0]}, {BALL_RESET_POS[1]}, {BALL_RESET_POS[2]})")

    def calibrate(self):
        print("\n" + "=" * 60)
        print("  GPS CALIBRATION - Finding optimal grasp angles")
        print("=" * 60)
        target = BALL_CONN_TARGET
        sp_hint = 0.79 + math.atan2(target[1], target[0])
        print(f"  Ball connector target: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
        print(f"  Shoulder-pan hint: {sp_hint:.3f} rad")

        poses = generate_calibration_poses(sp_hint)
        print(f"  Scanning {len(poses)} candidate poses ...\n")

        self.move_to(HOME, settle_ms=500)

        best_pose = None
        best_dist = float("inf")
        best_gps = None

        for i, pose in enumerate(poses):
            self.set_joints(pose)
            self.wait_reach(pose, timeout_ms=5000, threshold=0.12)
            self.wait_ms(350)
            gps = self.gps_pos()
            d = dist3(gps, target)
            if d < best_dist:
                best_dist = d
                best_pose = list(pose)
                best_gps = list(gps)
                print(f"  [{i+1:3d}/{len(poses)}] d={d:.3f}m *** NEW BEST  "
                      f"GPS=({gps[0]:.3f},{gps[1]:.3f},{gps[2]:.3f})")
            elif (i + 1) % 30 == 0:
                print(f"  [{i+1:3d}/{len(poses)}] best so far: {best_dist:.3f}m")

        self.move_to(HOME, settle_ms=500)

        above = list(best_pose)
        above[1] -= 0.30
        above[3] = math.pi / 2.0 - above[1] - above[2]

        print(f"\n  CALIBRATION RESULT:")
        print(f"  Best distance : {best_dist:.4f}m (tolerance: 0.25m)")
        print(f"  Grasp angles  : [{', '.join(f'{v:+.3f}' for v in best_pose)}]")
        print(f"  Above angles  : [{', '.join(f'{v:+.3f}' for v in above)}]")
        print(f"  GPS at grasp  : ({best_gps[0]:.4f}, {best_gps[1]:.4f}, {best_gps[2]:.4f})")
        status = "PASS" if best_dist <= 0.25 else "FAIL"
        print(f"  Status        : {status}")
        print("=" * 60)

        return best_pose, above, best_dist

    def grab(self):
        if not self.connector:
            print("  [ERROR] No connector!")
            return False
        presence = self.connector.getPresence()
        gps = self.gps_pos()
        d = dist3(gps, BALL_CONN_TARGET)
        print(f"\n  >> GRAB ATTEMPT")
        print(f"     Presence before lock: {presence}")
        print(f"     GPS:    ({gps[0]:.4f}, {gps[1]:.4f}, {gps[2]:.4f})")
        print(f"     Target: ({BALL_CONN_TARGET[0]:.4f}, {BALL_CONN_TARGET[1]:.4f}, {BALL_CONN_TARGET[2]:.4f})")
        print(f"     Distance: {d:.4f}m")

        print(f"  >> Closing fingers ...")
        self.fingers_close()
        self.wait_ms(800)

        self.connector.lock()
        self.wait_ms(1500)
        p2 = self.connector.getPresence()
        if p2 == 1:
            print(f"  >> LOCK SUCCESS (presence={p2})")
            return True

        print(f"  >> First lock no contact (presence={p2}), retrying ...")
        self.connector.unlock()
        self.wait_ms(500)
        self.connector.lock()
        self.wait_ms(1500)
        p3 = self.connector.getPresence()
        if p3 == 1:
            print(f"  >> Retry SUCCESS (presence={p3})")
            return True

        print(f"  >> Retry FAILED (presence={p3})")
        return False

    def run(self):
        print()
        print("=" * 60)
        print("  UR5e Robotiq 3F - Ball Pick-and-Place")
        print("  Connector + GPS calibration + finger animation")
        print("=" * 60)

        self.move_to(HOME, "HOME (init)")
        self.wait_ms(1000)

        ball_pos = self.get_ball_pos()
        print(f"\n  Ball initial: {ball_pos}")

        # === CALIBRATE ===
        grasp, above, cal_dist = self.calibrate()

        self.reset_ball()
        ball_pos = self.get_ball_pos()
        print(f"  Ball after reset: {ball_pos}")

        if cal_dist > 0.25:
            print(f"\n  [WARN] Calibration distance {cal_dist:.3f}m > 0.25m tolerance")

        # === PICK ===
        print(f"\n{'=' * 60}")
        print(f"  PICK SEQUENCE")
        print(f"{'=' * 60}")

        self.fingers_open()
        self.wait_ms(600)

        rot_home = list(HOME)
        rot_home[0] = grasp[0]
        self.move_to(rot_home, "ROTATE to pick direction", settle_ms=800)
        self.move_sequenced(above, "ABOVE pick", settle_ms=1000)
        self.move_to(grasp, "GRASP position")

        grabbed = self.grab()

        self.move_to(above, "LIFT")
        ball_pos = self.get_ball_pos()
        print(f"  Ball after lift: {ball_pos}")
        if ball_pos and ball_pos[2] > 0.85:
            print("  *** BALL IS LIFTED SUCCESSFULLY! ***")
        else:
            print("  Ball may NOT be lifted")

        self.move_sequenced(rot_home, "FOLD up", settle_ms=500)
        self.move_to(HOME, "HOME (transit)", settle_ms=800)

        # === PLACE ===
        print(f"\n{'=' * 60}")
        print(f"  PLACE SEQUENCE")
        print(f"{'=' * 60}")

        place_sp = 0.79 - math.pi
        place_above = list(above)
        place_above[0] = place_sp

        rot_home_place = list(HOME)
        rot_home_place[0] = place_sp

        self.move_to(rot_home_place, "ROTATE to place direction", settle_ms=1500)

        ball_pos = self.get_ball_pos()
        print(f"  Ball (carrying): {ball_pos}")

        self.move_sequenced(place_above, "ABOVE container", settle_ms=3000)

        gps_place = self.gps_pos()
        print(f"  GPS at release point: ({gps_place[0]:.4f}, {gps_place[1]:.4f}, {gps_place[2]:.4f})")
        print(f"  Container bounds: X[-0.64,-0.36] Y[-0.135,0.135]")
        x_in = -0.64 < gps_place[0] < -0.36
        y_in = -0.135 < gps_place[1] < 0.135
        print(f"  Within X: {x_in} ({gps_place[0]:.3f})  Within Y: {y_in} ({gps_place[1]:.3f})")

        print(f"  >> Opening fingers (clear path for ball) ...")
        self.fingers_open()
        self.wait_ms(1500)

        print(f"  >> Unlocking connector (releasing ball) ...")
        if self.connector:
            self.connector.unlock()
        self.wait_ms(3000)

        ball_pos = self.get_ball_pos()
        print(f"  Ball after release: {ball_pos}")

        self.move_to(place_above, "RETREAT up")
        self.move_sequenced(rot_home_place, "FOLD up", settle_ms=500)
        self.move_to(HOME, "HOME (done)", settle_ms=800)

        # === FINAL CHECK ===
        self.wait_ms(2000)
        ball_pos = self.get_ball_pos()
        print(f"\n{'=' * 60}")
        print(f"  FINAL RESULT")
        print(f"{'=' * 60}")
        print(f"  Ball final position: {ball_pos}")
        print(f"  Grabbed: {grabbed}")
        print(f"  Calibration distance: {cal_dist:.4f}m")

        if ball_pos:
            in_container = (
                -0.64 < ball_pos[0] < -0.36 and
                -0.135 < ball_pos[1] < 0.135 and
                ball_pos[2] > 0.73
            )
            on_place_table = (
                -0.80 < ball_pos[0] < -0.20 and
                -0.40 < ball_pos[1] < 0.40 and
                ball_pos[2] > 0.73
            )
            if in_container:
                print("  *** SUCCESS! Ball is in the container! ***")
            elif on_place_table:
                print("  Ball is on the place table (not exactly in container)")
            else:
                print(f"  Ball at unexpected position")
        else:
            print("  [ERROR] Cannot read ball position")

        print(f"\n  Idling. Check Webots 3D view for visual result.")
        while self.robot.step(self.ts) != -1:
            pass


if __name__ == "__main__":
    Controller().run()
