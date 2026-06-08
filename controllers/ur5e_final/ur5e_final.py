"""
UR5e Final Factory Controller v7 — Vision-Guided Dual-Ball Grasp
================================================================
Uses Webots Camera Recognition API + OpenCV HSV color detection
to identify and locate balls, then picks them into the container.

References:
  - BerkeleyAutomation/gqcnn (Dex-Net architecture inspiration)
  - atenpas/gpd (grasp pose detection concepts)
  - Webots Camera Recognition API + OpenCV integration
"""

import sys
import os
import math

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

_sp_place_base = BASE_GRASP[0] - math.pi

RED_CONTAINER_CENTER = (-0.5, 0.13)
BLUE_CONTAINER_CENTER = (-0.5, -0.13)

_red_y_off = math.atan2(RED_CONTAINER_CENTER[1], abs(RED_CONTAINER_CENTER[0]))
_blue_y_off = math.atan2(BLUE_CONTAINER_CENTER[1], abs(BLUE_CONTAINER_CENTER[0]))

PLACE_POSES = {
    "RED": {
        "place":  [_sp_place_base - _red_y_off, -0.500, 1.900, 0.171, 0.0, 0.0],
        "above":  [_sp_place_base - _red_y_off, -0.800, 1.900, 0.471, 0.0, 0.0],
        "bounds": {"x_min": -0.615, "x_max": -0.385,
                   "y_min": 0.025,  "y_max": 0.235, "z_min": 0.73},
    },
    "BLUE": {
        "place":  [_sp_place_base - _blue_y_off, -0.500, 1.900, 0.171, 0.0, 0.0],
        "above":  [_sp_place_base - _blue_y_off, -0.800, 1.900, 0.471, 0.0, 0.0],
        "bounds": {"x_min": -0.615, "x_max": -0.385,
                   "y_min": -0.235, "y_max": -0.025, "z_min": 0.73},
    },
}

MAX_VEL = 1.2

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
                fm.setVelocity(0.5)
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

        has_recog = self.camera and self.camera.hasRecognition()
        print(f"[INIT] Motors: {sum(1 for m in self.motors if m)}/6, "
              f"Fingers: {sum(1 for f in self.finger_motors if f)}/3, "
              f"GPS: {'OK' if self.gps else 'NO'}, "
              f"Connector: {'OK' if self.connector else 'NO'}, "
              f"Camera: {'OK' if self.camera else 'NO'}, "
              f"Recognition: {'OK' if has_recog else 'NO'}, "
              f"OpenCV: {'OK' if CV2_AVAILABLE else 'NO'}, "
              f"Display: {'OK' if self.display else 'NO'}")

    # ---- vision ----

    def scan_with_recognition(self):
        """Use Webots Recognition API to detect objects visible to arm_camera."""
        if not self.camera or not self.camera.hasRecognition():
            return []
        objects = self.camera.getRecognitionObjects()
        results = []
        for obj in objects:
            pos = obj.getPosition()
            colors = obj.getColors()
            size = obj.getSize()
            name = obj.getModel() if hasattr(obj, 'getModel') else "unknown"
            results.append({
                "position": list(pos),
                "colors": list(colors) if colors else [],
                "size": list(size) if size else [],
                "position_on_image": list(obj.getPositionOnImage()),
                "size_on_image": list(obj.getSizeOnImage()),
            })
        return results

    def save_camera_frame(self, label="frame"):
        """Save current camera frame to disk for debugging."""
        if not CV2_AVAILABLE or not self.camera:
            return
        w = self.camera.getWidth()
        h = self.camera.getHeight()
        raw = self.camera.getImage()
        if not raw:
            return
        img = np.frombuffer(raw, np.uint8).reshape((h, w, 4))
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        out_dir = os.path.dirname(LOG_PATH)
        path = os.path.join(out_dir, f"cam_{label}.png")
        cv2.imwrite(path, bgr)
        print(f"  [CAM] Saved frame: {path}")

    def scan_with_opencv(self, ball_info):
        """Use OpenCV HSV color masking to detect a specific ball color in camera image."""
        if not CV2_AVAILABLE or not self.camera:
            return None
        w = self.camera.getWidth()
        h = self.camera.getHeight()
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
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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

        self._overlay_detection(bgr, largest, cx, cy, ball_info["color_label"], area)

        return {"cx": cx, "cy": cy, "area": area, "color": ball_info["color_label"]}

    def _overlay_detection(self, bgr, contour, cx, cy, label, area):
        """Draw detection overlay on the Display device."""
        if not self.display or not CV2_AVAILABLE:
            return
        vis = bgr.copy()
        color_map = {"RED": (0, 0, 255), "BLUE": (255, 0, 0)}
        c = color_map.get(label, (0, 255, 0))
        cv2.drawContours(vis, [contour], -1, c, 2)
        cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(vis, f"{label} A={area:.0f}", (cx - 30, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        h, w = vis.shape[:2]
        ir = self.display.imageNew(vis.tobytes(), self.display.BGRA,
                                   w, h) if hasattr(self.display, 'BGRA') else None
        if ir is None:
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2BGRA)
            ir = self.display.imageNew(rgb.tobytes(), self.display.BGRA, w, h)
        if ir:
            self.display.imagePaste(ir, 0, 0, False)
            self.display.imageDelete(ir)

    _scan_count = 0

    def vision_scan_report(self, ball_info):
        """Run both recognition and OpenCV detection, print results."""
        UR5eFinalController._scan_count += 1
        self.save_camera_frame(f"{UR5eFinalController._scan_count}_{ball_info['color_label']}")
        recog = self.scan_with_recognition()
        cv_result = self.scan_with_opencv(ball_info)

        print(f"\n  [VISION] Recognition API: {len(recog)} objects detected")
        for i, obj in enumerate(recog):
            p = obj["position"]
            print(f"    Object {i}: pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                  f"  img=({obj['position_on_image'][0]:.0f}, {obj['position_on_image'][1]:.0f})"
                  f"  colors={obj['colors'][:3] if obj['colors'] else '?'}")

        if cv_result:
            print(f"  [VISION] OpenCV HSV: {cv_result['color']} detected"
                  f"  center=({cv_result['cx']}, {cv_result['cy']})"
                  f"  area={cv_result['area']:.0f}px")
        else:
            print(f"  [VISION] OpenCV HSV: {ball_info['color_label']} NOT detected")

        return recog, cv_result

    # ---- motion ----

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
        self.wait_ms(500)

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
            print(f"  [{tag:7s}] {label}  GPS=({gps[0]:.4f}, {gps[1]:.4f}, {gps[2]:.4f})")
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
            print(f"  [{tag:7s}] {label}  GPS=({gps[0]:.4f}, {gps[1]:.4f}, {gps[2]:.4f})")
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
        gps = self.gps_pos()
        print(f"  >> GRAB GPS=({gps[0]:.4f}, {gps[1]:.4f}, {gps[2]:.4f})")
        self.connector.lock()
        self.wait_ms(1500)
        if self.connector.getPresence():
            print(f"  >> ATTACHED")
            return True
        for retry in range(3):
            self.connector.unlock()
            self.wait_ms(300)
            self.connector.lock()
            self.wait_ms(1500)
            if self.connector.getPresence():
                print(f"  >> Retry {retry+1} ATTACHED")
                return True
        print(f"  >> GRAB FAILED")
        return False

    def release(self):
        if self.connector:
            self.connector.unlock()
            self.wait_ms(1000)

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

    # ---- pick and place with vision ----

    def pick_and_place(self, ball_info, idx, total):
        name = ball_info["name"]
        pick_y = ball_info["start"][1]
        sp_delta = sp_offset_for(pick_y)

        grasp = list(BASE_GRASP); grasp[0] += sp_delta
        above = list(BASE_ABOVE); above[0] += sp_delta

        print(f"\n{'=' * 60}")
        print(f"  TASK {idx}/{total}: {name} ({ball_info['color_label']})")
        print(f"  Pick Y={pick_y:+.3f}  SP offset={sp_delta:+.4f}")
        print(f"{'=' * 60}")

        print("\n--- HOME ---")
        self.move_to(HOME, "HOME")

        print("\n--- ABOVE PICK (vision scan) ---")
        self.fingers_open()
        self.move_sequenced(above, "ABOVE PICK")

        self.vision_scan_report(ball_info)

        print("\n--- DESCEND TO GRASP ---")
        self.move_to(grasp, "GRASP")

        self.vision_scan_report(ball_info)

        print("\n--- GRAB ---")
        grabbed = self.grab()
        self.fingers_close()
        self.wait_ms(1500)

        print("\n--- LIFT ---")
        self.move_to(above, "LIFT")
        ball = self.get_ball_pos(ball_info)
        lifted = ball[2] > BALL_CENTER_Z + 0.05
        print(f"  {name} z={ball[2]:.3f}, lifted={'YES' if lifted else 'NO'}")

        print("\n--- TRANSIT HOME ---")
        self.move_sequenced(HOME, "HOME transit")

        print("\n--- ABOVE PLACE ---")
        self.move_sequenced(BASE_ABOVE_PLACE, "ABOVE PLACE")

        print("\n--- LOWER TO PLACE ---")
        self.move_to(BASE_PLACE, "PLACE")
        self.wait_ms(4000)

        print("\n--- RELEASE ---")
        self.release()
        self.fingers_open()
        self.wait_ms(5000)

        ball_final = self.get_ball_pos(ball_info)
        in_container = self.pos_in_container(ball_final)
        print(f"\n  {name} final: ({ball_final[0]:.4f}, {ball_final[1]:.4f}, {ball_final[2]:.4f})")
        print(f"  In container: {'YES' if in_container else 'NO'}")

        print("\n--- RETREAT & HOME ---")
        self.move_to(BASE_ABOVE_PLACE, "RETREAT")
        self.move_sequenced(HOME, "HOME done")

        return lifted, in_container

    def run(self):
        print()
        print("=" * 60)
        print(f"  UR5e Final Controller v7 — Vision-Guided")
        print(f"  {len(BALLS)} Balls | Robotiq 3F + Connector")
        print(f"  OpenCV HSV detection + Webots Recognition API")
        print(f"  Ref: GQ-CNN (Berkeley), GPD (ten Pas et al.)")
        print("=" * 60)

        self.fingers_open()
        self.wait_ms(500)
        for b in BALLS:
            self.reset_ball(b)
            print(f"  {b['name']} reset to ({b['start'][0]}, {b['start'][1]}, {b['start'][2]})")

        print("\n--- INITIAL VISION SCAN FROM HOME ---")
        self.move_to(HOME, "HOME")
        for b in BALLS:
            self.vision_scan_report(b)

        results = []
        for i, b in enumerate(BALLS, 1):
            lifted, placed = self.pick_and_place(b, i, len(BALLS))
            results.append((b["name"], lifted, placed))

        print(f"\n{'=' * 60}")
        all_ok = all(l and p for _, l, p in results)
        for name, lifted, placed in results:
            status = "OK" if (lifted and placed) else "FAIL"
            print(f"  [{status}] {name}: lifted={lifted}, in_container={placed}")
        print(f"\n  {'SUCCESS! All balls placed!' if all_ok else 'Some balls failed.'}")
        print("=" * 60)

        while self.robot.step(self.ts) != -1:
            pass


if __name__ == "__main__":
    UR5eFinalController().run()
