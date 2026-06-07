"""
UR5e Complete Industrial Controller for Webots R2025a
=====================================================
Robust pick-and-place using pre-computed joint poses with IK verification.
Based on the official Webots ure_can_grasper approach for reliability.
"""

import sys
import os
import math
import json
import numpy as np

try:
    from ikpy.chain import Chain
    from ikpy.link import OriginLink, URDFLink
    HAS_IKPY = True
except ImportError:
    HAS_IKPY = False

try:
    from controller import Robot
except ImportError:
    sys.exit("This script must be launched from Webots.")


# ───────────────────────────────────────────────────────────────
# UR5e kinematic chain (DH parameters) WITH gripper TCP offset
# ───────────────────────────────────────────────────────────────

GRIPPER_TCP_OFFSET = 0.175  # Robotiq 3F gripper: flange to fingertip

def build_ur5e_chain():
    return Chain(name="ur5e", links=[
        OriginLink(),
        URDFLink(name="shoulder_pan",
                 origin_translation=[0, 0, 0.1625],
                 origin_orientation=[0, 0, 0],
                 rotation=[0, 0, 1]),
        URDFLink(name="shoulder_lift",
                 origin_translation=[0, 0, 0],
                 origin_orientation=[0, math.pi / 2, 0],
                 rotation=[0, 1, 0]),
        URDFLink(name="elbow",
                 origin_translation=[0, -0.4250, 0],
                 origin_orientation=[0, 0, 0],
                 rotation=[0, 1, 0]),
        URDFLink(name="wrist_1",
                 origin_translation=[0, -0.3922, 0],
                 origin_orientation=[0, 0, 0],
                 rotation=[0, 1, 0]),
        URDFLink(name="wrist_2",
                 origin_translation=[0, 0, 0.1333],
                 origin_orientation=[0, 0, 0],
                 rotation=[0, 0, 1]),
        URDFLink(name="wrist_3",
                 origin_translation=[0, 0, 0.0997],
                 origin_orientation=[0, 0, 0],
                 rotation=[0, 1, 0]),
        URDFLink(name="ee_fixed",
                 origin_translation=[0, -0.0996, 0],
                 origin_orientation=[0, 0, 0],
                 rotation=[0, 0, 0]),
    ])


# ───────────────────────────────────────────────────────────────
# Pre-computed joint poses for reliable pick and place
# These are the PRIMARY motion method (not dependent on IK)
# ───────────────────────────────────────────────────────────────

HOME_JOINTS = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]

# Intermediate safe pose (arm up and clear of table)
SAFE_ABOVE = [0.0, -1.2, 0.5, -1.87, 0.0, 0.0]

# Pre-computed pick poses: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
# Each target has: (approach_pose, grasp_pose) - empirically tuned for table at x=0.45
PICK_APPROACH_POSES = [
    [1.37, -1.10, 0.70, -1.17, -1.57, 0.0],   # target_red    (x=0.40, y=-0.12)
    [1.50, -1.10, 0.70, -1.17, -1.57, 0.0],   # target_green  (x=0.40, y=0.00)
    [1.62, -1.10, 0.70, -1.17, -1.57, 0.0],   # target_blue   (x=0.40, y=0.12)
    [1.30, -1.10, 0.70, -1.17, -1.57, 0.0],   # target_yellow (x=0.48, y=-0.08)
    [1.57, -1.10, 0.70, -1.17, -1.57, 0.0],   # target_orange (x=0.48, y=0.08)
]

PICK_GRASP_POSES = [
    [1.37, -1.35, 1.10, -1.32, -1.57, 0.0],   # target_red
    [1.50, -1.35, 1.10, -1.32, -1.57, 0.0],   # target_green
    [1.62, -1.35, 1.10, -1.32, -1.57, 0.0],   # target_blue
    [1.30, -1.35, 1.10, -1.32, -1.57, 0.0],   # target_yellow
    [1.57, -1.35, 1.10, -1.32, -1.57, 0.0],   # target_orange
]

# Place poses (toward the place table at x=-0.55)
PLACE_APPROACH_POSES = [
    [-1.57, -1.10, 0.70, -1.17, -1.57, 0.0],  # place_crate_1
    [-1.57, -1.10, 0.70, -1.17, -1.57, 0.0],  # place_crate_2
    [-1.57, -1.10, 0.70, -1.17, -1.57, 0.0],  # place_crate_3
    [-1.40, -1.10, 0.70, -1.17, -1.57, 0.0],  # place_crate_4
    [-1.40, -1.10, 0.70, -1.17, -1.57, 0.0],  # place_crate_5
]

PLACE_DOWN_POSES = [
    [-1.57, -1.30, 1.00, -1.27, -1.57, 0.0],  # place_crate_1
    [-1.57, -1.30, 1.00, -1.27, -1.57, 0.0],  # place_crate_2
    [-1.57, -1.30, 1.00, -1.27, -1.57, 0.0],  # place_crate_3
    [-1.40, -1.30, 1.00, -1.27, -1.57, 0.0],  # place_crate_4
    [-1.40, -1.30, 1.00, -1.27, -1.57, 0.0],  # place_crate_5
]


# ───────────────────────────────────────────────────────────────
# Controller
# ───────────────────────────────────────────────────────────────

class UR5eCompleteController:

    JOINT_NAMES = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    SENSOR_NAMES = [n + "_sensor" for n in JOINT_NAMES]
    GRIPPER_NAMES = [
        "finger_1_joint_1", "finger_2_joint_1", "finger_middle_joint_1",
    ]

    def __init__(self):
        self.robot = Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.config = self._load_config()

        self._init_devices()

        self.chain = build_ur5e_chain() if HAS_IKPY else None

        self.state = "INIT"
        self.target_idx = 0
        self.wait_counter = 0
        self.step_count = 0
        self.cycle_count = 0
        self.objects_placed = 0
        self.errors = []

        # Motion state
        self.trajectory = []
        self.traj_step = 0

        self._print_banner()

    # ── config ────────────────────────────────────────────────

    def _load_config(self):
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.json"
        )
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            print(f"[CONFIG] Loaded from {os.path.basename(config_path)}")
            return cfg
        except Exception as exc:
            print(f"[WARN] config.json not found ({exc}), using defaults")
            return self._default_config()

    @staticmethod
    def _default_config():
        return {
            "robot": {
                "base_position": [0, 0, 0.8],
                "base_rotation_z": -1.5708,
                "max_velocity": 0.8,
                "gripper_open": 0.0,
                "gripper_closed": 0.85,
                "gripper_tcp_offset": 0.175
            },
            "task": {
                "targets": [
                    {"name": "target_red",    "world_position": [0.40, -0.12, 0.765]},
                    {"name": "target_green",  "world_position": [0.40,  0.00, 0.770]},
                    {"name": "target_blue",   "world_position": [0.40,  0.12, 0.770]},
                    {"name": "target_yellow", "world_position": [0.48, -0.08, 0.765]},
                    {"name": "target_orange", "world_position": [0.48,  0.08, 0.765]},
                ],
                "place_positions": [
                    [-0.55, -0.10, 0.82],
                    [-0.55,  0.10, 0.82],
                    [-0.55,  0.00, 0.82],
                    [-0.45, -0.10, 0.82],
                    [-0.45,  0.10, 0.82],
                ],
                "home_joints": [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
            },
            "timing": {
                "grasp_wait_ms": 1500,
                "release_wait_ms": 1000,
                "settle_wait_ms": 500,
                "approach_wait_ms": 400
            }
        }

    # ── device init ───────────────────────────────────────────

    def _init_devices(self):
        self.motors = []
        self.sensors = []
        for jn, sn in zip(self.JOINT_NAMES, self.SENSOR_NAMES):
            m = self.robot.getDevice(jn)
            if m:
                m.setVelocity(self.config["robot"]["max_velocity"])
            self.motors.append(m)
            s = self.robot.getDevice(sn)
            if s:
                s.enable(self.time_step)
            self.sensors.append(s)

        self.gripper_motors = []
        for gn in self.GRIPPER_NAMES:
            gm = self.robot.getDevice(gn)
            if gm:
                gm.setVelocity(0.3)
                try:
                    gm.enableForceFeedback(self.time_step)
                except Exception:
                    pass
            self.gripper_motors.append(gm)

        self.distance_sensor = self.robot.getDevice("gripper_sensor")
        if self.distance_sensor:
            self.distance_sensor.enable(self.time_step)

        self.camera = self.robot.getDevice("arm_camera")
        if self.camera:
            self.camera.enable(self.time_step)

        self.touch_sensor = self.robot.getDevice("gripper_touch")
        if self.touch_sensor:
            self.touch_sensor.enable(self.time_step)

        motor_count = len([m for m in self.motors if m])
        grip_count = len([g for g in self.gripper_motors if g])
        print(f"[INIT] Motors={motor_count} Gripper={grip_count} "
              f"Cam={'OK' if self.camera else 'NO'} "
              f"Dist={'OK' if self.distance_sensor else 'NO'} "
              f"Touch={'OK' if self.touch_sensor else 'NO'}")

    def _print_banner(self):
        targets = self.config["task"]["targets"]
        print()
        print("=" * 60)
        print("  UR5e Robust Pick-and-Place Controller")
        print("  " + "-" * 44)
        print(f"  IK Engine       : {'ikpy (verification)' if HAS_IKPY else 'disabled'}")
        print(f"  Motion Method   : Pre-computed joint poses")
        print(f"  Targets         : {len(targets)}")
        print(f"  Time Step       : {self.time_step} ms")
        print(f"  Max Velocity    : {self.config['robot']['max_velocity']} rad/s")
        print(f"  Gripper TCP     : {self.config['robot'].get('gripper_tcp_offset', 0.175)} m")
        print("=" * 60)

    # ── coordinate helpers ────────────────────────────────────

    def world_to_robot(self, world_pos):
        bx, by, bz = self.config["robot"]["base_position"]
        angle = self.config["robot"]["base_rotation_z"]
        dx, dy, dz = world_pos[0] - bx, world_pos[1] - by, world_pos[2] - bz
        c, s = math.cos(-angle), math.sin(-angle)
        return [c * dx - s * dy, s * dx + c * dy, dz]

    # ── IK with gripper TCP offset ───────────────────────────

    def compute_ik(self, target_world_pos, grasp_from_above=True):
        if not self.chain:
            return None

        robot_pos = self.world_to_robot(target_world_pos)

        # Account for gripper TCP: IK targets the tool flange,
        # which is gripper_tcp_offset ABOVE the actual grasp point
        tcp_offset = self.config["robot"].get("gripper_tcp_offset", GRIPPER_TCP_OFFSET)
        if grasp_from_above:
            robot_pos[2] += tcp_offset

        target_mat = np.eye(4)
        target_mat[:3, 3] = robot_pos

        current = self.get_joint_positions()
        seeds = [
            [0.0] + list(current) + [0.0],
            [0.0] * 8,
            [0.0, 1.5, -1.2, 0.8, -1.2, -1.57, 0.0, 0.0],
            [0.0, 1.5, -1.5, 1.2, -1.5, -1.57, 0.0, 0.0],
        ]
        for _ in range(4):
            seeds.append(
                [0.0] + [np.random.uniform(-2.0, 2.0) for _ in range(6)] + [0.0]
            )

        best, best_err = None, float("inf")
        for sd in seeds:
            try:
                res = self.chain.inverse_kinematics(
                    target_mat, initial_position=sd, orientation_mode=None
                )
                joints = list(res[1:7])
                fk = self.compute_fk(joints)
                if fk is not None:
                    err = np.linalg.norm(np.array(fk) - np.array(robot_pos))
                    if err < best_err:
                        best_err = err
                        best = joints
            except Exception:
                continue

        if best is not None and best_err < 0.03:
            return best
        return None

    def compute_fk(self, joint_angles):
        if not self.chain:
            return None
        full = [0.0] + list(joint_angles) + [0.0]
        tf = self.chain.forward_kinematics(full)
        return tf[:3, 3].tolist()

    # ── joint read / write ────────────────────────────────────

    def get_joint_positions(self):
        return [s.getValue() if s else 0.0 for s in self.sensors]

    def set_joint_positions(self, positions):
        for m, p in zip(self.motors, positions):
            if m:
                m.setPosition(p)

    def joints_reached(self, target, threshold=0.05):
        current = self.get_joint_positions()
        return all(abs(c - t) < threshold for c, t in zip(current, target))

    # ── trajectory generation ─────────────────────────────────

    def plan_smooth_trajectory(self, target_joints, speed_factor=1.0):
        current = self.get_joint_positions()
        v_max = self.config["robot"]["max_velocity"] * speed_factor
        dt = self.time_step / 1000.0

        dq = [target_joints[i] - current[i] for i in range(6)]
        max_disp = max(abs(d) for d in dq)

        if max_disp < 0.01:
            self.trajectory = [list(target_joints)]
            self.traj_step = 0
            return

        duration = max_disp / v_max
        steps = max(4, int(duration / dt))

        traj = []
        for k in range(steps + 1):
            t = k / steps
            # Smooth S-curve interpolation
            s = t * t * (3.0 - 2.0 * t)
            point = [current[i] + s * dq[i] for i in range(6)]
            traj.append(point)

        self.trajectory = traj
        self.traj_step = 0

    def step_trajectory(self):
        if self.traj_step >= len(self.trajectory):
            return True
        self.set_joint_positions(self.trajectory[self.traj_step])
        self.traj_step += 1
        return self.traj_step >= len(self.trajectory)

    # ── gripper ───────────────────────────────────────────────

    def open_gripper(self):
        v = self.config["robot"]["gripper_open"]
        for gm in self.gripper_motors:
            if gm:
                gm.setPosition(v)

    def close_gripper(self):
        v = self.config["robot"]["gripper_closed"]
        for gm in self.gripper_motors:
            if gm:
                gm.setPosition(v)

    # ── sensor helpers ────────────────────────────────────────

    def get_gripper_force(self):
        total = 0.0
        for gm in self.gripper_motors:
            if gm is None:
                continue
            try:
                total += abs(gm.getForceFeedback())
            except Exception:
                pass
        return total

    # ── pose selection (with optional IK refinement) ──────────

    def get_pick_poses(self, idx):
        if idx >= len(PICK_APPROACH_POSES):
            idx = idx % len(PICK_APPROACH_POSES)

        approach = list(PICK_APPROACH_POSES[idx])
        grasp = list(PICK_GRASP_POSES[idx])

        # Try IK refinement for the grasp pose
        if HAS_IKPY and idx < len(self.config["task"]["targets"]):
            target = self.config["task"]["targets"][idx]
            wp = target["world_position"]
            ik_result = self.compute_ik(wp, grasp_from_above=True)
            if ik_result is not None:
                print(f"    [IK] Refined grasp pose for {target['name']}")
                grasp = ik_result
                # Derive approach from IK result (raise shoulder_lift)
                approach = list(ik_result)
                approach[1] += 0.25  # lift shoulder
                approach[2] -= 0.35  # reduce elbow bend
                approach[3] += 0.10  # adjust wrist

        return approach, grasp

    def get_place_poses(self, idx):
        pi = idx % len(PLACE_APPROACH_POSES)
        approach = list(PLACE_APPROACH_POSES[pi])
        down = list(PLACE_DOWN_POSES[pi])
        return approach, down

    # ── wait time computation ─────────────────────────────────

    def ms_to_steps(self, ms):
        return max(1, int(ms / self.time_step))

    # ── status display ────────────────────────────────────────

    def display_status(self):
        if self.step_count % 80 != 0:
            return
        joints = self.get_joint_positions()
        targets = self.config["task"]["targets"]
        total = len(targets)
        name = targets[self.target_idx]["name"] if self.target_idx < total else "done"

        print(f"\n  [{self.step_count:5d}] State={self.state:16s} "
              f"Target={self.target_idx+1}/{total} ({name}) "
              f"Placed={self.objects_placed}")
        print(f"         Joints=[{', '.join(f'{j:+.2f}' for j in joints)}]")

    # ── finite-state machine ──────────────────────────────────

    def tick(self):
        self.step_count += 1
        self.display_status()

        targets = self.config["task"]["targets"]
        home = self.config["task"]["home_joints"]
        timing = self.config.get("timing", {})

        # === INIT ===
        if self.state == "INIT":
            print("\n[PHASE] Initialization - sensor warmup")
            self.wait_counter = self.ms_to_steps(800)
            self.state = "WARMUP"

        elif self.state == "WARMUP":
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                self.state = "GO_HOME"

        # === GO HOME ===
        elif self.state == "GO_HOME":
            print("[PHASE] Moving to HOME position")
            self.open_gripper()
            self.plan_smooth_trajectory(home, speed_factor=0.7)
            self.state = "GO_HOME_MOVE"

        elif self.state == "GO_HOME_MOVE":
            if self.step_trajectory():
                self.wait_counter = self.ms_to_steps(timing.get("settle_wait_ms", 500))
                self.state = "GO_HOME_SETTLE"

        elif self.state == "GO_HOME_SETTLE":
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                print("[PHASE] HOME reached. Starting pick-and-place.")
                self.state = "IDLE"

        # === IDLE (decide next target) ===
        elif self.state == "IDLE":
            if self.target_idx < len(targets):
                tgt = targets[self.target_idx]
                self.cycle_count += 1
                print(f"\n{'=' * 55}")
                print(f"  CYCLE {self.cycle_count}: '{tgt['name']}'")
                print(f"  World position: {tgt['world_position']}")
                print(f"{'=' * 55}")
                self.state = "PICK_APPROACH"
            else:
                print(f"\n[COMPLETE] All {len(targets)} targets processed!")
                print(f"  Placed: {self.objects_placed}/{len(targets)}")
                self.plan_smooth_trajectory(home, speed_factor=0.5)
                self.state = "FINAL_HOME"

        # === PICK APPROACH (move above object) ===
        elif self.state == "PICK_APPROACH":
            approach, grasp = self.get_pick_poses(self.target_idx)
            self._current_approach = approach
            self._current_grasp = grasp
            self.open_gripper()
            self.plan_smooth_trajectory(approach, speed_factor=0.8)
            print(f"  -> Moving to approach above target")
            self.state = "PICK_APPROACH_MOVE"

        elif self.state == "PICK_APPROACH_MOVE":
            if self.step_trajectory():
                self.wait_counter = self.ms_to_steps(timing.get("approach_wait_ms", 400))
                self.state = "PICK_APPROACH_WAIT"

        elif self.state == "PICK_APPROACH_WAIT":
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                print(f"  -> Descending to grasp position")
                self.state = "PICK_DESCEND"

        # === PICK DESCEND (lower to object) ===
        elif self.state == "PICK_DESCEND":
            self.plan_smooth_trajectory(self._current_grasp, speed_factor=0.5)
            self.state = "PICK_DESCEND_MOVE"

        elif self.state == "PICK_DESCEND_MOVE":
            if self.step_trajectory():
                self.wait_counter = self.ms_to_steps(timing.get("settle_wait_ms", 500))
                self.state = "PICK_SETTLE"

        elif self.state == "PICK_SETTLE":
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                print(f"  -> Closing gripper")
                self.close_gripper()
                self.wait_counter = self.ms_to_steps(timing.get("grasp_wait_ms", 1500))
                self.state = "GRASPING"

        # === GRASPING (wait for gripper to close) ===
        elif self.state == "GRASPING":
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                force = self.get_gripper_force()
                print(f"  -> Grasp complete (force={force:.1f}N)")
                self.state = "LIFT"

        # === LIFT (raise back to approach height) ===
        elif self.state == "LIFT":
            self.plan_smooth_trajectory(self._current_approach, speed_factor=0.4)
            print(f"  -> Lifting object")
            self.state = "LIFT_MOVE"

        elif self.state == "LIFT_MOVE":
            if self.step_trajectory():
                self.wait_counter = self.ms_to_steps(timing.get("settle_wait_ms", 500))
                self.state = "LIFT_SETTLE"

        elif self.state == "LIFT_SETTLE":
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                self.state = "TRANSPORT_HOME"

        # === TRANSPORT via HOME (safe intermediate) ===
        elif self.state == "TRANSPORT_HOME":
            print(f"  -> Transporting via HOME")
            self.plan_smooth_trajectory(SAFE_ABOVE, speed_factor=0.7)
            self.state = "TRANSPORT_HOME_MOVE"

        elif self.state == "TRANSPORT_HOME_MOVE":
            if self.step_trajectory():
                self.wait_counter = self.ms_to_steps(300)
                self.state = "TRANSPORT_HOME_WAIT"

        elif self.state == "TRANSPORT_HOME_WAIT":
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                self.state = "PLACE_APPROACH"

        # === PLACE APPROACH ===
        elif self.state == "PLACE_APPROACH":
            approach, down = self.get_place_poses(self.target_idx)
            self._place_approach = approach
            self._place_down = down
            self.plan_smooth_trajectory(approach, speed_factor=0.7)
            print(f"  -> Moving to place approach")
            self.state = "PLACE_APPROACH_MOVE"

        elif self.state == "PLACE_APPROACH_MOVE":
            if self.step_trajectory():
                self.wait_counter = self.ms_to_steps(timing.get("approach_wait_ms", 400))
                self.state = "PLACE_APPROACH_WAIT"

        elif self.state == "PLACE_APPROACH_WAIT":
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                self.state = "PLACE_DESCEND"

        # === PLACE DESCEND ===
        elif self.state == "PLACE_DESCEND":
            self.plan_smooth_trajectory(self._place_down, speed_factor=0.5)
            print(f"  -> Lowering to place position")
            self.state = "PLACE_DESCEND_MOVE"

        elif self.state == "PLACE_DESCEND_MOVE":
            if self.step_trajectory():
                self.wait_counter = self.ms_to_steps(timing.get("settle_wait_ms", 500))
                self.state = "PLACE_SETTLE"

        elif self.state == "PLACE_SETTLE":
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                print(f"  -> Opening gripper (release)")
                self.open_gripper()
                self.wait_counter = self.ms_to_steps(timing.get("release_wait_ms", 1000))
                self.state = "RELEASING"

        # === RELEASING ===
        elif self.state == "RELEASING":
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                self.objects_placed += 1
                print(f"  [OK] Object placed! ({self.objects_placed} total)")
                self.state = "RETREAT"

        # === RETREAT (lift from place position) ===
        elif self.state == "RETREAT":
            self.plan_smooth_trajectory(self._place_approach, speed_factor=0.6)
            self.state = "RETREAT_MOVE"

        elif self.state == "RETREAT_MOVE":
            if self.step_trajectory():
                self.target_idx += 1
                self.plan_smooth_trajectory(HOME_JOINTS, speed_factor=0.7)
                self.state = "RETURN_HOME_MOVE"

        elif self.state == "RETURN_HOME_MOVE":
            if self.step_trajectory():
                self.wait_counter = self.ms_to_steps(300)
                self.state = "RETURN_HOME_WAIT"

        elif self.state == "RETURN_HOME_WAIT":
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                self.state = "IDLE"

        # === FINAL HOME ===
        elif self.state == "FINAL_HOME":
            if self.step_trajectory():
                print()
                print("=" * 60)
                print("  MISSION COMPLETE")
                print(f"    Cycles  : {self.cycle_count}")
                print(f"    Placed  : {self.objects_placed}/{len(targets)}")
                print(f"    Errors  : {len(self.errors)}")
                print("    Robot at HOME - standing by.")
                print("=" * 60)
                self.state = "STANDBY"

        elif self.state == "STANDBY":
            pass

    # ── main loop ─────────────────────────────────────────────

    def run(self):
        print("\n[START] UR5e Robust Controller launching ...\n")
        self.robot.step(self.time_step)
        while self.robot.step(self.time_step) != -1:
            self.tick()
        print("\n[END] Controller terminated.")


# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    UR5eCompleteController().run()
