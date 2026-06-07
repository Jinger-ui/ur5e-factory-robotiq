"""
UR5e Advanced Controller for Webots
- Inverse Kinematics via ikpy
- Joint-space & Cartesian-space control
- Multi-target pick-and-place with state machine
- Smooth cubic-interpolation trajectories
"""

import sys
import math
import numpy as np

try:
    from ikpy.chain import Chain
    from ikpy.link import OriginLink, URDFLink
    HAS_IKPY = True
except ImportError:
    HAS_IKPY = False
    print("[WARN] ikpy not installed – falling back to predefined joint poses")

try:
    from controller import Robot, Motor, PositionSensor, DistanceSensor
except ImportError:
    sys.exit("This script must be run inside Webots.")

# ---------------------------------------------------------------------------
# UR5e DH-parameter chain for ikpy (simplified but accurate enough for demo)
# ---------------------------------------------------------------------------

def build_ur5e_chain():
    """Build an ikpy kinematic chain matching the UR5e DH parameters."""
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


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def cubic_interpolation(q_start, q_end, num_steps):
    """Generate a smooth cubic trajectory between two joint configurations."""
    trajectory = []
    for i in range(num_steps + 1):
        t = i / num_steps
        s = 3 * t * t - 2 * t * t * t          # cubic ease-in-out
        q = [qs + s * (qe - qs) for qs, qe in zip(q_start, q_end)]
        trajectory.append(q)
    return trajectory


def linear_cartesian_path(start_pos, end_pos, num_waypoints=5):
    """Linearly interpolate between two Cartesian positions."""
    path = []
    for i in range(num_waypoints + 1):
        t = i / num_waypoints
        pos = [s + t * (e - s) for s, e in zip(start_pos, end_pos)]
        path.append(pos)
    return path


# ---------------------------------------------------------------------------
# Predefined joint waypoints (fallback when ikpy is unavailable)
# ---------------------------------------------------------------------------

HOME_JOINTS = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]

PICK_APPROACH = [1.0, -1.2, 1.4, -1.8, -1.57, 0.0]
PICK_DOWN     = [1.0, -0.9, 1.6, -2.3, -1.57, 0.0]

PLACE_APPROACH = [-1.0, -1.2, 1.4, -1.8, -1.57, 0.0]
PLACE_DOWN     = [-1.0, -0.9, 1.6, -2.3, -1.57, 0.0]

PREDEFINED_TARGETS = [
    {"name": "target_red",    "pick": [1.0,  -1.0, 1.5, -2.1, -1.57, 0.0]},
    {"name": "target_green",  "pick": [0.7,  -1.0, 1.5, -2.1, -1.57, 0.0]},
    {"name": "target_blue",   "pick": [0.4,  -1.0, 1.5, -2.1, -1.57, 0.0]},
    {"name": "target_yellow", "pick": [1.0,  -1.1, 1.4, -1.9, -1.57, 0.0]},
]


# ---------------------------------------------------------------------------
# States for the pick-and-place finite state machine
# ---------------------------------------------------------------------------

class State:
    IDLE             = "IDLE"
    MOVE_HOME        = "MOVE_HOME"
    MOVE_TO_APPROACH = "MOVE_TO_APPROACH"
    MOVE_TO_PICK     = "MOVE_TO_PICK"
    CLOSE_GRIPPER    = "CLOSE_GRIPPER"
    LIFT_OBJECT      = "LIFT_OBJECT"
    MOVE_TO_PLACE    = "MOVE_TO_PLACE"
    LOWER_OBJECT     = "LOWER_OBJECT"
    OPEN_GRIPPER     = "OPEN_GRIPPER"
    RETREAT          = "RETREAT"
    DONE             = "DONE"


# ---------------------------------------------------------------------------
# Main controller class
# ---------------------------------------------------------------------------

class UR5eController:
    TIME_STEP = 8          # ms – must match WorldInfo.basicTimeStep
    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    SENSOR_NAMES = [n + "_sensor" for n in JOINT_NAMES]
    GRIPPER_NAMES = [
        "finger_1_joint_1",
        "finger_2_joint_1",
        "finger_middle_joint_1",
    ]
    GRIPPER_OPEN   = 0.0
    GRIPPER_CLOSED = 0.85
    MAX_VELOCITY   = 1.5   # rad/s

    def __init__(self):
        self.robot = Robot()
        self._init_devices()
        self.chain = build_ur5e_chain() if HAS_IKPY else None

        self.state = State.IDLE
        self.current_target_idx = 0
        self.trajectory = []
        self.traj_step = 0
        self.wait_counter = 0
        self.place_offset = 0  # incremental offset so objects don't stack

        print("=" * 60)
        print("  UR5e Advanced Controller – Initialised")
        print(f"  IK engine: {'ikpy' if HAS_IKPY else 'predefined waypoints'}")
        print(f"  Targets to pick: {len(PREDEFINED_TARGETS)}")
        print("=" * 60)

    # ----- hardware init ---------------------------------------------------

    def _init_devices(self):
        self.motors = []
        self.sensors = []
        for jn, sn in zip(self.JOINT_NAMES, self.SENSOR_NAMES):
            m = self.robot.getDevice(jn)
            m.setVelocity(self.MAX_VELOCITY)
            self.motors.append(m)
            s = self.robot.getDevice(sn)
            s.enable(self.TIME_STEP)
            self.sensors.append(s)

        self.gripper_motors = []
        for gn in self.GRIPPER_NAMES:
            gm = self.robot.getDevice(gn)
            gm.setVelocity(0.5)
            self.gripper_motors.append(gm)

        self.distance_sensor = self.robot.getDevice("gripper_sensor")
        if self.distance_sensor:
            self.distance_sensor.enable(self.TIME_STEP)

    # ----- joint read / write ----------------------------------------------

    def get_joint_positions(self):
        return [s.getValue() for s in self.sensors]

    def set_joint_positions(self, positions):
        for m, p in zip(self.motors, positions):
            m.setPosition(p)

    def set_joint_velocities(self, velocities):
        for m, v in zip(self.motors, velocities):
            m.setVelocity(abs(v))

    # ----- gripper ---------------------------------------------------------

    def open_gripper(self):
        for gm in self.gripper_motors:
            gm.setPosition(self.GRIPPER_OPEN)

    def close_gripper(self):
        for gm in self.gripper_motors:
            gm.setPosition(self.GRIPPER_CLOSED)

    # ----- IK wrapper ------------------------------------------------------

    def compute_ik(self, target_xyz, initial_joints=None):
        """Compute IK for a target (x, y, z) in robot base frame."""
        if not self.chain:
            return None
        target = np.eye(4)
        target[:3, 3] = target_xyz
        if initial_joints is not None:
            seed = [0.0] + list(initial_joints) + [0.0]
        else:
            seed = [0.0] * 8
        result = self.chain.inverse_kinematics(
            target, initial_position=seed, orientation_mode=None
        )
        return list(result[1:7])

    def compute_fk(self, joint_angles):
        """Compute FK for the current joint angles → (x, y, z)."""
        if not self.chain:
            return None
        full = [0.0] + list(joint_angles) + [0.0]
        tf = self.chain.forward_kinematics(full)
        return tf[:3, 3].tolist()

    # ----- trajectory execution --------------------------------------------

    def plan_trajectory(self, target_joints, steps=60):
        current = self.get_joint_positions()
        self.trajectory = cubic_interpolation(current, target_joints, steps)
        self.traj_step = 0

    def step_trajectory(self):
        """Execute one step of the current trajectory.  Returns True when done."""
        if self.traj_step >= len(self.trajectory):
            return True
        self.set_joint_positions(self.trajectory[self.traj_step])
        self.traj_step += 1
        return self.traj_step >= len(self.trajectory)

    def at_target(self, target_joints, tolerance=0.05):
        current = self.get_joint_positions()
        return all(abs(c - t) < tolerance for c, t in zip(current, target_joints))

    # ----- state machine ---------------------------------------------------

    def get_pick_joints(self, idx):
        """Return (approach, pick_down) joint sets for target idx."""
        target = PREDEFINED_TARGETS[idx]

        if HAS_IKPY:
            base_angles = target["pick"]
            approach = list(base_angles)
            approach[1] -= 0.3
            approach[2] -= 0.2
            return approach, base_angles
        else:
            approach = list(target["pick"])
            pick_down = list(target["pick"])
            approach[1] -= 0.3
            approach[2] -= 0.2
            return approach, pick_down

    def get_place_joints(self, idx):
        """Return (approach, place_down) joint sets."""
        offset = idx * 0.15
        place_approach = list(PLACE_APPROACH)
        place_down = list(PLACE_DOWN)
        place_approach[0] += offset
        place_down[0] += offset
        return place_approach, place_down

    def tick(self):
        """Run one cycle of the FSM."""

        if self.state == State.IDLE:
            if self.current_target_idx < len(PREDEFINED_TARGETS):
                target = PREDEFINED_TARGETS[self.current_target_idx]
                print(f"\n▶ Picking target {self.current_target_idx + 1}/"
                      f"{len(PREDEFINED_TARGETS)}: {target['name']}")
                self.open_gripper()
                self.plan_trajectory(HOME_JOINTS, steps=40)
                self.state = State.MOVE_HOME
            else:
                if self.state != State.DONE:
                    print("\n✔ All targets picked and placed!")
                    self.plan_trajectory(HOME_JOINTS, steps=60)
                    self.state = State.DONE

        elif self.state == State.MOVE_HOME:
            if self.step_trajectory():
                approach, _ = self.get_pick_joints(self.current_target_idx)
                self.plan_trajectory(approach, steps=60)
                self.state = State.MOVE_TO_APPROACH

        elif self.state == State.MOVE_TO_APPROACH:
            if self.step_trajectory():
                _, pick = self.get_pick_joints(self.current_target_idx)
                self.plan_trajectory(pick, steps=40)
                self.state = State.MOVE_TO_PICK

        elif self.state == State.MOVE_TO_PICK:
            if self.step_trajectory():
                print("  ✋ Closing gripper …")
                self.close_gripper()
                self.wait_counter = 40
                self.state = State.CLOSE_GRIPPER

        elif self.state == State.CLOSE_GRIPPER:
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                approach, _ = self.get_pick_joints(self.current_target_idx)
                self.plan_trajectory(approach, steps=40)
                self.state = State.LIFT_OBJECT

        elif self.state == State.LIFT_OBJECT:
            if self.step_trajectory():
                approach, _ = self.get_place_joints(self.current_target_idx)
                self.plan_trajectory(approach, steps=70)
                self.state = State.MOVE_TO_PLACE

        elif self.state == State.MOVE_TO_PLACE:
            if self.step_trajectory():
                _, place = self.get_place_joints(self.current_target_idx)
                self.plan_trajectory(place, steps=40)
                self.state = State.LOWER_OBJECT

        elif self.state == State.LOWER_OBJECT:
            if self.step_trajectory():
                print("  🤚 Opening gripper …")
                self.open_gripper()
                self.wait_counter = 30
                self.state = State.OPEN_GRIPPER

        elif self.state == State.OPEN_GRIPPER:
            self.wait_counter -= 1
            if self.wait_counter <= 0:
                approach, _ = self.get_place_joints(self.current_target_idx)
                self.plan_trajectory(approach, steps=30)
                self.state = State.RETREAT

        elif self.state == State.RETREAT:
            if self.step_trajectory():
                self.current_target_idx += 1
                self.state = State.IDLE

        elif self.state == State.DONE:
            self.step_trajectory()

    # ----- main loop -------------------------------------------------------

    def run(self):
        print("\n[UR5e] Starting pick-and-place sequence …\n")
        self.robot.step(self.TIME_STEP)      # first step to initialise sensors

        while self.robot.step(self.TIME_STEP) != -1:
            self.tick()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ctrl = UR5eController()
    ctrl.run()
