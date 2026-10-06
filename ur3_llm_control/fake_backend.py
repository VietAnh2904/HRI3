"""Pure-python stand-in for Gazebo + MoveIt + the camera (no ROS).

FakeWorld          ground truth: where every cube really is, where the tool
                   is, how far the fingers are open, which cube is gripped.
FakeMotionBackend  moves the tool with the same UR kinematics (targets must
                   be reachable and inside joint limits); the gripper really
                   has to close around a cube to hold it (position/height/yaw
                   tolerances like a parallel gripper), a held cube moves with
                   the tool, a released cube lands on the table under it.
FakePerception     renders the table with synthetic_camera.render() and runs
                   the REAL vision pipeline (vision.detect_cubes) on it; if the
                   arm is not at home its links occlude the view.

Used by the unit tests and by offline_cli (real LLM, simulated robot).
"""
import math

import numpy as np

from .robot_skills import MotionBackend, Perception
from .skill_spec import Status
from .synthetic_camera import camera_model, render
from .ur_kinematics import URKinematics, tool_yaw
from .vision import detect_cubes


class FakeWorld:
    def __init__(self, scene, positions=None, yaws=None):
        self.scene = scene
        self.cubes = {n: tuple(o['spawn_xy']) for n, o in scene.objects.items()}
        if positions is not None:
            self.cubes = {n: tuple(p) for n, p in positions.items()}
        self.yaws = dict(yaws or {})
        self.held = None
        self.gap = 2 * float(scene.gripper.get('max_opening', 0.045))
        self.joints = list(scene.motion['home_joints'])


class FakeMotionBackend(MotionBackend):
    def __init__(self, scene, world=None, fail_on=None):
        self.scene = scene
        self.truth = world or FakeWorld(scene)
        self.kin = URKinematics(scene.ur_type)
        self.calls = []
        self.fail_on = set(fail_on or ())   # e.g. {'move_joints'} to inject failures
        self.scene_objects = {}

    # ---------------------------------------------------------- state
    def current_joints(self):
        return list(self.truth.joints)

    def _tool(self):
        t = self.kin.fk(self.truth.joints)
        return t[:3, 3], tool_yaw(t[:3, :3]), t[2, 2]

    def _tcp(self):
        p, yaw, _ = self._tool()
        return np.array([p[0], p[1], p[2] - self.scene.tcp_offset]), yaw

    def _carry(self):
        if self.truth.held:
            tcp, yaw = self._tcp()
            self.truth.cubes[self.truth.held] = (float(tcp[0]), float(tcp[1]))
            self.truth.yaws[self.truth.held] = yaw

    def _tcp_hits_something(self):
        """True if the fingers (tips at tcp - 0.015 m) would be inside a cube
        that is not the held one: a collision MoveIt would have rejected."""
        tcp, _ = self._tcp()
        tip_z = tcp[2] - 0.015
        top = self.scene.table_top + self.scene.cube_size
        if tip_z > top:
            return False
        for n, (x, y) in self.truth.cubes.items():
            if n == self.truth.held:
                continue
            if math.hypot(x - tcp[0], y - tcp[1]) < 0.012 and self.truth.gap < self.scene.cube_size:
                return True
        return False

    # ---------------------------------------------------------- motion
    def move_joints(self, joints):
        self.calls.append(('move_joints', tuple(round(float(j), 4) for j in joints)))
        if 'move_joints' in self.fail_on:
            return Status.PLANNING_FAILED
        q = np.asarray(joints, dtype=float)
        if np.any(np.abs(q) > 2 * math.pi + 1e-6) or abs(q[2]) > math.pi + 1e-6:
            return Status.PLANNING_FAILED
        self.truth.joints = list(q)
        self._carry()
        return Status.SUCCESS

    def move_vertical(self, z):
        p, yaw, zz = self._tool()
        self.calls.append(('move_vertical', round(float(z), 4)))
        if 'move_vertical' in self.fail_on:
            return Status.PLANNING_FAILED
        q = self.kin.ik((p[0], p[1], z), yaw, self.truth.joints)
        if q is None:
            return Status.PLANNING_FAILED
        self.truth.joints = list(q)
        self._carry()
        if self._tcp_hits_something():
            return Status.EXECUTION_FAILED
        return Status.SUCCESS

    # ---------------------------------------------------------- gripper
    def gripper_open(self):
        self.calls.append(('gripper_open',))
        if self.truth.held:
            tcp, yaw = self._tcp()
            obj = self.truth.held
            self.truth.held = None
            bottom = tcp[2] - self.scene.cube_size / 2
            if bottom - self.scene.table_top > 0.03:
                # dropped from high up: lands somewhere near
                self.truth.cubes[obj] = (float(tcp[0]) + 0.01, float(tcp[1]) - 0.01)
            else:
                self.truth.cubes[obj] = (float(tcp[0]), float(tcp[1]))
            self.truth.yaws[obj] = yaw
        self.truth.gap = 2 * float(self.scene.gripper.get('max_opening', 0.045))
        return Status.SUCCESS

    def gripper_close(self):
        self.calls.append(('gripper_close',))
        if 'gripper_close' in self.fail_on:
            self.truth.gap = 0.0
            return Status.SUCCESS, 0.0
        tcp, yaw = self._tcp()
        s = self.scene.cube_size
        for n, (x, y) in self.truth.cubes.items():
            dxy = math.hypot(x - tcp[0], y - tcp[1])
            dz = abs(self.scene.cube_center_z() - tcp[2])
            dyaw = abs(((yaw - self.truth.yaws.get(n, 0.0)) + math.pi / 4) % (math.pi / 2)
                       - math.pi / 4)
            if dxy < 0.015 and dz < 0.012 and dyaw < math.radians(12):
                self.truth.held = n
                self.truth.gap = s
                self.truth.cubes[n] = (float(tcp[0]), float(tcp[1]))
                self.truth.yaws[n] = yaw
                return Status.SUCCESS, s
        self.truth.gap = 0.0
        return Status.SUCCESS, 0.0

    def gripper_gap(self):
        return self.truth.gap

    # ---------------------------------------------------------- planning scene
    def attach(self, obj):
        self.calls.append(('attach', obj))
        self.scene_objects.pop(obj, None)
        return Status.SUCCESS

    def detach(self, obj):
        self.calls.append(('detach', obj))
        return Status.SUCCESS

    def scene_remove(self, obj):
        self.scene_objects.pop(obj, None)

    def scene_add_cube(self, obj, x, y, yaw):
        self.scene_objects[obj] = (x, y, yaw)

    def sync_objects(self, world):
        self.scene_objects = {o: (p[0], p[1], world.yaws.get(o, 0.0))
                              for o, p in world.positions.items()}


class FakePerception(Perception):
    def __init__(self, scene, truth: FakeWorld, noise=2.0):
        self.scene = scene
        self.truth = truth
        self.noise = noise
        self.K, self.T = camera_model(scene)
        self.kin = URKinematics(scene.ur_type)
        self.rng = np.random.default_rng(1)
        self.last_image = None

    def observe(self):
        visible = {n: p for n, p in self.truth.cubes.items() if n != self.truth.held}
        occluders = []
        home = np.asarray(self.scene.motion['home_joints'])
        if np.max(np.abs(np.asarray(self.truth.joints) - home)) > 0.03:
            for p in self.kin.joint_positions(self.truth.joints)[2:]:
                occluders.append((p[0], p[1], p[2], 0.05))
        img = render(self.scene, visible, self.truth.yaws, self.K, self.T,
                     noise=self.noise, rng=self.rng, occluders=occluders)
        self.last_image = img
        return detect_cubes(img, self.K, self.T, self.scene)
