"""Robot skills (Bai 03: real gripper + camera).

Every skill is a short deterministic sequence and returns an Outcome
(Status + detail).  All geometry (heights, approach distances, gripper yaw)
lives here and in scene.yaml - never in the LLM output.

Two abstract layers, so the same skills run in Gazebo and in the unit tests:
  MotionBackend : MoveIt 2 motions + the gripper controller + MoveIt planning
                  scene (moveit_backend.py  |  fake_backend.py)
  Perception    : camera -> VisionResult  (perception.py | fake_backend.py)

Sensing skills  : detect_objects, check_zone, find_object, find_free_position
Motion skills   : home, pick, place, move_above, move_to_zone
Gripper (internal, used by pick/place): open_gripper, close_gripper
"""
import math

import numpy as np

from .scene_model import TEMP, SceneConfig, WorldState
from .skill_spec import Outcome, Status
from .ur_kinematics import URKinematics, wrap


class MotionBackend:
    """Interface implemented by MoveItBackend and FakeMotionBackend."""

    def current_joints(self):                     # -> [6] or None
        raise NotImplementedError

    def move_joints(self, joints):                # plan (collision-aware) + execute -> Status
        raise NotImplementedError

    def move_vertical(self, z):                   # straight Cartesian line, tool keeps its pose
        raise NotImplementedError

    def gripper_open(self):                       # -> Status
        raise NotImplementedError

    def gripper_close(self):                      # -> (Status, gap between the fingers [m])
        raise NotImplementedError

    def gripper_gap(self):                        # -> float or None
        raise NotImplementedError

    # MoveIt planning scene (the fake backend just records the calls)
    def attach(self, obj):
        raise NotImplementedError

    def detach(self, obj):
        raise NotImplementedError

    def scene_remove(self, obj):
        raise NotImplementedError

    def scene_add_cube(self, obj, x, y, yaw):
        raise NotImplementedError

    def sync_objects(self, world):
        raise NotImplementedError


class Perception:
    def observe(self):                            # -> vision.VisionResult
        raise NotImplementedError


class RobotSkills:
    def __init__(self, scene: SceneConfig, backend: MotionBackend, perception: Perception,
                 world: WorldState, log=print):
        self.scene = scene
        self.backend = backend
        self.perception = perception
        self.world = world
        self.log = log
        self.m = scene.motion
        self.g = scene.gripper
        self.kin = URKinematics(scene.ur_type)
        self.free_spots = {}          # object -> (x, y) chosen by find_free_position
        self.last_vision = None

    # ------------------------------------------------------------ helpers
    def _at_home(self, tol=0.03):
        q = self.backend.current_joints()
        if q is None:
            return False
        return float(np.max(np.abs(np.asarray(q) - np.asarray(self.m['home_joints'])))) < tol

    def _ensure_observation_pose(self):
        """The camera is fixed above the table; the arm must be at home (out of
        its view) before an image is used."""
        if self._at_home():
            return Outcome(Status.SUCCESS)
        st = self.home()
        return st

    def _move_above_xy(self, x, y, yaw_ref):
        """Tool above (x, y) at approach height, gripper yaw = yaw_ref + k*90deg
        (the k that needs the least joint motion)."""
        seed = self.backend.current_joints()
        yaw, q = self.kin.best_grasp_yaw((x, y, self.scene.approach_tool_z()), yaw_ref, seed)
        if q is None:
            return Outcome(Status.PLANNING_FAILED, f'({x:.3f}, {y:.3f}) is out of reach'), None
        st = self.backend.move_joints(list(q))
        return Outcome(st), yaw

    def reachable(self, xy):
        """IK check of the approach and the contact pose over xy (gripper
        aligned with the table axes), used by find_free_position."""
        seed = self.m['home_joints']
        yaw, q = self.kin.best_grasp_yaw((xy[0], xy[1], self.scene.approach_tool_z()), 0.0, seed)
        if q is None:
            return False
        return self.kin.ik((xy[0], xy[1], self.scene.grasp_tool_z()), yaw, q) is not None

    def _vision_update(self):
        res = self.perception.observe()
        self.last_vision = res
        dets = {n: (d.x, d.y, d.yaw) for n, d in res.detections.items()}
        self.world.set_detections(dets)
        self.backend.sync_objects(self.world)
        for w in res.warnings:
            self.log(f'[camera] {w}')
        return res

    # ------------------------------------------------------------ sensing
    def detect_objects(self):
        st = self._ensure_observation_pose()
        if not st.ok:
            return st
        try:
            res = self._vision_update()
        except Exception as e:           # camera not publishing, TF missing ...
            return Outcome(Status.FAILED, f'camera: {e}')
        n = len(res.detections)
        return Outcome(Status.SUCCESS, f'{n}/{len(self.scene.objects)} blocks detected')

    def check_zone(self, zone):
        if zone not in self.scene.zones:
            return Outcome(Status.INVALID_ZONE, zone)
        st = self.detect_objects()
        if not st.ok:
            return st
        occ = self.world.occupant(zone)
        return Outcome(Status.SUCCESS, 'FREE' if occ is None else f'OCCUPIED by {occ}')

    def find_object(self, obj):
        if obj not in self.scene.objects:
            return Outcome(Status.INVALID_OBJECT, obj)
        if obj == self.world.held:
            return Outcome(Status.SUCCESS, 'in the gripper')
        st = self.detect_objects()
        if not st.ok:
            return st
        if not self.world.visible(obj):
            return Outcome(Status.NOT_FOUND, 'not seen by the camera')
        z = self.world.slot_of(obj)
        x, y = self.world.positions[obj]
        return Outcome(Status.SUCCESS, z if z else f'table ({x:.3f}, {y:.3f})')

    def find_free_position(self, obj, xy=None, src=None, avoid=()):
        """Choose (and remember) a temporary spot for obj.  `xy` may be given by
        the executor, which already resolved it while expanding the plan; it is
        re-checked against the current world state."""
        if obj not in self.scene.objects:
            return Outcome(Status.INVALID_OBJECT, obj)
        if xy is not None and not self._spot_is_free(obj, xy):
            xy = None                              # world changed: choose again
        if xy is None:
            xy = self.world.free_position(obj, reachable=self.reachable, avoid=avoid, src=src)
        if xy is None:
            return Outcome(Status.FAILED, 'no free, reachable spot on the table')
        self.free_spots[obj] = tuple(xy)
        return Outcome(Status.SUCCESS, f'({xy[0]:.3f}, {xy[1]:.3f})')

    def _spot_is_free(self, obj, xy):
        min_d = float(self.scene.free_space.get('min_object_distance', 0.10))
        for o, (x, y) in self.world.positions.items():
            if o not in (obj, self.world.held) and math.hypot(x - xy[0], y - xy[1]) < min_d:
                return False
        return True

    # ------------------------------------------------------------ gripper
    def open_gripper(self):
        return Outcome(self.backend.gripper_open())

    def close_gripper(self):
        """Close on a block; SUCCESS only if the fingers stopped at the block's
        width (i.e. a block is really between them)."""
        st, gap = self.backend.gripper_close()
        if st != Status.SUCCESS:
            return Outcome(st), gap
        tol = float(self.g.get('width_tolerance', 0.008))
        if gap is None or gap < self.scene.cube_size - tol:
            return Outcome(Status.GRASP_FAILED,
                           f'fingers closed to {100 * (gap or 0):.1f} cm: nothing grasped'), gap
        return Outcome(Status.SUCCESS, f'grip {100 * gap:.1f} cm'), gap

    # ------------------------------------------------------------ motion
    def home(self):
        return Outcome(self.backend.move_joints(list(self.m['home_joints'])))

    def move_above(self, obj):
        if obj not in self.scene.objects:
            return Outcome(Status.INVALID_OBJECT, obj)
        if obj == self.world.held or not self.world.visible(obj):
            return Outcome(Status.INVALID_STATE, f'{obj} is not visible on the table')
        x, y = self.world.positions[obj]
        st, _ = self._move_above_xy(x, y, self.world.yaws.get(obj, 0.0))
        return st

    def move_to_zone(self, zone):
        if zone not in self.scene.zones:
            return Outcome(Status.INVALID_ZONE, zone)
        x, y = self.scene.zone_xy(zone)
        st, _ = self._move_above_xy(x, y, 0.0)
        return st

    def pick(self, obj):
        if obj not in self.scene.objects:
            return Outcome(Status.INVALID_OBJECT, obj)
        if self.world.held is not None:
            return Outcome(Status.INVALID_STATE, f'gripper already holds {self.world.held}')
        if not self.world.visible(obj):
            return Outcome(Status.INVALID_STATE, f'{obj} was not detected by the camera')
        x, y = self.world.positions[obj]
        st = self.open_gripper()
        if not st.ok:
            return Outcome(st.status, 'could not open the gripper')
        st, yaw = self._move_above_xy(x, y, self.world.yaws.get(obj, 0.0))
        if not st.ok:
            return st
        # the open fingers will surround the block: take it out of MoveIt's
        # world for the straight descent (it is attached to the gripper next)
        self.backend.scene_remove(obj)
        st = Outcome(self.backend.move_vertical(self.scene.grasp_tool_z()))
        if not st.ok:
            self.backend.scene_add_cube(obj, x, y, self.world.yaws.get(obj, 0.0))
            return Outcome(st.status, 'descent to the block failed')
        grip, gap = self.close_gripper()
        if not grip.ok:
            self.backend.gripper_open()
            self.backend.move_vertical(self.scene.approach_tool_z())
            self.backend.scene_add_cube(obj, x, y, self.world.yaws.get(obj, 0.0))
            return grip
        self.backend.attach(obj)
        self.world.held = obj
        self.world.positions.pop(obj, None)
        st = Outcome(self.backend.move_vertical(self.scene.approach_tool_z()))
        if not st.ok:
            return Outcome(st.status, 'lift failed (block still in the gripper)')
        gap = self.backend.gripper_gap()
        tol = float(self.g.get('width_tolerance', 0.008))
        if gap is None or gap < self.scene.cube_size - tol:
            self.backend.detach(obj)
            self.world.held = None
            return Outcome(Status.GRASP_FAILED, f'{obj} slipped out of the gripper')
        return grip

    def place(self, obj, target, xy=None):
        """target = zone name or TEMP (then xy, or the spot chosen by
        find_free_position, is used)."""
        if obj not in self.scene.objects:
            return Outcome(Status.INVALID_OBJECT, obj)
        if self.world.held != obj:
            return Outcome(Status.INVALID_STATE,
                           f'gripper holds {self.world.held or "nothing"}, not {obj}')
        if target == TEMP:
            xy = xy or self.free_spots.get(obj)
            if xy is None:
                st = self.find_free_position(obj)
                if not st.ok:
                    return st
                xy = self.free_spots[obj]
            if not self._spot_is_free(obj, xy):
                return Outcome(Status.INVALID_STATE, f'temporary spot {xy} is no longer free')
            x, y = xy
        elif target in self.scene.zones:
            occ = self.world.occupant(target, exclude=obj)
            if occ is not None:          # the executor clears zones before picking
                return Outcome(Status.INVALID_STATE, f'{target} is occupied by {occ}')
            x, y = self.scene.zone_xy(target)
        else:
            return Outcome(Status.INVALID_ZONE, str(target))
        st, yaw = self._move_above_xy(x, y, 0.0)          # block square to the table axes
        if not st.ok:
            return st
        st = Outcome(self.backend.move_vertical(self.scene.place_tool_z()))
        if not st.ok:
            return Outcome(st.status, 'descent to the place pose failed')
        st = self.open_gripper()
        if not st.ok:
            return Outcome(st.status, 'gripper did not open')
        self.backend.detach(obj)
        self.world.held = None
        self.world.positions[obj] = (x, y)
        self.world.yaws[obj] = wrap(yaw)
        self.free_spots.pop(obj, None)
        st = Outcome(self.backend.move_vertical(self.scene.approach_tool_z()))
        self.backend.scene_add_cube(obj, x, y, yaw)
        if not st.ok:
            return Outcome(st.status, 'retreat after release failed')
        where = target if target != TEMP else f'({x:.3f}, {y:.3f})'
        return Outcome(Status.SUCCESS, f'released at {where}')
