"""MoveIt 2 motion backend + gripper controller (ROS 2 Humble, rclpy).

Talks to move_group through its standard ROS interfaces (moveit_py is not
released for Humble):
  /move_action              moveit_msgs/action/MoveGroup      plan + execute (OMPL)
  /compute_cartesian_path   moveit_msgs/srv/GetCartesianPath  straight approach/retreat
  /execute_trajectory       moveit_msgs/action/ExecuteTrajectory
  /apply_planning_scene     moveit_msgs/srv/ApplyPlanningScene table, blocks, attach/detach
Gripper (ros2_control, see config/ros2_controllers.yaml):
  /gripper_controller/joint_trajectory  trajectory_msgs/JointTrajectory  finger targets;
      the controller turns the position error into a finger FORCE (effort interface)
  /joint_states                         finger positions -> measured gap

The blocks are NEVER moved by this code: they only move because the gripper
fingers squeeze them (friction) while the arm moves.  MoveIt enforces joint
limits and checks self-collision and collision with the planning scene
(floor, table, the blocks seen by the camera, the held block) for every
motion, including the Cartesian segments (avoid_collisions=True).
"""
import math
import os
import threading
import time

import numpy as np

from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject, Constraints,
                             JointConstraint, MotionPlanRequest, MoveItErrorCodes,
                             PlanningOptions, PlanningScene, PlanningSceneComponents)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPlanningScene
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from builtin_interfaces.msg import Duration as DurationMsg
import tf2_ros
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .robot_skills import MotionBackend
from .skill_spec import Status
from .ur_kinematics import JOINT_NAMES, URKinematics, tool_yaw

_EXEC_ERRORS = {MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE,
                MoveItErrorCodes.CONTROL_FAILED, MoveItErrorCodes.TIMED_OUT,
                MoveItErrorCodes.PREEMPTED}

FINGER_JOINTS = ('left_finger_joint', 'right_finger_joint')
GRIPPER_LINKS = ['gripper_base_link', 'left_finger_link', 'right_finger_link']

# floor collision box (see setup_scene)
FLOOR_SIZE = 2.0
FLOOR_THICKNESS = 0.01
FLOOR_GAP = 0.005


# ------------------------------------------------------------------ helpers
def _quat(x, y, z, w):
    q = Quaternion()
    q.x, q.y, q.z, q.w = float(x), float(y), float(z), float(w)
    return q


def _yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _pose(xyz, q=(0.0, 0.0, 0.0, 1.0)):
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in xyz)
    p.orientation = _quat(*q)
    return p


def _box(size):
    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = [float(s) for s in size]
    return prim


def _set_if(msg, name, value):
    if hasattr(msg, name):
        setattr(msg, name, value)


def _quat_to_rot(x, y, z, w):
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


class MoveItBackend(MotionBackend):
    def __init__(self, node, scene, callback_group=None, tf_buffer=None):
        self.node = node
        self.scene = scene
        self.m = scene.motion
        self.g = scene.gripper
        self.frame = scene.frame_id
        self.ee = scene.ee_link
        self.group = scene.planning_group
        self.log = node.get_logger()

        cg = callback_group
        self.move_client = ActionClient(node, MoveGroup, '/move_action', callback_group=cg)
        self.exec_client = ActionClient(node, ExecuteTrajectory, '/execute_trajectory',
                                        callback_group=cg)
        self.cart_client = node.create_client(GetCartesianPath, '/compute_cartesian_path',
                                              callback_group=cg)
        self.scene_client = node.create_client(ApplyPlanningScene, '/apply_planning_scene',
                                               callback_group=cg)
        self.get_scene_client = node.create_client(GetPlanningScene, '/get_planning_scene',
                                                   callback_group=cg)
        self.gripper_pub = node.create_publisher(JointTrajectory,
                                                 '/gripper_controller/joint_trajectory', 10)
        if tf_buffer is None:
            tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(tf_buffer, node)
        self.tf_buffer = tf_buffer

        self._kin = URKinematics(scene.ur_type)
        self._joint_state = {}
        self._joint_vel = {}
        self._js_lock = threading.Lock()
        node.create_subscription(JointState, '/joint_states', self._on_joint_state,
                                 10, callback_group=cg)
        self._attached = None
        self._world_cubes = set()          # block ids currently in MoveIt's world

    # ---------------------------------------------------------------- infra
    @staticmethod
    def _wait(future, timeout):
        end = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > end:
                return None
            time.sleep(0.01)
        return future.result()

    def wait_for_servers(self, timeout=120.0):
        end = time.monotonic() + timeout
        pending = {'/move_action': self.move_client.server_is_ready,
                   '/execute_trajectory': self.exec_client.server_is_ready,
                   '/compute_cartesian_path': self.cart_client.service_is_ready,
                   '/apply_planning_scene': self.scene_client.service_is_ready,
                   '/gripper_controller/joint_trajectory (gripper_controller)':
                       lambda: self.gripper_pub.get_subscription_count() > 0,
                   '/joint_states (arm + fingers)':
                       lambda: self._joints(JOINT_NAMES + list(FINGER_JOINTS), 0.0) is not None}
        while time.monotonic() < end:
            missing = [n for n, ready in pending.items() if not ready()]
            if not missing:
                break
            time.sleep(0.5)
        else:
            return missing
        while time.monotonic() < end:           # TF world -> tool0 as well
            if self.tool_pose() is not None:
                return []
            time.sleep(0.5)
        return ['TF %s -> %s' % (self.frame, self.ee)]

    def _on_joint_state(self, msg):
        with self._js_lock:
            for i, name in enumerate(msg.name):
                self._joint_state[name] = msg.position[i]
                if i < len(msg.velocity):
                    self._joint_vel[name] = msg.velocity[i]

    def _joints(self, names, timeout=2.0):
        end = time.monotonic() + timeout
        while True:
            with self._js_lock:
                if all(n in self._joint_state for n in names):
                    return [self._joint_state[n] for n in names]
            if time.monotonic() >= end:
                return None
            time.sleep(0.02)

    def current_joints(self):
        return self._joints(JOINT_NAMES)

    def tool_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.frame, self.ee, Time(),
                                                timeout=Duration(seconds=0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        tr, r = t.transform.translation, t.transform.rotation
        return (tr.x, tr.y, tr.z), (r.x, r.y, r.z, r.w)

    def _apply(self, scene_msg):
        req = ApplyPlanningScene.Request()
        scene_msg.is_diff = True
        scene_msg.robot_state.is_diff = True
        req.scene = scene_msg
        res = self._wait(self.scene_client.call_async(req), 10.0)
        return res is not None and res.success

    # ------------------------------------------------------- planning scene
    def _collision_object(self, oid, size, xyz, q=(0, 0, 0, 1), op=CollisionObject.ADD,
                          frame=None):
        co = CollisionObject()
        co.header.frame_id = frame or self.frame
        co.id = oid
        co.pose = _pose(xyz, q)
        co.primitives = [_box(size)]
        co.primitive_poses = [_pose((0, 0, 0))]
        co.operation = op
        return co

    def _existing_objects(self):
        """(world object ids, attached object ids) currently in move_group."""
        if not self.get_scene_client.wait_for_service(timeout_sec=5.0):
            return None, None
        req = GetPlanningScene.Request()
        comp = PlanningSceneComponents
        req.components.components = comp.WORLD_OBJECT_NAMES | comp.ROBOT_STATE_ATTACHED_OBJECTS
        res = self._wait(self.get_scene_client.call_async(req), 10.0)
        if res is None:
            return None, None
        world = {co.id for co in res.scene.world.collision_objects}
        attached = {a.object.id for a in res.scene.robot_state.attached_collision_objects}
        return world, attached

    def setup_scene(self):
        """Floor and table (static); blocks left in MoveIt by a previous run
        (attached or not) are removed.  Blocks are added from the camera
        (sync_objects).

        The floor IS needed: move_group's robot model has no `ground_plane`
        link, so without it MoveIt could plan through the ground.  Its top
        face is floor_gap below z=0, so it never touches the robot base."""
        world, attached = self._existing_objects()
        blocks = set(self.scene.objects)
        if world is None:                    # service missing: remove blindly
            world, attached = blocks, set()
        ps = PlanningScene()
        for name in sorted(blocks & attached):
            aco = AttachedCollisionObject()
            aco.link_name = self.ee
            aco.object.id = name
            aco.object.operation = CollisionObject.REMOVE
            ps.robot_state.attached_collision_objects.append(aco)
        for name in sorted(blocks & world):
            ps.world.collision_objects.append(self._remove_msg(name))
        if ps.robot_state.attached_collision_objects or ps.world.collision_objects:
            self._apply(ps)
        self._world_cubes = set()
        self._attached = None
        tx, ty = self.scene.table['center']
        sx, sy = self.scene.table['size']
        h = self.scene.table_top
        gap = float(self.m.get('floor_gap', FLOOR_GAP))
        ps2 = PlanningScene()
        ps2.world.collision_objects = [
            self._collision_object('floor', (FLOOR_SIZE, FLOOR_SIZE, FLOOR_THICKNESS),
                                   (0.0, 0.0, -gap - FLOOR_THICKNESS / 2.0)),
            self._collision_object('table', (sx, sy, h), (tx, ty, h / 2.0)),
        ]
        return self._apply(ps2)

    def _remove_msg(self, obj):
        co = CollisionObject()
        co.header.frame_id = self.frame
        co.id = obj
        co.operation = CollisionObject.REMOVE
        return co

    def scene_add_cube(self, obj, x, y, yaw):
        s = self.scene.cube_size
        ps = PlanningScene()
        ps.world.collision_objects = [self._collision_object(
            obj, (s, s, s), (x, y, self.scene.cube_center_z()), _yaw_quat(yaw))]
        ok = self._apply(ps)
        if ok:
            self._world_cubes.add(obj)
        return ok

    def scene_remove(self, obj):
        if obj not in self._world_cubes:
            return True
        ps = PlanningScene()
        ps.world.collision_objects = [self._remove_msg(obj)]
        self._world_cubes.discard(obj)
        return self._apply(ps)

    def sync_objects(self, world):
        """Planning scene blocks := what the camera saw (the held block stays
        attached to the gripper)."""
        s = self.scene.cube_size
        ps = PlanningScene()
        present = set()
        for obj in self.scene.objects:
            if obj == self._attached:
                continue
            if obj in world.positions:
                x, y = world.positions[obj]
                ps.world.collision_objects.append(self._collision_object(
                    obj, (s, s, s), (x, y, self.scene.cube_center_z()),
                    _yaw_quat(world.yaws.get(obj, 0.0))))
                present.add(obj)
            elif obj in self._world_cubes:
                ps.world.collision_objects.append(self._remove_msg(obj))
        if not ps.world.collision_objects:
            return True
        ok = self._apply(ps)
        self._world_cubes = present
        return ok

    def attach(self, obj):
        """Held block = collision box attached to tool0 at the grasp centre.
        Slightly SMALLER than the block (negative padding `attached_shrink`):
        a block held at its resting height touches the table with zero
        clearance and MoveIt would report that contact as a collision
        (START_STATE_IN_COLLISION on the lift)."""
        s = self.scene.cube_size - float(self.m.get('attached_shrink', 0.008))
        self.scene_remove(obj)
        aco = AttachedCollisionObject()
        aco.link_name = self.ee
        aco.object = self._collision_object(obj, (s, s, s), (0, 0, self.scene.tcp_offset),
                                            frame=self.ee)
        aco.touch_links = [self.ee, 'wrist_3_link', 'flange'] + GRIPPER_LINKS
        ps = PlanningScene()
        ps.robot_state.attached_collision_objects = [aco]
        ok = self._apply(ps)
        if ok:
            self._attached = obj
        return Status.SUCCESS if ok else Status.FAILED

    def detach(self, obj):
        aco = AttachedCollisionObject()
        aco.link_name = self.ee
        aco.object.id = obj
        aco.object.operation = CollisionObject.REMOVE
        ps = PlanningScene()
        ps.robot_state.attached_collision_objects = [aco]
        self._attached = None
        return Status.SUCCESS if self._apply(ps) else Status.FAILED

    # ------------------------------------------------------------- gripper
    def _command_fingers(self, target):
        t = float(self.g.get('motion_time', 0.6))
        msg = JointTrajectory()
        msg.joint_names = list(FINGER_JOINTS)
        pt = JointTrajectoryPoint()
        pt.positions = [float(target), float(target)]
        pt.time_from_start = DurationMsg(sec=int(t), nanosec=int((t % 1.0) * 1e9))
        msg.points = [pt]
        self.gripper_pub.publish(msg)

    def gripper_gap(self):
        q = self._joints(list(FINGER_JOINTS), 1.0)
        if q is None:
            return None
        return float(q[0] + q[1])

    def _wait_fingers_still(self, timeout):
        """Wait until both fingers stopped moving (contact or joint limit)."""
        end = time.monotonic() + timeout
        time.sleep(float(self.g.get('motion_time', 0.6)))   # let the command take effect
        still_since = None
        last = None
        while time.monotonic() < end:
            gap = self.gripper_gap()
            if gap is not None and last is not None and abs(gap - last) < 2e-4:
                still_since = still_since or time.monotonic()
                if time.monotonic() - still_since > 0.4:
                    return gap
            else:
                still_since = None
            last = gap
            time.sleep(0.05)
        return self.gripper_gap()

    def gripper_open(self):
        self._command_fingers(float(self.g.get('max_opening', 0.045)))
        gap = self._wait_fingers_still(float(self.g.get('settle_time', 1.0)) + 2.0)
        full = 2 * float(self.g.get('max_opening', 0.045))
        if gap is None or gap < full - 0.01:
            self.log.warn(f'gripper did not open fully (gap {gap})')
            return Status.EXECUTION_FAILED
        return Status.SUCCESS

    def gripper_close(self):
        if os.environ.get('UR3_LLM_DEBUG'):
            tool = self.tool_pose()
            self.log.info(f'[debug] closing at tool0 '
                          f'{tuple(round(v, 4) for v in tool[0]) if tool else None}')
        self._command_fingers(float(self.g.get('closed_target', 0.0)))
        gap = self._wait_fingers_still(float(self.g.get('settle_time', 1.0)) + 2.0)
        if gap is None:
            return Status.EXECUTION_FAILED, None
        time.sleep(0.2)
        return Status.SUCCESS, self.gripper_gap()

    # --------------------------------------------------------------- motion
    def _base_request(self):
        req = MotionPlanRequest()
        req.group_name = self.group
        req.num_planning_attempts = int(self.m['planning_attempts'])
        req.allowed_planning_time = float(self.m['planning_time'])
        req.max_velocity_scaling_factor = float(self.m['velocity_scaling'])
        req.max_acceleration_scaling_factor = float(self.m['acceleration_scaling'])
        req.start_state.is_diff = True
        ws = req.workspace_parameters
        ws.header.frame_id = self.frame
        ws.min_corner.x, ws.min_corner.y, ws.min_corner.z = -1.0, -1.0, -0.05
        ws.max_corner.x, ws.max_corner.y, ws.max_corner.z = 1.0, 1.0, 1.2
        return req

    def _send_move_group(self, constraints):
        if not self.move_client.server_is_ready():
            return Status.PLANNING_FAILED
        goal = MoveGroup.Goal()
        goal.request = self._base_request()
        goal.request.goal_constraints = [constraints]
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        handle = self._wait(self.move_client.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            return Status.PLANNING_FAILED
        res = self._wait(handle.get_result_async(), 120.0)
        if res is None:
            return Status.EXECUTION_FAILED
        code = res.result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.log.warn(f'move_group error code {code}')
        return self._map(code)

    @staticmethod
    def _map(code):
        if code == MoveItErrorCodes.SUCCESS:
            return Status.SUCCESS
        if code in _EXEC_ERRORS:
            return Status.EXECUTION_FAILED
        return Status.PLANNING_FAILED

    def move_joints(self, joints):
        """Exact joint goal (the skills computed it with IK seeded from the
        current joint state, so the arm stays on its kinematic branch)."""
        c = Constraints()
        for name, val in zip(JOINT_NAMES, joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(val)
            # tight: OMPL samples the goal inside this band, and the IK target
            # (gripper yaw, grasp position) should be reached exactly
            jc.tolerance_above = jc.tolerance_below = 0.001
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        status = self._send_move_group(c)
        if status == Status.SUCCESS:
            # the next request (usually a Cartesian move) checks that the
            # robot is within 0.01 rad of where MoveIt thinks it stopped
            # (trajectory_execution.allowed_start_tolerance): let the arm
            # settle on the target and TF / planning scene monitor catch up.
            self._settle(joints)
        return status

    def _settle(self, target, tol=0.002, timeout=1.5):
        end = time.monotonic() + timeout
        time.sleep(0.2)
        err = None
        while time.monotonic() < end:
            q = self.current_joints()
            if q is not None:
                err = max(abs(a - b) for a, b in zip(q, target))
                if err < tol:
                    break
            time.sleep(0.05)
        if os.environ.get('UR3_LLM_DEBUG'):
            tool = self.tool_pose()
            want = self._kin.fk(target)[:3, 3]
            self.log.info(f'[debug] settled, joint error {err:.4f} rad, tool0 at '
                          f'{tuple(round(v, 4) for v in tool[0]) if tool else None}, '
                          f'target {tuple(round(float(v), 4) for v in want)}')
        time.sleep(0.1)

    def move_vertical(self, z):
        tool = self.tool_pose()
        if tool is None:
            return Status.FAILED
        (x, y, _), q = tool
        req = GetCartesianPath.Request()
        req.header.frame_id = self.frame
        req.start_state.is_diff = True
        req.group_name = self.group
        req.link_name = self.ee
        req.waypoints = [_pose((x, y, z), q)]
        req.max_step = float(self.m['cartesian_step'])
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        _set_if(req, 'max_velocity_scaling_factor', float(self.m['velocity_scaling']) * 0.5)
        _set_if(req, 'max_acceleration_scaling_factor', float(self.m['acceleration_scaling']))
        res = self._wait(self.cart_client.call_async(req), 15.0)
        low_fraction = (res is not None and res.error_code.val == MoveItErrorCodes.SUCCESS
                        and res.fraction < float(self.m['min_cartesian_fraction']))
        if res is None or (res.error_code.val != MoveItErrorCodes.SUCCESS) or low_fraction:
            if low_fraction:
                self.log.warn(f'Cartesian path only {res.fraction * 100:.0f}% feasible; '
                              f'retrying this vertical move via IK + joint-space planning')
            return self._move_vertical_ik_fallback((x, y, z), q)
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = res.solution
        handle = self._wait(self.exec_client.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            return Status.EXECUTION_FAILED
        out = self._wait(handle.get_result_async(), 60.0)
        if out is None:
            return Status.EXECUTION_FAILED
        status = self._map(out.result.error_code.val)
        if status == Status.SUCCESS:
            time.sleep(0.3)
            if os.environ.get('UR3_LLM_DEBUG'):
                t2 = self.tool_pose()
                self.log.info(f'[debug] vertical move to z={z:.4f} done, tool0 at '
                              f'{tuple(round(v, 4) for v in t2[0]) if t2 else None}, '
                              f'start ({x:.4f}, {y:.4f})')
        return status

    def _move_vertical_ik_fallback(self, xyz, q):
        """Straight-line path blocked (e.g. a singularity next to the start
        posture): solve IK for the target with the SAME tool yaw and let OMPL
        plan in joint space instead."""
        seed = self.current_joints()
        if seed is None:
            return Status.PLANNING_FAILED
        yaw = tool_yaw(np.array(_quat_to_rot(*q)))
        sol = self._kin.ik(xyz, yaw, seed)
        if sol is None:
            return Status.PLANNING_FAILED
        return self.move_joints(list(sol))
