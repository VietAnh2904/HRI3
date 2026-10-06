"""UR3/UR3e kinematics (pure numpy, no ROS).

Used for
  * choosing the joint goal of every free-space move (IK seeded from the
    robot's current joint state, so MoveIt always gets ONE exact target on
    the kinematic branch the arm is already in),
  * the offline reachability check of scene.yaml (check_scene),
  * the FakeMotionBackend used by the unit tests.

Planning (collision checking, joint limits, time parametrisation) and
execution are always done by MoveIt 2.

The chain mirrors ur_description/urdf/ur_macro.xacro:
world -> base_link -> base_link_inertia (rz=pi) -> shoulder ... -> tool0.
"""
import math

import numpy as np

# default_kinematics.yaml from ur_description (x, y, z, roll, pitch, yaw)
_PARAMS = {
    'ur3e': {
        'shoulder': (0, 0, 0.15185, 0, 0, 0),
        'upper_arm': (0, 0, 0, math.pi / 2, 0, 0),
        'forearm': (-0.24355, 0, 0, 0, 0, 0),
        'wrist_1': (-0.2132, 0, 0.13105, 0, 0, 0),
        'wrist_2': (0, -0.08535, 0, math.pi / 2, 0, 0),
        'wrist_3': (0, 0.0921, 0, math.pi / 2, math.pi, math.pi),
    },
    'ur3': {
        'shoulder': (0, 0, 0.1519, 0, 0, 0),
        'upper_arm': (0, 0, 0, math.pi / 2, 0, 0),
        'forearm': (-0.24365, 0, 0, 0, 0, 0),
        'wrist_1': (-0.21325, 0, 0.11235, 0, 0, 0),
        'wrist_2': (0, -0.08535, 0, math.pi / 2, 0, 0),
        'wrist_3': (0, 0.0819, 0, math.pi / 2, math.pi, math.pi),
    },
}

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
# joint_limits.yaml: +-2pi except elbow (+-pi)
JOINT_LOWER = np.array([-2 * math.pi, -2 * math.pi, -math.pi,
                        -2 * math.pi, -2 * math.pi, -2 * math.pi])
JOINT_UPPER = -JOINT_LOWER

# joints closer to the base move more mass / sweep more space: weigh them more
# when choosing between several valid IK solutions
_COST_WEIGHTS = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _tf(x, y, z, r, p, yw):
    t = np.eye(4)
    t[:3, :3] = _rpy(r, p, yw)
    t[:3, 3] = (x, y, z)
    return t


def _rz(q):
    t = np.eye(4)
    c, s = math.cos(q), math.sin(q)
    t[:2, :2] = [[c, -s], [s, c]]
    return t


def tool_down_rotation(yaw):
    """Rotation of tool0 pointing straight down (tool z = world -z) with the
    tool x axis at angle `yaw` in the table plane: Rz(yaw) * Rx(pi)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def tool_yaw(rot):
    """Yaw of a tool-down rotation matrix (inverse of tool_down_rotation)."""
    return math.atan2(rot[1, 0], rot[0, 0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _quat_of(m):
    """(w, xyz) of a rotation matrix (Shepperd's method)."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        v = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        v = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        v = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        v = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return w, v


class URKinematics:
    """Forward / numerical inverse kinematics of tool0 in the world frame."""

    def __init__(self, ur_type='ur3e'):
        if ur_type not in _PARAMS:
            raise ValueError(f'unsupported ur_type {ur_type}')
        self.ur_type = ur_type
        p = _PARAMS[ur_type]
        self._origins = [_tf(*p[k]) for k in (
            'shoulder', 'upper_arm', 'forearm', 'wrist_1', 'wrist_2', 'wrist_3')]
        self._base = _tf(0, 0, 0, 0, 0, math.pi)             # base_link_inertia
        self._tool = (_tf(0, 0, 0, 0, -math.pi / 2, -math.pi / 2)   # flange
                      @ _tf(0, 0, 0, math.pi / 2, 0, math.pi / 2))  # tool0

    def fk(self, q):
        t = self._base.copy()
        for origin, qi in zip(self._origins, q):
            t = t @ origin @ _rz(qi)
        return t @ self._tool

    def joint_positions(self, q):
        """World positions of shoulder, elbow, wrists and tool0 (for checks)."""
        t = self._base.copy()
        out = []
        for origin, qi in zip(self._origins, q):
            t = t @ origin @ _rz(qi)
            out.append(t[:3, 3].copy())
        out.append((t @ self._tool)[:3, 3].copy())
        return out

    # ------------------------------------------------------------------ IK
    def _error(self, q, pos, rot_des):
        t = self.fk(q)
        e_pos = pos - t[:3, 3]
        r = t[:3, :3]
        if rot_des is None:            # tool z -> world -z, yaw free
            e_rot = np.cross(r[:, 2], np.array([0.0, 0.0, -1.0]))
        else:
            # rotation vector of R_des * R^T via its quaternion (well defined
            # up to 180 deg, unlike the cross-product form)
            w, v = _quat_of(rot_des @ r.T)
            e_rot = 2.0 * (v if w >= 0.0 else -v)
        return np.concatenate([e_pos, e_rot])

    def _solve(self, pos, rot_des, seed, iters=200, tol=1e-5):
        q = np.array(seed, dtype=float)
        for _ in range(iters):
            e = self._error(q, pos, rot_des)
            if np.linalg.norm(e) < tol:
                break
            jac = np.zeros((6, 6))
            for j in range(6):
                dq = np.zeros(6)
                dq[j] = 1e-6
                jac[:, j] = (self._error(q + dq, pos, rot_des) - e) / -1e-6
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), e)
            q = q + np.clip(step, -0.3, 0.3)
        e = self._error(q, pos, rot_des)
        if np.linalg.norm(e) < tol * 10 and np.all(q >= JOINT_LOWER) and np.all(q <= JOINT_UPPER):
            return q
        return None

    def _canonical_seeds(self, pos, wrist3):
        """Elbow-up, wrist-down postures pointing at the target (the branch
        used everywhere in this package)."""
        pan0 = math.atan2(pos[1], pos[0])
        seeds = []
        for dpan in (0.0, 0.45, -0.45):
            for lift, elbow in ((-1.3, 1.6), (-0.9, 1.2), (-1.7, 2.1)):
                w1 = -(lift + elbow) - math.pi / 2
                seeds.append([pan0 + dpan, lift, elbow, w1, -math.pi / 2, wrist3])
        return seeds

    def ik(self, pos, yaw=None, seed=None, max_joint_jump=None):
        """tool0 at `pos` (x, y, z) pointing straight down; if `yaw` is given
        the tool x axis is at that angle (a parallel gripper must be aligned
        with the object), otherwise the yaw is free.

        Among all solutions found (seeded from `seed` = current joints first,
        then canonical elbow-up postures), returns the one closest to `seed`
        (weighted joint distance) so the robot never takes a detour through
        another kinematic branch.  Returns None if unreachable."""
        pos = np.asarray(pos, dtype=float)
        rot_des = None if yaw is None else tool_down_rotation(yaw)
        ref = None if seed is None else np.asarray(seed, dtype=float)
        seeds = []
        if ref is not None:
            seeds.append(ref)
        seeds += self._canonical_seeds(pos, 0.0 if ref is None else float(ref[5]))
        best, best_cost = None, None
        for s in seeds:
            q = self._solve(pos, rot_des, s)
            if q is None:
                continue
            if q[2] < 0.05:           # keep the elbow-up branch
                continue
            if ref is None:
                cost = float(np.sum(_COST_WEIGHTS * np.abs(q - np.asarray(s))))
            else:
                cost = float(np.sum(_COST_WEIGHTS * np.abs(q - ref)))
            if best is None or cost < best_cost:
                best, best_cost = q, cost
            if ref is not None and s is seeds[0] and cost < 1.0:
                break                  # the current-state seed already gave a near solution
        if best is not None and max_joint_jump is not None and ref is not None:
            if np.max(np.abs(best - ref)) > max_joint_jump:
                return None
        return best

    def ik_down(self, pos, seeds=None):
        """Backward compatible helper: tool down, free yaw."""
        seed = None if not seeds else seeds[0]
        return self.ik(pos, None, seed)

    def best_grasp_yaw(self, pos, object_yaw, seed=None):
        """A parallel gripper can grasp a cube from 4 directions (object_yaw +
        k*90deg).  Returns (yaw, joints) of the reachable one that needs the
        least joint motion from `seed`, or (None, None)."""
        best = (None, None, None)
        for k in range(4):
            yaw = wrap(object_yaw + k * math.pi / 2)
            q = self.ik(pos, yaw, seed)
            if q is None:
                continue
            ref = np.asarray(seed, dtype=float) if seed is not None else q
            cost = float(np.sum(_COST_WEIGHTS * np.abs(q - ref)))
            if best[0] is None or cost < best[2]:
                best = (yaw, q, cost)
        return best[0], best[1]
