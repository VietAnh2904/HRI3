"""Offline sanity check of scene.yaml (no ROS needed).

    python3 -m ur3_llm_control.check_scene [path/to/scene.yaml]
    ros2 run ur3_llm_control check_scene

Verifies that
  * every zone and every spawn position can be grasped / released with the
    gripper (tool down, 4 gripper yaws tried, joint limits),
  * blocks and zones are far enough apart for the open fingers,
  * the home pose points the tool down, is above the approach height and
    keeps the whole arm OUT of the camera's view of the table (home is also
    the observation pose),
  * the camera sees the whole table,
  * there is free table space for temporary positions,
  * the vision pipeline finds every block at its spawn position (synthetic
    image of the scene rendered with the camera model).
MoveIt still does the real (collision-aware) planning at run time; this only
catches bad layouts early.
"""
import itertools
import math
import sys

import numpy as np

from .paths import config_path
from .scene_model import SceneConfig, WorldState
from .synthetic_camera import camera_model, render
from .ur_kinematics import URKinematics
from .vision import detect_cubes, project


def default_scene_path():
    return config_path('scene.yaml')


def _arm_points(kin, q, gripper_len):
    pts = kin.joint_positions(q)
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        for s in np.linspace(0, 1, 8):
            out.append(a + (b - a) * s)
    t = kin.fk(q)
    for s in np.linspace(0, 1, 8):
        out.append(t[:3, 3] + t[:3, 2] * gripper_len * s)
    return np.array(out)


def check(scene: SceneConfig, ur_type=None):
    problems = []
    kin = URKinematics(ur_type or scene.ur_type)
    home = scene.motion['home_joints']
    heights = {'grasp': scene.grasp_tool_z(), 'place': scene.place_tool_z(),
               'approach': scene.approach_tool_z()}
    for name, (x, y) in scene.all_locations().items():
        for hname, z in heights.items():
            yaw, q = kin.best_grasp_yaw((x, y, z), 0.0, home)
            if q is None:
                problems.append(f'{name}: {hname} pose ({x:.3f}, {y:.3f}, {z:.3f}) unreachable')
        r = math.hypot(x, y)
        if r < 0.15:
            problems.append(f'{name}: too close to the robot base (r={r:.3f} m)')
    min_gap = float(scene.free_space.get('min_object_distance', 0.10))
    locs = scene.all_locations()
    for a, b in itertools.combinations(list(scene.objects), 2):
        (ax, ay), (bx, by) = locs[a], locs[b]
        if math.hypot(ax - bx, ay - by) < min_gap - 1e-9:
            problems.append(f'{a} and {b} are closer than {min_gap:.3f} m (fingers need room)')
    for a, b in itertools.combinations(list(scene.zones), 2):
        (ax, ay), (bx, by) = locs[a], locs[b]
        if max(abs(ax - bx), abs(ay - by)) < 2 * scene.zone_half(a) + 0.03:
            problems.append(f'zones {a} and {b} overlap / are too close')
    t = kin.fk(home)
    if t[2, 2] > -0.95:
        problems.append('home pose: tool is not pointing down')
    if t[2, 3] < scene.approach_tool_z():
        problems.append('home pose is lower than the approach height')
    tx, ty = scene.table['center']
    sx, sy = scene.table['size']
    if tx - sx / 2 < 0.09:
        problems.append('table overlaps the robot base (x_min < 0.09 m)')

    # ---- camera: sees the whole table, arm at home is outside that view
    if scene.camera:
        K, T = camera_model(scene)
        W, H = int(scene.camera['width']), int(scene.camera['height'])
        z = scene.table_top
        corners = np.array([[tx - sx / 2, ty - sy / 2, z], [tx + sx / 2, ty - sy / 2, z],
                            [tx + sx / 2, ty + sy / 2, z], [tx - sx / 2, ty + sy / 2, z]])
        uv = project(K, T, corners)
        if np.any(uv < 0) or np.any(uv[:, 0] >= W) or np.any(uv[:, 1] >= H):
            problems.append('camera does not see the whole table')
        cam = np.asarray(scene.camera['xyz'], dtype=float)
        pts = _arm_points(kin, home, scene.tcp_offset + 0.02)
        radius = 0.065                           # link radius incl. margin
        s = (cam[2] - z) / (cam[2] - pts[:, 2])
        xy = cam[:2] + (pts[:, :2] - cam[:2]) * s[:, None]
        grow = 0.03 + radius * s                 # roi_expand + projected link radius
        inside = ((xy[:, 0] > tx - sx / 2 - grow) & (xy[:, 0] < tx + sx / 2 + grow)
                  & (xy[:, 1] > ty - sy / 2 - grow) & (xy[:, 1] < ty + sy / 2 + grow))
        if np.any(inside):
            problems.append('home pose: the arm hides part of the table from the camera')
        # ---- vision finds every block at its spawn pose
        pos = {n: tuple(o['spawn_xy']) for n, o in scene.objects.items()}
        res = detect_cubes(render(scene, pos, noise=2.0), K, T, scene)
        for n, (x, y) in pos.items():
            d = res.detections.get(n)
            if d is None:
                problems.append(f'vision: {n} not detected in the synthetic image')
            elif math.hypot(d.x - x, d.y - y) > 0.005:
                problems.append(f'vision: {n} located {1000 * math.hypot(d.x - x, d.y - y):.1f} '
                                f'mm off')

    # ---- free space for temporary positions exists
    w = WorldState(scene)
    w.set_detections({n: (*o['spawn_xy'], 0.0) for n, o in scene.objects.items()})

    def reach(xy):
        return kin.best_grasp_yaw((xy[0], xy[1], scene.approach_tool_z()), 0.0, home)[1] is not None

    for obj in scene.objects:
        if w.free_position(obj, reachable=reach) is None:
            problems.append(f'no free table spot to park {obj}')
            break
    return problems, t


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else default_scene_path()
    scene = SceneConfig.from_file(path)
    np.set_printoptions(precision=3, suppress=True)
    all_ok = True
    for ur in ('ur3', 'ur3e'):
        problems, home = check(scene, ur)
        print(f'[{ur}] home tool0 position: {home[:3, 3]}')
        for p in problems:
            print(f'[{ur}] PROBLEM: {p}')
        print(f'[{ur}] ' + ('OK' if not problems else f'{len(problems)} problem(s)'))
        all_ok &= not problems if ur == scene.ur_type else True
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
