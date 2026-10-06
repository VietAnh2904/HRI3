"""Tiny software renderer of the workcell as seen by the overhead camera.

Used ONLY without ROS (unit tests and offline_cli): it lets the real vision
pipeline (vision.py) run on images drawn with exactly the camera model the
Gazebo camera uses.  In the ROS system the images come from Gazebo.
"""
import numpy as np

from .vision import (camera_matrix, cube_corners, optical_pose_from_config,
                     point_in_polygon_mask, project)

_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
_SHADE = [0.35, 1.0, 0.75, 0.6, 0.7, 0.8]      # bottom, top, sides


def camera_model(scene):
    cam = scene.camera
    K = camera_matrix(int(cam['width']), int(cam['height']), float(cam['horizontal_fov']))
    return K, optical_pose_from_config(cam)


def render(scene, positions, yaws=None, K=None, T=None, noise=0.0, rng=None,
           occluders=()):
    """RGB uint8 image of table, zones and cubes.

    positions: {object: (x, y)} (cubes resting on the table)
    occluders: list of (x, y, z, radius) dark grey discs (e.g. a robot link)."""
    if K is None or T is None:
        K, T = camera_model(scene)
    W, H = int(scene.camera['width']), int(scene.camera['height'])
    img = np.full((H, W, 3), 110, dtype=np.float32)          # floor
    tx, ty = scene.table['center']
    sx, sy = scene.table['size']
    z = scene.table_top
    table = np.array([[tx - sx / 2, ty - sy / 2, z], [tx + sx / 2, ty - sy / 2, z],
                      [tx + sx / 2, ty + sy / 2, z], [tx - sx / 2, ty + sy / 2, z]])
    img[point_in_polygon_mask((H, W), project(K, T, table))] = (175, 175, 178)
    for zone in scene.zones:
        zx, zy = scene.zone_xy(zone)
        for half, col in ((scene.zone_half(zone) + 0.005, (40, 40, 40)),
                          (scene.zone_half(zone), (235, 235, 235))):
            sq = np.array([[zx - half, zy - half, z + 0.001], [zx + half, zy - half, z + 0.001],
                           [zx + half, zy + half, z + 0.001], [zx - half, zy + half, z + 0.001]])
            img[point_in_polygon_mask((H, W), project(K, T, sq))] = col
    # painter's algorithm: far cubes first
    cam = T[:3, 3]
    far_first = {o: -float(np.hypot(p[0] - cam[0], p[1] - cam[1])) for o, p in positions.items()}
    order = sorted(positions, key=far_first.get)
    for obj in order:
        x, y = positions[obj]
        yaw = (yaws or {}).get(obj, 0.0)
        rgb = np.array(scene.objects[obj]['rgba'][:3]) * 255.0
        corners = cube_corners(x, y, yaw, z, scene.cube_size)
        uv = project(K, T, corners)
        faces = []
        for fi, f in enumerate(_FACES):
            ctr = corners[list(f)].mean(axis=0)
            faces.append((-np.linalg.norm(ctr - cam), fi, f))
        for _, fi, f in sorted(faces):
            poly = uv[list(f)]
            img[point_in_polygon_mask((H, W), poly)] = rgb * _SHADE[fi]
    for (ox, oy, oz, r) in occluders:
        n = 24
        ring = np.array([[ox + r * np.cos(a), oy + r * np.sin(a), oz]
                         for a in np.linspace(0, 2 * np.pi, n, endpoint=False)])
        img[point_in_polygon_mask((H, W), project(K, T, ring))] = (60, 60, 65)
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        img = img + rng.normal(0.0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
