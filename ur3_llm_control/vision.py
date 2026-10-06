"""Camera -> block poses (pure numpy, no ROS / OpenCV needed).

Pipeline for one RGB image of the fixed overhead camera:
  1. HSV colour segmentation, one hue range per block colour (scene.yaml
     `vision`), restricted to the table top as seen by the camera (ROI);
  2. connected components -> the largest blob of each colour;
  3. pose on the table from the blob, using the camera model
     (intrinsics K from /camera_info, extrinsics = TF world -> optical frame):
       * yaw  : minimum-area rectangle of the blob back-projected onto the
                plane of the cube's top face;
       * x, y : chosen so that the silhouette the camera WOULD see of a cube
                at (x, y, yaw) has the same centroid as the observed blob
                (Gauss-Newton on the projected 8 corners).  This removes the
                perspective bias of simply back-projecting the blob centroid
                (the camera also sees a side face of blocks away from the
                image centre).

Everything here is deterministic geometry, unit-tested on synthetic images
rendered with the same camera model (test/test_vision.py).
"""
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Detection:
    name: str
    color: str
    x: float
    y: float
    yaw: float
    area_px: int
    expected_area_px: float
    pixel: Tuple[float, float]
    bbox: Tuple[int, int, int, int]        # row0, col0, row1, col1

    @property
    def xy(self):
        return (self.x, self.y)

    @property
    def partial(self):
        return self.area_px < 0.6 * self.expected_area_px


@dataclass
class VisionResult:
    detections: Dict[str, Detection] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------- camera maths
def camera_matrix(width, height, hfov):
    """Pinhole K of a Gazebo camera (same formula as gazebo_ros_camera)."""
    fx = width / (2.0 * math.tan(hfov / 2.0))
    return np.array([[fx, 0.0, (width + 1) / 2.0],
                     [0.0, fx, (height + 1) / 2.0],
                     [0.0, 0.0, 1.0]])


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


# camera_link (Gazebo: looks along +x, image up = +z) -> optical frame
# (z forward, x right, y down): the usual rpy (-pi/2, 0, -pi/2)
LINK_TO_OPTICAL = rpy_matrix(-math.pi / 2, 0.0, -math.pi / 2)


def optical_pose_from_config(cam):
    """4x4 pose of the optical frame in the world, from scene.yaml `camera`
    (used by tests and as a fallback when TF is not available)."""
    t = np.eye(4)
    t[:3, :3] = rpy_matrix(*cam['rpy']) @ LINK_TO_OPTICAL
    t[:3, 3] = cam['xyz']
    return t


def project(K, T_world_cam, pts):
    """world points (N,3) -> pixel (N,2) as (u=col, v=row)."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    pc = (pts - t) @ R            # = R^T (p - t) for each row
    z = pc[:, 2:3]
    uv = (pc[:, :2] / z) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv


def backproject(K, T_world_cam, uv, z_plane):
    """pixels (N,2) (u, v) -> world points on the horizontal plane z = z_plane."""
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    d = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1],
                  np.ones(len(uv))], axis=1)
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    dw = d @ R.T
    s = (z_plane - t[2]) / dw[:, 2]
    return t + dw * s[:, None]


def cube_corners(x, y, yaw, z0, size):
    h = size / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for zz in (z0, z0 + size):
        for dx, dy in ((h, h), (-h, h), (-h, -h), (h, -h)):
            pts.append((x + c * dx - s * dy, y + s * dx + c * dy, zz))
    return np.array(pts)


def convex_hull(points):
    """Monotone chain; points (N,2) -> hull (M,2) counter-clockwise."""
    pts = sorted(set(map(tuple, np.round(np.asarray(points, dtype=float), 6))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_centroid_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2.0
    if abs(a) < 1e-12:
        return poly.mean(axis=0), 0.0
    cx = ((x + x1) * cr).sum() / (6 * a)
    cy = ((y + y1) * cr).sum() / (6 * a)
    return np.array([cx, cy]), abs(a)


def silhouette(K, T, x, y, yaw, z0, size):
    """(centroid (u, v), area px, hull) of the cube's image."""
    hull = convex_hull(project(K, T, cube_corners(x, y, yaw, z0, size)))
    c, a = polygon_centroid_area(hull)
    return c, a, hull


def point_in_polygon_mask(shape, poly):
    """Boolean image mask of the pixels whose centres lie inside a convex
    polygon given in (u, v) pixel coordinates (counter-clockwise or not)."""
    h, w = shape
    vv, uu = np.mgrid[0:h, 0:w]
    uu = uu + 0.5
    vv = vv + 0.5
    inside_pos = np.ones(shape, dtype=bool)
    inside_neg = np.ones(shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        cr = (x1 - x0) * (vv - y0) - (y1 - y0) * (uu - x0)
        inside_pos &= cr >= 0
        inside_neg &= cr <= 0
    return inside_pos | inside_neg


# ------------------------------------------------------------ segmentation
def rgb_to_hsv(img):
    """uint8 RGB (H, W, 3) -> H in [0, 180), S, V in [0, 255] (OpenCV scale)."""
    rgb = img[..., :3].astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    nz = diff > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = (60.0 * ((g[rm] - b[rm]) / diff[rm])) % 360.0
    h[gm] = 60.0 * ((b[gm] - r[gm]) / diff[gm]) + 120.0
    h[bm] = 60.0 * ((r[bm] - g[bm]) / diff[bm]) + 240.0
    s = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    return h / 2.0, s * 255.0, mx * 255.0


def color_mask(h, s, v, ranges, min_s, min_v):
    m = np.zeros(h.shape, dtype=bool)
    for lo, hi in ranges:
        m |= (h >= lo) & (h <= hi + 0.999)
    return m & (s >= min_s) & (v >= min_v)


def connected_components(mask):
    """4-connected components of a boolean mask -> list of (rows, cols)
    arrays, largest first.  Pure python BFS over the (few) set pixels."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return []
    w = mask.shape[1]
    todo = set((rows * w + cols).tolist())
    comps = []
    while todo:
        start = todo.pop()
        stack = [start]
        comp = [start]
        while stack:
            p = stack.pop()
            r, c = divmod(p, w)
            for q in ((p - w) if r > 0 else -1, (p + w), (p - 1) if c > 0 else -1,
                      (p + 1) if c < w - 1 else -1):
                if q >= 0 and q in todo:
                    todo.remove(q)
                    stack.append(q)
                    comp.append(q)
        arr = np.array(comp)
        comps.append((arr // w, arr % w))
    comps.sort(key=lambda rc: -len(rc[0]))
    return comps


# ---------------------------------------------------------------- pose
def min_area_yaw(xy):
    """Yaw in (-pi/4, pi/4] of the minimum-area rectangle around 2D points."""
    xy = np.asarray(xy, dtype=float)
    xy = xy - xy.mean(axis=0)

    def area(a):
        c, s = math.cos(a), math.sin(a)
        u = xy[:, 0] * c + xy[:, 1] * s
        w = -xy[:, 0] * s + xy[:, 1] * c
        return (u.max() - u.min()) * (w.max() - w.min())

    coarse = [math.radians(d) for d in range(-45, 45)]
    best = min(coarse, key=area)
    fine = [best + math.radians(0.1 * k) for k in range(-10, 11)]
    best = min(fine, key=area)
    best = (best + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return best


def estimate_cube_pose(rows, cols, K, T, table_z, size, iters=4):
    """(x, y, yaw) of a cube resting on the table from its blob pixels."""
    uv = np.stack([cols + 0.5, rows + 0.5], axis=1)
    c_obs = uv.mean(axis=0)
    # yaw from the blob seen on the top-face plane
    top = backproject(K, T, uv, table_z + size)
    yaw = min_area_yaw(top[:, :2])
    # initial guess: centroid on the mid-height plane
    x, y = backproject(K, T, c_obs[None, :], table_z + size / 2)[0, :2]
    for _ in range(iters):
        c0, _, _ = silhouette(K, T, x, y, yaw, table_z, size)
        e = c_obs - c0
        if np.linalg.norm(e) < 0.05:
            break
        d = 1e-3
        cx, _, _ = silhouette(K, T, x + d, y, yaw, table_z, size)
        cy, _, _ = silhouette(K, T, x, y + d, yaw, table_z, size)
        J = np.stack([(cx - c0) / d, (cy - c0) / d], axis=1)
        try:
            dx = np.linalg.solve(J, e)
        except np.linalg.LinAlgError:
            break
        x, y = x + float(dx[0]), y + float(dx[1])
    return float(x), float(y), float(yaw)


# ------------------------------------------------------------- top level
def table_roi(shape, K, T, scene, expand_m=0.03):
    """Pixels that see the table top (grown by `expand_m` so that blocks right
    at the edge, whose side faces project past the table outline, are kept)."""
    tx, ty = scene.table['center']
    sx, sy = scene.table['size']
    z = scene.table_top
    hx, hy = sx / 2 + expand_m, sy / 2 + expand_m
    corners = np.array([[tx - hx, ty - hy, z], [tx + hx, ty - hy, z],
                        [tx + hx, ty + hy, z], [tx - hx, ty + hy, z]])
    poly = project(K, T, corners)
    return point_in_polygon_mask(shape, poly), poly


def detect_cubes(img_rgb, K, T_world_cam, scene):
    """Detect every configured block in one RGB image.

    Returns VisionResult(detections={object_name: Detection}, warnings=[...]).
    Blocks that are not visible are absent from `detections`."""
    cfg = scene.vision
    h, s, v = rgb_to_hsv(img_rgb)
    roi, _ = table_roi(h.shape, K, T_world_cam, scene, float(cfg.get('roi_expand_m', 0.03)))
    min_area = int(cfg.get('min_area_px', 60))
    res = VisionResult()
    size = scene.cube_size
    for name, obj in scene.objects.items():
        color = obj['color']
        ranges = cfg.get('hue_ranges', {}).get(color)
        if not ranges:
            res.warnings.append(f'no hue range for {color}')
            continue
        m = color_mask(h, s, v, ranges, float(cfg.get('min_saturation', 110)),
                       float(cfg.get('min_value', 45))) & roi
        comps = [c for c in connected_components(m) if len(c[0]) >= min_area]
        if not comps:
            continue
        if len(comps) > 1:
            res.warnings.append(f'{len(comps)} {color} blobs, using the largest')
        rows, cols = comps[0]
        x, y, yaw = estimate_cube_pose(rows, cols, K, T_world_cam, scene.table_top, size)
        _, exp_area, _ = silhouette(K, T_world_cam, x, y, yaw, scene.table_top, size)
        det = Detection(name=name, color=color, x=x, y=y, yaw=yaw, area_px=int(len(rows)),
                        expected_area_px=float(exp_area),
                        pixel=(float(cols.mean() + 0.5), float(rows.mean() + 0.5)),
                        bbox=(int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())))
        if det.partial:
            res.warnings.append(f'{name}: only {det.area_px}/{exp_area:.0f} px visible '
                                f'(partly occluded?)')
        res.detections[name] = det
    return res


def annotate(img_rgb, result, scene, K, T):
    """Copy of the image with a box around every detection and the zones
    outlined (for /llm_robot/camera_debug)."""
    out = img_rgb[..., :3].copy()
    H, W = out.shape[:2]

    def box(r0, c0, r1, c1, col):
        r0, r1 = max(r0, 0), min(r1, H - 1)
        c0, c1 = max(c0, 0), min(c1, W - 1)
        out[r0, c0:c1 + 1] = col
        out[r1, c0:c1 + 1] = col
        out[r0:r1 + 1, c0] = col
        out[r0:r1 + 1, c1] = col

    for zone in scene.zones:
        zx, zy = scene.zone_xy(zone)
        hz = scene.zone_half(zone)
        pts = np.array([[zx - hz, zy - hz, scene.table_top], [zx + hz, zy + hz, scene.table_top],
                        [zx - hz, zy + hz, scene.table_top], [zx + hz, zy - hz, scene.table_top]])
        uv = project(K, T, pts)
        box(int(uv[:, 1].min()), int(uv[:, 0].min()), int(uv[:, 1].max()), int(uv[:, 0].max()),
            (255, 255, 255))
    for d in result.detections.values():
        r0, c0, r1, c1 = d.bbox
        box(r0 - 3, c0 - 3, r1 + 3, c1 + 3, (0, 255, 0) if not d.partial else (255, 128, 0))
        cu, cv = int(d.pixel[0]), int(d.pixel[1])
        out[max(cv - 1, 0):cv + 2, max(cu - 1, 0):cu + 2] = (0, 0, 0)
    try:                                   # labels only if OpenCV happens to be installed
        import cv2
        for d in result.detections.values():
            cv2.putText(out, d.name.replace('_cube', ''), (d.bbox[1], max(d.bbox[0] - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    except Exception:
        pass
    return out


def save_image(path, img):
    """Write an RGB image (PNG via OpenCV or PIL if available, else PPM)."""
    try:
        import cv2
        cv2.imwrite(path, img[..., ::-1])
        return path
    except Exception:
        pass
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
        return path
    except Exception:
        pass
    path = os.path.splitext(path)[0] + '.ppm'
    h, w = img.shape[:2]
    with open(path, 'wb') as f:
        f.write(f'P6 {w} {h} 255\n'.encode())
        f.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
    return path
