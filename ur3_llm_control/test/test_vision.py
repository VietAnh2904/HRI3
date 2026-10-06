"""Vision pipeline on synthetic images rendered with the camera model."""
import math
import os

import numpy as np
import pytest

from ur3_llm_control.scene_model import SceneConfig
from ur3_llm_control.synthetic_camera import camera_model, render
from ur3_llm_control.vision import (annotate, backproject, connected_components, detect_cubes,
                                    min_area_yaw, project, rgb_to_hsv)

CFG = os.path.join(os.path.dirname(__file__), '..', 'config', 'scene.yaml')


@pytest.fixture(scope='module')
def scene():
    return SceneConfig.from_file(CFG)


@pytest.fixture(scope='module')
def cam(scene):
    return camera_model(scene)


def _yaw_err(a, b):
    return abs(((a - b) + math.pi / 4) % (math.pi / 2) - math.pi / 4)


def test_project_backproject_roundtrip(cam):
    K, T = cam
    pts = np.array([[0.2, -0.3, 0.12], [0.45, 0.35, 0.10], [0.3, 0.0, 0.14]])
    uv = project(K, T, pts)
    for p, q in zip(pts, uv):
        back = backproject(K, T, q[None, :], p[2])[0]
        assert np.allclose(back, p, atol=1e-9)


def test_image_orientation(cam):
    """Camera straight down, image up = +x (away from the robot), image
    left = +y: a point further from the robot is higher in the image."""
    K, T = cam
    near, far, left = project(K, T, [[0.2, 0, 0.1], [0.4, 0, 0.1], [0.3, 0.2, 0.1]])
    assert far[1] < near[1]
    assert left[0] < project(K, T, [[0.3, 0.0, 0.1]])[0][0]


def test_hsv_matches_opencv_if_available():
    cv2 = pytest.importorskip('cv2')
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    h, s, v = rgb_to_hsv(img)
    ref = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(float)
    sat = ref[..., 1] > 40
    dh = np.abs(h - ref[..., 0])
    dh = np.minimum(dh, 180 - dh)
    assert np.percentile(dh[sat], 99) <= 1.0
    assert np.abs(s - ref[..., 1]).max() <= 1.5 and np.abs(v - ref[..., 2]).max() <= 1.0


def test_connected_components():
    m = np.zeros((20, 20), dtype=bool)
    m[2:6, 2:6] = True          # 16 px
    m[10:12, 10:19] = True      # 18 px
    m[0, 19] = True             # 1 px
    comps = connected_components(m)
    assert [len(c[0]) for c in comps] == [18, 16, 1]


def test_min_area_yaw():
    for yaw in (0.0, 0.3, -0.6, 0.78):
        c, s = math.cos(yaw), math.sin(yaw)
        g = np.array([[x, y] for x in np.linspace(-1, 1, 30) for y in np.linspace(-1, 1, 30)])
        pts = g @ np.array([[c, s], [-s, c]])
        assert _yaw_err(min_area_yaw(pts), yaw) < math.radians(0.5)


def test_detects_spawn_layout(scene, cam):
    K, T = cam
    pos = {n: tuple(o['spawn_xy']) for n, o in scene.objects.items()}
    res = detect_cubes(render(scene, pos, noise=3.0), K, T, scene)
    assert set(res.detections) == set(scene.objects) and not res.warnings
    for n, d in res.detections.items():
        assert math.hypot(d.x - pos[n][0], d.y - pos[n][1]) < 0.002, n
        assert _yaw_err(d.yaw, 0.0) < math.radians(2)
        assert not d.partial


def test_detects_random_layouts(scene, cam):
    """Random positions and yaws anywhere on the table (also near the edges
    where the camera sees a side face)."""
    K, T = cam
    rng = np.random.default_rng(7)
    tx, ty = scene.table['center']
    sx, sy = scene.table['size']
    worst = 0.0
    for trial in range(12):
        pos, yaws = {}, {}
        for n in scene.objects:
            for _ in range(200):
                p = (tx + rng.uniform(-sx / 2 + 0.03, sx / 2 - 0.03),
                     ty + rng.uniform(-sy / 2 + 0.03, sy / 2 - 0.03))
                if all(math.hypot(p[0] - q[0], p[1] - q[1]) > 0.07 for q in pos.values()):
                    break
            pos[n] = p
            yaws[n] = rng.uniform(-math.pi, math.pi)
        res = detect_cubes(render(scene, pos, yaws, noise=4.0, rng=rng), K, T, scene)
        assert set(res.detections) == set(scene.objects), trial
        for n, d in res.detections.items():
            err = math.hypot(d.x - pos[n][0], d.y - pos[n][1])
            worst = max(worst, err)
            assert err < 0.003, (trial, n, err)
            assert _yaw_err(d.yaw, yaws[n]) < math.radians(4), (trial, n)
    assert worst < 0.003


def test_empty_table_has_no_detections(scene, cam):
    K, T = cam
    res = detect_cubes(render(scene, {}, noise=5.0), K, T, scene)
    assert res.detections == {}


def test_occluded_block_is_missing_or_flagged(scene, cam):
    K, T = cam
    pos = {'red_cube': (0.30, 0.0)}
    img = render(scene, pos, occluders=[(0.30, 0.0, 0.40, 0.06)])
    assert 'red_cube' not in detect_cubes(img, K, T, scene).detections
    img = render(scene, pos, occluders=[(0.30, 0.018, 0.40, 0.012)])
    d = detect_cubes(img, K, T, scene).detections.get('red_cube')
    assert d is None or d.partial or math.hypot(d.x - 0.30, d.y) < 0.01


def test_blocks_off_the_table_are_ignored(scene, cam):
    K, T = cam
    img = render(scene, {'red_cube': (0.62, 0.0)})           # beyond the far edge
    assert detect_cubes(img, K, T, scene).detections == {}


def test_annotate_keeps_shape(scene, cam):
    K, T = cam
    pos = {n: tuple(o['spawn_xy']) for n, o in scene.objects.items()}
    img = render(scene, pos)
    out = annotate(img, detect_cubes(img, K, T, scene), scene, K, T)
    assert out.shape == img.shape and out.dtype == np.uint8 and (out != img).any()
