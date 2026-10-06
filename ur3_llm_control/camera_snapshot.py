"""Save what the overhead camera sees + what the vision pipeline detects.

    ros2 run ur3_llm_control camera_snapshot --ros-args -p use_sim_time:=true
    ros2 run ur3_llm_control camera_snapshot --ros-args -p use_sim_time:=true -p out:=/tmp/cam.png

Prints every detected block (position, yaw, zone) and the zone occupancy and
writes the raw and annotated images (PNG, needs python3-opencv or PIL; else
PPM).  Does not move the robot: put the arm at home first for a clear view.
"""
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .paths import config_path
from .perception import CameraPerception
from .scene_model import SceneConfig, WorldState
from .vision import annotate, save_image


def main(args=None):
    rclpy.init(args=args)
    node = Node('camera_snapshot')
    node.declare_parameter('scene_file', config_path('scene.yaml'))
    node.declare_parameter('out', '/tmp/ur3_camera.png')
    scene = SceneConfig.from_file(node.get_parameter('scene_file').value)
    cam = CameraPerception(node, scene)
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    end = time.monotonic() + 20.0
    while not cam.ready() and time.monotonic() < end:
        time.sleep(0.1)
    code = 0
    try:
        res = cam.observe()
        world = WorldState(scene)
        world.set_detections({n: (d.x, d.y, d.yaw) for n, d in res.detections.items()})
        for n, d in res.detections.items():
            print(f'{n:12s} x={d.x:.3f} y={d.y:.3f} yaw={np.degrees(d.yaw):6.1f} deg '
                  f'area={d.area_px}/{d.expected_area_px:.0f} px  -> {world.summary()[n]}')
        for n in scene.objects:
            if n not in res.detections:
                print(f'{n:12s} NOT DETECTED')
        print('zones: ' + ', '.join(f'{z}={o or "FREE"}' for z, o in world.zone_status().items()))
        for w in res.warnings:
            print('warning:', w)
        img, info = cam._latest()
        from .perception import image_to_numpy
        raw = image_to_numpy(img)
        K = np.array(info.k, dtype=float).reshape(3, 3)
        T = cam._camera_pose(img.header.frame_id)
        out = node.get_parameter('out').value
        p1 = save_image(out.replace('.png', '_raw.png'), raw)
        p2 = save_image(out, annotate(raw, res, scene, K, T))
        print(f'saved {p1} and {p2}')
    except Exception as e:
        print(f'camera_snapshot: {e}', file=sys.stderr)
        code = 1
    ex.shutdown(timeout_sec=2.0)
    node.destroy_node()
    rclpy.try_shutdown()
    sys.stdout.flush()
    os._exit(code)                  # see llm_robot_node.main()


if __name__ == '__main__':
    sys.exit(main())
