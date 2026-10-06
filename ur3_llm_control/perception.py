"""ROS side of the camera: image + camera_info + TF -> vision.detect_cubes.

  sub  /overhead_camera/image_raw      sensor_msgs/Image (rgb8 / bgr8)
  sub  /overhead_camera/camera_info    sensor_msgs/CameraInfo  (intrinsics K)
  TF   world -> <image frame_id>       (extrinsics: static transforms from
                                        sim.launch.py, same numbers as scene.yaml)
  pub  /llm_robot/camera_debug         sensor_msgs/Image, detections drawn in

observe() only uses an image taken AFTER it was called, so a picture shows the
scene once the arm has stopped at its observation (home) pose.
"""
import threading
import time

import numpy as np
from rclpy.duration import Duration
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros

from .robot_skills import Perception
from .vision import detect_cubes, annotate, optical_pose_from_config


def _quat_to_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def image_to_numpy(msg):
    enc = msg.encoding.lower()
    if enc not in ('rgb8', 'bgr8', 'rgba8', 'bgra8'):
        raise ValueError(f'unsupported image encoding {msg.encoding}')
    ch = 4 if enc.endswith('a8') else 3
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * ch].reshape(msg.height, msg.width, ch)[..., :3]
    if enc.startswith('bgr'):
        img = img[..., ::-1]
    return np.ascontiguousarray(img)


def numpy_to_image(img, stamp, frame_id):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    msg.encoding = 'rgb8'
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(img, dtype=np.uint8).tobytes()
    return msg


class CameraPerception(Perception):
    def __init__(self, node, scene, callback_group=None, tf_buffer=None,
                 image_topic=None, info_topic=None):
        self.node = node
        self.scene = scene
        cam = scene.camera
        name = cam.get('name', 'overhead_camera')
        image_topic = image_topic or f'/{name}/image_raw'
        info_topic = info_topic or f'/{name}/camera_info'
        self._lock = threading.Lock()
        self._img = None
        self._info = None
        # gazebo_ros_camera publishes RELIABLE; keep only the newest image
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        node.create_subscription(Image, image_topic, self._on_image, qos,
                                 callback_group=callback_group)
        node.create_subscription(CameraInfo, info_topic, self._on_info, qos,
                                 callback_group=callback_group)
        self.debug_pub = node.create_publisher(Image, '/llm_robot/camera_debug', 1)
        if tf_buffer is None:
            tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(tf_buffer, node)
        self.tf_buffer = tf_buffer
        self.image_topic = image_topic
        self.warned_tf = False

    def _on_image(self, msg):
        with self._lock:
            self._img = msg

    def _on_info(self, msg):
        with self._lock:
            self._info = msg

    def ready(self):
        with self._lock:
            return self._img is not None and self._info is not None

    def _latest(self):
        with self._lock:
            return self._img, self._info

    def _camera_pose(self, frame_id):
        try:
            t = self.tf_buffer.lookup_transform(self.scene.frame_id, frame_id, Time(),
                                                timeout=Duration(seconds=1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            if not self.warned_tf:
                self.node.get_logger().warn(
                    f'no TF {self.scene.frame_id} -> {frame_id} ({e}); using the camera pose '
                    f'from scene.yaml')
                self.warned_tf = True
            return optical_pose_from_config(self.scene.camera)
        T = np.eye(4)
        r = t.transform.rotation
        T[:3, :3] = _quat_to_matrix(r.x, r.y, r.z, r.w)
        tr = t.transform.translation
        T[:3, 3] = (tr.x, tr.y, tr.z)
        return T

    def observe(self, timeout=6.0):
        """Detections from the first image stamped after this call."""
        t_req = self.node.get_clock().now()
        end = time.monotonic() + timeout
        msg = info = None
        while time.monotonic() < end:
            msg, info = self._latest()
            if msg is not None and info is not None and \
                    Time.from_msg(msg.header.stamp) >= t_req:
                break
            time.sleep(0.03)
        else:
            msg, info = self._latest()
            if msg is None or info is None:
                raise RuntimeError(f'no image on {self.image_topic} (is the camera plugin '
                                   f'running?)')
            self.node.get_logger().warn(
                'no camera image newer than the request (use_sim_time:=true missing?); '
                'using the latest one')
        img = image_to_numpy(msg)
        K = np.array(info.k, dtype=float).reshape(3, 3)
        T = self._camera_pose(msg.header.frame_id or self.scene.camera.get(
            'optical_frame_id', 'camera_optical_frame'))
        res = detect_cubes(img, K, T, self.scene)
        try:
            dbg = annotate(img, res, self.scene, K, T)
            self.debug_pub.publish(numpy_to_image(dbg, msg.header.stamp, msg.header.frame_id))
        except Exception as e:           # debug output must never break a skill
            self.node.get_logger().warn(f'camera_debug: {e}')
        return res
