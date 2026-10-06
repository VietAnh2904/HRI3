"""ROS 2 node: natural-language control of the UR3/UR3e.

  ros2 run ur3_llm_control llm_robot_node                 # interactive prompt
  ros2 run ur3_llm_control llm_robot_node --ros-args -p command:="Put the red cube in zone B."
  ros2 run ur3_llm_control send_command "Move the blue cube to zone C."   # via topic

Topics
  sub  /llm_robot/command  std_msgs/String   natural-language command
  pub  /llm_robot/plan     std_msgs/String   validated JSON plan
  pub  /llm_robot/result   std_msgs/String   JSON execution report
  pub  /llm_robot/markers  visualization_msgs/MarkerArray   zone labels + detected blocks (RViz)
  pub  /llm_robot/camera_debug  sensor_msgs/Image           camera image with the detections
  sub  /overhead_camera/image_raw, /overhead_camera/camera_info   (camera)
"""
import json
import os
import queue
import sys
import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from .fake_backend import FakeMotionBackend, FakePerception, FakeWorld
from .llm_planner import LLMError, LLMPlanner, make_client_from_config
from .pipeline import CommandPipeline
from .robot_skills import RobotSkills
from .scene_model import SceneConfig, WorldState
from .student_task import describe
from .task_validator import TaskValidator


def _share(*parts):
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('ur3_llm_control'), *parts)


def out(msg=''):
    print(msg, flush=True)


class LLMRobotNode(Node):
    def __init__(self):
        super().__init__('llm_robot_node')
        self.declare_parameter('scene_file', _share('config', 'scene.yaml'))
        self.declare_parameter('student_file', _share('config', 'student_config.yaml'))
        self.declare_parameter('backend', 'moveit')        # moveit | fake
        self.declare_parameter('dry_run', False)
        self.declare_parameter('command', '')
        self.declare_parameter('interactive', True)
        self.declare_parameter('home_on_start', True)
        self.declare_parameter('llm_model', '')             # overrides student_config.yaml

        gp = self.get_parameter
        self.scene = SceneConfig.from_file(gp('scene_file').value)
        with open(gp('student_file').value, 'r', encoding='utf-8') as f:
            student = yaml.safe_load(f)
        llm_cfg = dict(student.get('llm', {}))
        if gp('llm_model').value:
            llm_cfg['model'] = gp('llm_model').value

        self.cb = ReentrantCallbackGroup()
        self.world = WorldState(self.scene)
        if gp('backend').value == 'fake':
            truth = FakeWorld(self.scene)
            self.backend = FakeMotionBackend(self.scene, truth)
            self.perception = FakePerception(self.scene, truth)
        else:
            import tf2_ros
            from .moveit_backend import MoveItBackend
            from .perception import CameraPerception
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.backend = MoveItBackend(self, self.scene, self.cb, self.tf_buffer)
            self.perception = CameraPerception(self, self.scene, self.cb, self.tf_buffer)
        self.skills = RobotSkills(self.scene, self.backend, self.perception, self.world, log=out)
        validator = TaskValidator(self.scene)
        client = make_client_from_config(llm_cfg)
        self.planner = LLMPlanner(client, self.scene, validator, student['student_name'],
                                  student['student_id'], llm_cfg.get('max_attempts', 2), log=out)
        self.pipeline = CommandPipeline(self.planner, self.skills, self.world, log=out)
        self.student_info = describe(student['student_name'], student['student_id'])
        self.llm_desc = f"{client.model} @ {client.url}"

        self.commands = queue.Queue()
        self.plan_pub = self.create_publisher(String, '/llm_robot/plan', 10)
        self.result_pub = self.create_publisher(String, '/llm_robot/result', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/llm_robot/markers', 1)
        self.create_subscription(String, '/llm_robot/command',
                                 lambda m: self.commands.put(m.data), 10,
                                 callback_group=self.cb)
        self.create_timer(1.0, self._publish_markers, callback_group=self.cb)

    # --------------------------------------------------------------- setup
    def prepare_robot(self):
        if isinstance(self.backend, FakeMotionBackend):
            st = self.skills.detect_objects()
            out(f'[setup] detect_objects() ... {st}')
            return st.ok
        out('[setup] waiting for MoveIt 2 (move_group), the gripper controller and the camera ...')
        missing = self.backend.wait_for_servers(timeout=180.0)
        if missing:
            out(f'[setup] ERROR: not available: {missing}. Is sim.launch.py running?')
            return False
        end = time.monotonic() + 60.0
        while not self.perception.ready() and time.monotonic() < end:
            time.sleep(0.2)
        if not self.perception.ready():
            out(f'[setup] ERROR: no image on {self.perception.image_topic}')
            return False
        if not self.backend.setup_scene():
            out('[setup] ERROR: could not update the MoveIt planning scene')
            return False
        out('[setup] planning scene ready (floor, table)')
        st = self.skills.open_gripper()
        out(f'[setup] open_gripper() ... {st}')
        if self.get_parameter('home_on_start').value:
            st = self.skills.home()
            out(f'[setup] home() ... {st}')
            if not st.ok:
                return False
        st = self.skills.detect_objects()
        out(f'[setup] detect_objects() ... {st}')
        if not st.ok:
            return False
        self.pipeline.print_world('[setup] camera sees:')
        return True

    def _publish_markers(self):
        arr = MarkerArray()
        top = self.scene.table_top
        for i, (name, z) in enumerate(self.scene.zones.items()):
            m = Marker()
            m.header.frame_id = self.scene.frame_id
            m.ns, m.id, m.type, m.action = 'zones', i, Marker.TEXT_VIEW_FACING, Marker.ADD
            m.pose.position.x, m.pose.position.y = map(float, z['xy'])
            m.pose.position.z = top + 0.12
            m.pose.orientation.w = 1.0
            m.scale.z = 0.03
            m.color.r = m.color.g = m.color.b = m.color.a = 1.0
            m.text = z['label']
            arr.markers.append(m)
            p = Marker()
            p.header.frame_id = self.scene.frame_id
            p.ns, p.id, p.type, p.action = 'zone_pads', i, Marker.CUBE, Marker.ADD
            p.pose.position.x, p.pose.position.y = map(float, z['xy'])
            p.pose.position.z = top + 0.001
            p.pose.orientation.w = 1.0
            p.scale.x = p.scale.y = float(z['size'])
            p.scale.z = 0.002
            p.color.r = p.color.g = p.color.b = 0.95
            p.color.a = 0.8
            arr.markers.append(p)
        # what the camera believes (labels above the detected blocks)
        for i, obj in enumerate(self.scene.objects):
            m = Marker()
            m.header.frame_id = self.scene.frame_id
            m.ns, m.id, m.type = 'detected', i, Marker.TEXT_VIEW_FACING
            pos = self.world.positions.get(obj)
            if pos is None:
                m.action = Marker.DELETE
            else:
                m.action = Marker.ADD
                m.pose.position.x, m.pose.position.y = float(pos[0]), float(pos[1])
                m.pose.position.z = top + self.scene.cube_size + 0.03
                m.pose.orientation.w = 1.0
                m.scale.z = 0.018
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.3, 1.0
                m.text = obj.replace('_cube', '')
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    # ----------------------------------------------------------- run a cmd
    def handle_command(self, command):
        report = self.pipeline.run(command, dry_run=self.get_parameter('dry_run').value)
        if report.get('plan'):
            self.plan_pub.publish(String(data=json.dumps({'plan': report['plan']})))
        self.result_pub.publish(String(data=json.dumps(report, ensure_ascii=False,
                                                       default=str)))
        return report


def _stdin_reader(q, stop, ready):
    while not stop.is_set():
        ready.wait()          # do not prompt while a command is being executed
        ready.clear()
        try:
            line = input('\nCommand> ')
        except (EOFError, KeyboardInterrupt):
            q.put('quit')
            return
        if line.strip():
            q.put(line.strip())
        else:
            ready.set()       # empty line: prompt again


def main(args=None):
    rclpy.init(args=args)
    try:
        node = LLMRobotNode()
    except (LLMError, OSError, KeyError, ValueError) as e:
        print(f'[llm_robot_node] configuration error: {e}', file=sys.stderr)
        rclpy.shutdown()
        return 1
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    out('=' * 60)
    out('UR3 LLM CONTROL')
    out(node.student_info)
    out(f'LLM (9Router): {node.llm_desc}')
    out('=' * 60)
    code = 0
    stop = threading.Event()
    ready = threading.Event()
    ready.set()
    try:
        if not node.prepare_robot():
            code = 1
        else:
            one_shot = node.get_parameter('command').value
            if one_shot:
                rep = node.handle_command(one_shot)
                code = 0 if rep['status'] in ('TASK SUCCESS', 'DRY RUN') else 2
            else:
                if node.get_parameter('interactive').value and sys.stdin.isatty():
                    out("Type a command ('state' shows the world state, 'quit' exits). "
                        "Commands on /llm_robot/command are accepted too.")
                    threading.Thread(target=_stdin_reader, args=(node.commands, stop, ready),
                                     daemon=True).start()
                else:
                    out('Waiting for commands on /llm_robot/command ...')
                while rclpy.ok():
                    try:
                        cmd = node.commands.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if cmd.lower() in ('quit', 'exit', 'q'):
                        break
                    if cmd.lower() == 'state':
                        st = node.skills.detect_objects()
                        out(f'detect_objects() ... {st}')
                        node.pipeline.print_world('World state (camera):')
                    else:
                        node.handle_command(cmd)
                    ready.set()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        _shutdown(executor, spin, node)
    # Leave without running the Python finaliser: with a MultiThreadedExecutor,
    # tf2 and camera subscriptions, rclpy's C++ objects can be destroyed from
    # the wrong thread at interpreter exit ("terminate called without an active
    # exception").  Everything is already shut down cleanly at this point.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _shutdown(executor, spin, node):
    try:
        executor.shutdown(timeout_sec=3.0)
    except Exception:
        pass
    spin.join(timeout=3.0)
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        rclpy.try_shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    sys.exit(main())
