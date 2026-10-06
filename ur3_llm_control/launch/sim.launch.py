"""Gazebo Classic + UR3/UR3e + gripper + camera + ros2_control + MoveIt 2 + RViz.

    ros2 launch ur3_llm_control sim.launch.py                   # ur_type from scene.yaml (ur3e)
    ros2 launch ur3_llm_control sim.launch.py ur_type:=ur3
    ros2 launch ur3_llm_control sim.launch.py gazebo_gui:=false   # lighter

Start-up order (each step waits for the previous one):
    gzserver (world from scene.yaml: table, zones, 5 blocks, overhead camera)
    + robot_state_publisher (UR + gripper, urdf/ur_gripper.urdf.xacro)
    + static TF world -> camera_link -> camera_optical_frame
      -> spawn_entity (robot "ur")
      -> joint_state_broadcaster -> joint_trajectory_controller -> gripper_controller
      -> MoveIt 2 move_group (+ RViz)   (launch/moveit.launch.py)
"""
import math
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

PKG = 'ur3_llm_control'
SUPPORTED = ('ur3', 'ur3e')


def _default_scene():
    return os.path.join(get_package_share_directory(PKG), 'config', 'scene.yaml')


def _scene_ur_type(path):
    try:
        with open(path, encoding='utf-8') as f:
            return str(yaml.safe_load(f).get('ur_type', 'ur3e'))
    except OSError:
        return 'ur3e'


def launch_setup(context, *args, **kwargs):
    from ur3_llm_control.scene_model import SceneConfig
    from ur3_llm_control.world_gen import generate

    ur_type = LaunchConfiguration('ur_type').perform(context)
    scene_file = LaunchConfiguration('scene_file').perform(context)
    gazebo_gui = LaunchConfiguration('gazebo_gui')
    launch_rviz = LaunchConfiguration('launch_rviz')
    safety_limits = LaunchConfiguration('safety_limits')
    share = get_package_share_directory(PKG)

    if ur_type not in SUPPORTED:
        raise RuntimeError(f'ur_type:={ur_type} is not supported (use one of {SUPPORTED})')

    scene = SceneConfig.from_file(scene_file)
    actions = []
    if scene.ur_type != ur_type:
        actions.append(LogInfo(msg=f'[sim.launch] WARNING: ur_type:={ur_type} but {scene_file} '
                                   f'says ur_type: {scene.ur_type}. Reachability was checked for '
                                   f'{scene.ur_type}; run `ros2 run {PKG} check_scene`.'))

    # ------------------------------------------------------------ world file
    world_path = os.path.join(tempfile.gettempdir(), 'ur3_llm_world.world')
    with open(world_path, 'w', encoding='utf-8') as f:
        f.write(generate(scene))
    actions.append(LogInfo(msg=f'[sim.launch] Gazebo world generated: {world_path}'))

    # ----------------------------------------------------- robot description
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]), ' ',
        os.path.join(share, 'urdf', 'ur_gripper.urdf.xacro'), ' ',
        'name:=ur', ' ',
        'ur_type:=', ur_type, ' ',
        'safety_limits:=', safety_limits, ' ',
        'sim_gazebo:=true', ' ',
        'simulation_controllers:=', os.path.join(share, 'config', 'ros2_controllers.yaml'), ' ',
        'initial_positions_file:=', os.path.join(share, 'config', 'initial_positions.yaml'), ' ',
        f'gripper_tcp:={scene.tcp_offset}',
    ])
    robot_description = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)}

    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher', output='both',
        parameters=[{'use_sim_time': True}, robot_description])

    # camera extrinsics (the same numbers the Gazebo world uses)
    cam = scene.camera
    x, y, z = (str(v) for v in cam['xyz'])
    r, p, yw = (str(v) for v in cam['rpy'])
    cam_link = cam.get('frame_id', 'camera_link')
    cam_opt = cam.get('optical_frame_id', 'camera_optical_frame')
    camera_tf = Node(
        package='tf2_ros', executable='static_transform_publisher', name='camera_link_tf',
        arguments=['--x', x, '--y', y, '--z', z, '--roll', r, '--pitch', p, '--yaw', yw,
                   '--frame-id', scene.frame_id, '--child-frame-id', cam_link],
        parameters=[{'use_sim_time': True}])
    camera_optical_tf = Node(
        package='tf2_ros', executable='static_transform_publisher', name='camera_optical_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--roll', str(-math.pi / 2), '--pitch', '0', '--yaw', str(-math.pi / 2),
                   '--frame-id', cam_link, '--child-frame-id', cam_opt],
        parameters=[{'use_sim_time': True}])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare('gazebo_ros'), '/launch', '/gazebo.launch.py']),
        launch_arguments={'world': world_path, 'gui': gazebo_gui, 'verbose': 'false'}.items())

    spawn_robot = Node(
        package='gazebo_ros', executable='spawn_entity.py', name='spawn_ur', output='screen',
        arguments=['-entity', 'ur', '-topic', 'robot_description', '-timeout', '120'])

    def spawner(name):
        return Node(package='controller_manager', executable='spawner', output='screen',
                    arguments=[name, '--controller-manager', '/controller_manager',
                               '--controller-manager-timeout', '120'])

    jsb_spawner = spawner('joint_state_broadcaster')
    jtc_spawner = spawner('joint_trajectory_controller')
    gripper_spawner = spawner('gripper_controller')

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'moveit.launch.py')),
        launch_arguments={
            'ur_type': ur_type,
            'safety_limits': safety_limits,
            'use_sim_time': 'true',
            'launch_rviz': launch_rviz,
            'scene_file': scene_file,
        }.items())

    actions += [
        robot_state_publisher,
        camera_tf,
        camera_optical_tf,
        gazebo,
        spawn_robot,
        RegisterEventHandler(OnProcessExit(target_action=spawn_robot, on_exit=[jsb_spawner])),
        RegisterEventHandler(OnProcessExit(target_action=jsb_spawner, on_exit=[jtc_spawner])),
        RegisterEventHandler(OnProcessExit(target_action=jtc_spawner, on_exit=[gripper_spawner])),
    ]
    if LaunchConfiguration('launch_moveit').perform(context).lower() == 'true':
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=gripper_spawner,
            on_exit=[LogInfo(msg='[sim.launch] controllers up -> starting MoveIt 2'), moveit])))
    return actions


def generate_launch_description():
    scene = _default_scene()
    return LaunchDescription([
        DeclareLaunchArgument('ur_type', default_value=_scene_ur_type(scene),
                              choices=list(SUPPORTED),
                              description='UR3 or UR3e (default: ur_type in scene.yaml)'),
        DeclareLaunchArgument('scene_file', default_value=scene,
                              description='scene description (table, blocks, zones, camera)'),
        DeclareLaunchArgument('gazebo_gui', default_value='true',
                              description='start the Gazebo client window'),
        DeclareLaunchArgument('launch_rviz', default_value='true',
                              description='start RViz with the MoveIt plugin'),
        DeclareLaunchArgument('safety_limits', default_value='true',
                              description='UR safety limits in the URDF'),
        DeclareLaunchArgument('launch_moveit', default_value='true',
                              description='start MoveIt 2 (false: Gazebo + controllers only)'),
        OpaqueFunction(function=launch_setup),
    ])
