"""MoveIt 2 move_group (+ RViz) for the UR3/UR3e WITH the gripper.

Same parameters as ur_moveit_config/launch/ur_moveit.launch.py, but the
robot model is urdf/ur_gripper.urdf.xacro and the semantic model
srdf/ur_gripper.srdf, so MoveIt checks collisions of the gripper fingers and
of the held block too.  Normally started by sim.launch.py.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PKG = 'ur3_llm_control'


def _load_yaml(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def launch_setup(context, *args, **kwargs):
    from ur3_llm_control.scene_model import SceneConfig

    share = get_package_share_directory(PKG)
    cfg = os.path.join(share, 'config')
    ur_type = LaunchConfiguration('ur_type').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    safety_limits = LaunchConfiguration('safety_limits')
    launch_rviz = LaunchConfiguration('launch_rviz')
    scene = SceneConfig.from_file(LaunchConfiguration('scene_file').perform(context))

    robot_description = {'robot_description': ParameterValue(Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]), ' ',
        os.path.join(share, 'urdf', 'ur_gripper.urdf.xacro'), ' ',
        'name:=ur', ' ',
        'ur_type:=', ur_type, ' ',
        'safety_limits:=', safety_limits, ' ',
        'sim_gazebo:=false', ' ',
        f'gripper_tcp:={scene.tcp_offset}',
    ]), value_type=str)}
    with open(os.path.join(share, 'srdf', 'ur_gripper.srdf'), encoding='utf-8') as f:
        robot_description_semantic = {'robot_description_semantic': f.read()}
    robot_description_kinematics = {
        'robot_description_kinematics': _load_yaml(os.path.join(cfg, 'kinematics.yaml'))}
    robot_description_planning = {
        'robot_description_planning': _load_yaml(os.path.join(cfg, 'moveit_joint_limits.yaml'))}

    ompl = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': (
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/FixStartStatePathConstraints'),
            'start_state_max_bounds_error': 0.1,
        }
    }
    ompl['move_group'].update(_load_yaml(os.path.join(cfg, 'ompl_planning.yaml')))

    moveit_controllers = {
        'moveit_simple_controller_manager': _load_yaml(os.path.join(cfg, 'moveit_controllers.yaml')),
        'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
    }
    trajectory_execution = {
        'moveit_manage_controllers': False,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
        'trajectory_execution.execution_duration_monitoring': False,
    }
    planning_scene_monitor = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
        'publish_robot_description_semantic': True,
    }

    move_group = Node(
        package='moveit_ros_move_group', executable='move_group', output='screen',
        parameters=[robot_description, robot_description_semantic, robot_description_kinematics,
                    robot_description_planning, ompl, trajectory_execution, moveit_controllers,
                    planning_scene_monitor, {'use_sim_time': use_sim_time}])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2_moveit', output='log',
        condition=IfCondition(launch_rviz),
        arguments=['-d', os.path.join(share, 'rviz', 'view_robot.rviz')],
        parameters=[robot_description, robot_description_semantic, ompl,
                    robot_description_kinematics, robot_description_planning,
                    {'use_sim_time': use_sim_time}])
    return [move_group, rviz]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('ur_type', default_value='ur3e', choices=['ur3', 'ur3e']),
        DeclareLaunchArgument('safety_limits', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),
        DeclareLaunchArgument(
            'scene_file',
            default_value=os.path.join(get_package_share_directory(PKG), 'config', 'scene.yaml')),
        OpaqueFunction(function=launch_setup),
    ])
