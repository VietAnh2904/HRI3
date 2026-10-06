"""Generate the Gazebo Classic world (SDF 1.6) from scene.yaml.

    python3 -m ur3_llm_control.world_gen config/scene.yaml > /tmp/ur3_llm.world

* table (static, neutral grey so it is never mistaken for a block colour),
* zone pads (visual only: white square with a dark border),
* 5 blocks = DYNAMIC rigid bodies (mass, inertia, gravity, friction): they
  only move when the gripper fingers squeeze and carry them,
* the overhead RGB camera (static model, gazebo_ros_camera plugin publishing
  /<name>/image_raw and /<name>/camera_info in frame <optical_frame_id>),
* ODE settings chosen for stable friction grasps (1 kHz, 100 solver
  iterations, small contact correction velocity).
"""
import sys

from .scene_model import SceneConfig


def _rgba(v):
    return ' '.join(f'{float(c):.3f}' for c in v)


def _visual_box(name, xyz, size, rgba):
    sx, sy, sz = size
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>{_rgba(rgba)}</ambient><diffuse>{_rgba(rgba)}</diffuse></material>
        </visual>
      </link>
    </model>"""


def _table(scene):
    tx, ty = scene.table['center']
    sx, sy = scene.table['size']
    h = scene.table_top
    rgba = (0.62, 0.62, 0.64, 1.0)
    return f"""
    <model name="work_table">
      <static>true</static>
      <pose>{tx} {ty} {h / 2} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx} {sy} {h}</size></box></geometry>
          <surface>
            <friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction>
            <contact><ode><kp>1000000.0</kp><kd>1.0</kd><min_depth>0.001</min_depth>
              <max_vel>0.0</max_vel></ode></contact>
          </surface>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx} {sy} {h}</size></box></geometry>
          <material><ambient>{_rgba(rgba)}</ambient><diffuse>{_rgba(rgba)}</diffuse></material>
        </visual>
      </link>
    </model>"""


def _cube(scene, name, obj):
    s = scene.cube_size
    m = scene.cube_mass
    i = m * s * s / 6.0
    x, y = obj['spawn_xy']
    z = scene.cube_center_z() + 0.0005        # dropped 0.5 mm, settles at start
    rgba = obj['rgba']
    return f"""
    <model name="{name}">
      <static>false</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>{m}</mass>
          <inertia><ixx>{i:.8f}</ixx><iyy>{i:.8f}</iyy><izz>{i:.8f}</izz>
            <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <collision name="collision">
          <geometry><box><size>{s} {s} {s}</size></box></geometry>
          <surface>
            <friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction>
            <contact><ode><kp>1000000.0</kp><kd>1.0</kd><min_depth>0.001</min_depth>
              <max_vel>0.0</max_vel></ode></contact>
          </surface>
        </collision>
        <visual name="visual">
          <geometry><box><size>{s} {s} {s}</size></box></geometry>
          <material><ambient>{_rgba(rgba)}</ambient><diffuse>{_rgba(rgba)}</diffuse>
            <specular>0.1 0.1 0.1 1</specular></material>
        </visual>
      </link>
    </model>"""


def _camera(scene):
    cam = scene.camera
    x, y, z = cam['xyz']
    r, p, yw = cam['rpy']
    name = cam.get('name', 'overhead_camera')
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z} {r} {p} {yw}</pose>
      <link name="link">
        <visual name="body">
          <pose>-0.03 0 0 0 0 0</pose>
          <geometry><box><size>0.06 0.08 0.05</size></box></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material>
        </visual>
        <sensor name="{name}" type="camera">
          <always_on>true</always_on>
          <update_rate>{float(cam.get('update_rate', 10.0))}</update_rate>
          <visualize>false</visualize>
          <camera name="{name}">
            <horizontal_fov>{float(cam['horizontal_fov'])}</horizontal_fov>
            <image>
              <width>{int(cam['width'])}</width>
              <height>{int(cam['height'])}</height>
              <format>R8G8B8</format>
            </image>
            <clip><near>0.05</near><far>5.0</far></clip>
          </camera>
          <plugin name="{name}_driver" filename="libgazebo_ros_camera.so">
            <camera_name>{name}</camera_name>
            <frame_name>{cam.get('optical_frame_id', 'camera_optical_frame')}</frame_name>
          </plugin>
        </sensor>
      </link>
    </model>"""


def generate(scene: SceneConfig) -> str:
    h = scene.table_top
    models = [_table(scene)]
    for name in scene.zones:
        x, y = scene.zone_xy(name)
        half = scene.zone_half(name)
        models.append(_visual_box(f'{name}_border', (x, y, h + 0.0005),
                                  (2 * half + 0.01, 2 * half + 0.01, 0.001), (0.15, 0.15, 0.15, 1)))
        models.append(_visual_box(f'{name}_pad', (x, y, h + 0.001),
                                  (2 * half, 2 * half, 0.0012), (0.95, 0.95, 0.95, 1)))
    for name, o in scene.objects.items():
        models.append(_cube(scene, name, o))
    if scene.camera:
        models.append(_camera(scene))
    cam = scene.camera.get('xyz', (0.3, 0.0, 1.0))
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="ur3_llm_world">
    <include><uri>model://ground_plane</uri></include>
    <include><uri>model://sun</uri></include>
    <gravity>0 0 -9.81</gravity>
    <physics name="grasp_physics" default="true" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver><type>quick</type><iters>100</iters><sor>1.3</sor></solver>
        <constraints>
          <cfm>0.0</cfm><erp>0.2</erp>
          <contact_max_correcting_vel>0.1</contact_max_correcting_vel>
          <contact_surface_layer>0.0005</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    <scene>
      <shadows>false</shadows>
      <ambient>0.55 0.55 0.55 1</ambient>
    </scene>
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>10.0</update_rate>
    </plugin>
    <gui><camera name="user_camera"><pose>1.25 -0.95 0.95 0 0.42 2.45</pose></camera></gui>
    <!-- overhead camera mount: pole beyond the far edge of the table (visual only,
         out of the robot's reach and outside the camera's view of the table) -->
    {_visual_box('camera_pole', (cam[0] + 0.40, cam[1], (cam[2] + 0.05) / 2),
                 (0.03, 0.03, cam[2] + 0.05), (0.3, 0.3, 0.3, 1))}
    {_visual_box('camera_arm', (cam[0] + 0.20, cam[1], cam[2] + 0.04), (0.43, 0.03, 0.03),
                 (0.3, 0.3, 0.3, 1))}
    {''.join(models)}
  </world>
</sdf>
"""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    print(generate(SceneConfig.from_file(argv[0])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
