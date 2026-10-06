#!/usr/bin/env bash
# Offline checks (no Gazebo / MoveIt needed). Run from anywhere:
#   /workspaces/ur_gazebo/src/ur3_llm_control/tools/run_all_checks.sh
# Exit code 0 = everything passed.
set -u
PKG="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PKG"
export PYTHONPATH="$PKG:${PYTHONPATH:-}"
fail=0
step() { echo; echo "=== $1"; }
ok()   { echo "    -> OK"; }
bad()  { echo "    -> FAILED"; fail=1; }

step "1/6 python syntax + style (package + launch files)"
python3 -m py_compile ur3_llm_control/*.py launch/*.py && ok || bad
if python3 -m flake8 --version >/dev/null 2>&1; then python3 -m flake8 ur3_llm_control launch test && echo "    flake8 clean" || bad; fi

step "2/6 unit + integration tests: skills, validator, executor, vision (pytest, no ROS)"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test -q && ok || bad

step "3/6 scene: reachability with the gripper, camera view, home out of view (UR3 + UR3e)"
python3 -m ur3_llm_control.check_scene config/scene.yaml && ok || bad

step "4/6 Gazebo world generated from scene.yaml is valid XML"
python3 - <<'EOF' && ok || bad
import xml.dom.minidom
from ur3_llm_control.scene_model import SceneConfig
from ur3_llm_control.world_gen import generate
xml.dom.minidom.parseString(generate(SceneConfig.from_file('config/scene.yaml')))
EOF

step "5/6 student config + LLM settings"
python3 - <<'EOF' && ok || bad
import os, yaml
from ur3_llm_control.student_task import describe
c = yaml.safe_load(open('config/student_config.yaml', encoding='utf-8'))
print(describe(c['student_name'], c['student_id']))
llm = c.get('llm', {})
print('model   :', os.environ.get('NINEROUTER_MODEL') or llm.get('model'))
print('base_url:', os.environ.get('NINEROUTER_BASE_URL') or llm.get('base_url'))
key = os.environ.get('NINEROUTER_API_KEY') or llm.get('api_key')
print('api key :', 'set' if key else 'NOT SET (export NINEROUTER_API_KEY=...)')
EOF

if command -v ros2 >/dev/null 2>&1 && ros2 pkg prefix ur3_llm_control >/dev/null 2>&1; then
  step "6/6 ROS 2 environment + robot description"
  for p in ur_description ur_moveit_config gazebo_ros gazebo_plugins gazebo_ros2_control \
           joint_trajectory_controller joint_state_broadcaster moveit_ros_move_group \
           moveit_planners_ompl moveit_simple_controller_manager xacro; do
    if ros2 pkg prefix "$p" >/dev/null 2>&1; then echo "    $p: found"; else echo "    $p: MISSING"; fail=1; fi
  done
  SHARE="$(ros2 pkg prefix ur3_llm_control)/share/ur3_llm_control"
  if xacro "$SHARE/urdf/ur_gripper.urdf.xacro" ur_type:=ur3e name:=ur sim_gazebo:=true \
       simulation_controllers:="$SHARE/config/ros2_controllers.yaml" > /tmp/ur3_llm_check.urdf; then
    if command -v check_urdf >/dev/null 2>&1; then
      check_urdf /tmp/ur3_llm_check.urdf >/dev/null && echo "    URDF (UR3e + gripper): OK" || bad
    else
      echo "    URDF generated (check_urdf not installed)"
    fi
  else
    bad
  fi
else
  echo
  echo "=== 6/6 skipped (package not built / ROS 2 not sourced)"
fi

echo
if [ $fail -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "SOME CHECKS FAILED"; fi
exit $fail
