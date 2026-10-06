"""Test the LLM part WITHOUT ROS / Gazebo (real 9Router, simulated robot + camera).

    python3 -m ur3_llm_control.offline_cli "Put the red cube in zone B."
    python3 -m ur3_llm_control.offline_cli            # interactive (state persists)

The robot is FakeMotionBackend (UR kinematics, a gripper that must really
close around a block) and the camera is FakePerception (synthetic image of
the table run through the REAL vision pipeline), starting from the spawn
layout of scene.yaml.  A quick way to check prompts, validator, conflict
handling and executor logic on any machine.
"""
import argparse
import sys

import yaml

from .fake_backend import FakeMotionBackend, FakePerception, FakeWorld
from .llm_planner import LLMError, LLMPlanner, make_client_from_config
from .paths import config_path
from .pipeline import CommandPipeline
from .robot_skills import RobotSkills
from .scene_model import SceneConfig, WorldState
from .student_task import describe
from .task_validator import TaskValidator


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('command', nargs='*')
    ap.add_argument('--scene', default=config_path('scene.yaml'))
    ap.add_argument('--student', default=config_path('student_config.yaml'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--save-image', default='',
                    help='write the last synthetic camera image to this .png/.ppm file')
    a = ap.parse_args(argv)

    scene = SceneConfig.from_file(a.scene)
    with open(a.student, encoding='utf-8') as f:
        student = yaml.safe_load(f)
    truth = FakeWorld(scene)
    world = WorldState(scene)
    perception = FakePerception(scene, truth)
    skills = RobotSkills(scene, FakeMotionBackend(scene, truth), perception, world)
    try:
        client = make_client_from_config(student.get('llm', {}))
    except LLMError as e:
        print(e)
        return 1
    planner = LLMPlanner(client, scene, TaskValidator(scene), student['student_name'],
                         student['student_id'], student.get('llm', {}).get('max_attempts', 2))
    pipe = CommandPipeline(planner, skills, world)
    print(describe(student['student_name'], student['student_id']))

    def save():
        if a.save_image and perception.last_image is not None:
            from .vision import save_image
            print('camera image:', save_image(a.save_image, perception.last_image))

    if a.command:
        rep = pipe.run(' '.join(a.command), dry_run=a.dry_run)
        save()
        return 0 if rep['status'] in ('TASK SUCCESS', 'DRY RUN') else 2
    while True:
        try:
            cmd = input('\nCommand> ').strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if cmd in ('quit', 'exit', 'q'):
            return 0
        if cmd == 'state':
            skills.detect_objects()
            pipe.print_world('World state (camera):')
        elif cmd:
            pipe.run(cmd, dry_run=a.dry_run)
            save()


if __name__ == '__main__':
    sys.exit(main())
