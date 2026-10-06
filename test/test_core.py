"""Unit + integration tests that need no ROS (run with: python3 -m pytest test).

The 'integration' tests run the real pipeline (prompt -> validator -> executor
-> skills -> vision) against FakeMotionBackend / FakePerception: a simulated
arm whose gripper must really close around a block, and a synthetic camera
image analysed by the real vision code.
"""
import http.server
import json
import math
import os
import threading

import numpy as np
import pytest

from ur3_llm_control.check_scene import check
from ur3_llm_control.fake_backend import FakeMotionBackend, FakePerception, FakeWorld
from ur3_llm_control.llm_planner import (LLMError, LLMPlanner, OpenAICompatibleClient,
                                         build_system_prompt, build_user_prompt, extract_json)
from ur3_llm_control.pipeline import CommandPipeline
from ur3_llm_control.robot_skills import RobotSkills
from ur3_llm_control.scene_model import TEMP, SceneConfig, WorldState
from ur3_llm_control.skill_executor import PlanExpansionError, SkillExecutor, expand_plan
from ur3_llm_control.skill_spec import SKILLS, Status
from ur3_llm_control.student_task import compute_p, zone_assignment
from ur3_llm_control.task_validator import TaskValidator
from ur3_llm_control.ur_kinematics import URKinematics, tool_yaw
from ur3_llm_control.world_gen import generate

HERE = os.path.dirname(__file__)
CFG = os.path.join(HERE, '..', 'config', 'scene.yaml')
SPAWN = {'red_cube': (0.22, -0.16), 'yellow_cube': (0.22, 0.0), 'green_cube': (0.22, 0.16),
         'blue_cube': (0.40, 0.0), 'purple_cube': (0.30, -0.30)}


@pytest.fixture(scope='module')
def scene():
    return SceneConfig.from_file(CFG)


@pytest.fixture
def world(scene):
    w = WorldState(scene)
    w.set_detections({n: (x, y, 0.0) for n, (x, y) in SPAWN.items()})
    return w


def P(*steps):
    out = []
    for s in steps:
        name, *args = s
        d = {'skill': name}
        if name in ('pick', 'move_above', 'find_object', 'find_free_position'):
            d['object'] = args[0]
        elif name == 'place':
            d['object'], d['zone'] = args
        elif name in ('move_to_zone', 'check_zone'):
            d['zone'] = args[0]
        out.append(d)
    return {'plan': out}


DEMO = P(('check_zone', 'zone_b'), ('pick', 'blue_cube'), ('find_free_position', 'blue_cube'),
         ('place', 'blue_cube', TEMP), ('pick', 'red_cube'), ('place', 'red_cube', 'zone_b'),
         ('home',))


class FakeLLM:
    """Scripted LLM: returns the given answers in order, records prompts."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.model, self.url = 'fake', 'fake://'
        self.seen = []

    def chat(self, messages):
        self.seen.append(messages)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a if isinstance(a, str) else json.dumps(a)


def make_planner(scene, llm, attempts=2, sid='23020754'):
    return LLMPlanner(llm, scene, TaskValidator(scene), 'Le Trong Nghia', sid,
                      max_attempts=attempts, log=lambda *_: None)


class Rig:
    """Simulated robot + camera + pipeline."""

    def __init__(self, scene, answers=(), positions=None, fail_on=None, sid='23020754'):
        self.truth = FakeWorld(scene, positions)
        self.backend = FakeMotionBackend(scene, self.truth, fail_on=fail_on)
        self.perception = FakePerception(scene, self.truth)
        self.world = WorldState(scene)
        self.lines = []
        self.skills = RobotSkills(scene, self.backend, self.perception, self.world,
                                  log=self.lines.append)
        self.llm = FakeLLM(*answers)
        self.pipe = CommandPipeline(make_planner(scene, self.llm, sid=sid), self.skills,
                                    self.world, log=self.lines.append)

    def run(self, command='cmd'):
        rep = self.pipe.run(command)
        return rep, '\n'.join(str(x) for x in self.lines)

    def truth_zone(self, obj, scene):
        x, y = self.truth.cubes[obj]
        for z in scene.zones:
            zx, zy = scene.zone_xy(z)
            if abs(x - zx) <= scene.zone_half(z) and abs(y - zy) <= scene.zone_half(z):
                return z
        return None


# ================================================================ student
@pytest.mark.parametrize('sid,p,a,b,c', [
    ('23020123', 5, 'blue', 'yellow', 'red'),     # example of the assignment
    ('23020104', 4, 'blue', 'red', 'yellow'),
    ('23020100', 0, 'red', 'yellow', 'blue'),
    ('23020754', 0, 'red', 'yellow', 'blue'),
    ('23020107', 1, 'red', 'blue', 'yellow'),
    ('23020108', 2, 'yellow', 'red', 'blue'),
    ('23020109', 3, 'yellow', 'blue', 'red'),
])
def test_student_mapping(sid, p, a, b, c):
    assert compute_p(sid) == p
    assert zone_assignment(sid) == {'zone_a': f'{a}_cube', 'zone_b': f'{b}_cube',
                                    'zone_c': f'{c}_cube'}


def test_student_id_invalid():
    with pytest.raises(ValueError):
        compute_p('2A')
    assert compute_p(23020199) == 99 % 6      # int accepted


# ================================================================== scene
def test_scene_has_five_blocks_three_zones(scene):
    assert len(scene.objects) == 5 and len(scene.zones) == 3
    assert {o['color'] for o in scene.objects.values()} == {
        'red', 'yellow', 'green', 'blue', 'purple'}
    assert scene.camera and scene.gripper


def test_demo_layout_blue_in_zone_b(scene):
    """The assignment's demo needs Zone B occupied by blue_cube at start."""
    w = WorldState(scene)
    w.set_detections({n: (*o['spawn_xy'], 0.0) for n, o in scene.objects.items()})
    assert w.occupant('zone_b') == 'blue_cube'
    assert w.occupant('zone_a') is None and w.occupant('zone_c') is None


def test_scene_check_passes_both_robots(scene):
    for ur in ('ur3', 'ur3e'):
        problems, home = check(scene, ur)
        assert problems == [], problems
        assert home[2, 2] < -0.99             # home: tool points down


def test_scene_rejects_bad_configs(scene):
    bad = dict(scene.raw)
    bad['objects'] = dict(bad['objects'], red_cube={'color': 'red', 'spawn_xy': [0.9, 0],
                                                    'rgba': [1, 0, 0, 1]})
    with pytest.raises(ValueError, match='outside'):
        SceneConfig(bad)
    bad = dict(scene.raw)
    bad['objects'] = dict(bad['objects'], red_cube={'color': 'blue', 'spawn_xy': [0.2, 0],
                                                    'rgba': [1, 0, 0, 1]})
    with pytest.raises(ValueError, match='colour'):
        SceneConfig(bad)
    bad = dict(scene.raw)
    bad['zones'] = dict(bad['zones'], **{TEMP: {'xy': [0.3, 0.3]}})
    with pytest.raises(ValueError, match='reserved'):
        SceneConfig(bad)


def test_world_file(scene):
    import xml.dom.minidom
    text = generate(scene)
    doc = xml.dom.minidom.parseString(text)
    names = {m.getAttribute('name') for m in doc.getElementsByTagName('model')}
    assert set(scene.objects) | {'work_table', 'overhead_camera'} <= names
    assert 'libgazebo_ros_camera.so' in text
    for m in doc.getElementsByTagName('model'):
        if m.getAttribute('name') in scene.objects:
            # real rigid bodies: not static, not kinematic, gravity on, with mass + friction
            static = m.getElementsByTagName('static')[0].firstChild.data
            assert static == 'false'
            assert not m.getElementsByTagName('kinematic')
            assert not m.getElementsByTagName('gravity')
            assert m.getElementsByTagName('mass') and m.getElementsByTagName('mu')


def test_nothing_teleports_objects():
    """Bai 03: the object pose may never be set directly."""
    root = os.path.join(HERE, '..', 'ur3_llm_control')
    for fn in os.listdir(root):
        if fn.endswith('.py'):
            src = open(os.path.join(root, fn), encoding='utf-8').read()
            assert 'set_entity_state' not in src, fn
            assert 'SetEntityState' not in src, fn
            assert 'SetModelState' not in src, fn


# ============================================================= kinematics
def test_ik_with_yaw_roundtrip(scene):
    kin = URKinematics('ur3e')
    home = scene.motion['home_joints']
    for xyz, yaw in (((0.30, 0.0, 0.31), 0.0), ((0.40, 0.16, 0.19), 0.3),
                     ((0.30, -0.30, 0.19), -1.2), ((0.22, 0.16, 0.19), 2.5)):
        q = kin.ik(xyz, yaw, home)
        assert q is not None
        t = kin.fk(q)
        assert np.allclose(t[:3, 3], xyz, atol=1e-4)
        assert t[2, 2] < -0.9999
        assert abs(((tool_yaw(t[:3, :3]) - yaw) + math.pi) % (2 * math.pi) - math.pi) < 1e-3
    assert kin.ik((0.9, 0.0, 0.19), 0.0, home) is None


def test_best_grasp_yaw_is_face_aligned(scene):
    kin = URKinematics('ur3e')
    yaw, q = kin.best_grasp_yaw((0.30, 0.0, 0.2), 0.35, scene.motion['home_joints'])
    d = ((yaw - 0.35) + math.pi / 4) % (math.pi / 2) - math.pi / 4
    assert abs(d) < 1e-6 and q is not None


def test_ik_stays_on_branch_near_seed(scene):
    """Seeded from the current joints, the IK must not jump to another
    kinematic branch (large detours / wrist flips in Gazebo)."""
    kin = URKinematics('ur3e')
    home = np.asarray(scene.motion['home_joints'])
    q1 = kin.ik((0.30, 0.0, scene.approach_tool_z()), 0.0, home)
    q2 = kin.ik((0.30, 0.0, scene.grasp_tool_z()), 0.0, q1)
    assert np.max(np.abs(q2 - q1)) < 0.6
    assert q1[2] > 0 and q2[2] > 0            # elbow up


# ================================================================== world
def test_occupancy_uses_block_footprint(scene, world):
    zx, zy = scene.zone_xy('zone_a')
    world.positions['green_cube'] = (zx, zy + scene.zone_half('zone_a') + 0.015)  # half inside
    assert world.slot_of('green_cube') is None
    assert world.occupant('zone_a') == 'green_cube'
    assert world.zone_status() == {'zone_a': 'green_cube', 'zone_b': 'blue_cube', 'zone_c': None}


def test_summary_marks_missing_blocks(scene):
    w = WorldState(scene)
    w.set_detections({'red_cube': (0.22, -0.16, 0.0)})
    s = w.summary()
    assert s['red_cube'].startswith('table') and s['blue_cube'] == 'not detected'


def test_free_position_rules(scene, world):
    kin = URKinematics(scene.ur_type)

    def reach(xy):
        return kin.best_grasp_yaw((xy[0], xy[1], scene.approach_tool_z()), 0.0,
                                  scene.motion['home_joints'])[1] is not None

    xy = world.free_position('blue_cube', reachable=reach)
    assert xy is not None and scene.on_table(xy, 0.02)
    for z in scene.zones:
        zx, zy = scene.zone_xy(z)
        assert max(abs(xy[0] - zx), abs(xy[1] - zy)) > scene.zone_half(z) + scene.cube_size / 2
    for o, p in world.positions.items():
        if o != 'blue_cube':
            assert math.hypot(xy[0] - p[0], xy[1] - p[1]) >= 0.10 - 1e-9
    assert reach(xy)


def test_free_position_none_when_table_full(scene, world):
    assert world.free_position('red_cube', reachable=lambda xy: False) is None


# ============================================================== validator
def test_valid_demo_plan(scene, world):
    r = TaskValidator(scene).validate(DEMO, world)
    assert r.ok, r.errors
    assert r.plan[3] == {'skill': 'place', 'object': 'blue_cube', 'zone': TEMP}
    assert not r.warnings


def test_validator_warns_on_occupied_zone(scene, world):
    r = TaskValidator(scene).validate(
        P(('pick', 'red_cube'), ('place', 'red_cube', 'zone_b'), ('home',)), world)
    assert r.ok and any('occupied by blue_cube' in w for w in r.warnings)


def test_normalises_case_spaces_and_aliases(scene, world):
    raw = {'plan': [{'skill': 'Pick', 'object': 'Red Cube'},
                    {'skill': 'place', 'object': 'red-cube', 'location': 'Zone A'}]}
    r = TaskValidator(scene).validate(raw, world)
    assert r.ok, r.errors
    assert r.plan[1] == {'skill': 'place', 'object': 'red_cube', 'zone': 'zone_a'}
    assert r.plan[-1] == {'skill': 'home'}


@pytest.mark.parametrize('raw,needle', [
    ([], 'JSON object'),
    ({'steps': []}, 'missing "plan"'),
    ({'plan': []}, 'no plan'),
    ({'plan': [], 'error': 'no orange cube'}, 'no orange cube'),
    (P(('fly', 'red_cube')), 'not allowed'),
    ({'plan': [{'skill': 'move_joints', 'joints': [0, 1, 2, 3, 4, 5]}]}, 'not allowed'),
    ({'plan': [{'skill': 'home', 'joint_positions': [0] * 6}]}, 'low-level'),
    ({'plan': [{'skill': 'pick', 'object': 'red_cube', 'trajectory': []}]}, 'low-level'),
    ({'plan': [{'skill': 'place', 'object': 'red_cube', 'x': 0.3, 'y': 0.1}]}, 'low-level'),
    ({'plan': [{'skill': 'pick', 'object': 'red_cube', 'speed': 2}]}, 'unexpected'),
    (P(('pick', 'orange_cube')), 'INVALID_OBJECT'),
    (P(('pick', 'red')), 'INVALID_OBJECT'),
    (P(('pick', 'red_cube'), ('place', 'red_cube', 'zone_d')), 'INVALID_ZONE'),
    (P(('check_zone', TEMP)), 'INVALID_ZONE'),
    (P(('move_to_zone', TEMP)), 'INVALID_ZONE'),
    ({'plan': [{'skill': 'pick'}]}, 'needs string parameter'),
    ({'plan': [{'skill': 'pick', 'object': 3}]}, 'needs string parameter'),
    ({'plan': ['pick red']}, 'must be an object'),
    (P(('place', 'red_cube', 'zone_a')), 'holds nothing'),
    (P(('pick', 'red_cube'), ('pick', 'blue_cube')), 'already holding'),
    (P(('pick', 'red_cube'), ('place', 'blue_cube', 'zone_a')), 'gripper holds red_cube'),
    (P(('pick', 'red_cube'), ('home',)), 'still holding'),
    (P(('pick', 'red_cube'), ('move_above', 'red_cube')), 'not visible'),
    ({'plan': [{'skill': 'home'}] * 31}, 'too long'),
])
def test_validator_rejects(scene, world, raw, needle):
    r = TaskValidator(scene).validate(raw, world)
    assert not r.ok
    assert any(needle in e for e in r.errors), r.errors


def test_validator_rejects_block_not_seen_by_camera(scene, world):
    del world.positions['purple_cube']
    r = TaskValidator(scene).validate(
        P(('pick', 'purple_cube'), ('place', 'purple_cube', 'zone_a')), world)
    assert not r.ok and 'not detected by the camera' in r.errors[0]


# ================================================================ parsing
@pytest.mark.parametrize('text', [
    '{"plan": [{"skill": "home"}]}',
    '```json\n{"plan": [{"skill": "home"}]}\n```',
    'Sure! Here is the plan:\n{"plan": [{"skill": "home"}]}\nHope it helps',
    '<think>{"plan": "draft"}</think>{"plan": [{"skill": "home"}]}',
])
def test_extract_json(text):
    assert extract_json(text) == {'plan': [{'skill': 'home'}]}


def test_extract_json_fails():
    with pytest.raises(ValueError):
        extract_json('I cannot do that')


def test_prompt_contains_whitelist_rules_and_personal_task(scene):
    p = build_system_prompt(scene, 'Nguyen Van An', '23020123')
    for name in SKILLS:
        assert f'{name}(' in p
    for token in ('red_cube', 'purple_cube', 'zone_c', TEMP, 'check_zone(zone)',
                  'P = last two digits mod 6 = 5', 'blue_cube -> zone_a',
                  'yellow_cube -> zone_b', 'red_cube -> zone_c', 'OCCUPIED by blue_cube',
                  'never output coordinates'):
        assert token in p, token


def test_user_prompt_has_camera_state(scene, world):
    text = build_user_prompt('Put the red cube in Zone B.', world)
    assert 'blue_cube: in zone_b' in text
    assert 'zone_b: OCCUPIED by blue_cube' in text and 'zone_a: FREE' in text
    assert 'red_cube: on the table' in text


# ================================================================ planner
def test_planner_retries_after_rejection(scene, world):
    llm = FakeLLM(P(('pick', 'red_cube'), ('place', 'red_cube', 'zone_9')), DEMO)
    r, _ = make_planner(scene, llm).plan('put red in B', world)
    assert r.ok and len(llm.seen) == 2
    assert 'rejected by the validator' in llm.seen[1][-1]['content']


def test_planner_gives_up(scene, world):
    llm = FakeLLM('not json', 'still not json')
    r, _ = make_planner(scene, llm).plan('dance', world)
    assert not r.ok and len(llm.seen) == 2


def test_planner_llm_refusal_is_final(scene, world):
    llm = FakeLLM({'plan': [], 'error': 'there is no orange cube'}, 'SHOULD NOT BE USED')
    r, _ = make_planner(scene, llm).plan('move the orange cube', world)
    assert not r.ok and len(llm.seen) == 1 and 'orange' in r.errors[0]


# ================================================================ expansion
def _free(sim, obj, src):
    return sim.free_position(obj, src=src)


def test_expand_auto_clears_occupied_zone(scene, world):
    plan = TaskValidator(scene).validate(
        P(('pick', 'red_cube'), ('place', 'red_cube', 'zone_b'), ('home',)), world).plan
    steps = expand_plan(plan, world, _free)
    assert [(s['skill'], s.get('object'), s.get('auto', False)) for s in steps[:3]] == [
        ('pick', 'blue_cube', True), ('find_free_position', 'blue_cube', True),
        ('place', 'blue_cube', True)]
    assert steps[2]['zone'] == TEMP and steps[2]['xy'] == steps[1]['xy']
    assert [s['skill'] for s in steps[3:]] == ['pick', 'place', 'home']


def test_expand_respects_explicit_clearing(scene, world):
    steps = expand_plan(DEMO['plan'], world, _free)
    assert not any(s.get('auto') for s in steps)
    assert steps[3]['xy'] == steps[2]['xy'] and steps[3]['zone'] == TEMP
    assert '_expect' in steps[0] and steps[0]['_expect']['zone_b'] == 'blue_cube'


def test_expand_place_temp_without_find_free_inserts_it(scene, world):
    plan = P(('pick', 'blue_cube'), ('place', 'blue_cube', TEMP), ('home',))['plan']
    steps = expand_plan(plan, world, _free)
    assert [s['skill'] for s in steps] == ['pick', 'find_free_position', 'place', 'home']
    assert steps[1]['auto'] and steps[2]['xy'] == steps[1]['xy']


def test_expand_reorders_to_avoid_temporary_moves(scene, world):
    """blue in B; plan asks yellow->B, then blue->C: run blue->C first."""
    plan = P(('check_zone', 'zone_b'), ('pick', 'yellow_cube'), ('place', 'yellow_cube', 'zone_b'),
             ('check_zone', 'zone_c'), ('pick', 'blue_cube'), ('place', 'blue_cube', 'zone_c'),
             ('home',))['plan']
    steps = expand_plan(plan, world, _free)
    assert not any(s.get('auto') for s in steps)
    assert [s.get('object') for s in steps if s['skill'] == 'pick'] == ['blue_cube', 'yellow_cube']


def test_expand_skips_block_already_in_place(scene, world):
    plan = P(('check_zone', 'zone_b'), ('pick', 'blue_cube'), ('place', 'blue_cube', 'zone_b'),
             ('home',))['plan']
    steps = expand_plan(plan, world, _free)
    assert steps[1]['skip'] and steps[2]['skip'] and not steps[3].get('skip')


def test_expand_no_free_space(scene, world):
    plan = P(('pick', 'red_cube'), ('place', 'red_cube', 'zone_b'))['plan']
    with pytest.raises(PlanExpansionError):
        expand_plan(plan, world, lambda *a: None)


# ======================================================= skills (simulated)
def test_skills_guard_invalid_inputs(scene):
    rig = Rig(scene)
    sk = rig.skills
    assert sk.detect_objects().ok
    assert sk.pick('orange_cube').status == Status.INVALID_OBJECT
    assert sk.place('red_cube', 'zone_z').status == Status.INVALID_STATE     # not held
    assert sk.check_zone(TEMP).status == Status.INVALID_ZONE
    assert sk.pick('red_cube').ok
    assert sk.pick('blue_cube').status == Status.INVALID_STATE              # hand full
    assert sk.move_above('red_cube').status == Status.INVALID_STATE         # it is in the hand
    assert sk.place('red_cube', 'zone_b').status == Status.INVALID_STATE    # B occupied
    assert sk.place('red_cube', 'zone_a').ok
    assert sk.home().ok


def test_camera_sees_spawn_layout(scene):
    rig = Rig(scene)
    assert rig.skills.detect_objects().detail == '5/5 blocks detected'
    for n, (x, y) in SPAWN.items():
        px, py = rig.world.positions[n]
        assert math.hypot(px - x, py - y) < 0.003, n
    assert rig.skills.check_zone('zone_b').detail == 'OCCUPIED by blue_cube'
    assert rig.skills.check_zone('zone_a').detail == 'FREE'
    assert rig.skills.find_object('purple_cube').detail.startswith('table')


def test_pick_needs_a_real_grasp(scene):
    """If the gripper closes on nothing, pick reports GRASP_FAILED."""
    rig = Rig(scene, fail_on={'gripper_close'})
    rig.skills.detect_objects()
    st = rig.skills.pick('red_cube')
    assert st.status == Status.GRASP_FAILED and rig.world.held is None
    assert rig.truth.held is None


def test_perception_moves_to_home_first(scene):
    """A camera skill called with the arm over the table first goes home
    (otherwise the arm would hide blocks)."""
    rig = Rig(scene)
    rig.skills.detect_objects()
    assert rig.skills.move_above('yellow_cube').ok
    assert rig.skills.detect_objects().detail == '5/5 blocks detected'
    assert rig.backend.calls[-1][0] == 'move_joints'
    assert np.allclose(rig.truth.joints, scene.motion['home_joints'])


# ============================================== full pipeline (simulated)
def test_demo_red_to_occupied_zone_b(scene):
    """The assignment's demo: Zone B holds blue_cube, 'Put the red cube in Zone B.'"""
    rig = Rig(scene, [DEMO])
    rep, text = rig.run('Put the red cube in Zone B.')
    assert rep['status'] == 'TASK SUCCESS', text
    for token in ('USER COMMAND:', 'CAMERA (detect_objects):', 'OCCUPIED (blue_cube)',
                  'LLM PLAN:', 'check_zone(zone_b)', 'place(blue_cube, temporary_position)',
                  'EXECUTION:', 'OCCUPIED by blue_cube', 'VERIFY (camera):',
                  'red_cube in zone_b', 'blue_cube on the table', 'TASK SUCCESS'):
        assert token in text, token
    # ground truth of the simulated world, not only the belief
    assert rig.truth_zone('red_cube', scene) == 'zone_b'
    assert rig.truth_zone('blue_cube', scene) is None
    assert rig.truth.held is None
    # the camera prompt told the LLM that B is occupied
    assert 'zone_b: OCCUPIED by blue_cube' in rig.llm.seen[0][1]['content']
    order = [s[0] for s in rep['steps']]
    assert order.index('pick(blue_cube)') < order.index('pick(red_cube)')


def test_demo_when_llm_forgets_to_clear_the_zone(scene):
    rig = Rig(scene, [P(('pick', 'red_cube'), ('place', 'red_cube', 'zone_b'), ('home',))])
    rep, text = rig.run('Put the red cube in Zone B.')
    assert rep['status'] == 'TASK SUCCESS', text
    assert 'pick(blue_cube)  [auto]' in text
    assert rig.truth_zone('red_cube', scene) == 'zone_b'
    assert rig.truth_zone('blue_cube', scene) is None


def test_arrange_by_student_id_with_five_blocks(scene):
    """23020754 -> P = 0 -> A red, B yellow, C blue.  Blue starts in B."""
    plan = P(('check_zone', 'zone_a'), ('pick', 'red_cube'), ('place', 'red_cube', 'zone_a'),
             ('check_zone', 'zone_b'), ('pick', 'yellow_cube'), ('place', 'yellow_cube', 'zone_b'),
             ('check_zone', 'zone_c'), ('pick', 'blue_cube'), ('place', 'blue_cube', 'zone_c'),
             ('home',))
    rig = Rig(scene, [plan])
    rep, text = rig.run('Arrange all objects according to my student ID.')
    assert rep['status'] == 'TASK SUCCESS', text
    for zone, obj in zone_assignment('23020754').items():
        assert rig.truth_zone(obj, scene) == zone
    assert '[auto]' not in text                       # reordered: blue->C before yellow->B


def test_arrange_clears_blocks_that_are_not_in_the_mapping(scene):
    """green sits in zone_a and purple in zone_c; red/yellow/blue go to A/B/C."""
    pos = dict(SPAWN, green_cube=scene.zone_xy('zone_a'), purple_cube=scene.zone_xy('zone_c'),
               blue_cube=(0.22, 0.30))
    plan = P(('pick', 'red_cube'), ('place', 'red_cube', 'zone_a'),
             ('pick', 'yellow_cube'), ('place', 'yellow_cube', 'zone_b'),
             ('pick', 'blue_cube'), ('place', 'blue_cube', 'zone_c'), ('home',))
    rig = Rig(scene, [plan], positions=pos)
    rep, text = rig.run('Sắp xếp các khối theo MSSV của tôi.')
    assert rep['status'] == 'TASK SUCCESS', text
    assert text.count('[auto]') >= 6                   # green and purple parked
    for zone, obj in zone_assignment('23020754').items():
        assert rig.truth_zone(obj, scene) == zone
    assert rig.truth_zone('green_cube', scene) is None
    assert rig.truth_zone('purple_cube', scene) is None


def test_swap_two_blocks_uses_a_temporary_position(scene):
    pos = dict(SPAWN, red_cube=scene.zone_xy('zone_a'))       # red in A, blue in B
    plan = P(('pick', 'red_cube'), ('place', 'red_cube', 'zone_b'),
             ('pick', 'blue_cube'), ('place', 'blue_cube', 'zone_a'), ('home',))
    rig = Rig(scene, [plan], positions=pos)
    rep, text = rig.run('swap red and blue')
    assert rep['status'] == 'TASK SUCCESS', text
    assert '[auto]' in text
    assert rig.truth_zone('red_cube', scene) == 'zone_b'
    assert rig.truth_zone('blue_cube', scene) == 'zone_a'


def test_closed_loop_replans_when_camera_sees_a_change(scene):
    """Someone drops the green block into zone_b after the first camera
    snapshot: check_zone sees it and the executor clears it first."""
    plan = P(('check_zone', 'zone_a'), ('pick', 'red_cube'), ('place', 'red_cube', 'zone_a'),
             ('check_zone', 'zone_c'), ('pick', 'yellow_cube'), ('place', 'yellow_cube', 'zone_c'),
             ('home',))
    rig = Rig(scene, [plan])
    orig = rig.skills.check_zone

    def check_zone_with_disturbance(zone):
        if zone == 'zone_c':
            rig.truth.cubes['green_cube'] = scene.zone_xy('zone_c')
        return orig(zone)

    rig.skills.check_zone = check_zone_with_disturbance
    rep, text = rig.run('red to A and yellow to C')
    assert rep['status'] == 'TASK SUCCESS', text
    assert 'camera shows a different zone occupancy' in text
    assert rig.truth_zone('yellow_cube', scene) == 'zone_c'
    assert rig.truth_zone('green_cube', scene) is None


def test_rejected_plan_does_not_move_the_robot(scene):
    rig = Rig(scene, [{'plan': [{'skill': 'set_joints', 'joints': [0] * 6}]},
                      {'plan': [{'skill': 'home', 'joint_positions': [0] * 6}]}])
    rep, text = rig.run('Rotate joint 1 by 90 degrees')
    assert rep['status'] == 'TASK REJECTED'
    assert 'low-level robot control is forbidden' in text
    assert 'nothing was executed' in text
    assert rig.backend.calls == []           # only the camera was used


def test_llm_refuses_unknown_object(scene):
    rig = Rig(scene, [{'plan': [], 'error': 'there is no orange cube'}])
    rep, text = rig.run('Move the orange cube to zone A.')
    assert rep['status'] == 'TASK REJECTED' and 'orange' in text
    assert rig.backend.calls == []


def test_grasp_failure_stops_the_task(scene):
    rig = Rig(scene, [DEMO], fail_on={'gripper_close'})
    rep, text = rig.run('Put the red cube in Zone B.')
    assert rep['status'] == 'TASK FAILED'
    labels = [s[1] for s in rep['steps']]
    assert labels[1].startswith(Status.GRASP_FAILED)
    assert all(lab == 'NOT EXECUTED' for lab in labels[2:])


def test_planning_failure_stops_the_task(scene):
    rig = Rig(scene, [DEMO], fail_on={'move_vertical'})
    rep, text = rig.run('Put the red cube in Zone B.')
    assert rep['status'] == 'TASK FAILED' and 'NOT EXECUTED' in text


def test_llm_unreachable(scene):
    rig = Rig(scene, [LLMError('cannot reach 9Router')])
    rep, text = rig.run('x')
    assert rep['status'] == 'TASK REJECTED' and 'LLM ERROR' in text


def test_verification_detects_a_wrong_final_state(scene):
    rig = Rig(scene, [DEMO])
    ex = rig.pipe.executor
    orig_verify = ex.verify

    def verify_after_bump(steps):
        rig.truth.cubes['red_cube'] = (0.30, 0.10)          # knocked out of zone_b
        return orig_verify(steps)

    ex.verify = verify_after_bump
    rep, text = rig.run('Put the red cube in Zone B.')
    assert rep['status'] == 'TASK FAILED' and 'WRONG' in text


def test_executor_free_position_is_reachable(scene):
    rig = Rig(scene)
    rig.skills.detect_objects()
    ex = SkillExecutor(rig.skills, log=lambda *_: None)
    xy = ex.free_fn(rig.world, 'blue_cube', rig.world.positions['blue_cube'])
    assert rig.skills.reachable(xy)


# ============================================= real HTTP client (mock 9Router)
class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.last = (self.path, self.headers.get('Authorization'), body)
        if self.headers.get('Authorization') != 'Bearer good-key':
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"unauthorized"}')
            return
        content = '```json\n{"plan":[{"skill":"home"}]}\n```'
        answer = {'choices': [{'message': {'role': 'assistant', 'content': content}}]}
        data = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def mock_router():
    srv = http.server.HTTPServer(('127.0.0.1', 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def test_http_client_ok(mock_router):
    url = f'http://127.0.0.1:{mock_router.server_port}/v1'
    c = OpenAICompatibleClient(url, 'good-key', 'oc/muse-spark-1.2-contributor-free')
    text = c.chat([{'role': 'user', 'content': 'hi'}])
    assert extract_json(text) == {'plan': [{'skill': 'home'}]}
    path, auth, body = mock_router.last
    assert path == '/v1/chat/completions' and body['model'] == 'oc/muse-spark-1.2-contributor-free'
    assert body['temperature'] == 0.0


def test_http_client_bad_key(mock_router):
    c = OpenAICompatibleClient(f'http://127.0.0.1:{mock_router.server_port}/v1', 'bad', 'm')
    with pytest.raises(LLMError, match='401'):
        c.chat([{'role': 'user', 'content': 'hi'}])


def test_http_client_unreachable():
    c = OpenAICompatibleClient('http://127.0.0.1:9/v1', 'k', 'm', timeout_s=2)
    with pytest.raises(LLMError, match='cannot reach'):
        c.chat([{'role': 'user', 'content': 'hi'}])


def test_http_client_ignores_proxy_for_localhost(mock_router, monkeypatch):
    """A local 9Router must be reached directly even if http_proxy is set."""
    monkeypatch.setenv('http_proxy', 'http://127.0.0.1:9')      # dead proxy
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:9')
    monkeypatch.delenv('no_proxy', raising=False)
    monkeypatch.delenv('NO_PROXY', raising=False)
    c = OpenAICompatibleClient(f'http://localhost:{mock_router.server_port}/v1',
                               'good-key', 'oc/muse-spark-1.2-contributor-free')
    assert extract_json(c.chat([{'role': 'user', 'content': 'hi'}]))['plan']


def test_config_requires_model_and_key(monkeypatch):
    from ur3_llm_control.llm_planner import make_client_from_config
    for v in ('NINEROUTER_API_KEY', 'NINEROUTER_MODEL', 'NINEROUTER_BASE_URL'):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(LLMError, match='API key'):
        make_client_from_config({'model': 'oc/muse-spark-1.2-contributor-free', 'api_key': ''})
    monkeypatch.setenv('NINEROUTER_API_KEY', 'sk-env')
    c = make_client_from_config({'model': 'oc/muse-spark-1.2-contributor-free', 'api_key': ''})
    assert c.api_key == 'sk-env' and c.model == 'oc/muse-spark-1.2-contributor-free'


def test_shipped_config_has_model_and_no_secret():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(HERE, '..', 'config', 'student_config.yaml'),
                              encoding='utf-8'))
    assert '/' in cfg['llm']['model'], 'llm.model must be "<provider>/<name>"'
    assert not cfg['llm']['api_key'], 'do not ship a 9Router key in the package'
    compute_p(cfg['student_id'])


# ======================================================= ROS files (static)
def _read(*parts):
    return open(os.path.join(HERE, '..', *parts), encoding='utf-8').read()


def test_launch_files_compile():
    import py_compile
    for name in ('sim.launch.py', 'moveit.launch.py', 'llm_robot.launch.py'):
        py_compile.compile(os.path.join(HERE, '..', 'launch', name), doraise=True)


def test_controllers_config():
    """Arm: velocity commands (a position command teleports the joints in
    gazebo_ros2_control and the fingers could not carry a block by friction).
    Gripper: PD on the finger positions whose OUTPUT IS A FORCE."""
    import yaml
    c = yaml.safe_load(_read('config', 'ros2_controllers.yaml'))
    cm = c['controller_manager']['ros__parameters']
    assert cm['gripper_controller']['type'] == 'joint_trajectory_controller/JointTrajectoryController'
    g = c['gripper_controller']['ros__parameters']
    assert g['joints'] == ['left_finger_joint', 'right_finger_joint']
    assert g['command_interfaces'] == ['effort']
    assert all(g['gains'][j]['p'] > 0 for j in g['joints'])
    arm = c['joint_trajectory_controller']['ros__parameters']
    assert arm['command_interfaces'] == ['velocity']
    assert set(arm['gains']) == set(arm['joints'])
    x = _read('urdf', 'parallel_gripper.xacro')
    assert '<command_interface name="effort"/>' in x


def test_initial_positions_equal_home(scene):
    import yaml
    ip = yaml.safe_load(_read('config', 'initial_positions.yaml'))
    names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
             'wrist_2_joint', 'wrist_3_joint']
    assert np.allclose([ip[n] for n in names], scene.motion['home_joints'], atol=1e-3)


def test_gripper_xacro_matches_scene(scene):
    x = _read('urdf', 'parallel_gripper.xacro')
    assert f'max_opening:={scene.gripper["max_opening"]}' in x
    assert 'tcp:=0.07' in x and abs(scene.tcp_offset - 0.07) < 1e-9
    srdf = _read('srdf', 'ur_gripper.srdf')
    assert 'group name="ur_manipulator"' in srdf
    for link in ('gripper_base_link', 'left_finger_link', 'right_finger_link'):
        assert link in srdf


def test_backend_adds_floor_below_robot_base():
    import ast
    src = _read('ur3_llm_control', 'moveit_backend.py')
    body = src.split('def setup_scene')[1].split('def scene_add_cube')[0]
    assert "_collision_object('floor'" in body
    consts = {n.targets[0].id: n.value.value for n in ast.parse(src).body
              if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith('FLOOR_')}
    assert -consts['FLOOR_GAP'] < -0.0008 and consts['FLOOR_GAP'] < 0.02


def test_held_block_clears_the_table(scene):
    """The held block (MoveIt attached box, shrunk) must not touch the table."""
    shrink = float(scene.motion['attached_shrink'])
    bottom = scene.cube_center_z() - (scene.cube_size - shrink) / 2.0
    assert bottom - scene.table_top >= float(scene.motion['position_tolerance'])


def test_fingertips_stay_above_the_table(scene):
    """finger tip = base (0.03) + finger (0.055) below tool0; at grasp height the
    tips must stay above the table top (else MoveIt rejects the descent)."""
    tip_below_tool0 = 0.03 + 0.055
    tip_z = scene.grasp_tool_z() - tip_below_tool0
    assert tip_z - scene.table_top > 0.002
