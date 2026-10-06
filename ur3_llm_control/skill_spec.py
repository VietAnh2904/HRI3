"""The ONLY skills the LLM may use (whitelist) and their parameters.

Shared by the prompt builder, the validator and the executor so the three can
never disagree.
"""
from collections import OrderedDict
from dataclasses import dataclass

from .scene_model import TEMP

SKILLS = OrderedDict([
    ('detect_objects', {'params': [], 'kind': 'sense',
                        'doc': 'look at the table with the camera and update where every block is'}),
    ('check_zone', {'params': ['zone'], 'kind': 'sense',
                    'doc': 'look at one zone with the camera: FREE or OCCUPIED by which block'}),
    ('find_object', {'params': ['object'], 'kind': 'sense',
                     'doc': 'locate one block with the camera'}),
    ('find_free_position', {'params': ['object'], 'kind': 'sense',
                            'doc': 'choose a free spot on the table (outside every zone) where '
                                   'the object can be parked; used by place(object, '
                                   f'"{TEMP}")'}),
    ('pick', {'params': ['object'], 'kind': 'motion',
              'doc': 'open the gripper, approach, grasp and lift the block (gripper must be empty)'}),
    ('place', {'params': ['object', 'zone'], 'kind': 'motion',
               'doc': 'put the held block into a zone, or into the free spot found by '
                      f'find_free_position when zone is "{TEMP}", and release it'}),
    ('move_above', {'params': ['object'], 'kind': 'motion',
                    'doc': 'move the gripper above a block (no grasp)'}),
    ('move_to_zone', {'params': ['zone'], 'kind': 'motion',
                      'doc': 'move the gripper above a zone (no release)'}),
    ('home', {'params': [], 'kind': 'motion',
              'doc': 'move the arm to the safe home pose (also the camera observation pose)'}),
])

SENSING = tuple(n for n, s in SKILLS.items() if s['kind'] == 'sense')
MAX_PLAN_STEPS = 30


class Status:
    SUCCESS = 'SUCCESS'
    SKIPPED = 'SKIPPED'
    FAILED = 'FAILED'
    INVALID_OBJECT = 'INVALID_OBJECT'
    INVALID_ZONE = 'INVALID_ZONE'
    INVALID_STATE = 'INVALID_STATE'
    NOT_FOUND = 'NOT_FOUND'
    GRASP_FAILED = 'GRASP_FAILED'
    PLANNING_FAILED = 'PLANNING_FAILED'
    EXECUTION_FAILED = 'EXECUTION_FAILED'

    OK = (SUCCESS, SKIPPED)


@dataclass
class Outcome:
    """Result of one skill: a Status plus a short human readable detail
    (e.g. 'OCCUPIED by blue_cube', 'grip 4.0 cm', 'at (0.300, -0.280)')."""
    status: str
    detail: str = ''

    @property
    def ok(self):
        return self.status in Status.OK

    def __str__(self):
        return f'{self.status} ({self.detail})' if self.detail else self.status


def outcome(x):
    return x if isinstance(x, Outcome) else Outcome(str(x))


def step_to_str(step: dict) -> str:
    params = SKILLS.get(step.get('skill'), {}).get('params', [])
    args = ', '.join(str(step.get(p)) for p in params)
    return f"{step.get('skill')}({args})"
