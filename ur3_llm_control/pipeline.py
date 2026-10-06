"""User command -> camera -> LLM -> validator -> executor -> camera check,
with the terminal report required by the assignment.  Independent of ROS
(the ROS node just feeds it commands, a MoveIt backend and the camera)."""
import json

from .llm_planner import LLMError
from .skill_executor import PlanExpansionError, SkillExecutor, format_line, label_of
from .skill_spec import step_to_str

BAR = '=' * 64


class CommandPipeline:
    def __init__(self, planner, skills, world, log=print):
        self.planner = planner
        self.skills = skills
        self.world = world
        self.log = log
        self.executor = SkillExecutor(skills, log=log)

    def print_world(self, title):
        log = self.log
        log(title)
        for obj, where in self.world.summary().items():
            log(f'  {obj:12s} {where}')
        zs = ', '.join(f'{z}: ' + ('FREE' if o is None else f'OCCUPIED ({o})')
                       for z, o in self.world.zone_status().items())
        log(f'  zones        {zs}')

    def run(self, command, dry_run=False):
        """Returns a dict report (also published on /llm_robot/result)."""
        log = self.log
        report = {'command': command, 'plan': [], 'status': 'TASK FAILED', 'steps': []}
        log(BAR)
        log('USER COMMAND:')
        log(command)
        log('')

        # -------------------------------------------------- 1 camera first
        st = self.skills.detect_objects()
        if not st.ok:
            log(f'CAMERA: detect_objects() {st}')
            log('')
            log('TASK FAILED (the camera is needed to plan)')
            report['error'] = str(st)
            log(BAR)
            return report
        self.print_world('CAMERA (detect_objects):')
        report['camera'] = self.world.summary()
        log('')

        # -------------------------------------------------- 2 LLM + validator
        try:
            result, answer = self.planner.plan(command, self.world)
        except LLMError as e:
            log(f'LLM ERROR: {e}')
            log('')
            log('TASK REJECTED')
            report['status'] = 'TASK REJECTED'
            report['error'] = str(e)
            log(BAR)
            return report

        if not result.ok:
            log('LLM PLAN: REJECTED BY VALIDATOR')
            for err in result.errors:
                log(f'  - {err}')
            log(f'  raw LLM answer: {answer.strip()[:400]}')
            log('')
            log('TASK REJECTED (nothing was executed)')
            report['status'] = 'TASK REJECTED'
            report['errors'] = result.errors
            log(BAR)
            return report

        report['plan'] = result.plan
        log('LLM PLAN:')
        for step in result.plan:
            log(step_to_str(step))
        for w in result.warnings:
            log(f'  (validator note: {w})')
        log('')
        log('JSON PLAN:')
        log(json.dumps({'plan': result.plan}, ensure_ascii=False))
        log('')

        # -------------------------------------------------- 3 expand
        try:
            steps = self.executor.expand(result.plan)
        except PlanExpansionError as e:
            log(f'EXECUTION ABORTED: {e}')
            log('')
            log('TASK FAILED')
            log(BAR)
            return report
        plain = [{k: v for k, v in s.items() if k in ('skill', 'object', 'zone')} for s in steps]
        if plain != result.plan or any(s.get('skip') for s in steps):
            log('EXECUTION PLAN (after zone check / conflict resolution):')
            for s in steps:
                tag = ''
                if s.get('auto'):
                    tag = '  <- clear target zone'
                if s.get('skip'):
                    tag = '  [skip: already in place]'
                if s['skill'] in ('find_free_position', 'place') and s.get('xy'):
                    tag += f'  -> ({s["xy"][0]:.3f}, {s["xy"][1]:.3f})'
                log(label_of(s) + tag)
            log('')

        if dry_run:
            log('DRY RUN: plan validated, robot not moved')
            report['status'] = 'DRY RUN'
            log(BAR)
            return report

        # -------------------------------------------------- 4 execute
        log('EXECUTION:')
        ok, results, done = self.executor.execute(steps)
        report['steps'] = [(lab, str(res)) for lab, res in results]
        log('')

        # -------------------------------------------------- 5 camera check
        if ok:
            vok, lines = self.executor.verify(done)
            log('VERIFY (camera):')
            for lab, txt in lines:
                log(format_line(lab, txt))
            if not lines:
                log('  (nothing was moved)')
            log('')
            ok = ok and vok
            report['verify'] = lines
        report['status'] = 'TASK SUCCESS' if ok else 'TASK FAILED'
        log(report['status'])
        self.print_world('World state (camera):')
        log(BAR)
        return report
