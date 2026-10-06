"""Skill executor: turns a validated plan into robot actions, closed-loop
with the camera.

1. expand_plan()  (deterministic, on a COPY of the camera-based world state)
     - reorders independent pick/place blocks so that a block whose target
       zone is free goes first (fewer temporary moves);
     - object already in its target zone          -> its pick/place are SKIPPED
     - target zone occupied by another block      -> that block is first moved
       to a free spot (auto-inserted  pick / find_free_position / place(...,
       temporary_position), shown with [auto])
     - every place(..., temporary_position) gets its concrete spot from
       find_free_position (resolved now, re-checked at run time).
2. execute()      runs the steps through RobotSkills, prints the status table
                  and stops at the first failure.  After every camera step
                  (detect_objects / check_zone / find_object) the observed zone
                  occupancy is compared with the prediction; if the world
                  changed, the remaining steps are expanded again from what
                  the camera sees now.
3. verify()       final camera check that every block is where the plan put it.
"""
import math

from .scene_model import TEMP
from .skill_spec import SENSING, Outcome, Status, step_to_str

LINE_WIDTH = 34


class PlanExpansionError(RuntimeError):
    pass


def _strip(step):
    return {k: v for k, v in step.items() if not k.startswith('_') and k not in (
        'auto', 'skip', 'xy')}


# ------------------------------------------------------------------ reorder
def _parse_blocks(plan):
    """plan = prefix (sensing steps) + blocks + suffix (home) where a block is
    [check_zone]* pick(o) [find_free_position(o)] place(o, L).
    Returns (prefix, blocks, suffix) or None if the plan has another shape."""
    i, n = 0, len(plan)
    prefix = []
    while i < n and plan[i]['skill'] in ('detect_objects', 'find_object'):
        prefix.append(plan[i])
        i += 1
    suffix = []
    end = n
    while end > i and plan[end - 1]['skill'] == 'home':
        end -= 1
    suffix = plan[end:]
    blocks = []
    while i < end:
        blk = []
        while i < end and plan[i]['skill'] == 'check_zone':
            blk.append(plan[i])
            i += 1
        if i >= end or plan[i]['skill'] != 'pick':
            return None
        obj = plan[i]['object']
        blk.append(plan[i])
        i += 1
        if i < end and plan[i]['skill'] == 'find_free_position' and plan[i]['object'] == obj:
            blk.append(plan[i])
            i += 1
        if i >= end or plan[i]['skill'] != 'place' or plan[i]['object'] != obj:
            return None
        blk.append(plan[i])
        i += 1
        blocks.append(blk)
    return prefix, blocks, suffix


def reorder_blocks(plan, world):
    parsed = _parse_blocks(plan)
    if parsed is None or len(parsed[1]) < 2:
        return list(plan)
    prefix, blocks, suffix = parsed
    sim = world.copy()
    ordered = []
    while blocks:
        choice = None
        for b in blocks:
            place = b[-1]
            if place['zone'] == TEMP or sim.occupant(place['zone'], exclude=place['object']) is None:
                choice = b
                break
        choice = choice or blocks[0]
        blocks.remove(choice)
        ordered += choice
        place = choice[-1]
        o = place['object']
        if place['zone'] == TEMP:
            sim.positions[o] = (99.0, 99.0)          # somewhere off the zones
        else:
            sim.positions[o] = sim.scene.zone_xy(place['zone'])
    return prefix + ordered + suffix


# ------------------------------------------------------------------ expand
def _find_place(plan, start, obj):
    for j in range(start, len(plan)):
        s = plan[j]
        if s['skill'] == 'place' and s['object'] == obj:
            return j
        if s['skill'] == 'pick':
            break
    return None


def expand_plan(plan, world, free_fn):
    """free_fn(sim_world, obj, src_xy) -> (x, y) or None: find_free_position
    on a simulated world.  Returns the list of executable steps; perception
    steps carry '_expect' = predicted zone occupancy at that moment."""
    plan = reorder_blocks([_strip(s) for s in plan], world)
    sim = world.copy()
    out = []
    skip_idx = set()
    pending_free = {}
    pick_src = {}
    for i, step in enumerate(plan):
        s = step['skill']
        if i in skip_idx:
            out.append(dict(step, skip=True))
            continue
        if s == 'pick':
            obj = step['object']
            j = _find_place(plan, i + 1, obj)
            if j is not None and plan[j]['zone'] != TEMP:
                zone = plan[j]['zone']
                if sim.slot_of(obj) == zone:           # already where it should go
                    out.append(dict(step, skip=True))
                    for k in range(i + 1, j + 1):
                        if plan[k]['skill'] in ('find_free_position', 'place'):
                            skip_idx.add(k)
                    continue
                occ = sim.occupant(zone, exclude=obj)
                if occ is not None:
                    xy = free_fn(sim, occ, sim.positions.get(occ))
                    if xy is None:
                        raise PlanExpansionError(
                            f'{zone} is occupied by {occ} and there is no free spot on the table')
                    out.append({'skill': 'pick', 'object': occ, 'auto': True})
                    out.append({'skill': 'find_free_position', 'object': occ, 'auto': True,
                                'xy': xy})
                    out.append({'skill': 'place', 'object': occ, 'zone': TEMP, 'auto': True,
                                'xy': xy})
                    sim.positions[occ] = xy
            if sim.visible(obj):
                pick_src[obj] = sim.positions[obj]
            sim.held = obj
            sim.positions.pop(obj, None)
            out.append(dict(step))
        elif s == 'find_free_position':
            obj = step['object']
            xy = free_fn(sim, obj, pick_src.get(obj, sim.positions.get(obj)))
            if xy is None:
                raise PlanExpansionError(f'no free spot on the table for {obj}')
            pending_free[obj] = xy
            out.append(dict(step, xy=xy))
        elif s == 'place':
            obj = step['object']
            zone = step['zone']
            if zone == TEMP:
                xy = pending_free.pop(obj, None)
                if xy is None:
                    xy = free_fn(sim, obj, pick_src.get(obj))
                    if xy is None:
                        raise PlanExpansionError(f'no free spot on the table for {obj}')
                    out.append({'skill': 'find_free_position', 'object': obj, 'auto': True,
                                'xy': xy})
                step = dict(step, xy=xy)
            else:
                xy = sim.scene.zone_xy(zone)
                step = dict(step)
            sim.positions[obj] = xy
            sim.yaws[obj] = 0.0
            if sim.held == obj:
                sim.held = None
            out.append(step)
        elif s in SENSING:
            out.append(dict(step, _expect=sim.zone_status()))
        else:
            out.append(dict(step))
    return out


def predicted_final(steps, world):
    """Where every block should be after the steps: {obj: zone | 'table'}."""
    final = {}
    for s in steps:
        if s.get('skip') or s['skill'] != 'place':
            continue
        final[s['object']] = s['zone'] if s['zone'] != TEMP else 'table'
    return final


def format_line(label, result):
    dots = '.' * max(2, LINE_WIDTH - len(label))
    return f'{label} {dots} {result}'


def label_of(step):
    lab = step_to_str(step)
    if step.get('auto'):
        lab += '  [auto]'
    return lab


class SkillExecutor:
    def __init__(self, skills, log=print):
        self.skills = skills
        self.log = log

    def free_fn(self, sim, obj, src):
        return sim.free_position(obj, reachable=self.skills.reachable, src=src)

    def expand(self, plan):
        return expand_plan(plan, self.skills.world, self.free_fn)

    def _run(self, step):
        k = self.skills
        s = step['skill']
        if s == 'home':
            return k.home()
        if s == 'detect_objects':
            return k.detect_objects()
        if s == 'check_zone':
            return k.check_zone(step['zone'])
        if s == 'find_object':
            return k.find_object(step['object'])
        if s == 'find_free_position':
            return k.find_free_position(step['object'], xy=step.get('xy'))
        if s == 'pick':
            return k.pick(step['object'])
        if s == 'place':
            return k.place(step['object'], step['zone'], xy=step.get('xy'))
        if s == 'move_above':
            return k.move_above(step['object'])
        if s == 'move_to_zone':
            return k.move_to_zone(step['zone'])
        return Outcome(Status.FAILED, f'unknown skill {s}')

    def execute(self, steps):
        """Returns (task_ok, [(label, Outcome), ...], executed_steps)."""
        results = []
        done = []
        steps = list(steps)
        i = 0
        failed = False
        while i < len(steps):
            step = steps[i]
            label = label_of(step)
            if failed:
                res = Outcome('NOT EXECUTED')
            elif step.get('skip'):
                res = Outcome(Status.SKIPPED, 'already in place')
            else:
                try:
                    res = self._run(step)
                except Exception as e:           # never leave the loop silently
                    self.log(f'[executor] exception in {label}: {e!r}')
                    res = Outcome(Status.FAILED, repr(e))
                if not res.ok:
                    failed = True
            results.append((label, res))
            done.append(step)
            self.log(format_line(label, res))
            # ---- closed loop: did the camera see what the plan expected?
            if (not failed and step['skill'] in SENSING and '_expect' in step
                    and step['skill'] != 'find_free_position'):
                now = self.skills.world.zone_status()
                if now != step['_expect']:
                    rest = [s for s in steps[i + 1:] if not s.get('auto')]
                    self.log('[executor] camera shows a different zone occupancy than '
                             'predicted: ' + ', '.join(f'{z}={o or "free"}' for z, o in now.items()))
                    try:
                        new_rest = self.expand(rest)
                    except Exception as e:
                        self.log(f'[executor] cannot re-plan: {e}')
                        failed = True
                        new_rest = rest
                    if not failed:
                        self.log('[executor] remaining steps re-planned from the camera view:')
                        for s in new_rest:
                            self.log('    ' + label_of(s) + ('  [skip]' if s.get('skip') else ''))
                    steps = steps[:i + 1] + new_rest
            i += 1
        return (not failed), results, done

    def verify(self, steps):
        """Camera check of the final state.  Returns (ok, [(label, text)])."""
        expected = predicted_final(steps, self.skills.world)
        res = self.skills.detect_objects()
        lines = []
        if not res.ok:
            return False, [('detect_objects()', str(res))]
        ok = True
        w = self.skills.world
        for obj, target in expected.items():
            if not w.visible(obj):
                lines.append((f'{obj} -> {target}', 'NOT SEEN'))
                ok = False
                continue
            actual = w.slot_of(obj)
            x, y = w.positions[obj]
            if target == 'table':
                good = actual is None
                txt = 'OK' if good else f'WRONG (in {actual})'
                lines.append((f'{obj} on the table', f'{txt} at ({x:.3f}, {y:.3f})'))
            else:
                zx, zy = w.scene.zone_xy(target)
                err = math.hypot(x - zx, y - zy)
                good = actual == target
                txt = (f'OK ({1000 * err:.0f} mm from centre)' if good
                       else f'WRONG (at {actual or "table"})')
                lines.append((f'{obj} in {target}', txt))
            ok &= good
        return ok, lines
