"""Scene configuration + world state.

Pure python (no ROS import) so it can be unit-tested anywhere.

SceneConfig  = fixed workcell geometry from scene.yaml (table, zones, camera,
               gripper, motion parameters).
WorldState   = what the system currently BELIEVES about the blocks.  It is
               filled from camera detections (detect_objects / check_zone /
               find_object) and updated by the skills after every successful
               pick / place.  Nothing in it comes from the spawn positions of
               scene.yaml.
"""
import copy
import math

import yaml

TEMP = 'temporary_position'     # symbolic place target resolved by find_free_position()


class SceneConfig:
    def __init__(self, data: dict):
        self.raw = data
        self.frame_id = data.get('frame_id', 'world')
        self.ur_type = data.get('ur_type', 'ur3e')
        self.planning_group = data.get('planning_group', 'ur_manipulator')
        self.ee_link = data.get('ee_link', 'tool0')
        self.table = data['table']
        self.table_top = float(self.table['height'])
        self.cube_size = float(data.get('cube_size', 0.04))
        self.cube_mass = float(data.get('cube_mass', 0.05))
        self.objects = data['objects']
        self.zones = data['zones']
        self.camera = data.get('camera', {})
        self.vision = data.get('vision', {})
        self.gripper = data.get('gripper', {})
        self.motion = data['motion']
        self.free_space = data.get('free_space', {})
        self._check()

    @classmethod
    def from_file(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            return cls(yaml.safe_load(f))

    def _check(self):
        names = list(self.objects) + list(self.zones)
        if len(names) != len(set(names)):
            raise ValueError('object / zone names must be unique')
        if TEMP in names:
            raise ValueError(f'"{TEMP}" is reserved')
        colors = [o['color'] for o in self.objects.values()]
        if len(colors) != len(set(colors)):
            raise ValueError('every object needs its own colour (the camera identifies blocks by colour)')
        hues = self.vision.get('hue_ranges', {})
        for c in colors:
            if hues and c not in hues:
                raise ValueError(f'no hue range for colour "{c}" in vision.hue_ranges')
        for name, xy in self.all_locations().items():
            if not self.on_table(xy):
                raise ValueError(f'{name} at {xy} is outside the table')
        if len(self.motion['home_joints']) != 6:
            raise ValueError('motion.home_joints needs 6 values')

    # -------------------------------------------------------------- helpers
    @property
    def object_names(self):
        return list(self.objects.keys())

    @property
    def zone_names(self):
        return list(self.zones.keys())

    def color_of(self, obj):
        return self.objects[obj]['color']

    def object_of_color(self, color):
        for n, o in self.objects.items():
            if o['color'] == color:
                return n
        return None

    def all_locations(self):
        locs = {n: tuple(o['spawn_xy']) for n, o in self.objects.items()}
        locs.update({n: tuple(z['xy']) for n, z in self.zones.items()})
        return locs

    def on_table(self, xy, margin=0.0):
        tx, ty = self.table['center']
        sx, sy = self.table['size']
        return (tx - sx / 2 + margin <= xy[0] <= tx + sx / 2 - margin
                and ty - sy / 2 + margin <= xy[1] <= ty + sy / 2 - margin)

    def zone_xy(self, zone):
        return tuple(self.zones[zone]['xy'])

    def zone_half(self, zone):
        return float(self.zones[zone].get('size', 0.09)) / 2.0

    # ---------------------------------------------------------- heights
    @property
    def tcp_offset(self):
        return float(self.gripper.get('tcp_offset', 0.07))

    def cube_center_z(self):
        return self.table_top + self.cube_size / 2.0

    def grasp_tool_z(self):
        """tool0 height when the grasp centre is at the centre of a cube
        resting on the table (tool pointing down)."""
        return self.cube_center_z() + self.tcp_offset

    def place_tool_z(self):
        return self.grasp_tool_z() + float(self.motion['place_clearance'])

    def approach_tool_z(self):
        return self.grasp_tool_z() + float(self.motion['approach_height'])


class WorldState:
    """Belief about the blocks: {name: (x, y)} + yaw, built from the camera.

    An object that the camera did not see is simply absent from `positions`
    (and the validator refuses to pick it)."""

    def __init__(self, scene: SceneConfig):
        self.scene = scene
        self.positions = {}          # name -> (x, y)
        self.yaws = {}               # name -> yaw of the cube faces [rad]
        self.held = None
        self.observed = False        # at least one camera observation so far

    def copy(self):
        return copy.deepcopy(self)

    def set_detections(self, detections):
        """detections: {name: (x, y, yaw)} from the camera.  Replaces the
        belief about every object that is not in the gripper."""
        held = self.held
        self.positions = {}
        self.yaws = {}
        for name, d in detections.items():
            if name == held:
                continue
            self.positions[name] = (float(d[0]), float(d[1]))
            self.yaws[name] = float(d[2]) if len(d) > 2 else 0.0
        self.observed = True

    def visible(self, obj):
        return obj in self.positions and obj != self.held

    # -------------------------------------------------------------- zones
    def slot_of(self, obj):
        """zone the object sits in (its centre inside the zone square), else None."""
        if not self.visible(obj):
            return None
        ox, oy = self.positions[obj]
        for zone in self.scene.zones:
            zx, zy = self.scene.zone_xy(zone)
            h = self.scene.zone_half(zone)
            if abs(ox - zx) <= h and abs(oy - zy) <= h:
                return zone
        return None

    def occupant(self, zone, exclude=None):
        """Object whose footprint overlaps the zone (not only its centre):
        a block half inside Zone B still blocks Zone B."""
        zx, zy = self.scene.zone_xy(zone)
        reach = self.scene.zone_half(zone) + self.scene.cube_size / 2.0
        best, best_d = None, None
        for obj, (ox, oy) in self.positions.items():
            if obj in (exclude, self.held):
                continue
            d = max(abs(ox - zx), abs(oy - zy))
            if d < reach and (best is None or d < best_d):
                best, best_d = obj, d
        return best

    def zone_status(self):
        return {z: self.occupant(z) for z in self.scene.zones}

    def summary(self):
        out = {}
        for obj in self.scene.objects:
            if obj == self.held:
                out[obj] = 'held by gripper'
            elif obj not in self.positions:
                out[obj] = 'not detected'
            else:
                z = self.slot_of(obj)
                if z:
                    out[obj] = z
                else:
                    x, y = self.positions[obj]
                    out[obj] = f'table ({x:.3f}, {y:.3f})'
        return out

    # --------------------------------------------------------- free space
    def free_position(self, obj, reachable=None, avoid=(), src=None):
        """A free spot on the table where `obj` can be parked.

        Candidates on a grid over the table, rejected if they are
          * off the table (margin) or outside the reachable annulus,
          * inside / too close to any zone (zones must stay usable),
          * closer than min_object_distance to any other block
            (the open gripper fingers need room), or to `avoid` points,
        then sorted by distance from `src` (default: the object's current
        position -> short transfer) and the first one accepted by `reachable(xy)` (IK check with
        the gripper) is returned.  None if the table is full."""
        fs = self.scene.free_space
        step = float(fs.get('grid_step', 0.02))
        min_obj = float(fs.get('min_object_distance', 0.10))
        zmargin = float(fs.get('zone_margin', 0.05))
        tmargin = float(fs.get('table_margin', 0.04)) + self.scene.cube_size / 2
        rmin = float(fs.get('min_radius', 0.20))
        rmax = float(fs.get('max_radius', 0.46))
        tx, ty = self.scene.table['center']
        sx, sy = self.scene.table['size']
        if src is None:
            src = self.positions.get(obj)
        others = [p for o, p in self.positions.items() if o not in (obj, self.held)]
        others += list(avoid)
        cands = []
        nx = int(round(sx / step))
        ny = int(round(sy / step))
        for i in range(nx + 1):
            for j in range(ny + 1):
                x = round(tx - sx / 2 + i * step, 4)
                y = round(ty - sy / 2 + j * step, 4)
                if not self.scene.on_table((x, y), tmargin):
                    continue
                r = math.hypot(x, y)
                if r < rmin or r > rmax:
                    continue
                bad = False
                for zone in self.scene.zones:
                    zx, zy = self.scene.zone_xy(zone)
                    lim = self.scene.zone_half(zone) + self.scene.cube_size / 2 + zmargin
                    if abs(x - zx) < lim and abs(y - zy) < lim:
                        bad = True
                        break
                if bad:
                    continue
                if any(math.hypot(x - ox, y - oy) < min_obj for ox, oy in others):
                    continue
                if src is not None:
                    score = math.hypot(x - src[0], y - src[1])
                else:
                    score = abs(r - 0.32)
                cands.append((score, x, y))
        cands.sort()
        for _, x, y in cands:
            if reachable is None or reachable((x, y)):
                return (x, y)
        return None
