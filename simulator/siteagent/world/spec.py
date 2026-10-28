"""Ground truth of a simulated site: what the world really is.

The agent never reads a SiteSpec. It only gets what its sensors return and what the (possibly stale or wrong) site
dossier claims. The bench and the evaluator read it. sample_site() generates a kitchen from a seed; reference_site()
is the fixed reference kitchen.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

# Five places on the U-shaped worktop where a module can stand. Each module faces the robot. Slots are directions (deg)
# on a ring around the nominal robot position, and the distance from the robot depends on the module: modules with a
# handle stand at the back (0.70 m), trays near the front edge (0.62 m). Earlier the front slot was at 0.62 m and the
# wings at 0.58 m for every module: a drawer or the dishwasher basket pulled toward the robot then brought its handle
# closer than the arm can hold it (a hand pointing down needs about 0.28 m from the base axis), and two neighbouring
# footprints overlapped. Slots are 58 deg apart so that the widest modules (3-drawer unit 0.53 m, landing tray 0.40 m)
# and a fully opened drawer or basket keep >= 2.6 cm from their neighbours. The worktop was enlarged to carry the ring.
from ..site_plan import SLOT_ANGLES  # noqa: E402 (public site plan, shared with the robot side)

MODULE_RADIUS = {"drawers": 0.70, "dishwasher": 0.70, "sink": 0.68, "bins": 0.64,
                 "landing": 0.64, "drying_mat": 0.64}
SLOTS = {k: (round(0.70 * math.cos(math.radians(a)), 4), round(0.70 * math.sin(math.radians(a)), 4))
         for k, a in SLOT_ANGLES.items()}


def slot_xy(slot: str, kind: str) -> tuple[float, float]:
    a = math.radians(SLOT_ANGLES[slot])
    r = MODULE_RADIUS[kind]
    return (r * math.cos(a), r * math.sin(a))


# Where a tool rests in a drawer: offset (along the drawer axis, sideways) from the drawer floor
# centre, toward the front so a partly opened drawer exposes it. One item per drawer.
DRAWER_ITEM_POS = {"sponge": (0.075, 0.0), "spray_bottle": (0.05, 0.0), "towel": (0.055, 0.0),
                   "fork": (0.06, 0.0), "spoon": (0.06, 0.0)}
# Tools left in the sink basin (sink frame offsets), distinct so they never overlap. Away
# from the side walls (the 21 cm palm must fit in the basin) and in front of the faucet spout.
SINK_SIDE_POS = {"sponge": (0.06, 0.07), "spray_bottle": (0.06, -0.07)}

DRAWER_ITEMS = ("sponge", "spray_bottle", "fork", "spoon", "towel")
DISH_KINDS = ("cup", "bowl", "fork", "spoon")

WORKTOP_COLORS = {
    "oak": (0.62, 0.47, 0.30), "white_laminate": (0.88, 0.88, 0.86),
    "steel": (0.70, 0.72, 0.75), "black_granite": (0.16, 0.16, 0.18),
    "green_tile": (0.42, 0.55, 0.47),
}


@dataclass
class Mount:
    """How the robot is installed. Nominal = level, base 5 cm above the worktop."""
    robot: str = "panda"
    x: float = 0.0
    y: float = 0.0
    height_offset: float = 0.05  # base height above the worktop, metres
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class Module:
    kind: str  # sink | dishwasher | drawers | landing | drying_mat | bins
    slot: str
    params: dict = field(default_factory=dict)


@dataclass
class Item:
    name: str
    kind: str
    where: str  # module name ("landing", "drawers/1", ...) or "worktop"
    pos: tuple[float, float] = (0.0, 0.0)  # local offset inside its container


@dataclass
class SiteSpec:
    seed: int
    worktop_height: float = 0.90
    worktop_color: str = "oak"
    light: float = 0.75
    mount: Mount = field(default_factory=Mount)
    modules: list[Module] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)

    def module(self, kind: str) -> Module | None:
        return next((m for m in self.modules if m.kind == kind), None)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "SiteSpec":
        d = json.loads(text)
        d["mount"] = Mount(**d["mount"])
        d["modules"] = [Module(**m) for m in d["modules"]]
        d["items"] = [Item(**{**i, "pos": tuple(i["pos"])}) for i in d["items"]]
        return SiteSpec(**d)


def reference_site() -> SiteSpec:
    """The site the dossier was written for: level robot, dishwasher on the left."""
    s = SiteSpec(seed=0)
    s.modules = [
        Module("dishwasher", "front_left"),
        Module("landing", "front_center"),
        Module("sink", "front_right"),
        Module("drawers", "left_wing", {"n": 2}),
        Module("bins", "right_wing"),
    ]
    s.items = [
        Item("cup_0", "cup", "landing", (0.08, -0.14)),
        Item("cup_1", "cup", "landing", (-0.08, 0.0)),
        Item("bowl_0", "bowl", "landing", (0.08, 0.14)),
        Item("sponge", "sponge", "drawers/0", DRAWER_ITEM_POS["sponge"]),
        Item("spray_bottle", "spray_bottle", "drawers/1", DRAWER_ITEM_POS["spray_bottle"]),
    ]
    return s


def sample_site(seed: int) -> SiteSpec:
    """A random site. Every axis below is something that changes between real cafés."""
    rng = np.random.default_rng(seed)
    s = SiteSpec(seed=seed)
    s.worktop_height = round(float(rng.uniform(0.86, 0.94)), 3)
    s.worktop_color = str(rng.choice(list(WORKTOP_COLORS)))
    s.light = round(float(rng.uniform(0.45, 1.0)), 2)

    level = rng.random() < 0.4
    tilt = lambda: 0.0 if level else round(float(np.clip(rng.normal(0, 2.0), -5, 5)), 2)
    s.mount = Mount(
        x=round(float(rng.uniform(-0.03, 0.03)), 3),
        y=round(float(rng.uniform(-0.03, 0.03)), 3),
        height_offset=round(float(rng.uniform(-0.02, 0.18)), 3),
        roll_deg=tilt(), pitch_deg=tilt(),
        yaw_deg=round(float(rng.uniform(-10, 10)), 2),
    )

    has_dishwasher = rng.random() < 0.65
    kinds = ["sink", "landing", "drawers", "dishwasher" if has_dishwasher else "drying_mat"]
    if rng.random() < 0.7:
        kinds.append("bins")
    slots = list(SLOTS)
    rng.shuffle(slots)
    s.modules = [Module(k, slots[i]) for i, k in enumerate(kinds)]
    n_drawers = int(rng.integers(2, 4))
    s.module("drawers").params = {"n": n_drawers}

    # Dirty dishes on the landing tray, on a jittered grid so they never overlap.
    n_dishes = int(rng.integers(2, 6))
    # Pitch 0.16 x 0.14 m and jitter +-5 mm (earlier: 0.12 x 0.10, +-1 cm): the hand's palm is
    # 21 cm long, it must clear the neighbours when it comes down on a dish.
    cells = [(dx, dy) for dx in (-0.08, 0.08) for dy in (-0.14, 0.0, 0.14)]
    rng.shuffle(cells)
    for i in range(n_dishes):
        kind = str(rng.choice(DISH_KINDS, p=[0.5, 0.2, 0.15, 0.15]))
        dx, dy = cells[i]
        s.items.append(Item(f"{kind}_{i}", kind, "landing",
                            (round(dx + rng.uniform(-0.005, 0.005), 3), round(dy + rng.uniform(-0.005, 0.005), 3))))

    # Tools: hidden in drawers (one item per drawer), sometimes left in the sink basin.
    for tool in ("sponge", "spray_bottle"):
        free_now = [k for k in range(n_drawers) if not any(i.where == f"drawers/{k}" for i in s.items)]
        if rng.random() < 0.8 and free_now:
            k = int(rng.choice(free_now))
            s.items.append(Item(tool, tool, f"drawers/{k}", DRAWER_ITEM_POS[tool]))
        else:
            s.items.append(Item(tool, tool, "sink_side", SINK_SIDE_POS[tool]))
    # A distractor in a free drawer: the agent must not confuse it with a tool.
    free = [k for k in range(n_drawers) if not any(i.where == f"drawers/{k}" for i in s.items)]
    if free and rng.random() < 0.6:
        s.items.append(Item("towel", "towel", f"drawers/{free[0]}", DRAWER_ITEM_POS["towel"]))
    return s


def mount_quat(m: Mount) -> np.ndarray:
    """Base orientation from roll, pitch, yaw in degrees (ZYX convention)."""
    r, p, y = (math.radians(a) for a in (m.roll_deg, m.pitch_deg, m.yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])
