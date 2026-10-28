"""Program A: the fixed reference program, written once for the reference kitchen.

Like an integrator would at installation: the declared drawer of the sponge and positions measured once on the
reference kitchen (world.spec.reference_site) are hard-coded below. No perception, no search, no adaptation. Sequence:
calibrate; open the declared drawer; pick the sponge where it lies, shifted by the measured drawer travel; scrub the
bowl at its reference position for 3 s at 4 N. The claim is "task_done" when scrub returns ok, else
"failed:<stage>:<reason>"; the evaluator decides.

The constants are simulator truth of the reference kitchen at rest (an idealized measurement with zero error, which
favors A). World frame, meters, z up. Not a comparator on generated kitchens, where the layout differs.
"""
from __future__ import annotations

import numpy as np

NAME = "A_fixed_reference"
DRAWER_HANDLE = (-0.3122, 0.4463, 1.03)  # rail handle centre of drawer_unit:0, drawer closed
PULL_DIR = (0.438371, -0.898794, 0.0)  # toward the robot (drawer unit +x)
SPONGE_CLOSED = (-0.3504, 0.5245, 0.9337)  # sponge centre with the drawer closed
SPONGE_GRASP_YAW = 0.4538  # finger closing axis across the sponge (rad, world)
BOWL_TOP = (0.56, -0.14, 0.9539)  # top surface of the bowl on the landing tray
OPEN_DISTANCE = 0.22
SCRUB_S, SCRUB_N = 3.0, 4.0


def run(robot) -> dict:
    """robot: the skill facade (programs.facade.SkillFacade). Returns the program's claim."""
    r = robot.calibrate()
    if not r.ok:
        return {"claim": f"failed:calibrate:{r.reason}"}
    pull = np.array(PULL_DIR)
    r = robot.open_drawer(np.array(DRAWER_HANDLE), pull, distance=OPEN_DISTANCE)
    if not r.ok:
        return {"claim": f"failed:open_drawer:{r.reason}"}
    moved = float(r.data.get("moved", OPEN_DISTANCE))
    sponge = np.array(SPONGE_CLOSED) + moved * pull
    r = robot.pick(sponge, "sponge", grasp="top", yaw=SPONGE_GRASP_YAW)
    if not r.ok:
        return {"claim": f"failed:pick:{r.reason}"}
    r = robot.scrub(np.array(BOWL_TOP), SCRUB_S, SCRUB_N)
    if not r.ok:
        return {"claim": f"failed:scrub:{r.reason}"}
    return {"claim": "task_done"}
