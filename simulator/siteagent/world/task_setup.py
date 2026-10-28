"""Task instance: a kitchen with exactly one bowl on the landing tray, the target of the scrub.

By a fixed rule: one bowl on the tray already, nothing changes; several, the first stays the target and the others
become cups at the same spots; none, the first dish on the tray becomes a bowl. The setup log says what changed; it is
bench data (the robot is told only the task text). The target is the only bowl on the tray, so a robot can tell it
apart by category; the evaluator knows it by its simulator name, which the robot never receives.
"""
from __future__ import annotations

import copy

from .spec import SiteSpec


def task_site(spec: SiteSpec) -> tuple[SiteSpec, list[dict], str]:
    """Returns (task instance spec, setup log, target bowl name)."""
    s = copy.deepcopy(spec)
    dishes = [i for i in s.items if i.where == "landing" and i.kind in ("cup", "bowl", "fork", "spoon")]
    if not dishes:
        raise ValueError("no dish on the landing tray: the task does not apply")
    dishes.sort(key=lambda i: int(i.name.rsplit("_", 1)[1]))
    bowls = [i for i in dishes if i.kind == "bowl"]
    log: list[dict] = []
    if bowls:
        target = bowls[0]
        for b in bowls[1:]:
            idx = b.name.rsplit("_", 1)[1]
            log.append({"change": "bowl_to_cup", "from": b.name, "to": f"cup_{idx}", "pos": list(b.pos)})
            b.kind, b.name = "cup", f"cup_{idx}"
    else:
        target = dishes[0]
        idx = target.name.rsplit("_", 1)[1]
        log.append({"change": "dish_to_bowl", "from": target.name, "to": f"bowl_{idx}", "pos": list(target.pos)})
        target.kind, target.name = "bowl", f"bowl_{idx}"
    names = [i.name for i in s.items]
    assert len(names) == len(set(names)), names
    return s, log, target.name
