"""Public site plan shared by the kitchen generator and the robot.

The slot bearings of the U-shaped worktop are part of the site plan given with any site: the robot uses them to name
the slot a perceived module stands in. world.spec imports them from here, so robot-side code has no import path into
the generator.
"""
from __future__ import annotations

# Five places on the U-shaped worktop where a module can stand, as directions (deg) on a ring
# around the nominal robot position (see world.spec for the history of these values).
SLOT_ANGLES = {"left_wing": 116.0, "front_left": 58.0, "front_center": 0.0,
               "front_right": -58.0, "right_wing": -116.0}
