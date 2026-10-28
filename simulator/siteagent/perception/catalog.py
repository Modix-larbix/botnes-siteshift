"""Public catalogs the robot's perception may use.

Catalog knowledge, like the object catalog in robot.skills: shapes of object categories, of module types and of the
drawer unit type. They hold no site and no simulator state. The same category words appear in the site dossier.
"""
from __future__ import annotations

# Object categories the simulated recogniser can name (ideal semantic recognition of visible
# pixels, declared in perception.sensor).
OBJECT_CATEGORIES = ("cup", "bowl", "fork", "spoon", "sponge", "spray_bottle", "towel")
# Dishes are transient (the day's dirty dishes), not part of the dossier; tools are stored.
DISH_CATEGORIES = ("cup", "bowl", "fork", "spoon")
TOOL_CATEGORIES = ("sponge", "spray_bottle", "towel")
HANDLE_CATEGORIES = ("drawer_handle", "dishwasher_handle")
MODULE_CATEGORIES = ("sink", "dishwasher", "drawer_unit", "landing_tray", "drying_mat", "bins")
# Dossier module kind -> perceived module category.
MODULE_KIND_TO_CATEGORY = {"sink": "sink", "dishwasher": "dishwasher", "drawers": "drawer_unit",
                           "landing": "landing_tray", "drying_mat": "drying_mat", "bins": "bins"}

# Object shape catalog for pose estimation: height of the highest visible surface above the
# object's reference point (its centre), and whether a yaw is estimated (elongated objects).
# "lying" entries: a spray bottle lying in a drawer.
OBJECT_SHAPES = {
    "cup": dict(top_above_centre=0.046, footprint=(0.06, 0.06), height=0.09, yaw=False),
    "bowl": dict(top_above_centre=0.023, footprint=(0.062, 0.062), height=0.044, yaw=False),
    "fork": dict(top_above_centre=0.006, footprint=(0.15, 0.02), height=0.012, yaw=True),
    "spoon": dict(top_above_centre=0.006, footprint=(0.14, 0.024), height=0.012, yaw=True),
    "sponge": dict(top_above_centre=0.022, footprint=(0.09, 0.06), height=0.036, yaw=True),
    "spray_bottle": dict(top_above_centre=0.115, footprint=(0.06, 0.044), height=0.23, yaw=True,
                         lying_top_above_centre=0.030, lying_min_length=0.12,
                         lying_footprint=(0.21, 0.06), lying_height=0.06),
    "towel": dict(top_above_centre=0.0125, footprint=(0.14, 0.064), height=0.025, yaw=True),
}
# Height above a floor at which region grids are also checked for visibility (perception
# .estimate): an object at least this tall standing on a grid point would have shown in the
# image if the point at this height was in direct view.
VISIBILITY_LAYER_H = 0.03

# Rail handle (drawer fronts and dishwasher basket): a 1.6 cm square bar.
HANDLE_HALF_THICKNESS = 0.008
# Rail length per handle category (catalog of the unit models; used for the robot-side visibility estimate).
HANDLE_LENGTH = {"drawer_handle": 0.10, "dishwasher_handle": 0.16}
# Round object categories (silhouette: a prism of ROUND_SIDES sides).
ROUND_CATEGORIES = ("cup", "bowl")

# Drawer unit type (catalog of the unit model, as a robot integrator would have it): interior
# floor relative to the handle centre. Along the drawer axis (away from the robot, i.e. against
# the pull direction) the floor spans 0.032 to 0.292 m behind the handle; across, +-0.069 m;
# the floor surface is 0.108 m below the handle centre. The unit top covers the part of the
# drawer still inside the unit.
DRAWER_FLOOR_BEHIND_HANDLE = (0.032, 0.292)
DRAWER_FLOOR_HALF_WIDTH = 0.069
DRAWER_FLOOR_BELOW_HANDLE = 0.108
DRAWER_PITCH = 0.17  # lateral spacing of drawers in a unit
# Front edge of the unit top, behind the handle of a CLOSED drawer (along the pull axis): the
# part of an opened drawer behind this line is under the top (used to tell whether a
# perceived object in a drawer leaves room for the hand).
DRAWER_TOP_EDGE_BEHIND_CLOSED_HANDLE = 0.012
DRAWER_MAX_OPENING = 0.24  # slide travel of a drawer

# Open surface regions of modules (half sizes in the module frame, +x toward the robot), and the
# surface height above the worktop. Used to compute how much of a region a view covered.
MODULE_SURFACES = {
    "landing_tray": dict(half=(0.16, 0.19), z=0.010),
    "sink": dict(half=(0.155, 0.185), z=0.012),
    "drying_mat": dict(half=(0.15, 0.18), z=0.008),
}
# surface-frame-1 (perception.estimate docstring, "Surface outline"): OUTER footprint of each surface module type, full
# sizes (m) along the module's x axis (toward the robot) and across, as MODULE_SURFACES: the outline (rim and walls
# included) that the robot fits to its own depth images to measure the module's pose (catalog of the module models, as a
# robot integrator would have it; the drying mat has no wall, its outline is its surface). Declared: a module that is
# not the catalog model does not fit, and its region is never cleared.
MODULE_OUTER_FOOTPRINTS = {
    "landing_tray": (0.34, 0.40),
    "sink": (0.34, 0.40),
    "drying_mat": (0.30, 0.36),
}

# Region identifiers derived from the dossier vocabulary (never from simulator names).
def drawer_region(k: int) -> str:
    return f"drawer_unit:{k}"


SURFACE_REGION = {"landing_tray": "landing_tray", "sink": "sink_basin", "drying_mat": "drying_mat"}
# Sink module type (catalog): a faucet at the back of the basin, its spout up to 0.38 m
# above the worktop and reaching over the back half of the basin. A hand looking down at the sink
# keeps its TCP SINK_HAND_CLEARANCE above it (the "look again" over the sink
# went down to a 0.32 m camera height and the hand hit the spout, 60 to 219 N, in 6 of 200
# scripted-chain episodes on seeds 23000-23039).
SINK_FAUCET_TOP = 0.38
SINK_HAND_CLEARANCE = 0.05
# The spout reaches to 2 cm behind the basin centre: a look at the sink is taken from at least
# SINK_LOOK_FORWARD in front of the basin centre (toward the robot). with the
# clearance alone, the hand still met the spout once on its way to a look point above the back
# of the basin (kitchen 23001, 286 N).
SINK_LOOK_FORWARD = 0.08
UNINSPECTED_REGIONS = {"dishwasher": "dishwasher_basket", "bins": "bins_interior"}
