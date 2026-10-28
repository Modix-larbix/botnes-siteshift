"""Development-only worlds (bench side), kept apart from siteagent/world/.

Nothing here is imported by the robot side (perception, sitemodel, robot, agent). A world_dev world is applied only
around a development episode, through world_dev.occlusion.applied() or world_dev.occlusion2.applied(), after their
refusal rules. The offline demo's masked situation uses occlusion-dev-1:towel_over.
"""
