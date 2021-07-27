import numpy as np
import random
from ai2thor.controller import Controller
from .controller import thor_get, _resolve
from .constants import V_ANGLES, H_ANGLES


def _reachable_thor_loc2d(controller):
    """
    Returns a tuple (x, z) where x and z are lists corresponding to x/z coordinates.
    You can obtain a set of 2d positions tuples by:
        `set(zip(x, z))`
    """
    # get reachable positions
    event = controller.step(action="GetReachablePositions")
    positions = event.metadata["actionReturn"]
    x = np.array([p['x'] for p in positions])
    y = np.array([p['y'] for p in positions])
    z = np.array([p['z'] for p in positions])
    return x, z

def thor_reachable_positions(controller, by_axes=False):
    """
    If `by_axes` is True, then returns x, z
    where x and z are both numpy arrays corresponding
    to the coordinates of the reachable positions.

    Otherwise, returns [(x,z) ... ] where x and z are
    floats for individual reachable position coordinates.
    """
    x, z = _reachable_thor_loc2d(controller)
    if by_axes:
        return x, z
    else:
        return [(x[i], z[i]) for i in range(len(x))]

def thor_agent_pose(event_or_controller, as_tuple=False):
    """Returns a tuple (pos, rot),
    pos: dict (x=, y=, z=)
    rot: dict (x=, y=, z=)
    The angles are in degrees and between 0 to 360 (ai2thor convention)
    """
    event = _resolve(event_or_controller)
    p = thor_get(event, "agent", "position")
    r = thor_get(event, "agent", "rotation")
    if as_tuple:
        return (p["x"], p["y"], p["z"]), (r["x"], r["y"], r["z"])
    else:
        return p, r

def thor_agent_position(event_or_controller):
    """Returns a tuple (pos, rot),
    pos: dict (x=, y=, z=)
    """
    event = _resolve(event_or_controller)
    position = thor_get(event, "agent", "position")
    return position

def thor_apply_pose(controller, pose):
    """Given a 2d pose (x,y,th), teleport the agent to that pose"""
    pos, rot = thor_agent_pose(controller)
    x, z, th = pose
    # if th != 0.0:
    #     import pdb; pdb.set_trace()
    controller.step("TeleportFull",
                    x=x, y=pos["y"], z=z,
                    rotation=dict(y=th))
    controller.step(action="Pass")  #https://github.com/allenai/ai2thor/issues/538

def thor_teleport(controller, position, rotation, horizon):
    """Calls the Teleport function with relevant parameters."""
    return controller.step(action="Teleport",
                           position=position,
                           rotation=rotation,
                           horizon=horizon,
                           standing=True)  # we don't deal with this


def thor_camera_pose(event_or_controller, get_tuples=False):
    """
    This is exactly the same as thor_agent_pose
    except that the pitch of the rotation is set
    to camera horizon. Everything else is the same.
    """
    event = _resolve(event_or_controller)
    position = thor_get(event, "agent", "position")
    rotation = thor_get(event, "agent", "rotation")
    assert abs(rotation["z"]) < 1e-3  # assert that there is no roll
    cameraHorizon = thor_get(event, "agent", "cameraHorizon")
    if get_tuples:
        return (position["x"], position["y"], position["z"]),\
            (cameraHorizon, rotation["y"], 0)
    else:
        return position, dict(x=cameraHorizon, y=rotation["y"], z=0)


def thor_camera_horizon(event_or_controller):
    event = _resolve(event_or_controller)
    cameraHorizon = thor_get(event, "agent", "cameraHorizon")
    return cameraHorizon


def thor_place_agent_randomly(controller,
                              v_angles=V_ANGLES,
                              h_angles=H_ANGLES):
    """Place the agent randomly in an environment;
    Both the position and rotation will be random,
    but valid.

    Args:
       controller_or_reachable_positions (list or or Controller)
       v_angles (list): List of valid pitch (tilt) angles
       h_angles (list): List of valid yaw (rotation) angles"""
    reachable_positions = thor_reachable_positions(controller)
    agent_pose = thor_agent_pose(controller.last_event, as_tuple=False)
    pos = random.sample(reachable_positions, 1)[0]
    pitch = random.sample(v_angles, 1)[0]
    yaw = random.sample(h_angles, 1)[0]
    return controller.step(action="Teleport",
                           position=dict(x=pos[0], y=agent_pose[0]['y'], z=pos[1]),
                           rotation=dict(x=agent_pose[1]['x'], y=yaw, z=agent_pose[1]['z']),
                           horizon=pitch,
                           standing=True)
