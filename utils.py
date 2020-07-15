"""Utility functions for RL training."""
import torch
import numpy as np


def discount(rewards, gamma):
    """
    Discount the reward trajectory.

    Parameters
    ----------
    rewards : list of float
        Reward trajectory.
    gamma : float
        Discount factor.

    Returns
    -------
    discounted_rewards : list of float
        Discounted reward trajectory.
    """
    R = 0.0
    discounted_rewards = []
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards, device=rewards.device)
    return discounted_rewards


def construct_state(world, robot, device="cpu"):
    """
    Construct state as a tensor given world and robot.

    State is the concatenation of:
        - XY coordinates of the goal (2 dim)
        - XY coordinates of the tip position (2 dim)
        - Pose information of other objects (7 * n_obj dim)
        - Joint angles (7 dim)

    Parameters
    ----------
    world : env.Environment object
        Environment instance.
    robot : torobo_wrapper.Torobo object
        Torobo instance.
    device : string or torch.device
        Device of state tensor.

    Returns
    -------
    state : torch.tensor
        State tensor.
    """
    tip_x = np.array(robot.get_tip_pos()[:2])
    joint_angles = robot.get_joint_angles()
    object_x = world.get_state().reshape(-1, 7)
    object_x[:, :2] = object_x[:, :2] - tip_x
    x = np.concatenate([object_x[0, :2], tip_x, joint_angles, object_x[1:].reshape(-1)])
    x = torch.tensor(x, dtype=torch.float, device=device)
    return x


def clip_to_rectangle(x, global_limits):
    """
    Clip x to global limits.

    Parameters
    ----------
    x : list of float
        Array to be clipped.
    global_limits : list of list of float
        Global operational limits. [[min_x, max_x], [min_y, max_y]].

    Returns
    -------
    clipped : list of float
        Clipped state.
    """
    clipped = x.copy()
    clipped[0] = np.clip(clipped[0], global_limits[0][0], global_limits[0][1])
    clipped[1] = np.clip(clipped[1], global_limits[1][0], global_limits[1][1])
    return clipped


def in_rectangle(x, rectangle):
    """
    Check whether x is in rectangle.

    Parameters
    ----------
    x : list of float
        2-dimensional point. [x, y]
    rectangle : list of list of float
        Rectangle limits. [[min_x, max_x], [min_y, max_y]]

    Returns
    -------
    result : bool
        True if point is in rectangle limits else False.
    """
    p = np.array(x)
    rec = np.array(rectangle)
    result = False
    if (rec[:, 0] < p).all() and (p < (rec[:, 1])).all():
        result = True
    return result
