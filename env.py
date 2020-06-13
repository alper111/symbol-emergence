"""Environments in Gazebo."""

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState, ModelStates


class Environment:
    """Tabletop environment wrapper."""

    def __init__(self, objects, rng_ranges):
        """
        Initialize environment with object names and limits.

        Parameters
        ----------
        objects : list of str
            Object names.
        rng_ranges : ndarray
            Valid position limits for random generation.
        """
        self.publisher = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self.objects = objects
        self.num_objects = len(objects)
        self.rng_ranges = rng_ranges
        self.prev_positions = None

    def get_reward(self):
        """
        Get reward for the current state.

        Parameters
        ----------
        None

        Returns
        -------
        reward : float
            Current reward value.
        """
        target = self.prev_positions[0][:2]
        cube = self.get_object_position(self.objects[1])[:2]
        distance_target = np.linalg.norm(target-cube, 2)
        reward = - distance_target
        return reward

    def get_state(self):
        """
        Get current state.

        State consists of three cartesian coordinates and four orientation
        coordinates in quaternion for each object.

        Parameters
        ----------
        None

        Returns
        -------
        state : ndarray
            Current state of objects.
        """
        msg = self.get_model_states()
        indices = list(map(lambda x: msg.name.index(x), self.objects))
        state = np.zeros((self.num_objects, 7))

        for i in range(self.num_objects):
            p = msg.pose[indices[i]]
            state[i] = [p.position.x, p.position.y, p.position.z,
                        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        # make target static
        state[0][:3] = self.prev_positions[0]
        return state.reshape(-1)

    def random(self):
        """
        Set environment to a random state.

        Each object is repositioned w.r.t. their limits defined in
        `rng_ranges`.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        current = []
        proposed = []
        for obj in self.objects:
            valid = False
            while not valid:
                valid = True
                limits = self.rng_ranges[obj]
                x = np.random.uniform(*limits[0])
                y = np.random.uniform(*limits[1])
                z = np.random.uniform(*limits[2])
                pos = np.array([x, y, z])
                # ensure that objects are apart from each other
                for p in current:
                    if np.linalg.norm(p[:2] - pos[:2]) < 0.1:
                        valid = False
                        break
            current.append(pos)
            proposed.append(pos)

        self.prev_positions = proposed.copy()
        for i, pos in enumerate(proposed):
            self.set_model_state(self.objects[i], pos, [0, 0, 0, 1])

    def load_prev_state(self):
        """
        Load environment to its last reset.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for i, pos in enumerate(self.prev_positions):
            self.set_model_state(self.objects[i], pos, [0, 0, 0, 1])

    def is_terminal(self):
        """
        Check whether the environment is at a terminal state.

        Parameters
        ----------
        None

        Returns
        -------
        state : bool
            True if the state is terminal. Otherwise, False.
        """
        cube_pos = self.get_object_position(self.objects[1])[:2]
        cube_limits = self.rng_ranges[self.objects[1]]
        target_pos = self.prev_positions[0][:2]
        distance_target = np.linalg.norm(target_pos-cube_pos, 2)
        if not ((cube_limits[:2, 0] < cube_pos).all() and (cube_pos < cube_limits[:2, 1]).all()):
            return True
        if distance_target < 0.05:
            return True

        return False

    def get_model_states(self):
        """
        Get pose information of objects from Gazebo.

        Parameters
        ----------
        None

        Returns
        -------
        msg : ModelStates
            Pose information of each object.
        """
        msg = rospy.wait_for_message("/model_states", ModelStates)
        return msg

    def get_object_position(self, name):
        """
        Get pose information of a single object.

        Parameters
        ----------
        name : str
            Name of the object.

        Returns
        -------
        pose : list of float
            Cartesian coordinates for position and quaternion coordinates for
            orientation.
        """
        msg = self.get_model_states()
        idx = msg.name.index(name)
        position = msg.pose[idx].position
        orientation = msg.pose[idx].orientation
        return [position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w]

    def set_model_state(self, name, pos, quat):
        """
        Set object's position and orientation.

        Parameters
        ----------
        name : str
            Name of the object.
        pos : list of float
            Cartesian coordinates of the object.
        quat : list of float
            Orientation of the object represented in quaternion coordinates.
        """
        msg = ModelState()
        msg.model_name = name
        msg.pose = Pose()
        msg.pose.position.x = pos[0]
        msg.pose.position.y = pos[1]
        msg.pose.position.z = pos[2]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        self.publisher.publish(msg)
