"""Environments in Gazebo."""

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState, ModelStates
import utils


class Environment:
    """Tabletop environment wrapper."""

    def __init__(self, robot, objects, rng_ranges):
        """
        Initialize environment with object names and limits.

        Parameters
        ----------
        robot : torobo_wrapper.Torobo
            Robot object.
        objects : list of str
            Object names.
        rng_ranges : numpy.ndarray
            Valid position limits for random generation.
        """
        self.robot = robot
        self.publisher = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self.objects = objects
        self.num_objects = len(objects)
        self.rng_ranges = rng_ranges
        self.prev_positions = None
        self.init_diff = None
        self.abs_limits = [[0.32, 0.51], [-0.1, 0.5]]

    def initialize(self):
        """Initialize arm position."""
        self.robot.go(np.radians([90, 0, 0, 0, 0, -90, 0]))
        rospy.sleep(2.0)
        self.robot.go(np.radians([90, 45, 0, 45, 0, -90, 0]))
        rospy.sleep(2.0)

    def zerorobotpose(self):
        """Reset arm position."""
        self.robot.go(np.radians([90, 45, 0, 45, 0, -90, 0]))
        rospy.sleep(2)
        self.robot.go(np.radians([90, 0, 0, 0, 0, -90, 0]))
        rospy.sleep(2)
        self.robot.go(np.radians([0, 0, 0, 0, 0, 0, 0]))
        rospy.sleep(2)

    def reset(self):
        """Reset environment."""
        self.robot.init_pose()
        rospy.sleep(2.0)
        current_angles = self.robot.get_joint_angles()
        angles = self.robot.compute_ik(current_angles, [0.40, -0.07, 1.17, np.pi, 0, 0])
        angles_temp = list(angles)
        angles_temp[5] = np.radians(-60)
        self.robot.go(angles_temp, time_from_start=2.0)
        rospy.sleep(2.0)
        self.robot.go(angles, time_from_start=2.0)
        rospy.sleep(2.0)
        self._random()
        rospy.sleep(0.1)

        target_pos = np.array(self.get_object_position(self.objects[0])[:2])
        obj_pos = np.array(self.get_object_position(self.objects[-1])[:2])
        self.init_diff = np.linalg.norm(target_pos - obj_pos, 2)
        return self.get_state()

    def get_state(self):
        """
        Construct state as a numpy vector.

        State is the concatenation of:
            - XY coordinates of the goal (2 dim)
            - XY coordinates of the tip position (2 dim)
            - Pose information of other objects (7 * n_obj dim)
            - Joint angles (7 dim)

        Parameters
        ----------
        None

        Returns
        -------
        state : numpy.ndarray
            State vector.
        """
        tip_x = np.array(self.robot.get_tip_pos()[:2])
        joint_angles = self.robot.get_joint_angles()
        object_x = self._get_object_state().reshape(-1, 7)
        object_x[:, :2] = object_x[:, :2] - tip_x
        x = np.concatenate([object_x[0, :2], tip_x, joint_angles, object_x[1:].reshape(-1)])
        return x

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
        obj_pos = self.get_object_position(self.objects[1])[:2]
        target_pos = self.prev_positions[0][:2]
        distance_target = np.linalg.norm(target_pos-obj_pos, 2)
        # TODO: set these into parameters in initialization.
        if distance_target < 0.001:
            return True

        if not utils.in_rectangle(obj_pos, self.abs_limits):
            return True

        return False

    def get_reward(self, init_diff):
        """
        Get reward for the current state.

        Parameters
        ----------
        init_diff : float
            Initial difference.

        Returns
        -------
        reward : float
            Current reward value.
        """
        target = self.prev_positions[0][:2]
        obj = self.get_object_position(self.objects[-1])[:2]
        final_diff = np.linalg.norm(target-obj, 2)
        reward = init_diff - final_diff
        if reward < 1e-4:
            reward = 0.0
        else:
            reward = 10 * reward
        return reward

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
        msg = self._get_model_states()
        idx = msg.name.index(name)
        position = msg.pose[idx].position
        orientation = msg.pose[idx].orientation
        return [position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w]

    def step(self, state, action, rate):
        """
        Act on the environment.

        Parameters
        ----------
        state : numpy.ndarray
            Current state of the environment.
        action : numpy.ndarray
            Action.
        rate : rospy.Rate
            Rospy simulation rate.

        Returns
        -------
        next_state : numpy.ndarray
            Next state of the environment.
        reward : float
            Reward.
        done : bool
            True if the environment is in the terminal state else false.
        success : bool
            True if action is done else false.
        """
        normalized_action = action * rate.sleep_dur.to_sec()
        x_next = state[2:4] + normalized_action
        x_clip = utils.clip_to_rectangle(x_next, self.abs_limits)
        action_aug = x_clip.tolist() + [1.17, np.pi, 0, 0]
        current_angles = self.robot.get_joint_angles()
        angles = self.robot.compute_ik(current_angles, action_aug)
        if angles != -31:
            self.robot.go(angles, time_from_start=rate.sleep_dur.to_sec())
            rate.sleep()
            success = True
        else:
            success = False

        done = self.is_terminal()
        if not utils.in_rectangle(x_next, self.abs_limits):
            done = True

        reward = self.get_reward(self.init_diff)
        next_state = self.get_state()
        return next_state, reward, done, success

    def _random(self):
        """Set objects to a random state w.r.t. limits."""
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
            self._set_model_state(self.objects[i], pos, [0, 0, 0, 1])

    def _get_object_state(self):
        """
        Get current state.

        State consists of three cartesian coordinates and four orientation
        coordinates in quaternion for each object.

        Parameters
        ----------
        None

        Returns
        -------
        state : numpy.ndarray
            Current state of objects.
        """
        msg = self._get_model_states()
        indices = list(map(lambda x: msg.name.index(x), self.objects))
        state = np.zeros((self.num_objects, 7))

        for i in range(self.num_objects):
            p = msg.pose[indices[i]]
            state[i] = [p.position.x, p.position.y, p.position.z,
                        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        # make target static
        state[0][:3] = self.prev_positions[0]
        return state.reshape(-1)

    def _get_model_states(self):
        """
        Get pose information of objects from Gazebo.

        Parameters
        ----------
        None

        Returns
        -------
        msg : gazebo_msgs.msg.ModelStates
            Pose information of each object.
        """
        msg = rospy.wait_for_message("/model_states", ModelStates)
        return msg

    def _load_prev_state(self):
        """Load environment to its last reset."""
        for i, pos in enumerate(self.prev_positions):
            self._set_model_state(self.objects[i], pos, [0, 0, 0, 1])

    def _set_model_state(self, name, pos, quat):
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
