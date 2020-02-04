import rospy
import actionlib
import numpy as np
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    JointTolerance
)
from moveit_msgs.msg import RobotState, PositionIKRequest
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation


class Torobo:
    def __init__(self):
        self.MAX_VELOCITY = np.radians([1500, 150, 180, 180, 200, 200, 200])
        self.ACTION_SERVICE = "/torobo/left_arm_controller/follow_joint_trajectory"
        self.JOINT_NAMES = ["left_arm/joint_" + str(i) for i in range(1, 8)]

        rospy.wait_for_service("/torobo/compute_fk")
        rospy.wait_for_service("/torobo/compute_ik")

        self.fk = rospy.ServiceProxy("/torobo/compute_fk", GetPositionFK)
        self.ik = rospy.ServiceProxy("/torobo/compute_ik", GetPositionIK)

        self.action_client = actionlib.SimpleActionClient(
            self.ACTION_SERVICE,
            FollowJointTrajectoryAction)
        self.action_client.wait_for_server()

    """
    Function for computing the forward kinematics of Torobo humanoid

    arguments:
        names: list of joint names (string[])
        joint_angles: list of joint angles (float[])
    returns:
        pose: x, y, z, roll, pitch, yaw (float[])
    """
    def compute_fk(self, joint_angles):
        header = Header(0, rospy.Time.now(), "world")
        rs = RobotState()
        rs.joint_state.name = self.JOINT_NAMES
        rs.joint_state.position = joint_angles
        response = self.fk(header, ["left_gripper/grasping_frame"], rs)
        pose = response.pose_stamped[0].pose
        euler = Rotation.from_quat([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w]).as_euler("xyz")

        return [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            euler[0],
            euler[1],
            euler[2]]

    """
    Function for computing the inverse kinematics of Torobo humanoid

    arguments:
        joint_angles: initial joint angles as seed (list of floats)
        x: x-axis coordinate (float)
        y: y-axis coordinate (float)
        z: z-axis coordinate (float)
        roll: rotation around x-axis in radian (float)
        pitch: rotation around y-axis in radian (float)
        yaw: rotation around z-axis in radian (float)
    returns:
        -31 if cannot find ik solution
    else
        joint_angles: list of joint angles (float[])
    """
    def compute_ik(self, joint_angles, target):
        quaternion = Rotation.from_euler(
            "zyx",
            [
                target[-1],
                target[-2],
                target[-3]
            ]).as_quat()
        # create request
        req = PositionIKRequest()
        req.timeout = rospy.Duration(0.1)
        req.ik_link_name = "left_gripper/grasping_frame"
        req.pose_stamped.header = Header()
        req.pose_stamped.header.frame_id = "world"
        req.pose_stamped.pose.position.x = target[0]
        req.pose_stamped.pose.position.y = target[1]
        req.pose_stamped.pose.position.z = target[2]
        req.pose_stamped.pose.orientation.x = quaternion[0]
        req.pose_stamped.pose.orientation.y = quaternion[1]
        req.pose_stamped.pose.orientation.z = quaternion[2]
        req.pose_stamped.pose.orientation.w = quaternion[3]
        req.robot_state.joint_state.name = self.JOINT_NAMES
        req.robot_state.joint_state.position = joint_angles
        req.group_name = "left_arm"
        req.avoid_collisions = False

        # get ik result
        res = self.ik(req)
        ik_result = res.solution
        if res.error_code.val == -31:
            rospy.loginfo("Cannot find IK solution")
            return -31

        # todo: make this generic
        return ik_result.joint_state.position[4:11]

    """
    Function for action to move the arm

    Parameters
    ----------
    action_client : actionlib.SimpleActionClient
        SimpleActionClient
    joint_names : list
        list of joint names
    positions : list
        list of joint's goal positions(radian)
    time_from_start : float
        transition time from start

    Returns
    -------
    None

    Throws
    ------
    e : rospy.ROSInterruptException
        action fail
    """
    def follow_joint_trajectory(
        self,
        positions,
        times=None,
        velocities=None,
        accelerations=None,
        efforts=None
    ):

        if times is None:
            joint_angles = np.array(self.get_joint_angles())
            difference = abs(joint_angles - positions[0])
            times = [(difference / self.MAX_VELOCITY).max()+0.1]
            for i in range(positions.shape[0]-1):
                diff = (positions[i] - positions[i+1])
                times.append(times[-1]+(diff / self.MAX_VELOCITY).max()+0.1)

        # Creates a goal.
        goal = FollowJointTrajectoryGoal()
        goal.goal_time_tolerance = rospy.Time(0.5)
        goal.trajectory.joint_names = self.JOINT_NAMES
        goal.path_tolerance = []
        goal.goal_tolerance = []
        # set tolerances
        for i in range(7):
            tolerance = JointTolerance()
            tolerance.name = self.JOINT_NAMES[i]
            tolerance.position = -1
            tolerance.velocity = -1
            tolerance.acceleration = -1
            goal.path_tolerance.append(tolerance)
            goal.goal_tolerance.append(tolerance)

        for i in range(positions.shape[0]):
            point = JointTrajectoryPoint()
            point.positions = positions[i]
            if velocities is not None:
                point.velocities = velocities[i]
            if accelerations is not None:
                point.accelerations = accelerations[i]
            if efforts is not None:
                point.effort = efforts[i]
            point.time_from_start = rospy.Duration(times[i])
            goal.trajectory.points.append(point)

        goal.trajectory.header = Header()
        goal.trajectory.header.stamp = rospy.Time.now()
        # Sends the goal.
        self.action_client.send_goal(goal)

        # Waits for the server.
        self.action_client.wait_for_result(timeout=rospy.Duration(times[-1] + 0.5))

        # Log state and result.
        state = self.action_client.get_state()
        result = self.action_client.get_result()
        if state != 3:
            rospy.loginfo("[state ]: " + str(state))
            rospy.loginfo("[result]: " + str(result))
        return state

    """
    Given a cartesian via-points, this method creates a
    path in the joint space with linear segments.

    arguments:
        points: via-points (2-dim np.array)
    returns:
        path: list of joint angles (float[][])
        failed: path is found or not (bool)
    """
    def create_cartesian_path(self, points):
        angles = self.get_joint_angles()
        path = []
        failed = False
        for i in range(points.shape[0]):
            angles = self.compute_ik(angles, points[i])
            if angles != -31:
                path.append(angles)
            else:
                failed = True
                break

        return path, failed

    """
    Given an angle, this method creates two
    via points which can be used later on with
    create_cartesian_path method.
    arguments:
        location: center point (float[])
        angle: angle (float)
        radius: radius (float)
    returns:
        via_points: via points (float[][])
    """
    def create_via_points(self, location, angle, radius):
        x = location[0]
        y = location[1]
        w1 = [x+radius*np.cos(angle), y+radius*np.sin(angle)] + location[2:]
        return [w1, location]

    """
    Moves robot to a pre-defined initial position from
    zero position.

    arguments:
        none
    returns:
        none
    """
    def initialize(self):
        # angles = np.radians([
        #     [0, 0, 90, 0, 0, -30, 0],
        #     [120, 0, 90, 45, 0, -30, 0],
        #     [120, 90, 90, -45, -90, -30, 0]
        # ])
        angles = np.radians([
            [120., 0., 60., 0., 0., 0., 0.],
            [120., 0., 60., 45., 0., 0., 0.],
            [120., 0., -20., 45., 0., 0., 0.],
            [120., 30., -20., 45., 0., 0., 0.]])
        self.follow_joint_trajectory(angles)

    """
    Moves robot to a pre-defined initial position.

    arguments:
        none
    returns:
        none
    """
    def init_pose(self):
        angles = np.radians([
            # [120, 90, 90, -45, -90, -30, 0]
            [120., 30., -20., 45., 0., 0., 0.]
        ])
        self.follow_joint_trajectory(angles)

    """
    Moves robot to zero angle position from a pre-defined
    initial position.

    arguments:
        none
    returns:
        none
    """
    def zero_pose(self):
        # angles = np.radians([
        #     [120, 0, 90, 45, 0, -30, 0],
        #     [0, 0, 90, 0, 0, -30, 0],
        #     [0, 0, 0, 0, 0, 0, 0]
        # ])
        angles = np.radians([
            [120., 0., -20., 45., 0., 0., 0.],
            [120., 0., 60, 45., 0., 0., 0.],
            [120., 0., 60, 0., 0., 0., 0.],
            [0, 0, 0, 0, 0, 0, 0]])
        self.follow_joint_trajectory(angles)

    """
    Get current joint angles

    arguments:
        none
    returns
        joint_angles: current joint angles (float[])
    """
    def get_joint_angles(self):
        msg = rospy.wait_for_message("/torobo/joint_states", JointState)
        joint_angles = msg.position[2:9]
        return joint_angles
