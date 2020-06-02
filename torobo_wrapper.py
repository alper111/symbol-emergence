import rospy
import numpy as np
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotState, PositionIKRequest
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation


class Torobo:
    """Wrapper class for Torobo."""

    def __init__(self):
        self.MAX_VELOCITY = np.radians([150, 150, 180, 180, 200, 200, 200])
        self.JOINT_NAMES = ["left_arm/joint_" + str(i) for i in range(1, 8)]

        rospy.wait_for_service("/torobo/compute_fk")
        rospy.wait_for_service("/torobo/compute_ik")

        self._fk = rospy.ServiceProxy("/torobo/compute_fk", GetPositionFK)
        self._ik = rospy.ServiceProxy("/torobo/compute_ik", GetPositionIK)
        self._publisher = rospy.Publisher("/torobo/left_arm_controller/command", JointTrajectory, queue_size=1)

    def compute_fk(self, joint_angles):
        header = Header(0, rospy.Time.now(), "world")
        rs = RobotState()
        rs.joint_state.name = self.JOINT_NAMES
        rs.joint_state.position = joint_angles
        response = self._fk(header, ["left_gripper/grasping_frame"], rs)
        pose = response.pose_stamped[0].pose
        quat = Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        euler = quat.as_euler("xyz")
        return [pose.position.x, pose.position.y, pose.position.z, euler[0], euler[1], euler[2]]

    def compute_ik(self, joint_angles, target):
        quaternion = Rotation.from_euler("zyx", [target[-1], target[-2], target[-3]]).as_quat()
        # create request
        req = PositionIKRequest()
        req.timeout = rospy.Duration(0.05)
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
        res = self._ik(req)
        ik_result = res.solution
        if res.error_code.val == -31:
            rospy.loginfo("Cannot find IK solution")
            return -31

        # todo: make this generic
        return ik_result.joint_state.position[4:11]

    def go(self, positions, velocities=None, accelerations=None, effort=None, time_from_start=2.0):
        msg = JointTrajectory()
        msg.joint_names = self.JOINT_NAMES
        joint_states = JointTrajectoryPoint()
        joint_states.positions = positions
        if velocities:
            joint_states.velocities = velocities
        if accelerations:
            joint_states.accelerations = accelerations
        if effort:
            joint_states.effort = effort
        joint_states.time_from_start = rospy.Duration.from_sec(time_from_start)
        msg.points.append(joint_states)
        self._publisher.publish(msg)

    def convert_to_joint(self, points):
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

    def init_pose(self):
        self.go(np.radians([90, 45, 0, 45, 0, 0, 0]))

    def zero_pose(self):
        self.go(np.radians([0, 0, 0, 0, 0, 0, 0]))

    def get_joint_angles(self):
        msg = rospy.wait_for_message("/torobo/joint_states", JointState)
        joint_angles = msg.position[2:9]
        return joint_angles

    def get_tip_pos(self):
        angles = self.get_joint_angles()
        x = self.compute_fk(angles)
        return x
