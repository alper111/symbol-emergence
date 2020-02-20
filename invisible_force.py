import rospy
from gazebo_msgs.srv import ApplyBodyWrench
from geometry_msgs.msg import Wrench


class InvisibleHand:

    def __init__(self):
        self.service_name = "gazebo/apply_body_wrench"
        rospy.wait_for_service(self.service_name)
        self.service = rospy.ServiceProxy(self.service_name, ApplyBodyWrench)

    def apply_force(self, name, x, y, z):
        wrench = Wrench()
        wrench.force.x = x
        wrench.force.y = y

        self.service(
            body_name=name,
            wrench=wrench,
            start_time=rospy.Time.now(),
            duration=rospy.Duration(0.001)
        )
