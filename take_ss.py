"""Take screenshots of states."""
import os
import argparse
import rospy
import numpy as np
import torobo_wrapper
import env

parser = argparse.ArgumentParser("Record states.")
parser.add_argument("-s", help="state file", type=str, required=True)
parser.add_argument("-o", help="output folder", type=str, required=True)
args = parser.parse_args()

if not os.path.exists(args.o):
    os.makedirs(args.o)

# INITIALIZE ROSNODE
rospy.init_node("test_node", anonymous=True)
rate = rospy.Rate(100)
rospy.sleep(1.0)
robot = torobo_wrapper.Torobo()
rospy.sleep(1.0)

# INITIALIZE ARM POSITION
robot.go(np.radians([90, 0, 0, 0, 0, -90, 0]))
rospy.sleep(2)
robot.go(np.radians([90, 45, 0, 45, 0, -90, 0]))
rospy.sleep(2)

# INITIALIZE ENVIRONMENT
objects = ["target_plate", "small_cube"]
random_ranges = {
    "target_plate": np.array([[0.32, 0.52], [0.30, 0.50], [1.125, 1.125]]),
    "small_cube": np.array([[0.32, 0.52], [0.0, 0.15], [1.155, 1.155]]),
}
world = env.Environment(objects=objects, rng_ranges=random_ranges)
rospy.sleep(0.5)

states = np.load(args.s, allow_pickle=True)
for i, s in enumerate(states):
    robot.go(s["robot"], time_from_start=2.0)
    rospy.sleep(3.0)
    world.set_model_state("target_plate", s["target_plate"][:3], s["target_plate"][3:])
    world.set_model_state("small_cube", s["small_cube"][:3], s["small_cube"][3:])
    rospy.sleep(0.1)
    os.system("import -window Gazebo %s/%d.jpg" % (args.o, i))
