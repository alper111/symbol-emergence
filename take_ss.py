"""Take screenshots of states."""
import os
import argparse
import torch
import rospy
import numpy as np
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

# INITIALIZE ENVIRONMENT
objects = ["target_plate", "small_cube"]
random_ranges = {
    "target_plate": np.array([[0.32, 0.52], [0.30, 0.50], [1.125, 1.125]]),
    "small_cube": np.array([[0.32, 0.52], [0.0, 0.15], [1.155, 1.155]]),
}
world = env.Environment(objects=objects, rng_ranges=random_ranges)
rospy.sleep(0.5)

states = torch.load(args.s)
for i, s in enumerate(states):
    s = s.tolist()
    world.set_model_state("target_plate", s[:3], s[3:7])
    world.set_model_state("small_cube", s[7:10], s[10:14])
    os.system("import -window Gazebo %s/%d.jpg" % (args.o, i))
