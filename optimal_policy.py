import rospy
import env
import invisible_force
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def policy(state):
    xf = 0
    yf = 0
    if state[2] < 0:
        xf = -5
    else:
        xf = 5
    return xf, yf


rospy.init_node("optimal_policy", anonymous=True)
rospy.sleep(1.0)
hand = invisible_force.InvisibleHand()

objects = ["white_ball"]
random_ranges = np.array([[0.0, 0.0, 0., 0.40, 0.40, 1.09]])

world = env.Environment(objects=objects, rng_ranges=random_ranges)
reward_history = []
while True:
    it = 0
    world.random()
    stationary = False
    hit = True
    start_time = rospy.get_time()
    while not stationary and it < 100:
        white_pos = world.get_object_position("white_ball")
        skip = False
        if white_pos[0] < 0.35 or white_pos[0] > 0.45:
            skip = True
            hit = True

        if not skip and hit:
            state = world.get_state()
            xf, yf = policy(state)
            # xf = 5 * np.cos(angle.item())
            # yf = 5 * np.sin(angle.item())
            hand.apply_force(name="white_ball::link", x=xf, y=yf, z=0.0)
            stationary = world.is_stationary()
            it += 1
            hit = False
        else:
            stationary = world.is_stationary()

    reward_history.append(it)
    np.save("rewards.npy", reward_history)

    if it > 0:
        print("reward: %d" % it)

plt.plot(reward_history)
pp = PdfPages("reward.pdf")
pp.savefig()
pp.close()
