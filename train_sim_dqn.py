import argparse
import os
import rospy
import torch
import models
import env
import invisible_force
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


parser = argparse.ArgumentParser("train simulation")
parser.add_argument("-save", help="save path", type=str)
parser.add_argument("-load", help="load path", type=str)
parser.add_argument("-e", help="number of episodes.", type=int)
parser.add_argument("-bs", help="batch size", type=int)
parser.add_argument("-hid", help="hidden dim of mlps.", type=int)
parser.add_argument("-update_iter", help="update iter.", type=int)
args = parser.parse_args()

rospy.init_node("training_node", anonymous=True)
rospy.sleep(1.0)
hand = invisible_force.InvisibleHand()

objects = ["white_ball"]
random_ranges = np.array([[0.0, 0.8, 0., 0.40, -0.10, 1.09]])

world = env.Environment(objects=objects, rng_ranges=random_ranges)
angle_dim = 8
force_dim = 6
action_dim = angle_dim * force_dim
warmup = 100
state_dim = 4 * len(objects)
lr = 0.001

agent = models.DQN(state_dim, args.hid, action_dim, 2, 1000, lr, args.bs, 1.0, 0.99, 100, torch.device("cpu"))

if args.load is not None:
    print("loading from %s" % args.load)
    agent.load(args.load, ext="_last")

if not os.path.exists(args.save):
    os.makedirs(args.save)
os.chdir(args.save)

print("="*10+"Q network"+"="*10)
print(agent.Q)
print("Training starts...")
reward_history = []
running_avg = []
update_step = 0
for epi in range(args.e):
    it = 0
    avgloss = 0.0
    world.random()
    stationary = False
    start_time = rospy.get_time()
    prev_state = None
    prev_action = None
    hit = True
    while not stationary and it < 100:
        white_pos = world.get_object_position("white_ball")
        skip = False
        if white_pos[0] < 0.37 or white_pos[0] > 0.43:
            skip = True
            hit = True

        if not skip and hit:
            state = torch.tensor(world.get_state(), dtype=torch.float)
            action = agent.action(state, epsgreedy=True)
            force = action // angle_dim
            angle = action % angle_dim
            angle = angle.float() * ((2*np.pi)/angle_dim)
            xf = force.float() * np.cos(angle.item())
            yf = force.float() * np.sin(angle.item())
            # print("Action: %d, Force: %d, Angle: %.1f, x: %.2f, y: %.2f" % (action, force, np.degrees(angle), xf, yf))

            if torch.isnan(action).any():
                print("got action nan.")
                exit()
            hand.apply_force(name="white_ball::link", x=xf, y=yf, z=0.0)
            stationary = world.is_stationary()
            it += 1
            if prev_state is not None:
                reward = torch.tensor(1.0)
                terminal = torch.tensor(False)
                agent.record(prev_state, prev_action, reward, state, terminal)
            prev_state = state
            prev_action = action
            hit = False

            if agent.memory.size >= warmup:
                loss = agent.update()
            else:
                loss = 0.0
            avgloss += loss
        else:
            stationary = world.is_stationary()

    if it > 0:
        state = torch.tensor(world.get_state(), dtype=torch.float)
        reward = torch.tensor(0.0)
        terminal = torch.tensor(True)
        agent.record(prev_state, prev_action, reward, state, terminal)
        if agent.memory.size >= warmup:
            agent.update()
            agent.decay_epsilon(0.999, 0.1)

    reward_history.append(it)
    np.save("rewards.npy", reward_history)

    if it > 0:
        print("Episode %d, reward: %d, loss: %.3f, update count: %d" % (epi+1, it, avgloss/it, agent.num_updates))
        with torch.no_grad():
            agent.save(".", ext="_last")
            if (epi + 1) % 50 == 0:
                agent.save(".", ext="_{0}".format(epi+1))

plt.plot(reward_history)
pp = PdfPages("reward.pdf")
pp.savefig()
pp.close()
