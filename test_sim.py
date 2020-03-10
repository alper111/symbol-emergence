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

parser = argparse.ArgumentParser("test simulation")
parser.add_argument("-ckpt", help="path to the folder", type=str)
args = parser.parse_args()


rospy.init_node("test_node", anonymous=True)
rospy.sleep(1.0)
hand = invisible_force.InvisibleHand()

objects = ["white_ball"]  # "red_ball", "yellow_ball"]
random_ranges = np.array([
    # [0.0, 0.0, 0., 0.41, 0.20, 1.09],
    # [0.0, 0.80, 0., 0.85, -0.10, 1.09],
    [0.20, 0.60, 0., 0.31, -0.10, 1.09],
    # [0.30, 0.80, 0., 0.7, -0.10, 1.09],
    # [0.0, 0.0, 0.0, 0.82, 0.40, 1.09],
])

world = env.Environment(objects=objects, rng_ranges=random_ranges)
episode = 10
action_dim = 2
obs_dim = 4 * len(objects)
hidden_dim = 32

policy_network = models.MLP_gaussian([obs_dim, hidden_dim, hidden_dim, hidden_dim, 2*action_dim])
value_network = models.ValueNetwork([obs_dim, hidden_dim, hidden_dim, hidden_dim, 1])

policy_network.load_state_dict(torch.load(os.path.join(args.ckpt, "policy_net_last.ckpt")))
value_network.load_state_dict(torch.load(os.path.join(args.ckpt, "value_net_last.ckpt")))

print("="*10+"POLICY NETWORK"+"="*10)
print(policy_network)
print("="*10+"VALUE NETWORK"+"="*10)
print(value_network)
print("Training starts...")
reward_history = []
for epi in range(episode):
    rewards = []
    it = 0
    world.random()
    stationary = False
    while not stationary:
        obs = torch.tensor(world.get_state(), dtype=torch.float)
        with torch.no_grad():
            out = policy_network(obs)
            action = out[..., :out.shape[-1]//2].detach()

        white_pos = world.get_object_position("white_ball")
        skip = False
        if white_pos[0] < 0.31 or white_pos[0] > 0.51 or white_pos[1] < -0.1 or white_pos[1] > 0.5:
            skip = True

        if torch.isnan(action).any():
            print("got action nan.")
            exit()

        if not skip:
            hand.apply_force(name="white_ball::link", x=action[0].item(), y=action[1].item(), z=0.0)

        rospy.sleep(0.001)
        stationary = world.is_stationary()
        reward = 0.1
        rewards.append(reward)
        it += 1

    # reshaping
    rewards = torch.tensor(rewards, dtype=torch.float)
    cumrew = rewards.sum().item()
    reward_history.append(cumrew)

    # value_estimate = value_network(observations).reshape(-1)
    # if (torch.isnan(value_estimate)).any():
    #     print("got estimate nan.")
    #     exit()
    # advantage = (rewards - value_estimate).detach()

plt.plot(reward_history)
pp = PdfPages("save/reward.pdf")
pp.savefig()
pp.close()
