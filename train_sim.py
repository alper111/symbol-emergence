import argparse
import os
import rospy
import torch
import models
import env
import utils
import invisible_force
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


parser = argparse.ArgumentParser("train simulation")
parser.add_argument("-save", help="save path", type=str)
parser.add_argument("-load", help="load path", type=str)
parser.add_argument("-e", help="number of episodes.", type=int)
parser.add_argument("-bs", help="batch size", type=int)
parser.add_argument("-k", help="optimize policy for k epochs.", type=int)
parser.add_argument("-eps", help="epsilon of ppo.", type=float)
parser.add_argument("-hid", help="hidden dim of mlps.", type=int)
parser.add_argument("-update_iter", help="update iter.", type=int)
args = parser.parse_args()

rospy.init_node("training_node", anonymous=True)
rospy.sleep(1.0)
hand = invisible_force.InvisibleHand()

objects = ["white_ball"]
random_ranges = np.array([[0.0, 0.80, 0., 0.40, -0.10, 1.09]])

world = env.Environment(objects=objects, rng_ranges=random_ranges)
action_dim = 1
obs_dim = 4 * len(objects)
c_clip = 1.0
c_v = 0.5
c_ent = 0.01
lr = 0.001
agent = models.PPOAgent(obs_dim, args.hid, 2*action_dim, "gaussian", 2, torch.device("cpu"), lr,
                        args.k, args.bs, args.eps, c_clip, c_v, c_ent)

if args.load is not None:
    print("loading from %s" % args.load)
    agent.load(args.load, ext="_last")

if not os.path.exists(args.save):
    os.makedirs(args.save)
os.chdir(args.save)

print("="*10+"POLICY NETWORK"+"="*10)
print(agent.policy)
print("="*10+"VALUE NETWORK"+"="*10)
print(agent.value)
print("Training starts...")
reward_history = []
for epi in range(args.e):
    observations = []
    actions = []
    logprobs = []
    it = 0
    world.random()
    stationary = False
    start_time = rospy.get_time()
    while not stationary and it < 20:
        white_pos = world.get_object_position("white_ball")
        skip = False
        if white_pos[0] < 0.38 or white_pos[0] > 0.42:
            skip = True

        if not skip:
            obs = torch.tensor(world.get_state(), dtype=torch.float)
            observations.append(obs)
            action, logprob = agent.action(obs)
            # [-1, 1] -> [-pi/2, pi/2]
            action = action * (np.pi/2)
            xf = np.cos(action.item())
            yf = np.sin(action.item())

            if torch.isnan(action).any():
                print("got action nan.")
                exit()
            hand.apply_force(name="white_ball::link", x=xf, y=yf, z=0.0)
            stationary = world.is_stationary()

            # bookkeeping
            actions.append(action)
            logprobs.append(logprob)
            it += 1
        else:
            stationary = world.is_stationary()

    end_time = rospy.get_time()
    reward = (end_time - start_time)
    rewards = [reward] * it

    if len(actions) < 1:
        continue

    # reshaping
    observations = torch.stack(observations)
    actions = torch.stack(actions)
    logprobs = torch.stack(logprobs).detach()
    rewards = torch.tensor(rewards, dtype=torch.float)
    cumrew = rewards[0].item()
    rewards = utils.discount(rewards)
    reward_history.append(cumrew)
    with torch.no_grad():
        values = agent.value(observations).reshape(-1)

    # add to memory
    for i in range(observations.shape[0]):
        agent.memory.append(observations[i], actions[i], logprobs[i], rewards[i], values[i])
    np.save("rewards.npy", reward_history)

    print("Episode %d, reward: %.2f, it: %d, memory: %d" % (epi+1, cumrew, it, agent.memory.size))
    if agent.memory.size >= args.update_iter:
        policy_loss, value_loss, entropy_loss = agent.update()
        print("p loss: %.3f, v loss: %.3f, e loss: %.3f" % (policy_loss, value_loss, entropy_loss))
        agent.memory.clear()
        # last checkpoint for debugging
        with torch.no_grad():
            agent.save(".", ext="_last")
            if (epi + 1) % 50 == 0:
                agent.save(".", ext="_{0}".format(epi+1))

plt.plot(reward_history)
pp = PdfPages("reward.pdf")
pp.savefig()
pp.close()
