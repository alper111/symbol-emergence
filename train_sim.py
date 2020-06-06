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
action_dim = 2
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
running_avg = []
update_step = 0
for epi in range(args.e):
    observations = []
    actions = []
    logprobs = []
    it = 0
    world.random()
    stationary = False
    start_time = rospy.get_time()
    hit = True
    while not stationary and it < 100:
        white_pos = world.get_object_position("white_ball")
        skip = False
        if white_pos[0] < 0.37 or white_pos[0] > 0.43:
            skip = True
            hit = True

        if not skip and hit:
            obs = torch.tensor(world.get_state(), dtype=torch.float)
            observations.append(obs)
            action, logprob = agent.action(obs)
            # clamp to [-1, 1], since it can get values outside this range
            action.clamp_(-1., 1.)
            # [-1, 1] -> [0, 5]
            amplitude = (action[0]+1) * (5/2)
            # [-1, 1] -> [-pi, pi]
            angle = action[1] * np.pi
            xf = amplitude * np.cos(angle.item())
            yf = amplitude * np.sin(angle.item())

            if torch.isnan(action).any():
                print("got action nan.")
                exit()
            hand.apply_force(name="white_ball::link", x=xf, y=yf, z=0.0)
            stationary = world.is_stationary()

            # bookkeeping
            actions.append(action)
            logprobs.append(logprob)
            it += 1
            hit = False
        else:
            stationary = world.is_stationary()

    end_time = rospy.get_time()
    # reward = (end_time - start_time)
    # rewards = [reward] * it
    rewards = [it] * it

    if len(actions) < 1:
        continue

    # reshaping
    observations = torch.stack(observations)
    actions = torch.stack(actions)
    logprobs = torch.stack(logprobs).detach()
    rewards = torch.tensor(rewards, dtype=torch.float)
    cumrew = rewards[0].item()
    rewards = utils.discount(rewards, gamma=0.99)
    reward_history.append(cumrew)
    with torch.no_grad():
        values = agent.value(observations).reshape(-1)

    # add to memory
    for i in range(observations.shape[0]):
        agent.record(observations[i], actions[i], logprobs[i], rewards[i], values[i])
    np.save("rewards.npy", reward_history)

    print("Episode %d, reward: %.2f, it: %d, memory: %d" % (epi+1, cumrew, it, agent.memory.size))
    if agent.memory.size >= args.update_iter:
        loss = agent.update()
        agent.reset_memory()
        print("loss: %.3f" % loss)

        running_avg.append(np.mean(reward_history[update_step:]))
        np.save("running_avg.npy", running_avg)
        update_step = epi
        # last checkpoint for debugging
        with torch.no_grad():
            agent.save(".", ext="_last")
            if (epi + 1) % 50 == 0:
                agent.save(".", ext="_{0}".format(epi+1))

plt.plot(reward_history)
pp = PdfPages("reward.pdf")
pp.savefig()
pp.close()
