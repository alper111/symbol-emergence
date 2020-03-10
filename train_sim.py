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


def discount(rewards):
    R = 0.0
    discounted_rewards = []
    for r in rewards[::1]:
        R = r + 0.99 * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards)
    return discounted_rewards


parser = argparse.ArgumentParser("train simulation")
parser.add_argument("-save", help="save path", type=str)
parser.add_argument("-load", help="load path", type=str)
parser.add_argument("-e", help="number of episodes.", type=int)
parser.add_argument("-r", help="number of rollouts.", type=int)
parser.add_argument("-bs", help="batch size", type=int)
parser.add_argument("-k", help="optimize policy for k epochs.", type=int)
parser.add_argument("-eps", help="epsilon of ppo.", type=float)
parser.add_argument("-hid", help="hidden dim of mlps.", type=int)
args = parser.parse_args()

assert args.bs <= args.r

rospy.init_node("training_node", anonymous=True)
rospy.sleep(1.0)
memory = models.Memory(10000)
hand = invisible_force.InvisibleHand()


# objects = ["white_ball", "red_ball", "yellow_ball"]
objects = ["white_ball"]
random_ranges = np.array([
    # [0.0, 0.0, 0., 0.41, 0.20, 1.09],
    # [0.0, 0.80, 0., 0.85, -0.10, 1.09],
    # [0.20, 0.60, 0., 0.31, -0.10, 1.09],
    [0.0, 0.80, 0., 0.40, -0.10, 1.09],
    # [0.30, 0.80, 0., 0.7, -0.10, 1.09],
    # [0.30, 0.80, 0., 0.7, -0.10, 1.09],
    # [0.0, 0.0, 0.0, 0.82, 0.40, 1.09],
])

world = env.Environment(objects=objects, rng_ranges=random_ranges)
args.r = args.r
action_dim = 2
obs_dim = 4 * len(objects)

policy_network = models.MLP_gaussian([obs_dim, args.hid, args.hid, args.hid, 2*action_dim])
old_policy = models.MLP_gaussian([obs_dim, args.hid, args.hid, args.hid, 2*action_dim])
value_network = models.ValueNetwork([obs_dim, args.hid, args.hid, args.hid, 1])

if args.load is not None:
    print("loading from %s" % args.load)
    policy_network.load_state_dict(torch.load(os.path.join(args.load, "policy_net_last.ckpt")))
    value_network.load_state_dict(torch.load(os.path.join(args.load, "value_net_last.ckpt")))

old_policy.load_state_dict(policy_network.state_dict())

if not os.path.exists(args.save):
    os.makedirs(args.save)
os.chdir(args.save)

optim_policy = torch.optim.Adam(lr=0.001, params=policy_network.parameters(), amsgrad=True)
optim_value = torch.optim.Adam(lr=0.001, params=value_network.parameters(), amsgrad=True)
criterion = torch.nn.MSELoss(reduction="mean")

print("="*10+"POLICY NETWORK"+"="*10)
print(policy_network)
print("="*10+"VALUE NETWORK"+"="*10)
print(value_network)
print("Training starts...")
reward_history = []
for epi in range(args.e):
    for roll in range(args.r):
        rewards = []
        actions = []
        observations = []
        it = 0
        world.random()
        stationary = False
        start_time = rospy.get_time()
        while not stationary and it < 1000:
            white_pos = world.get_object_position("white_ball")
            skip = False
            if white_pos[0] < 0.39 or white_pos[0] > 0.41:
                skip = True

            if not skip:
                obs = torch.tensor(world.get_state(), dtype=torch.float)
                action, _ = policy_network.action(obs)

                if torch.isnan(action).any():
                    print("got action nan.")
                    exit()
                hand.apply_force(name="white_ball::link", x=action[0].item(), y=action[1].item(), z=0.0)
                stationary = world.is_stationary()

                end_time = rospy.get_time()
                reward = (end_time - start_time) * 0.1
                start_time = end_time

                # bookkeeping
                actions.append(action)
                observations.append(obs)
                rewards.append(reward)
                it += 1
            else:
                stationary = world.is_stationary()

        end_time = rospy.get_time()
        reward = (end_time - start_time) * 0.1
        rewards.append(reward)
        rewards = rewards[1:]

        if len(actions) < 1:
            continue

        # reshaping
        rewards = torch.tensor(rewards, dtype=torch.float)
        cumrew = rewards.sum().item()
        rewards = discount(rewards)
        actions = torch.stack(actions)
        observations = torch.stack(observations)
        reward_history.append(cumrew)

        # add to memory
        for i in range(observations.shape[0]):
            memory.append(observations[i], actions[i], rewards[i])
        np.save("rewards.npy", reward_history)

    # last checkpoint for debugging
    with torch.no_grad():
        torch.save(policy_network.state_dict(), "policy_net_last.ckpt")
        torch.save(value_network.state_dict(), "value_net_last.ckpt")
        if (epi + 1) % 50 == 0:
            torch.save(policy_network.state_dict(), "policy_net_{0}.ckpt".format(epi+1))
            torch.save(value_network.state_dict(), "value_net_{0}.ckpt".format(epi+1))

    # OPTIMIZATION
    # optimize the policy network
    for i in range(args.k):
        optim_policy.zero_grad()
        state, action, reward = memory.sample_n(args.bs)
        v_bar = value_network(state).reshape(-1).detach()
        old_logp = old_policy.logprob(state, action).sum(dim=-1).detach()
        new_logp = policy_network.logprob(state, action).sum(dim=-1)
        ratio = torch.exp(new_logp - old_logp)
        adv = (reward - v_bar)
        surr1 = ratio * adv
        surr2 = ratio.clamp(1.0 - args.eps, 1.0 + args.eps) * adv
        policy_loss = - torch.min(surr1, surr2).mean()
        policy_loss.backward()
        # clip gradient to (-10, 10)
        for p in policy_network.parameters():
            p.grad.data.clamp_(-10., 10.)
        optim_policy.step()
    # update old policy network
    old_policy.load_state_dict(policy_network.state_dict())

    # optimize the value network
    for i in range(args.k):
        optim_value.zero_grad()
        state, action, reward = memory.sample_n(args.bs)
        v_bar = value_network(state).reshape(-1)
        value_loss = criterion(v_bar, reward)
        value_loss.backward()
        # clip gradient to (-10, 10)
        for p in value_network.parameters():
            p.grad.data.clamp_(-10., 10.)
        optim_value.step()
    print("Episode: %d, reward: %.3f, adv: %.3f, p loss: %.3f, v loss: %.3f, mem: %d"
          % (
              epi+1,
              np.mean(reward_history[epi*args.r:(epi+1)*args.r]),
              adv.mean().item(),
              policy_loss.item(),
              value_loss.item(),
              memory.size))
    memory.clear()

plt.plot(reward_history)
pp = PdfPages("reward.pdf")
pp.savefig()
pp.close()
