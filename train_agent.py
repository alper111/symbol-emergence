"""Reinforcement learning agent training script."""
import os
import argparse
import time
import yaml
import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torobo_wrapper
import models
import env
import utils

parser = argparse.ArgumentParser("Train PPO Agent.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

# READ SETTINGS
opts = yaml.safe_load(open(args.opts, "r"))
if not os.path.exists(opts["save"]):
    os.makedirs(opts["save"])
opts["time"] = time.asctime(time.localtime(time.time()))
file = open(os.path.join(opts["save"], "opts.yml"), "w")
yaml.dump(opts, file)
file.close()
print(yaml.dump(opts))


# INITIALIZE PPO AGENT
model = models.PPOAgent(state_dim=opts["state_dim"], hidden_dim=opts["hidden_dim"], action_dim=opts["action_dim"],
                        dist="gaussian", num_layers=opts["num_layers"], device=opts["device"], lr_p=opts["lr_p"],
                        lr_v=opts["lr_v"], K=opts["K"], batch_size=opts["batch_size"], eps=opts["eps"],
                        c_clip=opts["c_clip"], c_v=opts["c_v"], c_ent=opts["c_ent"])

# OPTIONAL, LOAD AGENT
if "load" in opts.keys():
    model.load(path=opts["load"], ext="_last")

# INITIALIZE ROSNODE
rospy.init_node("training_node", anonymous=True)
rate = rospy.Rate(100)
rospy.sleep(1.0)
robot = torobo_wrapper.Torobo()
rospy.sleep(1.0)

# INITIALIZE ENVIRONMENT
world = env.Environment(robot=robot, objects=opts["objects"], rng_ranges=opts["ranges"])
rospy.sleep(0.5)
world.initialize()

print("="*10+"POLICY NETWORK"+"="*10)
print(model.policy)
print("="*10+"VALUE NETWORK"+"="*10)
print(model.value)
print("Training starts...")
reward_history = []
temp_history = []
it = 0
update_count = 0
while update_count < opts["episode"]:

    states = []
    actions = []
    logprobs = []
    rewards = []

    obs = torch.tensor(world.reset(), dtype=torch.float, device=opts["device"])
    skip = False
    for t in range(opts["max_timesteps"]):
        action, logprob = model.action(obs)
        act_tanh = torch.tanh(action)
        states.append(obs)
        obs, reward, done, success = world.step(obs.cpu().numpy(), act_tanh.cpu().numpy(), rate)
        obs = torch.tensor(obs, dtype=torch.float, device=opts["device"])
        if not success:
            skip = True
            break

        # COLLECT REWARD AND BOOKKEEPING
        actions.append(action)
        logprobs.append(logprob)
        if done or (t == (opts["max_timesteps"]-1)):
            rewards.append(reward)
            break
        else:
            rewards.append(0.0)

    if t < 2:
        skip = True
    if skip:
        print("Breaking the episode")
        continue

    it += 1
    states.append(obs)
    if not done:
        rewards.append(model.value(obs).item())
    else:
        rewards.append(0.0)

    # RESHAPING
    states = torch.stack(states)
    actions = torch.stack(actions)
    logprobs = torch.stack(logprobs).detach()
    rewards = torch.tensor(rewards, dtype=torch.float, device=opts["device"])
    with torch.no_grad():
        values = model.value(states).reshape(-1)
        if done:
            values[-1] = 0.0
        advantages = rewards[:-1] + opts["gamma"] * values[1:] - values[:-1]
    discounted_adv = utils.discount(advantages, opts["gamma"] * opts["lambda"])
    cumrew = rewards[:-1].sum().item()
    rewards = utils.discount(rewards, gamma=opts["gamma"])[:-1]
    temp_history.append(cumrew)

    print("Episode: %d, reward: %.3f, std: %.3f, %.3f" % (it, cumrew, *torch.exp(model.log_std).detach()))

    # ADD TO MEMORY
    for i in range(states.shape[0]-1):
        model.record(states[i], actions[i], logprobs[i], rewards[i], discounted_adv[i])

    # UPDATE
    if model.memory.size >= opts["update_iter"]:
        loss = model.update()
        model.reset_memory()
        model.save(opts["save"], ext=str(it))
        model.save(opts["save"], ext="_last")
        reward_history.append(np.mean(temp_history))
        temp_history = []
        print("Update: %d, Avg. Rew.: %.3f, loss=%.3f" % (update_count, reward_history[-1], loss))
        np.save(os.path.join(opts["save"], "rewards.npy"), reward_history)
        update_count += 1

world.zerorobotpose()

plt.plot(reward_history)
pp = PdfPages(os.path.join(opts["save"], "reward.pdf"))
pp.savefig()
pp.close()
