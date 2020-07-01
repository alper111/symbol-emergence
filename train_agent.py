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

# INITIALIZE ROSNODE
rospy.init_node("training_node", anonymous=True)
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
world = env.Environment(objects=opts["objects"], rng_ranges=opts["ranges"])
rospy.sleep(0.5)

print("="*10+"POLICY NETWORK"+"="*10)
print(model.policy)
print("="*10+"VALUE NETWORK"+"="*10)
print(model.value)
print("Training starts...")
reward_history = []
for epi in range(opts["episode"]):

    states = []
    actions = []
    logprobs = []
    rewards = []

    world.random()
    rospy.sleep(0.1)
    # MOVE TO INITIAL POSITION
    target_pos = np.array(world.get_object_position(opts["objects"][0])[:2])
    cube_pos = np.array(world.get_object_position(opts["objects"][1])[:2])
    diff = target_pos - cube_pos
    diff_n = 0.08 * (diff / np.linalg.norm(diff, 2))
    start_pos = cube_pos - diff_n
    robot.go(np.radians([90, 45, 0, 45, 0, -30, 0]))
    rospy.sleep(2)
    ##
    action_aug = start_pos.tolist() + [1.17, np.pi, 0, 0]
    angles = robot.compute_ik(robot.get_joint_angles(), action_aug)
    if angles != -31:
        robot.go(angles, time_from_start=2.0)
        rospy.sleep(2.0)
    else:
        print("Skipping")
        continue

    skip = False
    for t in range(opts["max_timesteps"]):
        # GET STATE
        x = utils.construct_state(world, robot, opts["device"])
        states.append(x)

        # ACT
        action, logprob = model.action(x)
        # action.clamp_(-1., 1.)
        normalized_action = action * 0.005
        x_next = np.array(x[2:4]) + normalized_action.cpu().numpy()
        xmin = max(opts["ranges"][opts["objects"][1]][0, 0] - 0.08, 0.32)
        xmax = min(opts["ranges"][opts["objects"][1]][0, 1] + 0.08, 0.51)
        ymin = max(opts["ranges"][opts["objects"][1]][1, 0] - 0.08, -0.10)
        ymax = min(opts["ranges"][opts["objects"][1]][1, 1] + 0.08, 0.50)
        x_next[0] = np.clip(x_next[0], xmin, xmax)
        x_next[1] = np.clip(x_next[1], ymin, ymax)
        action_aug = x_next.tolist() + [1.17, np.pi, 0, 0]
        angles = robot.compute_ik(robot.get_joint_angles(), action_aug)
        if angles != -31:
            robot.go(angles, time_from_start=rate.sleep_dur.to_sec())
            rate.sleep()
        else:
            skip = True
            break
        done = world.is_terminal()

        # COLLECT REWARD AND BOOKKEEPING
        actions.append(action)
        logprobs.append(logprob)
        if done or (t == (opts["max_timesteps"]-1)):
            start_diff = np.linalg.norm(diff, 2)
            target_pos = np.array(world.get_object_position(opts["objects"][0])[:2])
            cube_pos = np.array(world.get_object_position(opts["objects"][1])[:2])
            end_diff = np.linalg.norm(target_pos - cube_pos, 2)
            reward = start_diff - end_diff
            if reward < 1e-4:
                reward = 0.0
            rewards.append(reward*10)
            break
        else:
            rewards.append(0.0)
    if len(actions) < 10:
        skip = True

    if skip:
        print("Breaking the episode")
        continue

    # CONSTRUCT STATE T+1
    x = utils.construct_state(world, robot, opts["device"])
    states.append(x)
    if not done:
        rewards.append(model.value(x).item())
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
    reward_history.append(cumrew)

    if (torch.isnan(values)).any():
        print("got estimate nan.")
        exit()

    print("Episode: %d, reward: %.3f, std: %.3f, %.3f" % (epi+1, cumrew, *torch.exp(model.log_std).detach()))

    # ADD TO MEMORY
    for i in range(states.shape[0]-1):
        model.record(states[i], actions[i], logprobs[i], rewards[i], discounted_adv[i])
    np.save(os.path.join(opts["save"], "rewards.npy"), reward_history)

    # UPDATE
    if model.memory.size >= opts["update_iter"]:
        loss = model.update()
        model.reset_memory()
        model.save(opts["save"], ext=str(epi+1))
        model.save(opts["save"], ext="_last")
        print("Episode: %d, reward: %.3f, loss=%.3f" % (epi+1, cumrew, loss))

# RESET ARM POSITION
robot.go(np.radians([90, 45, 0, 45, 0, -90, 0]))
rospy.sleep(2)
robot.go(np.radians([90, 0, 0, 0, 0, -90, 0]))
rospy.sleep(2)
robot.go(np.radians([0, 0, 0, 0, 0, 0, 0]))
rospy.sleep(2)

plt.plot(reward_history)
pp = PdfPages(os.path.join(opts["save"], "reward.pdf"))
pp.savefig()
pp.close()
