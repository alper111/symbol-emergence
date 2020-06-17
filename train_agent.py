"""Reinforcement learning agent training script."""
import os
import rospy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torobo_wrapper
import models
import env
import utils

# HYPERPARAMETERS
OUT_FOLDER = "out/ppo_gae_8"
if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)
device = torch.device("cuda:0") if torch.cuda.is_available() else False
episode = 20000
K = 80
eps = 0.2
c_clip = 1.0
c_v = 0.5
c_ent = 0.01
lr_p = 0.0003
lr_v = 0.001
# Discount factor
GAMMA = 0.99
# GAE tradeoff parameter
LAMBDA = 0.95
# absolute positions + orientations (7*2), joint angles (7), tip position (2), relative positions (2*2)
state_dim = 28
hidden_dim = 128
action_dim = 2
batch_size = -1
max_timesteps = 200
update_iter = 20 * max_timesteps


# INITIALIZE PPO AGENT
model = models.PPOAgent(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, dist="gaussian",
                        num_layers=3, device=device, lr_p=lr_p, lr_v=lr_v, K=K, batch_size=batch_size, eps=eps,
                        c_clip=c_clip, c_v=c_v, c_ent=c_ent)

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
objects = ["target_plate", "small_cube", "obstacle"]
random_ranges = {
    "target_plate": np.array([[0.32, 0.52], [0.30, 0.50], [1.125, 1.125]]),
    "small_cube": np.array([[0.32, 0.52], [0.0, 0.15], [1.155, 1.155]]),
    "obstacle": np.array([[0.32, 0.52], [0.25, 0.25], [1.155, 1.155]])
}
world = env.Environment(objects=objects, rng_ranges=random_ranges)
rospy.sleep(0.5)

print("="*10+"POLICY NETWORK"+"="*10)
print(model.policy)
print("="*10+"VALUE NETWORK"+"="*10)
print(model.value)
print("Training starts...")
reward_history = []
for epi in range(episode):
    states = []
    actions = []
    logprobs = []
    rewards = []
    world.random()
    rospy.sleep(0.1)
    target_pos = np.array(world.get_object_position(objects[0])[:2])
    cube_pos = np.array(world.get_object_position(objects[1])[:2])
    diff = target_pos - cube_pos
    diff_n = 0.08 * (diff / np.linalg.norm(diff, 2))
    start_pos = cube_pos - diff_n
    ##
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
    for t in range(max_timesteps):
        # GET STATE
        tip_x = np.array(robot.get_tip_pos()[:2])
        joint_angles = robot.get_joint_angles()
        object_x = world.get_state().reshape(-1)
        # object_x = world.get_state().reshape(-1, 7)
        # rel_object_x = object_x[:, :2] - tip_x"
        # x = np.concatenate([object_x.reshape(-1), rel_object_x.reshape(-1), tip_x, joint_angles])
        x = np.concatenate([object_x, joint_angles])
        x = torch.tensor(x, dtype=torch.float, device=device)
        states.append(x)
        # ACT
        action, logprob = model.action(x)
        # action.clamp_(-1., 1.)
        normalized_action = action * 0.005
        x_next = np.array(tip_x) + normalized_action.cpu().numpy()
        xmin = max(random_ranges["small_cube"][0, 0] - 0.08, 0.32)
        xmax = min(random_ranges["small_cube"][0, 1] + 0.08, 0.51)
        ymin = max(random_ranges["small_cube"][1, 0] - 0.08, -0.10)
        ymax = min(random_ranges["small_cube"][1, 1] + 0.08, 0.50)
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
        if done or (t == (max_timesteps-1)):
            start_diff = np.linalg.norm(diff, 2)
            target_pos = np.array(world.get_object_position(objects[0])[:2])
            cube_pos = np.array(world.get_object_position(objects[1])[:2])
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
    object_x = world.get_state().reshape(-1)
    joint_angles = robot.get_joint_angles()
    # tip_x = np.array(robot.get_tip_pos()[:2])
    # object_x = world.get_state().reshape(-1, 7)
    # rel_object_x = object_x[:, :2] - tip_x
    # x = np.concatenate([object_x.reshape(-1), rel_object_x.reshape(-1), tip_x, joint_angles])
    x = np.concatenate([object_x, joint_angles])
    x = torch.tensor(x, dtype=torch.float, device=device)
    states.append(x)
    if not done:
        rewards.append(model.value(x).item())
    else:
        rewards.append(0.0)

    # RESHAPING
    states = torch.stack(states)
    actions = torch.stack(actions)
    logprobs = torch.stack(logprobs).detach()
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    with torch.no_grad():
        values = model.value(states).reshape(-1)
        if done:
            values[-1] = 0.0
        advantages = rewards[:-1] + GAMMA * values[1:] - values[:-1]
    discounted_adv = utils.discount(advantages, GAMMA * LAMBDA)
    cumrew = rewards[:-1].sum().item()
    rewards = utils.discount(rewards, gamma=GAMMA)[:-1]
    reward_history.append(cumrew)

    if (torch.isnan(values)).any():
        print("got estimate nan.")
        exit()

    print("Episode: %d, reward: %.3f, std: %.3f, %.3f" % (epi+1, cumrew, *torch.exp(model.log_std).detach()))

    # ADD TO MEMORY
    for i in range(states.shape[0]-1):
        model.record(states[i], actions[i], logprobs[i], rewards[i], discounted_adv[i])
    np.save(os.path.join(OUT_FOLDER, "rewards.npy"), reward_history)

    # UPDATE
    if model.memory.size >= update_iter:
        loss = model.update()
        model.reset_memory()
        model.save(OUT_FOLDER, ext=str(epi+1))
        print("Episode: %d, reward: %.3f, loss=%.3f" % (epi+1, cumrew, loss))

# RESET ARM POSITION
robot.go(np.radians([90, 45, 0, 45, 0, -90, 0]))
rospy.sleep(2)
robot.go(np.radians([90, 0, 0, 0, 0, -90, 0]))
rospy.sleep(2)
robot.go(np.radians([0, 0, 0, 0, 0, 0, 0]))
rospy.sleep(2)

plt.plot(reward_history)
pp = PdfPages(os.path.join(OUT_FOLDER, "reward.pdf"))
pp.savefig()
pp.close()
