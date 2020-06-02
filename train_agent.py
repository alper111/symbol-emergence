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


OUT_FOLDER = "out/ppo_1"
if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)
device = torch.device("cuda:0") if torch.cuda.is_available() else False
episode = 20000
K = 4
batch_size = -1
eps = 0.2
update_iter = 2000
c_clip = 1.0
c_v = 0.5
c_ent = 0.01
lr = 0.002
state_dim = 4+7
hidden_dim = 128
action_dim = 2
max_timesteps = 200

model = models.PPOAgent(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=2*action_dim, dist="gaussian",
                        num_layers=2, device=device, lr=lr, K=K, batch_size=batch_size, eps=eps, c_clip=c_clip,
                        c_v=c_v, c_ent=c_ent)

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
# robot.go(np.radians([90, 45, 0, 45, 0, -30, 0]))
# rospy.sleep(2)

objects = ["target_plate", "small_cube"]
random_ranges = np.array([
    [0.00, 0.00, 0., 0.4, 0.3, 1.125],
    # [0.19, 0.60, 0., 0.32, -0.10, 1.140],
    [0.19, 0.50, 0., 0.32, 0.0, 1.155],
    # [0.00, 0.00, 0., 0.4, 0.1, 1.135],
])


world = env.Environment(objects=objects, rng_ranges=random_ranges)
rospy.sleep(0.5)
# world.random()

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
    cube_pos = world.get_object_position(objects[1])
    cube_pos[1] = cube_pos[1] - 0.07
    ##
    robot.go(np.radians([90, 45, 0, 45, 0, -30, 0]))
    rospy.sleep(2)
    ##
    action_aug = cube_pos + [1.17, np.pi, 0, 0]
    angles = robot.compute_ik(robot.get_joint_angles(), action_aug)
    if angles != -31:
        robot.go(angles, time_from_start=2.0)
        rospy.sleep(2.0)
    else:
        print("Skipping")
        continue

    for t in range(max_timesteps):
        x = world.get_state().reshape(-1, 2)
        tip_pos = np.array(robot.get_tip_pos()[:2])
        joint_angles = robot.get_joint_angles()
        x = (x - tip_pos).reshape(-1)
        x = np.concatenate([x, joint_angles])
        x = torch.tensor(x, dtype=torch.float, device=device)
        states.append(x)
        action, logprob = model.action(x)
        action.clamp_(-1., 1.)
        action *= 0.1
        x_next = np.array(tip_pos) + action.cpu().numpy()
        x_next[0] = np.clip(x_next[0], 0.32, 0.51)
        x_next[1] = np.clip(x_next[1], -0.1, 0.5)
        action_aug = x_next.tolist() + [1.17, np.pi, 0, 0]
        angles = robot.compute_ik(robot.get_joint_angles(), action_aug)
        if angles != -31:
            robot.go(angles, time_from_start=2.0)
            rate.sleep()

        actions.append(action)
        logprobs.append(logprob)
        if world.is_terminal() or (t == max_timesteps-1):
            reward = world.get_reward(arm_position=tip_pos)
            rewards.append(reward)
            break
        else:
            rewards.append(0.0)

    # reshaping
    states = torch.stack(states)
    actions = torch.stack(actions)
    logprobs = torch.stack(logprobs).detach()
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    cumrew = rewards.sum().item()
    rewards = utils.discount(rewards, gamma=0.99)
    reward_history.append(cumrew)
    with torch.no_grad():
        values = model.value(states).reshape(-1)

    if (torch.isnan(values)).any():
        print("got estimate nan.")
        exit()

    print("Episode: %d, reward: %.3f" % (epi+1, cumrew))
    # add to memory
    for i in range(states.shape[0]):
        model.record(states[i], actions[i], logprobs[i], rewards[i], values[i])
    np.save(os.path.join(OUT_FOLDER, "rewards.npy"), reward_history)
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
