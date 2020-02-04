import rospy
import torch
import torobo
import models
import env
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def act(env, agent, action, radius):
    white_pos = env.get_object_position("white_ball")
    w1 = white_pos + [1.17, np.pi, 0, 0]
    via = agent.create_via_points(w1, action, radius)
    ans, failed = agent.create_cartesian_path(np.array(via))
    if not failed:
        agent.follow_joint_trajectory(np.array(ans))
        return True
    else:
        return False


rospy.init_node("test_node", anonymous=True)
rospy.sleep(1.0)
agent = torobo.Torobo()
agent.initialize()

objects = ["white_ball", "red_ball", "yellow_ball"]
random_ranges = np.array([
    [0.0, 0.0, 0., 0.41, 0.20, 1.09],
    # [0.0, 0.80, 0., 0.85, -0.10, 1.09],
    # [0.20, 0.60, 0., 0.31, -0.10, 1.09],
    [0.30, 0.80, 0., 0.7, -0.10, 1.09],
    [0.30, 0.80, 0., 0.7, -0.10, 1.09],
])

world = env.Environment(objects=objects, rng_ranges=random_ranges)
episode = 1000
action_dim = 1
obs_dim = 2 * len(objects)
hidden_dim = 128

policy_network = models.MLP_gaussian([
    obs_dim,
    hidden_dim,
    hidden_dim,
    2*action_dim])

value_network = models.ValueNetwork([obs_dim, hidden_dim, hidden_dim, 1])
policy_network.load_state_dict(torch.load("save/ppo_3top_iyigiden/policy_net_last.ckpt"))
value_network.load_state_dict(torch.load("save/ppo_3top_iyigiden/value_net_last.ckpt"))

print("="*10+"POLICY NETWORK"+"="*10)
print(policy_network)
print("="*10+"VALUE NETWORK"+"="*10)
print(value_network)
print("Testing starts...")
reward_history = []
for epi in range(episode):
    world.random()
    obs = torch.tensor(world.get_state(), dtype=torch.float)
    out = policy_network(obs)
    action = out[..., :out.shape[-1]//2].detach()

    if torch.isnan(action):
        print("got action nan.")
        exit()
    start_time = rospy.get_time()

    is_success = act(world, agent, action, 0.08)
    agent.init_pose()
    if not is_success:
        print("white ball:", world.get_object_position("white_ball"))
        print("Action failed.")
        continue

    stationary = world.is_stationary()
    end_time = rospy.get_time()
    while not stationary and (end_time - start_time) < 20.0:
        rospy.sleep(0.02)
        stationary = world.is_stationary()
        end_time = rospy.get_time()
    end_time = rospy.get_time()
    obs = torch.tensor(world.get_state(), dtype=torch.float)
    reward = world.get_reward()
    reward_history.append(reward)

agent.zero_pose()
plt.hist(reward_history)
pp = PdfPages("save/test_reward.pdf")
pp.savefig()
pp.close()
