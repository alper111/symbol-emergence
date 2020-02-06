import rospy
import torch
import torobo
import models
import env
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


rospy.init_node("training_node", anonymous=True)
rospy.sleep(1.0)
agent = torobo.Torobo()
agent.initialize()
memory = models.Memory(100)

objects = ["white_ball", "red_ball", "yellow_ball"]
random_ranges = np.array([
    [0.0, 0.0, 0., 0.41, 0.20, 1.09],
    # [0.0, 0.80, 0., 0.85, -0.10, 1.09],
    # [0.20, 0.60, 0., 0.31, -0.10, 1.09],
    [0.30, 0.80, 0., 0.7, -0.10, 1.09],
    [0.30, 0.80, 0., 0.7, -0.10, 1.09],
])

world = env.Environment(objects=objects, rng_ranges=random_ranges)
episode = 10000
action_dim = 1
obs_dim = 2 * len(objects)
hidden_dim = 128

policy_network = models.SoftTree(
    in_features=obs_dim,
    out_features=2*action_dim,
    depth=3)

value_network = models.ValueNetwork([obs_dim, hidden_dim, hidden_dim, 1])

optim_policy = torch.optim.Adam(
    lr=0.0003,
    params=policy_network.parameters(),
    amsgrad=True)

optim_value = torch.optim.Adam(
    lr=0.001,
    params=value_network.parameters(),
    amsgrad=True
)
criterion = torch.nn.MSELoss(reduction="sum")

print("="*10+"POLICY NETWORK"+"="*10)
print(policy_network)
print("="*10+"VALUE NETWORK"+"="*10)
print(value_network)
print("Training starts...")
reward_history = []
for epi in range(episode):
    logprobs = []
    rewards = []
    actions = []
    observations = []
    world.random()
    rospy.sleep(0.5)
    world.load_prev_state()
    obs = torch.tensor([world.get_state()], dtype=torch.float)
    action, logprob = policy_network.action(obs)
    curr_max = -1e10
    best_action = None
    best_reward = None
    for action_leaf, logprob_leaf in zip(action.squeeze(1), logprob.squeeze(1)):
        if torch.isnan(action_leaf):
            print("got action nan.")
            exit()

        start_time = rospy.get_time()
        is_success = act(world, agent, action_leaf, 0.08)
        agent.init_pose()
        if not is_success:
            print("white ball:", world.get_object_position("white_ball"))
            print("Action failed.")
            continue

        actions.append(action_leaf)
        observations.append(obs[0])
        stationary = world.is_stationary()
        end_time = rospy.get_time()
        # rospy.loginfo("waiting world to be stationary")
        while not stationary and (end_time - start_time) < 20.0:
            rospy.sleep(0.02)
            stationary = world.is_stationary()
            end_time = rospy.get_time()
        end_time = rospy.get_time()
        reward = world.get_reward() - 0.1 * (end_time - start_time)

        # bookkeeping
        rewards.append(reward)
        logprobs.append(logprob_leaf.sum())
        world.load_prev_state()
        rospy.sleep(0.2)

        if logprob_leaf.sum() > curr_max:
            curr_max = logprob_leaf
            best_action = action_leaf
            best_reward = reward

    if len(rewards) < 1:
        print("Episode failed. No valid rollout.")
        continue

    # rospy.loginfo("optimization steps")
    # reshaping
    rewards = torch.tensor(rewards, dtype=torch.float)
    cumrew = rewards.sum().item()
    logprobs = torch.stack(logprobs)
    actions = torch.stack(actions)
    observations = torch.stack(observations)
    reward_history.append(cumrew)

    value_estimate = value_network(observations).reshape(-1)
    if (torch.isnan(value_estimate)).any():
        print("got estimate nan.")
        exit()
    advantage = (rewards - value_estimate).detach()

    # add to memory
    with torch.no_grad():
        for i in range(observations.shape[0]):
            memory.append(
                observations[i],
                actions[i],
                rewards[i],
                logprobs[i].clone(),
                value_estimate[i].clone()
            )

    # rospy.loginfo("optimizing policy net")
    for k in range(10):
        # optimize the policy network
        optim_policy.zero_grad()
        old_s, old_a, old_r, old_logp, old_v = memory.sample_n(policy_network.leaf_count)
        logp = policy_network.logprob(old_s, old_a.unsqueeze(0)).sum(dim=-1)
        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * (old_r - old_v)
        surr2 = ratio.clamp(0.8, 1.2) * (old_r - old_v)
        policy_loss = - torch.min(surr1, surr2).mean()
        # policy_loss = -(logprobs * advantage).sum()
        policy_loss.backward()
        optim_policy.step()

    # optimize the value network
    optim_value.zero_grad()
    value_loss = criterion(value_estimate, rewards)
    value_loss.backward()
    optim_value.step()
    print("Episode: %d, reward: %.3f, adv: %.3f, p loss: %.3f, v loss: %.3f, mem: %d, best: %.3f"
          % (epi+1, cumrew, advantage.sum().item(), policy_loss.item(), value_loss.item(), memory.size, best_reward))

    # last checkpoint for debugging
    with torch.no_grad():
        torch.save(policy_network.state_dict(), "save/policy_net_last.ckpt")
        torch.save(value_network.state_dict(), "save/value_net_last.ckpt")

    # save models
    if (epi+1) % 50 == 0:
        with torch.no_grad():
            torch.save(policy_network.state_dict(), "save/policy_net_{0}.ckpt".format(epi+1))
            torch.save(value_network.state_dict(), "save/value_net_{0}.ckpt".format(epi+1))
    np.save("save/rewards.npy", reward_history)

agent.zero_pose()
plt.plot(reward_history)
pp = PdfPages("save/reward.pdf")
pp.savefig()
pp.close()
