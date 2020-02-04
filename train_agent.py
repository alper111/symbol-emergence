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
memory = models.Memory(200)

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
rollout = 8
action_dim = 1
obs_dim = 2 * len(objects)
hidden_dim = 128

policy_network = models.MLP_gaussian([
    obs_dim,
    hidden_dim,
    hidden_dim,
    2*action_dim])

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
criterion = torch.nn.SmoothL1Loss(reduction="sum")

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
    it = 0

    while it < rollout:
        world.random()
        obs = torch.tensor(world.get_state(), dtype=torch.float)
        if (epi+1) % 10 == 0:
            with torch.no_grad():
                print("TESTING WITH MEAN ACTION")
                out = policy_network(obs)
                action = out[..., :out.shape[-1]//2].detach()
                logprob = policy_network.logprob(obs, action)
        else:
            action, logprob = policy_network.action(obs)

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

        actions.append(action)
        observations.append(obs)
        stationary = world.is_stationary()
        end_time = rospy.get_time()

        while not stationary and (end_time - start_time) < 20.0:
            rospy.sleep(0.02)
            stationary = world.is_stationary()
            end_time = rospy.get_time()
        end_time = rospy.get_time()
        obs = torch.tensor(world.get_state(), dtype=torch.float)
        reward = world.get_reward() - 0.1 * (end_time - start_time)

        # bookkeeping
        rewards.append(reward)
        logprobs.append(logprob.sum())
        it += 1

    if len(rewards) < 1:
        print("Episode failed. No valid rollout.")
        continue

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
    if (epi+1) % 10 != 0:
        with torch.no_grad():
            for i in range(observations.shape[0]):
                memory.append(
                    observations[i],
                    actions[i],
                    rewards[i],
                    logprobs[i].clone(),
                    value_estimate[i].clone()
                )

        for k in range(10):
            # optimize the policy network
            optim_policy.zero_grad()
            # if memory.size < 64:
            #     old_s, old_a, old_r, old_logp, old_v = memory.sample_n(rollout)
            # else:
            old_s, old_a, old_r, old_logp, old_v = memory.sample_n(rollout)
            logp = policy_network.logprob(old_s, old_a).sum(dim=-1)
            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * (old_r - old_v)
            surr2 = ratio.clamp(0.8, 1.2) * (old_r - old_v)
            policy_loss = - torch.min(surr1, surr2).mean()
            # policy_loss = -(logprobs * advantage).sum()
            policy_loss.backward()
            optim_policy.step()

        # rospy.loginfo("optimizing value net")
        # optimize the value network
        optim_value.zero_grad()
        value_loss = criterion(value_estimate, rewards)
        value_loss.backward()
        optim_value.step()
        print("Episode: %d, reward: %.3f, adv: %.3f, p loss: %.3f, v loss: %.3f, mem: %d"
            % (epi+1, cumrew, advantage.sum().item(), policy_loss.item(), value_loss.item(), memory.size))

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
    else:
        print("Episode: %d, reward: %.3f, adv: %.3f" % (epi+1, cumrew, advantage.sum().item()))

agent.zero_pose()
plt.plot(reward_history)
pp = PdfPages("save/reward.pdf")
pp.savefig()
pp.close()
