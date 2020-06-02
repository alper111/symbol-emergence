import torch
import models
import utils
import gym
import numpy as np


device = torch.device("cuda:0") if torch.cuda.is_available() else False
env = gym.make("HalfCheetah-v2")
render = False
solved_reward = 1000
K = 4
batch_size = -1
eps = 0.2
update_iter = 10000
c_clip = 1.0
c_v = 0.5

c_ent = 0.01
lr = 0.001
hidden_dim = 256
max_timesteps = 1000
obs_dim = utils.get_dim(env.observation_space.shape)
# discrete action space
if env.action_space.dtype == "int64":
    action_dim = env.action_space.n
    dist = "categorical"
# continuous action space
else:
    action_dim = 2 * env.action_space.shape[0]
    dist = "gaussian"

agent = models.PPOAgent(obs_dim, hidden_dim, action_dim, dist, 2, device,
                        lr, K, batch_size, eps, c_clip, c_v, c_ent)

if len(env.observation_space.shape) > 1:
    flatten = models.Flatten([-1, -2, -3])

print("="*10+"POLICY NETWORK"+"="*10)
print(agent.policy)
print("="*10+"VALUE NETWORK"+"="*10)
print(agent.value)
print("Training starts...")
reward_history = []
solved_counter = 0
epi = 0
while solved_counter < 10:
    epi += 1
    observations = []
    actions = []
    logprobs = []
    rewards = []
    it = 0
    obs = torch.tensor(env.reset(), dtype=torch.float, device=device)
    if len(env.observation_space.shape) > 1:
        obs = flatten(obs)
    done = False
    for t in range(max_timesteps):
        if render:
            env.render()
        observations.append(obs)
        action, logprob = agent.action(obs)
        if dist == "gaussian":
            action = action.cpu().numpy()
        else:
            action = action.item()
        obs, reward, done, info = env.step(action)
        obs = torch.tensor(obs, dtype=torch.float, device=device)
        if len(env.observation_space.shape) > 1:
            obs = flatten(obs)
        actions.append(torch.tensor(action, device=device))
        logprobs.append(logprob)
        rewards.append(reward)
        it += 1
        if done:
            break

    # reshaping
    observations = torch.stack(observations)
    actions = torch.stack(actions)
    logprobs = torch.stack(logprobs).detach()
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    cumrew = rewards.sum().item()
    rewards = utils.discount(rewards, gamma=0.99)
    reward_history.append(cumrew)
    with torch.no_grad():
        values = agent.value(observations).reshape(-1)

    if cumrew >= solved_reward:
        solved_counter += 1
    else:
        solved_counter = 0

    # add to memory
    for i in range(observations.shape[0]):
        agent.record(observations[i], actions[i], logprobs[i], rewards[i], values[i])
    np.save("save/rewards.npy", reward_history)
    if agent.memory.size >= update_iter:
        loss = agent.update()
        agent.reset_memory()
        print("Episode: %d, reward: %d, it: %d, loss= %.3f" % (epi, cumrew, it, loss))


obs = torch.tensor(env.reset(), dtype=torch.float, device=device)
if len(env.observation_space.shape) > 1:
    obs = flatten(obs)
done = False
while not done:
    env.render()
    action, logprob = agent.action(obs)
    if dist == "gaussian":
        action = action.cpu().numpy()
    else:
        action = action.item()
    obs, reward, done, info = env.step(action)
    obs = torch.tensor(obs, dtype=torch.float, device=device)
    if len(env.observation_space.shape) > 1:
        obs = flatten(obs)
    it += 1
