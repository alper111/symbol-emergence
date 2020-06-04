import torch
import models
import utils
import gym
import numpy as np


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")
env = gym.make("FetchPush-v1", reward_type="dense")
render = False
solved_reward = 0
rollout = 100
batch_size = -1
lr = 0.001
hidden_dim = 256
max_timesteps = 50
obs_dim = utils.get_dim(env.observation_space["observation"].shape)
flatten_on = True if len(env.observation_space["observation"].shape) > 1 else False
# discrete action space
if env.action_space.dtype == "int64":
    action_dim = env.action_space.n
    dist = "categorical"
# continuous action space
else:
    action_dim = 2 * env.action_space.shape[0]
    dist = "gaussian"

agent = models.PGAgent(obs_dim, hidden_dim, action_dim, dist, 2, device, lr, batch_size)


if flatten_on:
    flatten = models.Flatten([-1, -2, -3])

print("="*10+"POLICY NETWORK"+"="*10)
print(agent.policy)
print("Training starts...")
reward_history = []
solved_counter = 0
epi = 0
while solved_counter < 10:
    epi += 1
    logprobs = []
    rewards = []
    it = 0
    obs = torch.tensor(env.reset(), dtype=torch.float, device=device)
    if flatten_on:
        obs = flatten(obs)
    done = False
    for t in range(max_timesteps):
        if render:
            env.render()
        action, logprob = agent.action(obs)
        if dist == "gaussian":
            action = action.cpu().numpy()
        else:
            action = action.item()
        obs, reward, done, info = env.step(action)
        obs = torch.tensor(obs, dtype=torch.float, device=device)
        if flatten_on:
            obs = flatten(obs)
        logprobs.append(logprob)
        rewards.append(reward)
        it += 1
        if done:
            break

    # reshaping
    logprobs = torch.stack(logprobs)
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    cumrew = rewards.sum().item()
    rewards = utils.discount(rewards, gamma=0.99)
    reward_history.append(cumrew)

    if cumrew >= solved_reward:
        solved_counter += 1
    else:
        solved_counter = 0

    # add to memory
    for i in range(logprobs.shape[0]):
        agent.record(logprobs[i], rewards[i])
    np.save("save/rewards.npy", reward_history)
    if (epi % rollout) == 0:
        loss = agent.update()
        agent.reset_memory()
        print("Episode: %d, reward: %d, it: %d, loss= %.3f" % (epi, cumrew, it, loss))


obs = torch.tensor(env.reset(), dtype=torch.float, device=device)
if flatten_on:
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
    if flatten_on:
        obs = flatten(obs)
    it += 1
