import torch
import models
import utils
import gym
import numpy as np


device = torch.device("cuda:0") if torch.cuda.is_available() else False
env = gym.make("CartPole-v1")
render = False
solved_reward = 200
batch_size = 32
eps = 1.0
lr = 0.0001
hidden_dim = 64
max_timesteps = 200
warmup = 500
obs_dim = utils.get_dim(env.observation_space.shape)
action_dim = env.action_space.n

agent = models.DQN(obs_dim, hidden_dim, action_dim, 2, 10000, lr, batch_size, 1.0, 0.999, 100, device)

if len(env.observation_space.shape) > 1:
    flatten = models.Flatten([-1, -2, -3])

print("="*10+"Q network"+"="*10)
print(agent.Q)
print("Training starts...")
reward_history = []
solved_counter = 0
epi = 0
while solved_counter < 10:
    epi += 1
    it = 0
    cumrew = 0
    avgloss = 0.0
    obs = torch.tensor(env.reset(), dtype=torch.float, device=device)
    if len(env.observation_space.shape) > 1:
        obs = flatten(obs)
    done = False
    for t in range(max_timesteps):
        action = agent.action(obs, epsgreedy=True)
        next_obs, reward, done, info = env.step(action.item())
        next_obs = torch.tensor(next_obs, dtype=torch.float, device=device)
        if len(env.observation_space.shape) > 1:
            next_obs = flatten(next_obs)

        agent.record(obs, action, torch.tensor(reward, dtype=torch.float, device=device), next_obs, torch.tensor(done))
        # warmup
        if agent.memory.size >= warmup:
            loss = agent.update()
        else:
            loss = 0.0
        obs = next_obs
        it += 1
        cumrew += reward
        avgloss += loss
        if done:
            break

    if agent.memory.size >= warmup:
        agent.decay_epsilon(0.999, 0.1)
    reward_history.append(cumrew)
    if cumrew >= solved_reward:
        solved_counter += 1
    else:
        solved_counter = 0

    np.save("save/rewards.npy", reward_history)
    if epi % 50 == 0:
        print("Episode: %d, reward: %d, it: %d, loss: %.3f, epsilon: %.3f, total updates: %d" %
              (epi, cumrew, it, avgloss/it, agent.eps, agent.num_updates))
