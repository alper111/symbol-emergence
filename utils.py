import torch


def discount(rewards):
    R = 0.0
    discounted_rewards = []
    for r in reversed(rewards):
        R = r + 0.99 * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards, device=rewards.device)
    return discounted_rewards


def get_dim(dimensions):
    N = 1
    for d in dimensions:
        N = N * d
    return N
