import torch
import numpy as np


class MLP_gaussian(torch.nn.Module):
    def __init__(self, layers, activation=torch.nn.ReLU()):
        super(MLP_gaussian, self).__init__()
        model = []
        in_dim = layers[0]
        for l in layers[1:-1]:
            model.append(torch.nn.Linear(in_features=in_dim, out_features=l))
            model.append(activation)
            in_dim = l
        model.append(torch.nn.Linear(
            in_features=in_dim,
            out_features=layers[-1]))
        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        dim = out.shape[-1]
        out[..., :dim//2] = torch.tanh(out[..., :dim//2]) * 5
        out[..., dim//2:] = torch.exp(out[..., dim//2:])
        return out

    def action(self, x):
        m = self.dist(x)
        action = m.sample()
        return action, m.log_prob(action)

    def logprob(self, s, a):
        m = self.dist(s)
        return m.log_prob(a)

    def dist(self, x):
        out = self.forward(x)
        dims = out.shape[-1]
        mu = out[..., :dims//2]
        std = out[..., dims//2:]
        m = torch.distributions.normal.Normal(mu, std)
        return m


class MLP_categorical(torch.nn.Module):
    def __init__(self, layers, activation=torch.nn.ReLU()):
        super(MLP_categorical, self).__init__()
        model = []
        in_dim = layers[0]
        for l in layers[1:-1]:
            model.append(torch.nn.Linear(in_features=in_dim, out_features=l))
            model.append(activation)
            in_dim = l
        model.append(torch.nn.Linear(
            in_features=in_dim,
            out_features=layers[-1]))
        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

    def action(self, x):
        out = self.forward(x)
        m = torch.distributions.multinomial.Categorical(logits=out)
        action = m.sample()
        return action, m.log_prob(action)


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, layers, activation=torch.nn.ReLU()):
        super(QNetwork, self).__init__()
        self.state_proj = torch.nn.Linear(in_features=state_dim, out_features=layers[0]//2)
        self.action_proj = torch.nn.Linear(in_features=action_dim, out_features=layers[0]//2)

        model = []
        in_dim = layers[0]
        for l in layers[1:-1]:
            model.append(torch.nn.Linear(in_features=in_dim, out_features=l))
            model.append(activation)
            in_dim = l
        model.append(torch.nn.Linear(in_features=layers[-2], out_features=layers[-1]))
        self.model = torch.nn.Sequential(*model)

    def forward(self, state, action):
        s = self.state_proj(state)
        a = self.action_proj(action)
        fused = torch.cat([s, a], dim=-1)
        return self.model(fused)


class ValueNetwork(torch.nn.Module):
    def __init__(self, layers, activation=torch.nn.ReLU()):
        super(ValueNetwork, self).__init__()
        model = []
        in_dim = layers[0]
        for l in layers[1:-1]:
            model.append(torch.nn.Linear(in_features=in_dim, out_features=l))
            model.append(activation)
            in_dim = l
        model.append(torch.nn.Linear(in_features=in_dim, out_features=layers[-1]))
        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Memory:
    def __init__(self, buffer_length):
        self.states = []
        self.actions = []
        self.rewards = []
        self.size = 0
        self.buffer_length = buffer_length

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        self.size = 0

    def append(self, state, action, reward):
        if self.size == self.buffer_length:
            # self.states = self.states[self.buffer_length//10:]
            # self.actions = self.actions[self.buffer_length//10:]
            # self.rewards = self.rewards[self.buffer_length//10:]
            # self.logprobs = self.logprobs[self.buffer_length//10:]
            # self.values = self.values[self.buffer_length//10:]
            # self.size -= self.buffer_length // 10
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.rewards = self.rewards[1:]
            self.logprobs = self.logprobs[1:]
            self.values = self.values[1:]
            self.size -= 1
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.size += 1

    def peek_n(self, n, from_start=False):
        if from_start:
            idx = list(range(self.size-n, self.size))
        else:
            idx = list(range(n))
        return self.get_by_idx(idx)

    def sample_n(self, n):
        r = np.random.permutation(self.size)
        idx = r[:n]
        return self.get_by_idx(idx)

    def get_by_idx(self, idx):
        s = torch.stack([self.states[i] for i in idx])
        a = torch.stack([self.actions[i] for i in idx])
        r = torch.stack([self.rewards[i] for i in idx])
        return (s, a, r)

    def get_all(self):
        idx = list(range(self.size))
        return self.get_by_idx(idx)
