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
        out[..., :dim//2] = torch.tanh(out[..., :dim//2]) * np.pi
        out[..., dim//2:] = 1.5 * torch.sigmoid(out[..., dim//2:])
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


class SoftTree(torch.nn.Module):

    def __init__(self,
                 in_features,
                 out_features,
                 depth,
                 dropout=0.0):

        super(SoftTree, self).__init__()
        self.depth = depth
        self.in_features = in_features
        self.out_features = out_features
        self.leaf_count = int(2**depth)
        self.gate_count = int(self.leaf_count - 1)
        self.gw = torch.nn.Parameter(
            torch.nn.init.kaiming_normal_(
                torch.empty(self.gate_count, in_features),
                nonlinearity="sigmoid").t()
            )
        self.gb = torch.nn.Parameter(torch.zeros(self.gate_count))
        # dropout rate for gating weights.
        self.drop = torch.nn.Dropout(p=dropout)
        self.pw = torch.nn.init.kaiming_normal_(torch.empty(out_features * self.leaf_count, in_features), nonlinearity="linear")
        # [leaf, in, out]
        self.pw = torch.nn.Parameter(self.pw.view(out_features, self.leaf_count, in_features).permute(1, 2, 0))
        # [leaf, 1, out]
        self.pb = torch.nn.Parameter(torch.zeros(self.leaf_count, 1, out_features))

    def forward(self, x):
        leaf_probs = self.leaf_probability(x)
        # input = [1, batch, dim]
        x = x.view(1, x.shape[0], x.shape[1])
        out = torch.matmul(x, self.pw)+self.pb
        dim = out.shape[-1]
        out[..., :dim//2] = torch.tanh(out[..., :dim//2]) * np.pi
        out[..., dim//2:] = torch.sigmoid(out[..., dim//2:])
        result = (out, leaf_probs)
        return result

    def extra_repr(self):
        return "in_features=%d, out_features=%d, depth=%d" % (
            self.in_features,
            self.out_features,
            self.depth)

    def node_densities(self, x):
        gw_ = self.drop(self.gw)
        gatings = torch.sigmoid(torch.add(torch.matmul(x, gw_), self.gb))
        node_densities = torch.ones(x.shape[0], 2**(self.depth+1)-1, device=x.device)
        it = 1
        for d in range(1, self.depth+1):
            for i in range(2**d):
                parent_index = (it+1) // 2 - 1
                child_way = (it+1) % 2
                if child_way == 0:
                    parent_gating = gatings[:, parent_index]
                else:
                    parent_gating = 1 - gatings[:, parent_index]
                parent_density = node_densities[:, parent_index].clone()
                node_densities[:, it] = (parent_density * parent_gating)
                it += 1
        return node_densities

    def gatings(self, x):
        return torch.sigmoid(torch.add(torch.matmul(x, self.gw), self.gb))

    def leaf_probability(self, x):
        return self.node_densities(x)[:, -self.leaf_count:].T

    def total_path_value(self, z, index, level=None):
        gatings = self.gatings(z)
        gateways = np.binary_repr(index, width=self.depth)
        L = 0.
        current = 0
        if level is None:
            level = self.depth

        for i in range(level):
            if int(gateways[i]) == 0:
                L += gatings[:, current].mean()
                current = 2 * current + 1
            else:
                L += (1 - gatings[:, current]).mean()
                current = 2 * current + 2
        return L

    def action(self, x):
        m = self.dist(x)
        action = m.sample()
        return action, m.log_prob(action)

    def logprob(self, s, a):
        m = self.dist(s)
        leaf_probs = self.leaf_probability(s).unsqueeze(-1)
        return m.log_prob(a) + torch.log(leaf_probs)

    def dist(self, x):
        out, leaves = self.forward(x)
        dims = out.shape[-1]
        mu = out[..., :dims//2]
        std = out[..., dims//2:]
        m = torch.distributions.normal.Normal(mu, std)
        return m


class Memory:
    def __init__(self, buffer_length):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.values = []
        self.size = 0
        self.buffer_length = buffer_length

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.logprobs[:]
        del self.values[:]
        self.size = 0

    def append(self, state, action, reward, logprob, value):
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
        self.logprobs.append(logprob)
        self.values.append(value)
        self.size += 1

    def peek_n(self, n):
        idx = list(range(self.size-n, self.size))
        return self.get_by_idx(idx)

    def sample_n(self, n):
        r = np.random.permutation(self.size)
        idx = r[:n]
        return self.get_by_idx(idx)

    def get_by_idx(self, idx):
        s = torch.stack([self.states[i] for i in idx])
        a = torch.stack([self.actions[i] for i in idx])
        r = torch.stack([self.rewards[i] for i in idx])
        logp = torch.stack([self.logprobs[i] for i in idx])
        v = torch.stack([self.values[i] for i in idx])
        return (s, a, r, logp, v)

    def get_all(self):
        idx = list(range(self.size))
        return self.get_by_idx(idx)
