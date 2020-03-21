import torch
import math
import os


class PPOAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, dist, num_layers,  device, lr, K, batch_size,
                 eps, c_clip, c_v, c_ent):
        self.K = K
        self.batch_size = batch_size
        self.dst = dist
        self.memory = Memory(buffer_length=-1)
        self.eps = eps
        self.c_clip = c_clip
        self.c_v = c_v
        self.c_ent = c_ent
        if self.dst == "gaussian" and action_dim > 1:
            self.multivariate = True
        else:
            self.multivariate = False

        policy_layer = [state_dim] + [hidden_dim] * num_layers + [action_dim]
        value_layer = [state_dim] + [hidden_dim] * num_layers + [1]

        self.policy = MLP(layer_info=policy_layer, activation=torch.nn.ReLU(), std=None, normalization=None)
        self.value = MLP(layer_info=value_layer, activation=torch.nn.ReLU(), std=None, normalization=None)
        self.policy.to(device)
        self.value.to(device)
        self.optimizer = torch.optim.Adam(lr=lr, params=[{"params": self.policy.parameters()},
                                                         {"params": self.value.parameters()}], amsgrad=True)
        self.criterion = torch.nn.MSELoss()

    def dist(self, x):
        out = self.policy(x)
        if self.dst == "categorical":
            m = torch.distributions.multinomial.Categorical(logits=out)
        else:
            dim = out.shape[-1]
            mu = torch.tanh(out[..., :dim//2])
            logstd = out[..., dim//2:]
            std = torch.exp(logstd).clamp(1e-5, 2.0)
            m = torch.distributions.normal.Normal(mu, std)
        return m

    def logprob(self, s, a):
        m = self.dist(s)
        logprob = m.log_prob(a)
        entropy = m.entropy()
        if self.multivariate:
            logprob = logprob.sum(dim=-1)
            entropy = entropy.sum(dim=-1)
        return logprob, entropy

    def action(self, x):
        m = self.dist(x)
        action = m.sample()
        logprob = m.log_prob(action)
        if self.multivariate:
            logprob = logprob.sum(dim=-1)
        return action, logprob

    def record(self, state, action, logprob, reward, value):
        self.memory.append(state, action, logprob, reward, value)

    def reset_memory(self):
        self.memory.clear()

    def update(self):
        ploss = 0.0
        vloss = 0.0
        eloss = 0.0
        for i in range(self.K):
            if self.batch_size == -1:
                state, action, logprob, reward, value = self.memory.get_all()
            else:
                state, action, logprob, reward, value = self.memory.sample_n(self.batch_size)

            reward = (reward - reward.mean()) / (reward.std() + 1e-5)
            v_bar = self.value(state).reshape(-1)
            adv = reward - value
            new_logp, entropy = self.logprob(state, action)
            ratio = torch.exp(new_logp - logprob)
            surr1 = ratio * adv
            surr2 = ratio.clamp(1.0 - self.eps, 1.0 + self.eps) * adv
            policy_loss = - self.c_clip * torch.min(surr1, surr2)
            value_loss = self.c_v * self.criterion(v_bar, reward)
            entropy_loss = - self.c_ent * entropy
            loss = (policy_loss + value_loss + entropy_loss).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            ploss += policy_loss.data.mean()
            vloss += value_loss.data.mean()
            eloss += entropy_loss.data.mean()
        return ploss/self.K, vloss/self.K, eloss/self.K

    def save(self, path, ext=None):
        pname = "policy"
        vname = "value"
        if ext:
            pname = pname + ext + ".ckpt"
            vname = vname + ext + ".ckpt"
        pname = os.path.join(path, pname)
        vname = os.path.join(path, vname)
        torch.save(self.policy.eval().cpu().state_dict(), pname)
        torch.save(self.value.eval().cpu().state_dict(), vname)

    def load(self, path, ext=None):
        pname = "policy"
        vname = "value"
        if ext:
            pname = pname + ext + ".ckpt"
            vname = vname + ext + ".ckpt"
        pname = os.path.join(path, pname)
        vname = os.path.join(path, vname)
        self.policy.load_state_dict(torch.load(pname))
        self.value.load_state_dict(torch.load(vname))


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
        return self.model(x)

    def action(self, x):
        m = self.dist(x)
        action = m.sample()
        # burayı düzelt
        logprob = m.log_prob(action).sum(dim=-1)
        return action.cpu().numpy(), logprob

    def logprob(self, s, a):
        m = self.dist(s)
        logprob = m.log_prob(a)
        entropy = m.entropy()
        if len(logprob.shape) > 1:
            logprob = logprob.sum(dim=-1)
        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim=-1)
        return logprob, entropy

    def dist(self, x):
        out = self.forward(x)
        dims = out.shape[-1]
        mu = torch.tanh(out[..., :dims//2]) * 5
        std = torch.exp(out[..., dims//2:]).clamp(1e-5, 2.0)
        m = torch.distributions.normal.Normal(mu, std)
        return m

    def entropy(self, x):
        m = self.dist(x)
        entropy = m.entropy()
        if len(entropy.shape) > 1:
            entropy = entropy.sum(dim=-1)
        return entropy


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
        m = self.dist(x)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def logprob(self, s, a):
        m = self.dist(s)
        return m.log_prob(a), m.entropy()

    def dist(self, x):
        out = self.forward(x)
        m = torch.distributions.multinomial.Categorical(logits=out)
        return m

    def entropy(self, x):
        m = self.dist(x)
        return m.entropy()


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
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.size = 0
        self.buffer_length = buffer_length

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        self.size = 0

    def append(self, state, action, logprob, reward, value):
        if self.buffer_length != -1 and self.size == self.buffer_length:
            self.states = self.states[1:]
            self.actions = self.actions[1:]
            self.logprobs = self.logprobs[1:]
            self.rewards = self.rewards[1:]
            self.values = self.values[1:]
            self.size -= 1
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.size += 1

    def peek_n(self, n, from_start=False):
        if from_start:
            idx = list(range(self.size-n, self.size))
        else:
            idx = list(range(n))
        return self.get_by_idx(idx)

    def sample_n(self, n):
        r = torch.randperm(self.size)
        idx = r[:n]
        return self.get_by_idx(idx)

    def get_by_idx(self, idx):
        s = torch.stack([self.states[i] for i in idx])
        a = torch.stack([self.actions[i] for i in idx])
        lp = torch.stack([self.logprobs[i] for i in idx])
        r = torch.stack([self.rewards[i] for i in idx])
        v = torch.stack([self.values[i] for i in idx])
        return (s, a, lp, r, v)

    def get_all(self):
        idx = list(range(self.size))
        return self.get_by_idx(idx)


class Flatten(torch.nn.Module):
    def __init__(self, dims):
        super(Flatten, self).__init__()
        self.dims = dims

    def forward(self, x):
        dim = 1
        for d in self.dims:
            dim *= x.shape[d]
        return x.reshape(-1, dim)


class MLP(torch.nn.Module):
    '''multi-layer perceptron with batch norm option'''
    def __init__(self, layer_info, activation, std=None, normalization=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_info[0]
        for l in layer_info[1:-1]:
            layers.append(Linear(in_features=in_dim, out_features=l, std=std, normalization=normalization))
            layers.append(activation)
            in_dim = l
        layers.append(Linear(in_features=in_dim, out_features=layer_info[-1], std=std, normalization=None))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, y=None):
        return self.layers(x)


class Linear(torch.nn.Module):
    '''linear layer with optional batch normalization or layer normalization'''
    def __init__(self, in_features, out_features, std=None, normalization=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.normalization = normalization
        if normalization == 'batch_norm':
            self.normalization_func = torch.nn.BatchNorm1d(num_features=self.out_features)
        elif normalization == 'layer_norm':
            self.normalization_func = torch.nn.LayerNorm(normalized_shape=self.out_features)

        if std is not None:
            self.weight.data.normal_(0., std)
            self.bias.data.normal_(0., std)
        else:
            # he initialization for ReLU activaiton
            stdv = math.sqrt(2 / self.weight.size(1))
            self.weight.data.normal_(0., stdv)
            self.bias.data.zero_()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.normalization:
            x = self.normalization_func(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, normalization={}'.format(
            self.in_features, self.out_features, self.normalization
        )
