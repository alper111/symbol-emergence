import torch
import math
import os


class PPOAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, dist, num_layers, device, lr, K, batch_size,
                 eps, c_clip, c_v, c_ent):
        self.device = device
        self.K = K
        self.batch_size = batch_size
        self.dst = dist
        self.memory = Memory(keys=["state", "action", "reward", "logprob", "value"], buffer_length=-1)
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

        self.policy = MLP(layer_info=policy_layer)
        self.value = MLP(layer_info=value_layer)
        self.policy.to(device)
        self.value.to(device)
        self.optimizer = torch.optim.Adam(
            lr=lr,
            params=[{"params": self.policy.parameters()}, {"params": self.value.parameters()}],
            amsgrad=True)
        self.criterion = torch.nn.MSELoss()

    def dist(self, x):
        out = self.policy(x)
        if self.dst == "categorical":
            m = torch.distributions.multinomial.Categorical(logits=out)
        else:
            dim = out.shape[-1]
            mu = torch.tanh(out[..., :dim//2])
            logstd = out[..., dim//2:]
            std = 0.2 + torch.nn.functional.softplus(logstd)
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

    def action(self, x, std=True):
        m = self.dist(x)
        if std:
            action = m.sample()
        else:
            with torch.no_grad():
                out = self.policy(x)
                action = torch.tanh(out[..., :out.shape[-1]//2])
        logprob = m.log_prob(action)
        if self.multivariate:
            logprob = logprob.sum(dim=-1)
        return action, logprob

    def record(self, state, action, logprob, reward, value):
        dic = {"state": state, "action": action, "reward": reward, "logprob": logprob, "value": value}
        self.memory.append(dic)

    def reset_memory(self):
        self.memory.clear()

    def loss(self, dic):
        state = dic["state"]
        action = dic["action"]
        logprob = dic["logprob"]
        reward = dic["reward"]
        value = dic["value"]

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
        return loss

    def update(self):
        avg_loss = 0.0
        for i in range(self.K):
            if self.batch_size == -1:
                res = self.memory.get_all()
            else:
                res = self.memory.sample_n(self.batch_size)

            loss = self.loss(res)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            avg_loss += loss.item()
        return avg_loss

    def save(self, path, ext=None):
        if not os.path.exists(path):
            os.makedirs(path)
        pname = "policy"
        vname = "value"
        if ext:
            pname = pname + ext + ".ckpt"
            vname = vname + ext + ".ckpt"
        pname = os.path.join(path, pname)
        vname = os.path.join(path, vname)
        torch.save(self.policy.eval().cpu().state_dict(), pname)
        torch.save(self.value.eval().cpu().state_dict(), vname)
        self.policy.train().to(self.device)
        self.value.train().to(self.device)

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


class PGAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, dist, num_layers, device, lr, batch_size):
        self.device = device
        self.batch_size = batch_size
        self.dst = dist
        self.memory = Memory(keys=["logprob", "reward"], buffer_length=-1)
        if self.dst == "gaussian" and action_dim > 1:
            self.multivariate = True
        else:
            self.multivariate = False

        policy_layer = [state_dim] + [hidden_dim] * num_layers + [action_dim]

        self.policy = MLP(layer_info=policy_layer)
        self.policy.to(device)
        self.optimizer = torch.optim.Adam(lr=lr, params=self.policy.parameters(), amsgrad=True)

    def dist(self, x):
        out = self.policy(x)
        if self.dst == "categorical":
            m = torch.distributions.multinomial.Categorical(logits=out)
        else:
            dim = out.shape[-1]
            mu = torch.tanh(out[..., :dim//2])
            logstd = out[..., dim//2:]
            std = 0.1 + 0.9 * torch.nn.functional.softplus(logstd)
            m = torch.distributions.normal.Normal(mu, std)
        return m

    def logprob(self, s, a):
        m = self.dist(s)
        logprob = m.log_prob(a)
        if self.multivariate:
            logprob = logprob.sum(dim=-1)
        return logprob

    def action(self, x, std=True):
        m = self.dist(x)
        if std:
            action = m.sample()
        else:
            with torch.no_grad():
                out = self.policy(x)
                action = torch.tanh(out[..., :out.shape[-1]//2])
        logprob = m.log_prob(action)
        if self.multivariate:
            logprob = logprob.sum(dim=-1)
        return action, logprob

    def record(self, logprob, reward):
        dic = {"logprob": logprob, "reward": reward}
        self.memory.append(dic)

    def reset_memory(self):
        self.memory.clear()

    def loss(self, dic):
        logprob = dic["logprob"]
        reward = dic["reward"]
        reward = (reward - reward.mean()) / (reward.std() + 1e-5)
        loss = -reward * logprob
        return loss.mean()

    def update(self):
        avg_loss = 0.0
        if self.batch_size == -1:
            res = self.memory.get_all()
        else:
            res = self.memory.sample_n(self.batch_size)

        loss = self.loss(res)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        avg_loss += loss.item()
        return avg_loss

    def save(self, path, ext=None):
        if not os.path.exists(path):
            os.makedirs(path)
        pname = "policy"
        if ext:
            pname = pname + ext + ".ckpt"
        pname = os.path.join(path, pname)
        torch.save(self.policy.eval().cpu().state_dict(), pname)
        self.policy.train().to(self.device)

    def load(self, path, ext=None):
        pname = "policy"
        if ext:
            pname = pname + ext + ".ckpt"
        pname = os.path.join(path, pname)
        self.policy.load_state_dict(torch.load(pname))


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, num_layers, memory_size, lr, batch_size,
                 eps, gamma, q_update, device):
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.memory = Memory(keys=["state", "action", "reward", "next_state", "terminal"], buffer_length=memory_size)
        self.eps = eps
        self.gamma = gamma
        self.num_updates = 0
        self.q_update = q_update

        layer = [state_dim] + [hidden_dim] * num_layers + [action_dim]
        self.Q = MLP(layer_info=layer)
        self.Q_target = MLP(layer_info=layer)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q.to(device)
        self.Q_target.to(device)

        for p in self.Q_target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(lr=lr, params=self.Q.parameters(), amsgrad=True)
        self.criterion = torch.nn.MSELoss()

    def action(self, x, epsgreedy=False):
        q_values = self.Q(x)
        if epsgreedy:
            r = torch.rand(1)
            if r < self.eps:
                action = torch.randint(0, self.action_dim, (), device=x.device)
            else:
                action = q_values.argmax(dim=-1)
        else:
            action = q_values.argmax(dim=-1)
        return action

    def loss(self, dic):
        state = dic["state"]
        action = dic["action"]
        reward = dic["reward"]
        next_state = dic["next_state"]
        terminal = dic["terminal"].squeeze()

        pred = self.Q(state)[torch.arange(action.shape[0]), action]
        nextval, _ = self.Q_target(next_state).max(dim=-1)
        target = torch.zeros(pred.shape[0], device=pred.device)
        term_idx = (terminal == True)
        target[term_idx] = reward[term_idx].squeeze()
        target[~term_idx] = self.gamma * nextval[~term_idx] + reward[~term_idx].squeeze()
        # target = self.gamma * nextval + reward.squeeze()
        return self.criterion(pred, target)

    def record(self, state, action, reward, next_state, terminal):
        dic = {"state": state, "action": action, "reward": reward, "next_state": next_state, "terminal": terminal}
        self.memory.append(dic)

    def reset_memory(self):
        self.memory.clear()

    def update(self):
        dic = self.memory.sample_n(self.batch_size)
        loss = self.loss(dic)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.num_updates += 1
        if (self.num_updates % self.q_update) == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
        return loss.item()

    def decay_epsilon(self, rate, final):
        self.eps = max(self.eps*rate, final)

    def save(self, path, ext=None):
        name = "dqn"
        if ext:
            name = name + ext + ".ckpt"
        name = os.path.join(path, name)
        torch.save(self.Q.eval().cpu().state_dict(), name)

    def load(self, path, ext=None):
        name = "dqn"
        if ext:
            name = name + ext + ".ckpt"
        name = os.path.join(path, name)
        self.Q.load_state_dict(torch.load(name))


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
    def __init__(self, keys, buffer_length=-1):
        self.buffer = {}
        self.keys = keys
        for key in keys:
            self.buffer[key] = []
        self.buffer_length = buffer_length
        self.size = 0

    def clear(self):
        for key in self.keys:
            del self.buffer[key][:]
        self.size = 0

    def append(self, dic):
        if self.buffer_length != -1 and self.size == self.buffer_length:
            for key in self.keys:
                self.buffer[key] = self.buffer[key][1:]
            self.size -= 1
        for key in self.keys:
            self.buffer[key].append(dic[key])
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
        res = {}
        for key in self.keys:
            res[key] = torch.stack([self.buffer[key][i] for i in idx])
        return res

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
    """ multi-layer perceptron with batch norm option """
    def __init__(self, layer_info, activation=torch.nn.ReLU(), std=None, batch_norm=False, indrop=None, hiddrop=None):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_info[0]
        for i, unit in enumerate(layer_info[1:-1]):
            if i == 0 and indrop:
                layers.append(torch.nn.Dropout(indrop))
            elif i > 0 and hiddrop:
                layers.append(torch.nn.Dropout(hiddrop))
            layers.append(Linear(in_features=in_dim, out_features=unit, std=std, batch_norm=batch_norm, gain=2))
            layers.append(activation)
            in_dim = unit
        layers.append(Linear(in_features=in_dim, out_features=layer_info[-1], batch_norm=False))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Linear(torch.nn.Module):
    """ linear layer with optional batch normalization. """
    def __init__(self, in_features, out_features, std=None, batch_norm=False, gain=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(num_features=self.out_features)

        if std is not None:
            self.weight.data.normal_(0., std)
            self.bias.data.normal_(0., std)
        else:
            # defaults to linear activation
            if gain is None:
                gain = 1
            stdv = math.sqrt(gain / self.weight.size(1))
            self.weight.data.normal_(0., stdv)
            self.bias.data.zero_()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        if hasattr(self, "batch_norm"):
            x = self.batch_norm(x)
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}".format(self.in_features, self.out_features)
