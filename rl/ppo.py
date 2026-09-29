"""PPO with action masking (single discrete action space)."""
import numpy as np
import torch
import torch.nn as nn

NEG = -1e9


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=384):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)

    def forward(self, obs, mask):
        h = self.trunk(obs)
        logits = self.pi(h).masked_fill(~mask, NEG)
        return logits, self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, mask, greedy=False):
        logits, v = self(torch.as_tensor(obs), torch.as_tensor(mask))
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if greedy else dist.sample()
        return a.numpy(), dist.log_prob(a).numpy(), v.numpy()


def gae(rewards, values, dones, last_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


def ppo_update(model, opt, batch, epochs=4, minibatch=1024, clip=0.2, vf_coef=0.5, ent_coef=0.01, max_grad=0.5):
    obs, mask, act, old_logp, adv, ret = [torch.as_tensor(x) for x in batch]
    n = len(obs)
    stats = {}
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, minibatch):
            i = perm[s:s + minibatch]
            logits, v = model(obs[i], mask[i])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act[i])
            ratio = (logp - old_logp[i]).exp()
            a = adv[i]
            a = (a - a.mean()) / (a.std() + 1e-8)
            pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            vl = 0.5 * (v - ret[i]).pow(2).mean()
            ent = dist.entropy().mean()
            loss = pg + vf_coef * vl - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            opt.step()
            stats = {"pg": pg.item(), "vl": vl.item(), "ent": ent.item()}
    return stats
