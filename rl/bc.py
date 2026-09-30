"""Behaviour-clone the scripted teacher into the policy network.
    python -m rl.bc --episodes 600 --out checkpoints/bc.pt
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from rl.env import N_ACTIONS, TFTEnv
from rl.ppo import NEG, ActorCritic
from rl.teacher import teacher_action


def collect(episodes, seed, eps=0.05, set_dir=None):
    rng = random.Random(seed)
    O, M, A, places = [], [], [], []
    for ep in range(episodes):
        env = TFTEnv(seed=seed + ep * 31, set_dir=set_dir)
        o, info = env.reset()
        done = False
        while not done:
            mask = info["action_mask"]
            a = teacher_action(env, mask)
            O.append(o), M.append(mask), A.append(a)  # label = teacher's choice, even when we then explore
            if rng.random() < eps:
                a = rng.choice(np.flatnonzero(mask))
            o, _, done, _, info = env.step(a)
        places.append(info["placement"])
    return np.stack(O), np.stack(M), np.array(A), places


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set-dir", default=None)
    ap.add_argument("--out", default="checkpoints/bc.pt")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    t = time.time()
    O, M, A, places = collect(args.episodes, args.seed, set_dir=args.set_dir)
    print(f"{len(A)} samples from {args.episodes} teacher games in {time.time() - t:.0f}s "
          f"(teacher avg place under exploration noise {np.mean(places):.2f})", flush=True)
    model = ActorCritic(O.shape[1], N_ACTIONS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    O, M, A = map(torch.as_tensor, (O, M, A))
    n = len(A)
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        tot = acc = 0.0
        for s in range(0, n, 512):
            i = perm[s:s + 512]
            logits, _ = model(O[i], M[i])
            loss = F.cross_entropy(logits, A[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(i)
            acc += (logits.argmax(-1) == A[i]).float().sum().item()
        print(f"epoch {epoch + 1} loss {tot / n:.3f} acc {acc / n:.3f}", flush=True)
    torch.save({"model": model.state_dict(), "obs_dim": O.shape[1], "steps": 0}, args.out)


if __name__ == "__main__":
    main()
