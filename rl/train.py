"""Train PPO against rule bots.   python -m rl.train --steps 5000000"""
import argparse
import collections
import time

import numpy as np
import torch

from bots.rule_bots import RandomBot
from rl.env import EVAL_LINEUP, N_ACTIONS, TFTEnv
from rl.ppo import ActorCritic, gae, ppo_update
from rl.vec import VecEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs-per-worker", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/ppo.pt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--set-dir", default=None, help="season data dir (default: built-in placeholder set)")
    ap.add_argument("--opponents", choices=["random", "eval", "mix"], default="mix",
                    help="random: 7 random bots (curriculum start); eval: 3 econ/3 reroll/1 random; mix: sampled")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    device_note = "cpu"  # env/bot simulation dominates; the net is small

    lineup = {"random": [RandomBot] * 7, "eval": EVAL_LINEUP, "mix": None}[args.opponents]
    venv = VecEnv(args.workers, args.envs_per_worker, seed=args.seed * 100_000, lineup=lineup, set_dir=args.set_dir)
    obs_dim = venv.obs.shape[1]
    model = ActorCritic(obs_dim, N_ACTIONS)
    if args.resume:
        model.load_state_dict(torch.load(args.resume)["model"])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    T, E = args.horizon, venv.n
    n_iters = args.steps // (T * E)
    recent = collections.deque(maxlen=300)
    t0, total = time.time(), 0
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for it in range(1, n_iters + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * (1 - (it - 1) / n_iters)
        ob = np.zeros((T, E, obs_dim), np.float32)
        mk = np.zeros((T, E, N_ACTIONS), bool)
        ac = np.zeros((T, E), np.int64)
        lp = np.zeros((T, E), np.float32)
        rw = np.zeros((T, E), np.float32)
        dn = np.zeros((T, E), np.float32)
        vl = np.zeros((T, E), np.float32)
        for t in range(T):
            ob[t], mk[t] = venv.obs, venv.masks
            ac[t], lp[t], vl[t] = model.act(venv.obs, venv.masks)
            _, _, rw[t], d, places = venv.step(ac[t])
            dn[t] = d
            recent.extend(places)
        total += T * E
        with torch.no_grad():
            _, last_v = model(torch.as_tensor(venv.obs), torch.as_tensor(venv.masks))
        adv = np.zeros_like(rw)
        ret = np.zeros_like(rw)
        for e in range(E):
            adv[:, e], ret[:, e] = gae(rw[:, e], vl[:, e], dn[:, e], last_v[e].item(), args.gamma, args.lam)
        flat = lambda x: x.reshape(T * E, *x.shape[2:])
        stats = ppo_update(model, opt, [flat(ob), flat(mk), flat(ac), flat(lp), flat(adv), flat(ret)], ent_coef=args.ent)
        if it % 5 == 0 or it == n_iters:
            sps = total / (time.time() - t0)
            avg = np.mean(recent) if recent else float("nan")
            print(f"it {it}/{n_iters} steps {total} sps {sps:.0f} avg_place(last {len(recent)}) {avg:.2f} "
                  f"pg {stats['pg']:.3f} vl {stats['vl']:.3f} ent {stats['ent']:.2f}", flush=True)
            torch.save({"model": model.state_dict(), "obs_dim": obs_dim, "steps": total}, args.out)
    venv.close()


if __name__ == "__main__":
    main()
