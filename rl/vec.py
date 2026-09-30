"""Small multiprocess vector env: each worker owns several TFTEnvs and auto-resets on episode end."""
import multiprocessing as mp

import numpy as np

from rl.env import TFTEnv


def _worker(conn, seeds, lineup, set_dir):
    envs = [TFTEnv(seed=s, lineup=lineup, set_dir=set_dir) for s in seeds]
    obs, masks = zip(*[(lambda o, i: (o, i["action_mask"]))(*e.reset()) for e in envs])
    obs, masks = list(obs), list(masks)
    conn.send((np.stack(obs), np.stack(masks)))
    while True:
        cmd, data = conn.recv()
        if cmd == "close":
            return
        rews, dones, places = [], [], []
        for k, (e, a) in enumerate(zip(envs, data)):
            o, r, d, _, info = e.step(a)
            rews.append(r)
            dones.append(d)
            if d:
                places.append(info["placement"])
                o, i = e.reset()
                info = i
            obs[k], masks[k] = o, info["action_mask"]
        conn.send((np.stack(obs), np.stack(masks), np.array(rews, np.float32), np.array(dones), places))


class VecEnv:
    def __init__(self, n_workers, envs_per_worker, seed=0, lineup=None, set_dir=None):
        ctx = mp.get_context("fork")
        self.conns, self.procs = [], []
        for w in range(n_workers):
            parent, child = ctx.Pipe()
            seeds = [seed + w * 1000 + k * 17 for k in range(envs_per_worker)]
            p = ctx.Process(target=_worker, args=(child, seeds, lineup, set_dir), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)
        self.n = n_workers * envs_per_worker
        self.per = envs_per_worker
        res = [c.recv() for c in self.conns]
        self.obs = np.concatenate([r[0] for r in res])
        self.masks = np.concatenate([r[1] for r in res])

    def step(self, actions):
        for w, c in enumerate(self.conns):
            c.send(("step", actions[w * self.per:(w + 1) * self.per]))
        res = [c.recv() for c in self.conns]
        self.obs = np.concatenate([r[0] for r in res])
        self.masks = np.concatenate([r[1] for r in res])
        rew = np.concatenate([r[2] for r in res])
        done = np.concatenate([r[3] for r in res])
        places = [p for r in res for p in r[4]]
        return self.obs, self.masks, rew, done, places

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
