"""Evaluate a checkpoint (or random policy) against the rule-bot lineup, next to the bots' own results.

    python -m rl.evaluate --ckpt checkpoints/ppo.pt --games 200
"""
import argparse
import collections
import random

import numpy as np
import torch

from bots.rule_bots import EconBot, RandomBot, RerollBot
from rl.env import EVAL_LINEUP, N_ACTIONS, TFTEnv
from rl.ppo import ActorCritic
from tft_sim.data import GameData
from tft_sim.game import Game


def eval_policy(model, games, seed=10_000, greedy=False, set_dir=None):
    rng = np.random.default_rng(seed)
    places = []
    for g in range(games):
        env = TFTEnv(seed=seed + g, lineup=EVAL_LINEUP, set_dir=set_dir)
        o, info = env.reset()
        done = False
        while not done:
            m = info["action_mask"]
            if model is None:
                a = rng.choice(np.flatnonzero(m))
            else:
                a = model.act(o[None], m[None], greedy=greedy)[0][0]
            o, _, done, _, info = env.step(a)
        places.append(info["placement"])
    return places


def bot_baseline(games, seed=10_000, set_dir=None):
    """Seat 0 is an EconBot/RerollBot/RandomBot in turn, opponents as in EVAL_LINEUP."""
    out = {}
    for cls in (EconBot, RerollBot, RandomBot):
        pl = []
        for g in range(games):
            game = Game(seed=seed + g, data=GameData(set_dir))
            agents = [cls(seed + g)] + [c(seed + g * 10 + i) for i, c in enumerate(EVAL_LINEUP)]
            pl.append(game.run(agents)[0])
        out[cls.name] = pl
    return out


def summarize(name, pl):
    pl = np.asarray(pl)
    print(f"{name:14s} avg place {pl.mean():.2f}  top4 {np.mean(pl <= 4):.0%}  win {np.mean(pl == 1):.0%}  (n={len(pl)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ppo.pt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--set-dir", default=None)
    args = ap.parse_args()
    ck = torch.load(args.ckpt)
    model = ActorCritic(ck["obs_dim"], N_ACTIONS)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint trained for {ck.get('steps', '?')} steps; lineup 3 econ / 3 reroll / 1 random")
    summarize("ppo", eval_policy(model, args.games, greedy=args.greedy, set_dir=args.set_dir))
    summarize("uniform-random", eval_policy(None, args.games, set_dir=args.set_dir))
    if args.baselines:
        for n, pl in bot_baseline(args.games, set_dir=args.set_dir).items():
            summarize(f"{n} (seat 0)", pl)


if __name__ == "__main__":
    main()
