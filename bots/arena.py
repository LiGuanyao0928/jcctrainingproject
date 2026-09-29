"""Run many games of rule bots against each other and report balance + speed.

    python -m bots.arena --games 1000 --workers 4
"""
import argparse
import collections
import random
import time
from multiprocessing import Pool

from bots.rule_bots import EconBot, RandomBot, RerollBot
from tft_sim.game import Game

LINEUP = [EconBot, EconBot, EconBot, RerollBot, RerollBot, RerollBot, RandomBot, RandomBot]


def play(seed):
    rng = random.Random(seed)
    classes = LINEUP[:]
    rng.shuffle(classes)
    game = Game(seed=seed)
    agents = [c(seed * 100 + i) for i, c in enumerate(classes)]
    placements = game.run(agents)
    rows = [(classes[i].name, placements[i]) for i in range(len(classes))]
    return rows, game.round_idx, game.stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t = time.time()
    seeds = range(args.seed, args.seed + args.games)
    with Pool(args.workers) as pool:
        results = pool.map(play, seeds, chunksize=8)
    dt = time.time() - t
    place = collections.defaultdict(list)
    wins = collections.Counter()
    for rows, _, _ in results:
        for name, pl in rows:
            place[name].append(pl)
            wins[name] += pl == 1
    print(f"{args.games} games in {dt:.1f}s ({args.games / dt:.1f} games/s, {args.workers} workers)")
    print(f"avg rounds/game {sum(r for _, r, _ in results) / len(results):.1f}; "
          f"final stage histogram {dict(sorted(collections.Counter(s for _, _, s in results).items()))}")
    for name, v in sorted(place.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        top4 = sum(1 for x in v if x <= 4) / len(v)
        print(f"{name:7s} avg place {sum(v) / len(v):.2f}  top4 {top4:.0%}  "
              f"win/seat {wins[name] / len(v):.1%}  (n={len(v)})")


if __name__ == "__main__":
    main()
