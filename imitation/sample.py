"""Synthetic matches in the Riot match-v1 shape, for trying the pipeline/web UI without an API key.
Everything here is made up (ids start with SYN_) and carries no real player data.

    python -m imitation.sample --n 40 --out imitation_data/raw
"""
import argparse
import json
import random
from pathlib import Path

UNITS = [f"SYN_Unit{i}" for i in range(1, 25)]
TRAITS = ["SYN_Brawler", "SYN_Arcane", "SYN_Sniper", "SYN_Shade", "SYN_Bulwark"]
ITEMS = ["SYN_Blade", "SYN_Staff", "SYN_Plate", "SYN_Cloak", "SYN_Orb"]
AUGMENTS = [f"SYN_Aug{i}" for i in range(1, 13)]


def fake_match(rng, n):
    participants = []
    order = list(range(1, 9))
    rng.shuffle(order)
    for p, placement in enumerate(order):
        comp_trait = TRAITS[(n + p) % len(TRAITS)]
        strength = 9 - placement  # better placement -> more units of the "meta" trait and more 2-3 stars
        units = []
        for k in range(rng.randint(6, 9)):
            units.append({"character_id": rng.choice(UNITS), "tier": 1 + (rng.random() < strength / 12) + (rng.random() < strength / 40),
                          "itemNames": rng.sample(ITEMS, rng.randint(0, 3))})
        participants.append({
            "puuid": f"syn-player-{n}-{p}", "placement": placement, "level": min(10, 6 + strength // 3),
            "augments": rng.sample(AUGMENTS, 3), "units": units,
            "traits": [{"name": comp_trait, "num_units": 2 + strength // 3, "tier_current": 1 + (strength > 5)},
                       {"name": rng.choice(TRAITS), "num_units": 2, "tier_current": 1}],
        })
    return {"metadata": {"match_id": f"SYN_{n:05d}"}, "info": {"tft_set_number": 0, "queue_id": 1100, "participants": participants}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="imitation_data/raw")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for n in range(args.n):
        (out / f"SYN_{n:05d}.json").write_text(json.dumps(fake_match(rng, n)))
    print(f"wrote {args.n} synthetic matches to {out}")


if __name__ == "__main__":
    main()
