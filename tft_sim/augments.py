"""Augments: 3-choice picks at the rounds listed in rules.json (default 2-1 silver, 3-2 gold, 4-2 prismatic).

augments.json: {tier: [{id, name, gold?, xp?, free_rerolls?, components?, buff?: {stat: value}}]}
Effects beyond these generic ones need code; see the season import report for what was skipped.
"""
import json
from pathlib import Path

NUM_OFFERS = 3
DEFAULT_PATH = Path(__file__).parent / "data" / "augments.json"


def load(path=DEFAULT_PATH):
    return json.loads(Path(path).read_text())


AUGMENTS = load()


def offer(rng, tier, augments=None):
    pool = (augments or AUGMENTS)[tier]
    return rng.sample(pool, min(NUM_OFFERS, len(pool)))
