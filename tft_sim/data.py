"""Game data and rules. Everything season-specific is JSON in a "set dir" (default: tft_sim/data):

  champions.json  traits.json  [items.json]  [augments.json]  [rules.json]

Missing optional files fall back to the defaults in tft_sim/data. Structural constants that hold for every
season (board/bench size, shop size, player count) stay here as module constants.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import augments as augments_mod
from .items import ItemBook

DATA_DIR = Path(__file__).parent / "data"

NUM_PLAYERS = 8
START_HP = 100
BENCH_SIZE = 9
BOARD_SLOTS = 28  # 4 rows x 7 cols; rows 0-1 front, rows 2-3 back
BOARD_COLS = 7
SHOP_SIZE = 5
MAX_ITEMS_PER_UNIT = 3


@dataclass
class Rules:
    pool_size: dict
    shop_odds: dict
    xp_to_level: dict
    max_level: int = 9
    reroll_cost: int = 2
    xp_cost: int = 4
    xp_per_buy: int = 4
    free_xp_per_round: int = 2
    interest_per: int = 10
    interest_cap: int = 5
    streak_bonus_table: list = field(default_factory=lambda: [[5, 3], [4, 2], [2, 1]])
    stage_damage_table: dict = field(default_factory=dict)
    stage_damage_default: int = 15
    star_mult: float = 1.8
    augment_rounds: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d):
        ints = lambda m: {int(k): v for k, v in m.items()}
        return cls(
            pool_size=ints(d["pool_size"]), shop_odds={int(k): tuple(v) for k, v in d["shop_odds"].items()},
            xp_to_level=ints(d["xp_to_level"]), max_level=d["max_level"], reroll_cost=d["reroll_cost"],
            xp_cost=d["xp_cost"], xp_per_buy=d["xp_per_buy"], free_xp_per_round=d["free_xp_per_round"],
            interest_per=d["interest_per"], interest_cap=d["interest_cap"],
            streak_bonus_table=[list(x) for x in d["streak_bonus"]],
            stage_damage_table=ints(d["stage_damage"]), stage_damage_default=d["stage_damage_default"],
            star_mult=d["star_mult"], augment_rounds={(a, b): t for a, b, t in d["augment_rounds"]})

    def streak_bonus(self, streak):
        for need, bonus in sorted(self.streak_bonus_table, reverse=True):
            if streak >= need:
                return bonus
        return 0

    def stage_damage(self, stage):
        return self.stage_damage_table.get(stage, self.stage_damage_default)

    def odds(self, level):
        return self.shop_odds[min(level, max(self.shop_odds))]


DEFAULT_RULES = Rules.from_dict(json.loads((DATA_DIR / "rules.json").read_text()))

# defaults kept as module-level names for code/tests that don't carry a GameData around
POOL_SIZE, SHOP_ODDS, XP_TO_LEVEL = DEFAULT_RULES.pool_size, DEFAULT_RULES.shop_odds, DEFAULT_RULES.xp_to_level
MAX_LEVEL, REROLL_COST, XP_COST = DEFAULT_RULES.max_level, DEFAULT_RULES.reroll_cost, DEFAULT_RULES.xp_cost
XP_PER_BUY, FREE_XP_PER_ROUND = DEFAULT_RULES.xp_per_buy, DEFAULT_RULES.free_xp_per_round
INTEREST_PER, INTEREST_CAP, STAR_MULT = DEFAULT_RULES.interest_per, DEFAULT_RULES.interest_cap, DEFAULT_RULES.star_mult
streak_bonus, stage_damage = DEFAULT_RULES.streak_bonus, DEFAULT_RULES.stage_damage


def sell_value(cost: int, star: int) -> int:
    copies = 3 ** (star - 1)
    if star > 1 and cost > 1:
        return cost * copies - 1
    return cost * copies


@dataclass(frozen=True)
class Champion:
    name: str
    cost: int
    traits: tuple
    hp: float
    atk: float
    armor: float
    aspd: float
    range: int
    mana: int
    mana_start: int
    skill_type: str
    skill_value: float
    mr: float = 0.0
    crit_chance: float = 0.0
    crit_mult: float = 1.4
    skill_dmg: str = "magic"   # physical | magic | true (damage-type skills)
    skill_extra: float = 0.0    # buff_as: attack-speed fraction; stun: duration in ticks


class GameData:
    def __init__(self, set_dir=None):
        d = Path(set_dir) if set_dir else DATA_DIR
        pick = lambda name: d / name if (d / name).exists() else DATA_DIR / name
        self.set_dir = d
        self.champions = {}
        for c in json.loads(pick("champions.json").read_text()):
            sk = c["skill"]
            self.champions[c["name"]] = Champion(
                c["name"], c["cost"], tuple(c["traits"]), c["hp"], c["atk"], c["armor"],
                c["aspd"], c["range"], c["mana"], c["mana_start"], sk["type"], sk["value"],
                mr=c.get("mr", c["armor"]), crit_chance=c.get("crit_chance", 0.0), crit_mult=c.get("crit_mult", 1.4),
                skill_dmg=sk.get("damage_type", "magic"), skill_extra=sk.get("extra", 0.0))
        raw = json.loads(pick("traits.json").read_text())
        # trait -> sorted list of (breakpoint, effects)
        self.traits = {t: sorted((int(k), v) for k, v in bp.items()) for t, bp in raw.items()}
        self.rules = Rules.from_dict(json.loads(pick("rules.json").read_text()))
        self.items = ItemBook.load(pick("items.json"))
        self.augments = augments_mod.load(pick("augments.json"))
        self.by_cost = {}
        for c in self.champions.values():
            self.by_cost.setdefault(c.cost, []).append(c.name)
        missing = [c for c in self.by_cost if c not in self.rules.pool_size]
        if missing:
            raise ValueError(f"rules.json pool_size has no entry for costs {missing}")

    def pool_size(self, name):
        return self.rules.pool_size[self.champions[name].cost]

    def active_trait_effects(self, names):
        """names: iterable of champion names on board -> (list of effect dicts, trait counts); unique champs only."""
        counts = {}
        for n in set(names):
            for t in self.champions[n].traits:
                counts[t] = counts.get(t, 0) + 1
        effects = []
        for t, n in counts.items():
            best = None
            for bp, eff in self.traits.get(t, []):
                if n >= bp:
                    best = eff
            if best:
                effects.append(best)
        return effects, counts
