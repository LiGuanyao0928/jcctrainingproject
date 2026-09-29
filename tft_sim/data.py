"""Static game data and rule tables. Champion/trait data lives in replaceable JSON files."""
import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

NUM_PLAYERS = 8
START_HP = 100
MAX_LEVEL = 9
BENCH_SIZE = 9
BOARD_SLOTS = 28  # 4 rows x 7 cols; rows 0-1 front, rows 2-3 back
BOARD_COLS = 7
SHOP_SIZE = 5
REROLL_COST = 2
XP_COST = 4
XP_PER_BUY = 4
MAX_ITEMS_PER_UNIT = 3

# copies of each champion in the shared pool, by cost
POOL_SIZE = {1: 29, 2: 22, 3: 18, 4: 12, 5: 10}

# shop odds (percent) for cost 1..5, indexed by level
SHOP_ODDS = {
    1: (100, 0, 0, 0, 0), 2: (100, 0, 0, 0, 0), 3: (75, 25, 0, 0, 0),
    4: (55, 30, 15, 0, 0), 5: (45, 33, 20, 2, 0), 6: (30, 40, 25, 5, 0),
    7: (19, 30, 40, 10, 1), 8: (18, 25, 36, 18, 3), 9: (10, 20, 25, 35, 10),
}

# xp needed to go from level-1 to level
XP_TO_LEVEL = {2: 2, 3: 2, 4: 6, 5: 10, 6: 20, 7: 36, 8: 48, 9: 80}

INTEREST_PER = 10
INTEREST_CAP = 5
FREE_XP_PER_ROUND = 2


def streak_bonus(streak: int) -> int:
    if streak >= 5:
        return 3
    if streak == 4:
        return 2
    if streak >= 2:
        return 1
    return 0


STAGE_DAMAGE = {1: 0, 2: 2, 3: 3, 4: 5, 5: 8, 6: 10}
STAGE_DAMAGE_DEFAULT = 15
STAR_MULT = 1.8


def stage_damage(stage: int) -> int:
    return STAGE_DAMAGE.get(stage, STAGE_DAMAGE_DEFAULT)


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


class GameData:
    def __init__(self, champions_path=None, traits_path=None):
        champions_path = champions_path or DATA_DIR / "champions.json"
        traits_path = traits_path or DATA_DIR / "traits.json"
        self.champions = {}
        for c in json.loads(Path(champions_path).read_text()):
            self.champions[c["name"]] = Champion(
                c["name"], c["cost"], tuple(c["traits"]), c["hp"], c["atk"], c["armor"],
                c["aspd"], c["range"], c["mana"], c["mana_start"],
                c["skill"]["type"], c["skill"]["value"])
        raw = json.loads(Path(traits_path).read_text())
        # trait -> sorted list of (breakpoint, effects)
        self.traits = {t: sorted((int(k), v) for k, v in bp.items()) for t, bp in raw.items()}
        self.by_cost = {}
        for c in self.champions.values():
            self.by_cost.setdefault(c.cost, []).append(c.name)

    def pool_size(self, name):
        return POOL_SIZE[self.champions[name].cost]

    def active_trait_effects(self, names):
        """names: iterable of champion names on board -> list of effect dicts (unique champs only)."""
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
