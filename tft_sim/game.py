"""Shared-pool auto battler core: pool, shops, economy, merging, items, augments, carousel, 8-player rounds."""
import random
from dataclasses import dataclass, field

from . import augments as aug
from . import carousel as car
from . import combat
from .data import (BENCH_SIZE, BOARD_SLOTS, FREE_XP_PER_ROUND, INTEREST_CAP, INTEREST_PER,
                   MAX_ITEMS_PER_UNIT, MAX_LEVEL, NUM_PLAYERS, REROLL_COST, SHOP_ODDS, SHOP_SIZE,
                   START_HP, XP_COST, XP_PER_BUY, XP_TO_LEVEL, GameData, sell_value, stage_damage,
                   streak_bonus)
from .items import COMPONENTS, combine, is_component

MAX_STAGE = 9
# locations: 0..BENCH_SIZE-1 = bench index, BENCH_SIZE+slot = board slot
def board_loc(slot):
    return BENCH_SIZE + slot


@dataclass
class Unit:
    name: str
    star: int = 1
    items: list = field(default_factory=list)


@dataclass
class Player:
    id: int
    hp: int = START_HP
    gold: int = 0
    level: int = 1
    xp: int = 0
    bench: list = field(default_factory=lambda: [None] * BENCH_SIZE)
    board: dict = field(default_factory=dict)
    shop: list = field(default_factory=lambda: [None] * SHOP_SIZE)
    inventory: list = field(default_factory=list)
    win_streak: int = 0
    lose_streak: int = 0
    alive: bool = True
    placement: int = 0
    augments: list = field(default_factory=list)
    buffs: dict = field(default_factory=dict)
    free_rerolls: int = 0

    def get(self, loc):
        return self.bench[loc] if loc < BENCH_SIZE else self.board.get(loc - BENCH_SIZE)

    def set(self, loc, unit):
        if loc < BENCH_SIZE:
            self.bench[loc] = unit
        elif unit is None:
            self.board.pop(loc - BENCH_SIZE, None)
        else:
            self.board[loc - BENCH_SIZE] = unit

    def units(self):
        return [u for u in self.bench if u] + list(self.board.values())


def _merge_rank(entry):
    loc = entry[0]
    if loc is None:  # the not-yet-placed purchase is always consumed last
        return (2, 0)
    return (1 if loc < BENCH_SIZE else 0, loc)


class Agent:
    """Default agent: does nothing, takes the first option."""
    def act(self, game, player): pass
    def pick_augment(self, game, player, options): return 0
    def pick_carousel(self, game, player, options): return 0


def build_schedule():
    rounds = [(1, 1, "carousel"), (1, 2, "pve"), (1, 3, "pve"), (1, 4, "pve")]
    for s in range(2, MAX_STAGE + 1):
        for r in range(1, 8):
            kind = "carousel" if r == 4 else "pve" if r == 7 else "pvp"
            rounds.append((s, r, kind))
    return rounds


SCHEDULE = build_schedule()


def base_income(stage, rnd):
    if stage == 1:
        return {2: 2, 3: 2, 4: 3}.get(rnd, 0)
    return 4 if (stage, rnd) == (2, 1) else 5


class Game:
    def __init__(self, seed=None, data=None, num_players=NUM_PLAYERS):
        self.rng = random.Random(seed)
        self.data = data or GameData()
        self.players = [Player(i) for i in range(num_players)]
        self.pool = {n: self.data.pool_size(n) for n in self.data.champions}
        self.round_idx = 0
        self.stage, self.rnd, self.kind = SCHEDULE[0]
        self.finished = False
        self.next_placement = num_players
        self.carousel = []

    # ---------- pool / shop ----------
    def _draw(self, level):
        odds = SHOP_ODDS[level]
        cost = self.rng.choices(range(1, 6), weights=odds)[0]
        for c in [cost] + [x for x in range(1, 6) if x != cost]:
            names = [n for n in self.data.by_cost[c] if self.pool[n] > 0]
            if names:
                name = self.rng.choices(names, weights=[self.pool[n] for n in names])[0]
                self.pool[name] -= 1
                return name
        return None

    def _return_shop(self, p):
        for n in p.shop:
            if n:
                self.pool[n] += 1
        p.shop = [None] * SHOP_SIZE

    def refresh_shop(self, p):
        self._return_shop(p)
        p.shop = [self._draw(p.level) for _ in range(SHOP_SIZE)]

    def reroll(self, p):
        if p.free_rerolls > 0:
            p.free_rerolls -= 1
        elif p.gold >= REROLL_COST:
            p.gold -= REROLL_COST
        else:
            return False
        self.refresh_shop(p)
        return True

    def buy_xp(self, p):
        if p.gold < XP_COST or p.level >= MAX_LEVEL:
            return False
        p.gold -= XP_COST
        self.add_xp(p, XP_PER_BUY)
        return True

    def add_xp(self, p, n):
        p.xp += n
        while p.level < MAX_LEVEL and p.xp >= XP_TO_LEVEL[p.level + 1]:
            p.xp -= XP_TO_LEVEL[p.level + 1]
            p.level += 1
        if p.level >= MAX_LEVEL:
            p.xp = 0

    # ---------- units ----------
    def buy(self, p, i):
        name = p.shop[i]
        if name is None:
            return False
        cost = self.data.champions[name].cost
        if p.gold < cost or not self._acquire(p, name):
            return False
        p.gold -= cost
        p.shop[i] = None
        return True

    def _acquire(self, p, name):
        """Add a 1-star unit (merging if that completes a triple). False if there is no room."""
        pending = Unit(name)
        free = p.bench.index(None) if None in p.bench else None
        if free is not None:
            p.bench[free] = pending
            self._merge(p, name)
            return True
        # bench full: only OK if this copy completes a triple
        if sum(1 for u in p.units() if u.name == name and u.star == 1) < 2:
            return False
        self._merge(p, name, pending)
        return True

    def _merge(self, p, name, pending=None):
        for star in (1, 2):
            while True:
                found = [(l, p.get(l)) for l in range(BENCH_SIZE + BOARD_SLOTS)
                         if p.get(l) and p.get(l).name == name and p.get(l).star == star]
                if pending is not None and star == 1:
                    found.append((None, pending))
                if len(found) < 3:
                    break
                # merged unit prefers a board location, then the earliest bench slot
                found.sort(key=_merge_rank)
                take = found[:3]
                new = Unit(name, star + 1)
                for l, u in take:
                    new.items += u.items
                    if l is not None:
                        p.set(l, None)
                if pending is not None and any(u is pending for _, u in take):
                    pending = None
                new.items, extra = new.items[:MAX_ITEMS_PER_UNIT], new.items[MAX_ITEMS_PER_UNIT:]
                p.inventory += extra
                p.set(next(l for l, _ in take if l is not None), new)

    def sell(self, p, loc):
        u = p.get(loc)
        if u is None:
            return False
        p.gold += sell_value(self.data.champions[u.name].cost, u.star)
        self.pool[u.name] += 3 ** (u.star - 1)
        p.inventory += u.items
        p.set(loc, None)
        return True

    def move(self, p, src, dst):
        """Move/swap between any two locations, respecting board size = level."""
        if src == dst or p.get(src) is None:
            return False
        if not (0 <= dst < BENCH_SIZE + BOARD_SLOTS):
            return False
        a, b = p.get(src), p.get(dst)
        on_board = lambda l: l >= BENCH_SIZE
        if on_board(dst) and not on_board(src) and b is None and len(p.board) >= p.level:
            return False
        p.set(src, b)
        p.set(dst, a)
        return True

    def equip(self, p, inv_idx, loc):
        u = p.get(loc)
        if u is None or inv_idx >= len(p.inventory):
            return False
        item = p.inventory[inv_idx]
        if u.items and is_component(item) and is_component(u.items[-1]):
            u.items[-1] = combine(u.items[-1], item)
        elif len(u.items) < MAX_ITEMS_PER_UNIT:
            u.items.append(item)
        else:
            return False
        p.inventory.pop(inv_idx)
        return True

    def copies(self, name):
        n = self.pool[name]
        for p in self.players:
            n += sum(1 for s in p.shop if s == name)
            n += sum(3 ** (u.star - 1) for u in p.units() if u.name == name)
        n += sum(1 for c in self.carousel if c and c[0] == name)
        return n

    # ---------- augments / carousel ----------
    def apply_augment(self, p, a):
        p.augments.append(a["id"])
        p.gold += a.get("gold", 0)
        self.add_xp(p, a.get("xp", 0))
        p.free_rerolls += a.get("free_rerolls", 0)
        for _ in range(a.get("components", 0)):
            p.inventory.append(self.rng.choice(list(COMPONENTS)))
        for k, v in a.get("buff", {}).items():
            p.buffs[k] = p.buffs.get(k, 0) + v

    def run_carousel(self, agents):
        costs = car.carousel_costs(self.stage)
        self.carousel = []
        for _ in range(car.CAROUSEL_SIZE):
            names = [n for c in costs for n in self.data.by_cost[c] if self.pool[n] > 0]
            if not names:
                break
            name = self.rng.choices(names, weights=[self.pool[n] for n in names])[0]
            self.pool[name] -= 1
            self.carousel.append((name, self.rng.choice(list(COMPONENTS))))
        for p in car.pick_order(self.players):
            if not self.carousel:
                break
            i = agents[p.id].pick_carousel(self, p, list(self.carousel))
            i = i if 0 <= i < len(self.carousel) else 0
            name, item = self.carousel.pop(i)
            p.inventory.append(item)
            self._give_unit(p, name)
        for name, _ in self.carousel:
            self.pool[name] += 1
        self.carousel = []

    def _give_unit(self, p, name):
        if not self._acquire(p, name):
            # bench full and no merge: refund to the pool for the champion's gold cost
            self.pool[name] += 1
            p.gold += self.data.champions[name].cost

    # ---------- round flow ----------
    def alive_players(self):
        return [p for p in self.players if p.alive]

    def _income(self, p):
        base = base_income(self.stage, self.rnd)
        interest = min(INTEREST_CAP, p.gold // INTEREST_PER)
        streak = streak_bonus(max(p.win_streak, p.lose_streak))
        p.gold += base + interest + streak
        if (self.stage, self.rnd) != (1, 1):
            self.add_xp(p, FREE_XP_PER_ROUND)

    def _board_units(self, p, side):
        return combat.build_units(side, p.board, self.data, p.buffs)

    def play_round(self, agents):
        if self.finished:
            return
        stage, rnd, kind = self.stage, self.rnd, self.kind
        alive = self.alive_players()
        for p in alive:
            self._income(p)
            self.refresh_shop(p)
        tier = aug.AUGMENT_ROUNDS.get((stage, rnd))
        if tier:
            for p in alive:
                options = aug.offer(self.rng, tier)
                i = agents[p.id].pick_augment(self, p, options)
                self.apply_augment(p, options[i if 0 <= i < len(options) else 0])
        if kind == "carousel":
            self.run_carousel(agents)
        for p in alive:
            agents[p.id].act(self, p)
        if kind == "pvp":
            self._pvp(alive)
        elif kind == "pve":
            self._pve(alive)
        self._advance()

    def _pve(self, alive):
        for p in alive:
            res = combat.simulate(self._board_units(p, 0), combat.creep_team(1, self.stage))
            if res.winner == 0:
                p.inventory.append(self.rng.choice(list(COMPONENTS)))

    def _pvp(self, alive):
        ids = [p.id for p in alive]
        self.rng.shuffle(ids)
        fights = [(ids[i], ids[i + 1], False) for i in range(0, len(ids) - 1, 2)]
        if len(ids) % 2 and len(ids) > 1:  # odd one out fights a ghost copy of a random opponent
            fights.append((ids[-1], self.rng.choice(ids[:-1]), True))
        outcomes = []
        for a, b, ghost in fights:
            pa, pb = self.players[a], self.players[b]
            res = combat.simulate(self._board_units(pa, 0), self._board_units(pb, 1))
            outcomes.append((pa, pb, ghost, res))
        for pa, pb, ghost, res in outcomes:
            if res.winner < 0:
                continue
            win, lose = (pa, pb) if res.winner == 0 else (pb, pa)
            dmg = stage_damage(self.stage) + res.survivors[res.winner]
            win_is_ghost = ghost and win is pb
            lose_is_ghost = ghost and lose is pb
            if not win_is_ghost:
                win.win_streak += 1
                win.lose_streak = 0
            if not lose_is_ghost:
                lose.hp -= dmg
                lose.lose_streak += 1
                lose.win_streak = 0

    def _advance(self):
        newly = [p for p in self.players if p.alive and p.hp <= 0]
        n_alive = len(self.alive_players())
        for p in sorted(newly, key=lambda p: (p.hp, -p.id)):
            self._eliminate(p, n_alive)
            n_alive -= 1
        remaining = self.alive_players()
        if len(remaining) <= 1 or self.round_idx + 1 >= len(SCHEDULE):
            for i, p in enumerate(sorted(remaining, key=lambda p: (-p.hp, p.id))):
                p.placement = 1 + i
            self.finished = True
            return
        self.round_idx += 1
        self.stage, self.rnd, self.kind = SCHEDULE[self.round_idx]

    def _eliminate(self, p, placement):
        p.alive = False
        p.placement = placement
        self._return_shop(p)
        for u in p.units():
            self.pool[u.name] += 3 ** (u.star - 1)
        p.bench = [None] * BENCH_SIZE
        p.board = {}

    def run(self, agents):
        while not self.finished:
            self.play_round(agents)
        return {p.id: p.placement for p in self.players}
