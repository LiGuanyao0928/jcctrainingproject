"""Rule-based opponents: economy (运营流), reroll (D牌流) and random."""
import random

from tft_sim.data import BENCH_SIZE, BOARD_COLS
from tft_sim.game import Agent, board_loc

COL_ORDER = [3, 2, 4, 1, 5, 0, 6]  # centre-out


# ---------- shared helpers ----------
def copies_owned(p, name):
    return sum(3 ** (u.star - 1) for u in p.units() if u.name == name)


def unit_power(game, u):
    return game.data.champions[u.name].cost * 1.8 ** (u.star - 1) + 0.3 * len(u.items)


def focus_traits(game, p, k=2):
    score = {}
    for u in p.units():
        for t in game.data.champions[u.name].traits:
            score[t] = score.get(t, 0) + unit_power(game, u)
    return {t for t, _ in sorted(score.items(), key=lambda x: -x[1])[:k]}


def best_lineup(game, p):
    """Greedy pick of up to `level` units, rewarding trait breakpoints."""
    pool = p.units()
    chosen, names = [], set()

    def gain(u):
        g = unit_power(game, u)
        if u.name not in names:
            for t in game.data.champions[u.name].traits:
                n = 1 + sum(1 for nm in names if t in game.data.champions[nm].traits)
                if any(bp == n for bp, _ in game.data.traits.get(t, [])):
                    g += 2.5
        return g

    while pool and len(chosen) < p.level:
        best = max(pool, key=gain)
        pool.remove(best)
        chosen.append(best)
        names.add(best.name)
    return chosen


def place(game, p, chosen):
    melee = sorted((u for u in chosen if game.data.champions[u.name].range <= 1),
                   key=lambda u: -game.data.champions[u.name].hp)
    ranged = [u for u in chosen if game.data.champions[u.name].range > 1]
    layout = {}
    for rows, group in (((1, 0), melee), ((2, 3), ranged)):
        for i, u in enumerate(group):
            layout[rows[i // BOARD_COLS % 2] * BOARD_COLS + COL_ORDER[i % BOARD_COLS]] = u
    return layout


def tidy(game, p):
    """Field the best lineup and spread items over the strongest units."""
    chosen = best_lineup(game, p)
    layout = place(game, p, chosen)
    if len(layout) == len(chosen):
        game.arrange(p, layout)
    carriers = sorted(p.board.items(), key=lambda kv: -unit_power(game, kv[1]))
    for slot, u in carriers:
        while p.inventory and len(u.items) < 3:
            if not game.equip(p, 0, board_loc(slot)):
                break


def card_score(game, p, name, focus):
    ch = game.data.champions[name]
    have = copies_owned(p, name)
    s = ch.cost
    if have >= 2:
        s += 100
    elif have == 1:
        s += 40
    s += 6 * sum(1 for t in ch.traits if t in focus)
    return s


def make_room(game, p, keep_names):
    """Sell the weakest bench unit that isn't part of a pair, if the bench is full."""
    if None in p.bench:
        return True
    cand = [(i, u) for i, u in enumerate(p.bench) if u and u.name not in keep_names and copies_owned(p, u.name) == 1]
    if not cand:
        return False
    i, _ = min(cand, key=lambda x: unit_power(game, x[1]))
    return game.sell(p, i)


def buy_best(game, p, floor, min_score, max_cost=5):
    """Buy the best-scoring card while gold stays >= floor (merges ignore the floor)."""
    bought = False
    while True:
        focus = focus_traits(game, p)
        options = []
        for i, name in enumerate(p.shop):
            if not name:
                continue
            ch = game.data.champions[name]
            sc = card_score(game, p, name, focus)
            merge = copies_owned(p, name) >= 2
            if ch.cost > max_cost and copies_owned(p, name) == 0:
                continue
            if sc < min_score or p.gold < ch.cost or (not merge and p.gold - ch.cost < floor):
                continue
            options.append((sc, i))
        if not options:
            return bought
        sc, i = max(options)
        name = p.shop[i]
        if None not in p.bench and copies_owned(p, name) < 2:
            if not make_room(game, p, focus_names(game, p, focus)):
                return bought
        if not game.buy(p, i):
            return bought
        bought = True


def focus_names(game, p, focus):
    return {u.name for u in p.units() if focus & set(game.data.champions[u.name].traits)}


def level_up(game, p, target, floor):
    while p.level < target and p.gold - game.rules.xp_cost >= floor:
        if not game.buy_xp(p):
            break


def roll(game, p, floor, min_score, max_cost=5):
    while p.gold - game.rules.reroll_cost >= floor:
        if not game.reroll(p):
            break
        buy_best(game, p, floor, min_score, max_cost)


def pick_by(keys):
    def picker(options):
        def score(a):
            for i, k in enumerate(keys):
                if k in a or k in a.get("buff", {}):
                    return len(keys) - i
            return 0
        return max(range(len(options)), key=lambda i: score(options[i]))
    return picker


# ---------- bots ----------
class RuleBot(Agent):
    name = "rule"
    augment_keys = ("gold", "xp")

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def pick_augment(self, game, p, options):
        return pick_by(self.augment_keys)(options)

    def pick_carousel(self, game, p, options):
        focus = focus_traits(game, p)
        return max(range(len(options)),
                   key=lambda i: card_score(game, p, options[i][0], focus))

    def shop(self, game, p):
        raise NotImplementedError

    def act(self, game, p):
        self.shop(game, p)
        tidy(game, p)


class EconBot(RuleBot):
    """Hold 50 gold for interest, level on schedule, roll down when weak."""
    name = "econ"
    TARGET = {1: 3, 2: 5, 3: 7, 4: 8}

    def shop(self, game, p):
        target = self.TARGET.get(game.stage, 9)
        if game.stage == 2 and game.rnd >= 5:
            target = 6
        floor = 50 if game.stage >= 3 else 30 if game.stage == 2 else 10
        if p.hp < 40:
            floor = 0
        buy_best(game, p, floor, min_score=40)
        if len(p.units()) < p.level:  # never leave the board empty
            buy_best(game, p, 0, min_score=0, max_cost=max(1, p.level - 1))
        level_up(game, p, target, floor)
        if p.level >= target and p.gold > floor + 10 or p.hp < 40:
            roll(game, p, floor, 30)
        buy_best(game, p, floor, min_score=8)


class RerollBot(RuleBot):
    """Stay low-level and roll for 1-3 cost three-stars, levelling only late."""
    name = "reroll"
    augment_keys = ("free_rerolls", "gold", "aspd_pct")
    TARGET = {1: 3, 2: 5, 3: 6, 4: 7, 5: 8}

    def shop(self, game, p):
        target = self.TARGET.get(game.stage, 9)
        floor = 20 if game.stage <= 2 else 0
        max_cost = 3 if game.stage <= 4 else 5
        buy_best(game, p, floor, min_score=8, max_cost=max_cost)
        if len(p.units()) < p.level:
            buy_best(game, p, 0, min_score=0, max_cost=3)
        level_up(game, p, target, floor + 10 if game.stage <= 3 else 0)
        if game.stage >= 3 or (game.stage == 2 and game.rnd >= 5) or p.hp < 50:
            roll(game, p, floor, 8, max_cost=max_cost)


class RandomBot(RuleBot):
    name = "random"
    augment_keys = ()

    def pick_augment(self, game, p, options):
        return self.rng.randrange(len(options))

    def pick_carousel(self, game, p, options):
        return self.rng.randrange(len(options))

    def shop(self, game, p):
        for _ in range(8):
            r = self.rng.random()
            if r < 0.55:
                game.buy(p, self.rng.randrange(5))
            elif r < 0.75:
                game.reroll(p)
            elif r < 0.9:
                game.buy_xp(p)
            elif p.bench[0] is not None:
                game.sell(p, self.rng.randrange(BENCH_SIZE))
