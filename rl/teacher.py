"""Scripted economy policy expressed in the env's action space; used to warm-start the network
(behaviour cloning) because plain PPO from scratch never leaves last place."""
from bots.rule_bots import card_score, copies_owned, focus_traits, unit_power
from rl.env import AUG0, BUY0, CAR0, END, REROLL, SELL_BENCH0, XP
from tft_sim.data import REROLL_COST, XP_COST

TARGET = {1: 3, 2: 5, 3: 7, 4: 8}


def teacher_action(env, mask):
    g, me = env.game, env.me
    if env.phase == "aug":
        keys = ("gold", "xp")
        best = max(range(len(env.options)), key=lambda i: sum(2 if k in env.options[i] else 0 for k in keys) - (0 if any(k in env.options[i] for k in keys) else 1))
        return AUG0 + best
    if env.phase == "car":
        focus = focus_traits(g, me)
        return CAR0 + max(range(len(g.carousel)), key=lambda i: card_score(g, me, g.carousel[i][0], focus))
    stage = g.stage
    target = TARGET.get(stage, 9)
    if stage == 2 and g.rnd >= 5:
        target = 6
    floor = 50 if stage >= 3 else 30 if stage == 2 else 10
    if me.hp < 40:
        floor = 0
    focus = focus_traits(g, me)
    n_units = len(me.units())

    def best_buy(min_score, floor_):
        opts = []
        for i, name in enumerate(me.shop):
            if not name or not mask[BUY0 + i]:
                continue
            sc = card_score(g, me, name, focus)
            merge = copies_owned(me, name) >= 2
            cost = g.data.champions[name].cost
            if sc >= min_score and (merge or me.gold - cost >= floor_):
                opts.append((sc, i))
        return max(opts)[1] if opts else None

    i = best_buy(40, floor)
    if i is None and n_units < me.level:
        i = best_buy(0, 0)
    if i is not None:
        return BUY0 + i
    # bench full but a wanted card exists: free a slot first
    if None not in me.bench:
        pass
    if me.level < target and me.gold - XP_COST >= floor and mask[XP]:
        return XP
    if (me.level >= target and me.gold > floor + 10 or me.hp < 40) and me.gold - REROLL_COST >= floor and mask[REROLL]:
        return REROLL
    i = best_buy(8, floor)
    if i is not None:
        return BUY0 + i
    if None not in me.bench:  # keep the bench from clogging: sell the weakest single copy
        cands = [(unit_power(g, u), k) for k, u in enumerate(me.bench) if u and copies_owned(me, u.name) == 1
                 and not (focus & set(g.data.champions[u.name].traits))]
        if cands and any(n and card_score(g, me, n, focus) >= 40 for n in me.shop):
            return SELL_BENCH0 + min(cands)[1]
    return END
