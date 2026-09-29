"""Gymnasium env: the learner sits in seat 0 against 7 rule bots. One discrete action space
covers every phase (planning, augment pick, carousel pick); `action_masks()` gives the legal ones."""
import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bots.rule_bots import COL_ORDER, EconBot, RandomBot, RerollBot, copies_owned
from tft_sim import augments as aug
from tft_sim.data import BENCH_SIZE, BOARD_COLS, MAX_LEVEL, REROLL_COST, XP_COST, XP_TO_LEVEL, GameData
from tft_sim.game import Game, board_loc
from tft_sim.items import COMPONENTS, is_component

N = 9  # bench slots / addressable board units
END, REROLL, XP = 0, 1, 2
BUY0 = 3                       # 5 shop slots
SELL_BENCH0 = BUY0 + 5         # 9
SELL_BOARD0 = SELL_BENCH0 + N  # 9
UP0 = SELL_BOARD0 + N          # bench -> board, 9
DOWN0 = UP0 + N                # board -> bench, 9
EQUIP0 = DOWN0 + N             # inventory[0] onto board unit k, 9
AUG0 = EQUIP0 + N              # 3
CAR0 = AUG0 + 3                # 9
N_ACTIONS = CAR0 + 9

TRAITS = ["Assassin", "Bruiser", "Guardian", "Mage", "Ranger", "Warrior"]
UNIT_F = 10
BOT_MIX = ((EconBot, 0.4), (RerollBot, 0.4), (RandomBot, 0.2))
EVAL_LINEUP = [EconBot] * 3 + [RerollBot] * 3 + [RandomBot]


def _champ_feat(data):
    out = {}
    for n, c in data.champions.items():
        out[n] = (c.cost / 5.0, [1.0 if t in c.traits else 0.0 for t in TRAITS])
    return out


class TFTEnv(gym.Env):
    metadata = {}

    def __init__(self, seed=None, lineup=None, shaping=1.0, max_actions=40):
        self.data = GameData()
        self.cf = _champ_feat(self.data)
        self.lineup = lineup            # fixed list of 7 bot classes, else sampled from BOT_MIX
        self.shaping = shaping
        self.max_actions = max_actions
        self._seed = seed
        self._n_eps = 0
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.obs_dim = len(self._obs_from_blank())
        self.observation_space = spaces.Box(-1.0, 5.0, (self.obs_dim,), np.float32)

    # ---------- episode ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        base = (self._seed if self._seed is not None else random.randrange(2 ** 31)) + self._n_eps * 7919
        self._n_eps += 1
        rng = random.Random(base)
        classes = self.lineup or rng.choices([c for c, _ in BOT_MIX], weights=[w for _, w in BOT_MIX], k=7)
        self.game = Game(seed=base, data=self.data)
        self.me = self.game.players[0]
        self.bots = {i + 1: c(base + i) for i, c in enumerate(classes)}
        self.reward_acc = 0.0
        self.phase = "plan"
        self.options = []
        self.n_actions = 0
        self.done = False
        self.placement = 0
        self._gen = self._flow()
        next(self._gen)
        return self.obs(), {"action_mask": self.action_masks()}

    def _flow(self):
        g, me = self.game, self.me
        while not g.finished and me.alive:
            alive = g.start_round()
            hp0 = me.hp
            tier = aug.AUGMENT_ROUNDS.get((g.stage, g.rnd))
            if tier:
                offers = {p.id: aug.offer(g.rng, tier) for p in alive}
                for p in alive:
                    if p.id:
                        opts = offers[p.id]
                        i = self.bots[p.id].pick_augment(g, p, opts)
                        g.apply_augment(p, opts[i if 0 <= i < len(opts) else 0])
                self.options, self.phase = offers[0], "aug"
                a = yield
                g.apply_augment(me, self.options[a - AUG0])
            if g.kind == "carousel":
                for p in g.carousel_begin():
                    if not g.carousel:
                        break
                    if p.id == 0:
                        self.phase = "car"
                        a = yield
                        g.carousel_take(me, a - CAR0)
                    else:
                        g.carousel_take(p, self.bots[p.id].pick_carousel(g, p, list(g.carousel)))
                g.carousel_end()
            for p in alive:
                if p.id:
                    self.bots[p.id].act(g, p)
            self.phase, self.n_actions = "plan", 0
            while (yield) != END:
                pass
            fill = len(me.board) / max(1, me.level)
            g.finish_round(alive)
            # dense signal: hp lost this round, plus a small bonus for using every board slot
            self.reward_acc += self.shaping * (me.hp - hp0) / 100.0 + 0.01 * fill * self.shaping

    def step(self, action):
        action = int(action)
        assert self.action_masks()[action], f"illegal action {action} in phase {self.phase}"
        finished = False
        try:
            if self.phase == "plan" and action != END:
                self._apply(action)
                self.n_actions += 1
                if self.n_actions >= self.max_actions:
                    self._gen.send(END)
            else:
                self._gen.send(action)
        except StopIteration:
            finished = True
        reward, self.reward_acc = self.reward_acc, 0.0
        info = {}
        if finished:
            self.done = True
            self.placement = self.me.placement
            reward += (4.5 - self.placement) / 3.5
            info = {"placement": self.placement, "rounds": self.game.round_idx}
        info["action_mask"] = self.action_masks()
        return self.obs(), reward, finished, False, info

    # ---------- actions ----------
    def board_slots(self):
        return sorted(self.me.board)[:N]

    def _auto_slot(self, unit):
        rng = self.data.champions[unit.name].range
        rows = (1, 0) if rng <= 1 else (2, 3)
        for r in rows:
            for c in COL_ORDER:
                if r * BOARD_COLS + c not in self.me.board:
                    return r * BOARD_COLS + c
        return None

    def _apply(self, a):
        g, me = self.game, self.me
        if a == REROLL:
            g.reroll(me)
        elif a == XP:
            g.buy_xp(me)
        elif BUY0 <= a < SELL_BENCH0:
            g.buy(me, a - BUY0)
        elif SELL_BENCH0 <= a < SELL_BOARD0:
            g.sell(me, a - SELL_BENCH0)
        elif SELL_BOARD0 <= a < UP0:
            g.sell(me, board_loc(self.board_slots()[a - SELL_BOARD0]))
        elif UP0 <= a < DOWN0:
            i = a - UP0
            g.move(me, i, board_loc(self._auto_slot(me.bench[i])))
        elif DOWN0 <= a < EQUIP0:
            g.move(me, board_loc(self.board_slots()[a - DOWN0]), me.bench.index(None))
        elif EQUIP0 <= a < AUG0:
            g.equip(me, 0, board_loc(self.board_slots()[a - EQUIP0]))

    def action_masks(self):
        m = np.zeros(N_ACTIONS, dtype=bool)
        g, me = self.game, self.me
        if self.phase == "aug":
            m[AUG0:AUG0 + len(self.options)] = True
            return m
        if self.phase == "car":
            m[CAR0:CAR0 + len(g.carousel)] = True
            return m
        m[END] = True
        if self.n_actions >= self.max_actions:
            return m
        m[REROLL] = me.gold >= REROLL_COST or me.free_rerolls > 0
        m[XP] = me.gold >= XP_COST and me.level < MAX_LEVEL
        for i, name in enumerate(me.shop):
            if name and me.gold >= self.data.champions[name].cost and g.can_acquire(me, name):
                m[BUY0 + i] = True
        slots = self.board_slots()
        has_free_bench = None in me.bench
        for i, u in enumerate(me.bench):
            if u:
                m[SELL_BENCH0 + i] = True
                if len(me.board) < me.level and self._auto_slot(u) is not None:
                    m[UP0 + i] = True
        for k, s in enumerate(slots):
            m[SELL_BOARD0 + k] = True
            m[DOWN0 + k] = has_free_bench
            if me.inventory:
                u = me.board[s]
                m[EQUIP0 + k] = len(u.items) < 3 or (is_component(me.inventory[0]) and is_component(u.items[-1]))
        return m

    # ---------- observation ----------
    def _unit_feat(self, u):
        if u is None:
            return [0.0] * UNIT_F
        cost, tr = self.cf[u.name]
        return [1.0, cost, *tr, u.star / 3.0, len(u.items) / 3.0]

    def _card_feat(self, name):
        if name is None:
            return [0.0] * UNIT_F
        cost, tr = self.cf[name]
        have = copies_owned(self.me, name)
        return [1.0, cost, *tr, have / 9.0, float(have >= 2)]

    def _obs_from_blank(self):
        self.game = Game(seed=0, data=self.data)
        self.me = self.game.players[0]
        self.phase, self.options, self.n_actions = "plan", [], 0
        return self.obs()

    def obs(self):
        g, me = self.game, self.me
        f = []
        kind = [g.kind == k for k in ("pvp", "pve", "carousel")]
        phase = [self.phase == k for k in ("plan", "aug", "car")]
        nxt = XP_TO_LEVEL.get(me.level + 1)
        opp = sorted(p.hp for p in g.players[1:])
        comps = [sum(1 for i in me.inventory if i == c) for c in COMPONENTS]
        effects, counts = self.data.active_trait_effects(u.name for u in me.board.values())
        f += [g.stage / 8, g.rnd / 7, *map(float, kind), *map(float, phase), me.gold / 100, min(me.gold // 10, 5) / 5,
              me.level / 9, (me.xp / nxt) if nxt else 1.0, me.hp / 100, me.win_streak / 10, me.lose_streak / 10,
              len(g.alive_players()) / 8, len(me.board) / max(1, me.level), sum(u is None for u in me.bench) / 9,
              len(me.inventory) / 10, sum(1 for i in me.inventory if not is_component(i)) / 10,
              me.free_rerolls / 4, self.n_actions / self.max_actions]
        f += [c / 4 for c in comps]
        f += [counts.get(t, 0) / 6 for t in TRAITS]
        f += [h / 100 for h in opp] + [0.0] * (7 - len(opp))
        for name in me.shop:
            f += self._card_feat(name)
        for u in me.bench:
            f += self._unit_feat(u)
        slots = self.board_slots()
        for k in range(N):
            f += self._unit_feat(me.board[slots[k]] if k < len(slots) else None)
        cars = [c[0] for c in g.carousel] if self.phase == "car" else []
        for k in range(N):
            f += self._card_feat(cars[k] if k < len(cars) else None)
        for k in range(3):
            a = self.options[k] if k < len(self.options) else {}
            b = a.get("buff", {})
            f += [a.get("gold", 0) / 24, a.get("xp", 0) / 40, a.get("free_rerolls", 0) / 4, a.get("components", 0) / 3,
                  b.get("atk_pct", 0), b.get("hp_pct", 0), b.get("aspd_pct", 0), b.get("ap", 0) / 40]
        return np.asarray(f, dtype=np.float32)
