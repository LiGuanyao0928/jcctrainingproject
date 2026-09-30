"""Fast tick-based auto battler. 1 tick = 0.1s; an aspd of 1.0 attacks every 10 ticks."""
from dataclasses import dataclass, field

from .data import BOARD_COLS

MAX_TICKS = 300
MANA_PER_ATTACK = 10


@dataclass
class CUnit:
    side: int
    name: str
    slot: int
    hp: float
    atk: float
    armor: float
    interval: int
    range: int
    max_mana: int
    mana: float
    skill_type: str
    skill_value: float
    ap: float = 0.0
    mr: float = 0.0
    crit_avg: float = 1.0       # expected damage multiplier from crits (keeps combat deterministic)
    skill_dmg: str = "magic"
    skill_extra: float = 0.0
    buff_until: int = 0         # tick until which the attack-speed buff (skill_type buff_as) is active
    base_interval: int = 0
    assassin: bool = False
    star: int = 1
    shield: float = 0.0
    cd: int = 0
    max_hp: float = field(init=False)

    def __post_init__(self):
        self.max_hp = self.hp
        self.cd = self.interval
        self.base_interval = self.interval

    @property
    def alive(self):
        return self.hp > 0

    @property
    def front(self):
        return self.slot // BOARD_COLS < 2


@dataclass
class CombatResult:
    winner: int  # 0, 1 or -1 for draw
    survivors: tuple
    ticks: int


def build_units(side, board, data, buffs=None):
    """board: {slot: Unit}. buffs: extra multipliers from augments (atk_pct, hp_pct, armor, aspd_pct, ap)."""
    buffs = buffs or {}
    effects, counts = data.active_trait_effects(u.name for u in board.values())
    total = dict(buffs)
    for e in effects:
        for k, v in e.items():
            total[k] = total.get(k, 0) + v
    units = []
    for slot in sorted(board):
        u = board[slot]
        ch = data.champions[u.name]
        m = data.rules.star_mult ** (u.star - 1)
        stats = {}
        for it in u.items:
            for k, v in data.items.stats(it).items():
                stats[k] = stats.get(k, 0) + v
        g = lambda k: total.get(k, 0) + stats.get(k, 0)
        aspd = ch.aspd * (1 + g("aspd_pct"))
        units.append(CUnit(
            side=side, name=u.name, slot=slot, star=u.star,
            hp=(ch.hp * m) * (1 + g("hp_pct")) + stats.get("hp", 0) + total.get("hp", 0),
            atk=(ch.atk * m) * (1 + g("atk_pct")) + g("atk"),
            armor=ch.armor + g("armor"), mr=ch.mr + g("mr"),
            crit_avg=1 + min(1.0, ch.crit_chance + g("crit_chance")) * (ch.crit_mult - 1),
            skill_dmg=ch.skill_dmg, skill_extra=ch.skill_extra,
            interval=max(2, round(10 / aspd)), range=ch.range,
            max_mana=max(10, ch.mana - int(stats.get("mana", 0))),
            mana=ch.mana_start + g("mana_start"),
            skill_type=ch.skill_type, skill_value=ch.skill_value * m,
            ap=g("ap"), assassin="Assassin" in ch.traits))
    return units


def _reduce(dmg, armor):
    return dmg * 100.0 / (100.0 + armor)


def _skill_damage(dmg, kind, target):
    if kind == "true":
        return dmg
    return _reduce(dmg, target.armor if kind == "physical" else target.mr)


def _hit(target, dmg):
    if target.shield > 0:
        absorbed = min(target.shield, dmg)
        target.shield -= absorbed
        dmg -= absorbed
    target.hp -= dmg


def _pick_target(u, enemies):
    alive = [e for e in enemies if e.hp > 0]
    if not alive:
        return None
    if u.assassin:
        return max(alive, key=lambda e: (e.slot // BOARD_COLS, e.slot))
    front = [e for e in alive if e.front]
    pool = front or alive
    return min(pool, key=lambda e: (abs(e.slot % BOARD_COLS - u.slot % BOARD_COLS), e.slot))


BUFF_TICKS = 50


def _cast(u, allies, enemies, target, tick):
    v = u.skill_value * (1 + u.ap / 100.0)
    t = u.skill_type
    if t == "damage":
        _hit(target, _skill_damage(v, u.skill_dmg, target))
    elif t == "aoe":
        for e in enemies:
            if e.hp > 0:
                _hit(e, _skill_damage(v * 0.6, u.skill_dmg, e))
    elif t == "stun":
        _hit(target, _skill_damage(v, u.skill_dmg, target))
        target.cd += int(u.skill_extra)
    elif t == "buff_as":
        u.buff_until = tick + BUFF_TICKS
        u.interval = max(2, round(u.base_interval / (1 + u.skill_extra)))
    elif t == "shield":
        u.shield += v
    elif t == "heal":
        hurt = min((a for a in allies if a.hp > 0), key=lambda a: a.hp / a.max_hp)
        hurt.hp = min(hurt.max_hp, hurt.hp + v)


def simulate(units_a, units_b, max_ticks=MAX_TICKS) -> CombatResult:
    teams = (units_a, units_b)
    order = sorted(units_a + units_b, key=lambda u: (u.slot, u.side))
    tick = 0
    for tick in range(1, max_ticks + 1):
        for u in order:
            if u.hp <= 0:
                continue
            u.cd -= 1
            if u.cd > 0:
                continue
            enemies = teams[1 - u.side]
            target = _pick_target(u, enemies)
            if target is None:
                break
            if u.buff_until and tick >= u.buff_until:
                u.buff_until, u.interval = 0, u.base_interval
            u.cd = u.interval
            _hit(target, _reduce(u.atk * u.crit_avg, target.armor))
            u.mana += MANA_PER_ATTACK
            if u.mana >= u.max_mana:
                u.mana = 0
                _cast(u, teams[u.side], enemies, target, tick)
        sa = sum(1 for u in units_a if u.hp > 0)
        sb = sum(1 for u in units_b if u.hp > 0)
        if sa == 0 or sb == 0:
            break
    sa = sum(1 for u in units_a if u.hp > 0)
    sb = sum(1 for u in units_b if u.hp > 0)
    if sa and not sb:
        w = 0
    elif sb and not sa:
        w = 1
    else:
        w = -1
    return CombatResult(w, (sa, sb), tick)


def creep_team(side, stage, n=None):
    n = n or min(8, 2 + stage)
    scale = 1 + 0.6 * (stage - 1)
    return [CUnit(side=side, name="creep", slot=i, hp=350 * scale, atk=25 * scale,
                  armor=20, mr=20, interval=12, range=1, max_mana=999, mana=0,
                  skill_type="damage", skill_value=0) for i in range(n)]
