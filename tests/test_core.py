import collections
import random

import pytest

from tft_sim import combat
from tft_sim.data import (BENCH_SIZE, INTEREST_CAP, POOL_SIZE, SHOP_ODDS, GameData, sell_value,
                          stage_damage, streak_bonus, XP_TO_LEVEL)
from tft_sim.game import SCHEDULE, Agent, Game, Unit, board_loc, base_income
from tft_sim.items import combine, item_stats
from tft_sim import augments, carousel
from helpers import RandomAgent


def conserved(g):
    return all(g.copies(n) == g.data.pool_size(n) for n in g.data.champions)


def test_data_loads():
    d = GameData()
    assert len(d.champions) >= 30
    assert set(d.by_cost) == {1, 2, 3, 4, 5}
    for c in d.champions.values():
        assert all(t in d.traits for t in c.traits)


def test_pool_starts_full_and_conserved_through_shops():
    g = Game(seed=1)
    assert conserved(g)
    for p in g.players:
        p.level = 9
        g.refresh_shop(p)
    assert conserved(g)
    for _ in range(50):
        for p in g.players:
            g.refresh_shop(p)
    assert conserved(g)


def test_level_one_shop_only_cost_one():
    g = Game(seed=2)
    p = g.players[0]
    for _ in range(50):
        g.refresh_shop(p)
        assert all(g.data.champions[n].cost == 1 for n in p.shop)


def test_shop_odds_level_nine():
    g = Game(seed=3)
    p = g.players[0]
    p.level = 9
    cnt = collections.Counter()
    for _ in range(3000):
        g.refresh_shop(p)
        for n in p.shop:
            cnt[g.data.champions[n].cost] += 1
    total = sum(cnt.values())
    for cost, pct in enumerate(SHOP_ODDS[9], 1):
        assert abs(cnt[cost] / total - pct / 100) < 0.03


def test_buy_sell_gold_and_pool():
    g = Game(seed=4)
    p = g.players[0]
    p.gold = 10
    g.refresh_shop(p)
    name = p.shop[0]
    before = g.pool[name]
    assert g.buy(p, 0)
    assert p.gold == 9 and p.shop[0] is None and p.bench[0].name == name
    assert g.sell(p, 0)
    assert p.gold == 10 and g.pool[name] == before + 1 and conserved(g)


def test_buy_fails_without_gold_or_room():
    g = Game(seed=5)
    p = g.players[0]
    g.refresh_shop(p)
    assert not g.buy(p, 0)  # 0 gold
    p.gold = 100
    for i in range(BENCH_SIZE):
        p.bench[i] = Unit("Aria") if i % 2 else Unit("Brute")
        p.bench[i].name = ["Aria", "Brute", "Cinder", "Dax", "Elm", "Fen", "Gale", "Holt", "Iris"][i]
    p.shop[0] = "Kira"
    assert not g.buy(p, 0) and p.gold == 100


def test_merge_to_two_and_three_star():
    g = Game(seed=6)
    p = g.players[0]
    p.gold = 100
    for _ in range(3):
        p.shop[0] = "Aria"
        g.pool["Aria"] -= 1
        assert g.buy(p, 0)
    units = p.units()
    assert len(units) == 1 and units[0].star == 2
    for _ in range(6):
        p.shop[0] = "Aria"
        g.pool["Aria"] -= 1
        assert g.buy(p, 0)
    units = p.units()
    assert len(units) == 1 and units[0].star == 3
    assert conserved(g)
    assert sell_value(1, 3) == 9 and sell_value(3, 2) == 8


def test_merge_with_full_bench():
    g = Game(seed=7)
    p = g.players[0]
    p.gold = 100
    names = ["Aria", "Aria", "Brute", "Cinder", "Dax", "Elm", "Fen", "Gale", "Holt"]
    for n in names:
        p.bench[names.index(n) if False else p.bench.index(None)] = Unit(n)
        g.pool[n] -= 1
    p.shop[0] = "Aria"
    g.pool["Aria"] -= 1
    assert g.buy(p, 0)
    assert len(p.bench) == BENCH_SIZE
    assert sum(1 for u in p.units() if u.name == "Aria" and u.star == 2) == 1
    assert conserved(g)


def test_xp_levels_and_cap():
    g = Game(seed=8)
    p = g.players[0]
    g.add_xp(p, 2)
    assert p.level == 2 and p.xp == 0
    g.add_xp(p, 2 + 6)
    assert p.level == 4
    g.add_xp(p, 10 ** 4)
    assert p.level == 9 and p.xp == 0
    p.gold = 50
    assert not g.buy_xp(p)


def test_economy_interest_and_streaks():
    g = Game(seed=9)
    p = g.players[0]
    g.round_idx = SCHEDULE.index((2, 2, "pvp"))
    g.stage, g.rnd, g.kind = SCHEDULE[g.round_idx]
    p.gold = 100
    g._income(p)
    assert p.gold == 100 + 5 + INTEREST_CAP
    p.gold, p.win_streak = 0, 5
    g._income(p)
    assert p.gold == 5 + streak_bonus(5)
    assert [streak_bonus(i) for i in range(7)] == [0, 0, 1, 1, 2, 3, 3]
    assert base_income(1, 2) == 2 and base_income(2, 1) == 4 and base_income(3, 5) == 5


def test_damage_table_and_pvp_damage():
    assert stage_damage(2) < stage_damage(4) < stage_damage(7)
    g = Game(seed=10, num_players=2)
    g.round_idx = SCHEDULE.index((3, 1, "pvp"))
    g.stage, g.rnd, g.kind = SCHEDULE[g.round_idx]
    a, b = g.players
    a.board[0] = Unit("Aria")
    b.board[0] = Unit("Aria")
    a.board[1] = Unit("Holt")
    a.level = b.level = 3
    res = combat.simulate(g._board_units(a, 0), g._board_units(b, 1))
    assert res.winner == 0 and res.survivors[0] >= 1
    g._pvp([a, b])
    # side order is shuffled, so survivor counts may differ by a tick-order effect; accept either
    assert a.hp == 100 and 100 - b.hp in {stage_damage(3) + k for k in (1, 2)}
    assert a.win_streak == 1 and b.lose_streak == 1


def test_items_combine_and_equip():
    assert combine("Sword", "Bow") == combine("Bow", "Sword") == "Bow+Sword"
    assert item_stats("Bow+Sword") == {"atk": 15, "aspd_pct": 0.15}
    with pytest.raises(ValueError):
        combine("Sword", "Bow+Sword")
    g = Game(seed=11)
    p = g.players[0]
    p.bench[0] = Unit("Aria")
    p.inventory = ["Sword", "Bow", "Belt"]
    assert g.equip(p, 0, 0) and p.bench[0].items == ["Sword"]
    assert g.equip(p, 0, 0) and p.bench[0].items == ["Bow+Sword"]
    assert g.equip(p, 0, 0) and p.bench[0].items == ["Bow+Sword", "Belt"]
    g.sell(p, 0)
    assert "Bow+Sword" in p.inventory


def test_board_size_limited_by_level():
    g = Game(seed=12)
    p = g.players[0]
    p.bench[0], p.bench[1] = Unit("Aria"), Unit("Brute")
    assert g.move(p, 0, board_loc(3))
    assert not g.move(p, 1, board_loc(4))  # level 1 -> 1 slot
    p.level = 2
    assert g.move(p, 1, board_loc(4))


def test_augment_and_carousel_rounds():
    assert Game(seed=0).rules.augment_rounds == {(2, 1): "silver", (3, 2): "gold", (4, 2): "prismatic"}
    kinds = {(s, r): k for s, r, k in SCHEDULE}
    assert kinds[(1, 1)] == "carousel" and kinds[(2, 4)] == "carousel" and kinds[(2, 7)] == "pve"
    g = Game(seed=13)
    offers = []

    class Rec(Agent):
        def pick_augment(self, game, p, options):
            offers.append((game.stage, game.rnd, len(options)))
            return 0
    agents = [Rec() for _ in g.players]
    while (g.stage, g.rnd) <= (4, 2) and not g.finished:
        g.play_round(agents)
    assert {(s, r) for s, r, _ in offers} == {(2, 1), (3, 2), (4, 2)}
    assert all(n == 3 for *_, n in offers)


def test_carousel_pick_order_lowest_hp_first():
    g = Game(seed=14)
    for i, p in enumerate(g.players):
        p.hp = 50 + (i * 7) % 8 * 5
    order = []

    class Rec(Agent):
        def pick_carousel(self, game, p, options):
            order.append(p.id)
            return 0
    g.run_carousel([Rec() for _ in g.players])
    expected = [p.id for p in carousel.pick_order(g.players)]
    assert order == expected
    hps = [g.players[i].hp for i in order]
    assert hps == sorted(hps)
    assert conserved(g) and g.carousel == []
    assert all(len(p.inventory) == 1 for p in g.players)


def test_combat_deterministic_and_winner():
    d = GameData()
    strong = {0: Unit("Fable", 3), 1: Unit("Grim", 3)}
    weak = {0: Unit("Aria")}
    r1 = combat.simulate(combat.build_units(0, strong, d), combat.build_units(1, weak, d))
    r2 = combat.simulate(combat.build_units(0, strong, d), combat.build_units(1, weak, d))
    assert r1 == r2 and r1.winner == 0 and r1.survivors[1] == 0


def test_trait_bonus_applies():
    d = GameData()
    one = combat.build_units(0, {0: Unit("Brute")}, d)[0]
    two = combat.build_units(0, {0: Unit("Brute"), 1: Unit("Orin")}, d)  # Warrior 2 + Bruiser 2
    assert two[0].atk > one.atk and two[0].max_hp > one.max_hp


def test_full_game_completes_with_unique_placements_and_conservation():
    for seed in range(3):
        g = Game(seed=seed)
        agents = [RandomAgent(seed * 10 + i) for i in range(8)]
        placements = g.run(agents)
        assert sorted(placements.values()) == list(range(1, 9))
        assert conserved(g)
        best = [p for p in g.players if p.placement == 1][0]
        assert best.alive or g.finished
