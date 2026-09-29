import collections

from bots.arena import play
from bots.rule_bots import EconBot, RandomBot, RerollBot, best_lineup, place
from tft_sim.game import Game, Unit
from test_core import conserved


def test_arrange_rules():
    g = Game(seed=1)
    p = g.players[0]
    p.level = 3
    a, b, c, d = Unit("Aria"), Unit("Brute"), Unit("Cinder"), Unit("Dax")
    p.bench[:4] = [a, b, c, d]
    assert not g.arrange(p, {0: a, 1: b, 2: c, 3: d})      # over board size
    assert not g.arrange(p, {0: a, 1: a})                   # duplicate unit
    assert not g.arrange(p, {0: Unit("Aria")})              # not this player's unit
    assert g.arrange(p, {0: a, 1: b, 2: c})
    assert set(p.board) == {0, 1, 2} and p.bench[0] is d and len(p.bench) == 9
    assert g.arrange(p, {5: d})
    assert [u for u in p.bench if u] == [a, b, c] and len(p.units()) == 4


def test_lineup_placement_front_melee_back_ranged():
    g = Game(seed=2)
    p = g.players[0]
    p.level = 4
    p.bench[:4] = [Unit("Aria"), Unit("Brute"), Unit("Elm"), Unit("Cinder")]
    layout = place(g, p, best_lineup(g, p))
    assert len(layout) == 4
    for slot, u in layout.items():
        rng = g.data.champions[u.name].range
        assert (slot // 7 < 2) == (rng <= 1)


def test_bots_finish_games_and_conserve_pool():
    for seed in range(4):
        g = Game(seed=seed)
        agents = [c(i) for i, c in enumerate([EconBot, RerollBot, RandomBot, EconBot] * 2)]
        placements = g.run(agents)
        assert sorted(placements.values()) == list(range(1, 9))
        assert conserved(g)


def test_rule_bots_beat_random_on_average():
    placed = collections.defaultdict(list)
    for seed in range(60):
        rows, _, _ = play(seed)
        for name, pl in rows:
            placed[name].append(pl)
    avg = {k: sum(v) / len(v) for k, v in placed.items()}
    assert avg["econ"] < avg["random"] - 1 and avg["reroll"] < avg["random"] - 0.5
    assert abs(avg["econ"] - avg["reroll"]) < 2.0  # neither strategy completely dominates
