import json

import numpy as np
import pytest

from bots.rule_bots import EconBot, RandomBot, RerollBot
from fixtures import make_cdragon_dump, write_set
from rl.env import N_ACTIONS, TFTEnv
from tft_sim import combat
from tft_sim.combat import CUnit, simulate
from tft_sim.data import DEFAULT_RULES, GameData, Rules
from tft_sim.game import Game, Unit
from tft_sim.items import ItemBook
from tft_sim.season_import import convert, map_stat, pick_set


# ---------- importer ----------
def test_pick_set_newest_and_errors():
    d = make_cdragon_dump(99)
    assert pick_set(d)[0] == 99 and pick_set(d, 99)[0] == 99
    with pytest.raises(ValueError, match="not in dump"):
        pick_set(d, 5)
    with pytest.raises(ValueError):
        pick_set({})
    d["sets"] = {"100": d["setData"][0]}  # also understands the dict-style "sets"
    assert pick_set(d)[0] == 100


def test_map_stat_rules():
    assert map_stat("AD", 0.1) == ("atk_pct", 0.1) and map_stat("AD", 10) == ("atk", 10)
    assert map_stat("BonusHealth", 0.2) == ("hp_pct", 0.2) and map_stat("Health", 150) == ("hp", 150)
    assert map_stat("MagicResist", 20) == ("mr", 20) and map_stat("AP", 25) == ("ap", 25)
    assert map_stat("SpecialThing", 3) is None and map_stat("AD", 0) is None and map_stat("AS", "x") is None


def test_convert_champions_traits_items(tmp_path):
    no, champs, traits, items, report = convert(make_cdragon_dump(), 99, tmp_path)
    names = {c["name"] for c in champs}
    assert len(champs) == 24 and all(n.startswith("TFT99_") for n in names)
    assert "TFT99_Dummy" in report["champions_skipped"] and "TFT99_NoTraits" in report["champions_skipped"]
    c0 = next(c for c in champs if c["name"] == "TFT99_Champ0")
    assert c0["skill"] == {"type": "damage", "value": 250.0, "damage_type": "magic"}
    assert next(c for c in champs if c["name"] == "TFT99_Champ1")["skill"]["type"] == "shield"
    assert "TFT99_Champ2" in report["skills_default"]
    assert traits["Bruiser"] == {"2": {"hp_pct": 0.2}, "4": {"hp_pct": 0.5}}
    assert traits["Shade"]["2"] == {"armor": 25, "crit_chance": 0.1}
    assert traits["Warden"]["2"] == {} and "Warden 2" in report["trait_no_effect"]
    assert "TFT_Item_Spatula" not in items["components"] and len(items["components"]) == 7
    assert set(items["completed"]) == {"TFT_Item_Edge", "TFT_Item_Guard", "TFT_Item_Bane"}
    assert "TFT_Item_Weird" in report["items_special"]
    assert (tmp_path / "IMPORT_REPORT.md").read_text().startswith("# Import report: set 99")


def test_imported_set_loads_and_runs_full_games(tmp_path):
    out = write_set(tmp_path)
    data = GameData(out)
    assert len(data.champions) == 24 and data.rules.max_level == 9  # rules.json falls back to defaults
    assert "TFT_Item_Edge" in data.items.completed and data.augments["silver"]
    for seed in range(2):
        g = Game(seed=seed, data=data)
        agents = [c(i) for i, c in enumerate([EconBot, RerollBot, RandomBot, EconBot] * 2)]
        placements = g.run(agents)
        assert sorted(placements.values()) == list(range(1, 9))
        assert all(g.copies(n) == data.pool_size(n) for n in data.champions)


def test_env_on_imported_set_with_level_ten(tmp_path):
    rules = json.loads(open("tft_sim/data/rules.json").read())
    rules["max_level"] = 10
    rules["xp_to_level"]["10"] = 84
    rules["shop_odds"]["10"] = [5, 10, 20, 40, 25]
    out = write_set(tmp_path, rules=rules)
    env = TFTEnv(seed=1, set_dir=out)
    assert env.action_space.n == N_ACTIONS and env.trait_names == sorted(GameData(out).traits)
    o, info = env.reset()
    env.me.level, env.me.xp = 10, 0
    assert env.game.rules.odds(10) == (5, 10, 20, 40, 25)
    rng = np.random.default_rng(0)
    done, n = False, 0
    while not done and n < 400:
        o, r, done, _, info = env.step(rng.choice(np.flatnonzero(info["action_mask"])))
        assert o.shape == env.observation_space.shape and np.isfinite(o).all()
        n += 1
    assert env.game.rules.xp_to_level[10] == 84


# ---------- generic mechanics ----------
def cu(side, slot, hp=1000, atk=100, armor=0, mr=0, **kw):
    base = dict(side=side, name="x", slot=slot, hp=hp, atk=atk, armor=armor, mr=mr, interval=10, range=1,
                max_mana=999, mana=0, skill_type="damage", skill_value=0)
    base.update(kw)
    return CUnit(**base)


def test_magic_skill_uses_mr_physical_uses_armor_true_ignores():
    def dealt(dmg_type, armor, mr):
        a = cu(0, 0, atk=0, max_mana=10, mana=10, skill_value=200, skill_dmg=dmg_type)
        b = cu(1, 0, hp=10_000, armor=armor, mr=mr, atk=0)
        simulate([a], [b], max_ticks=12)
        return 10_000 - b.hp
    assert dealt("magic", 100, 0) == pytest.approx(200)
    assert dealt("magic", 0, 100) == pytest.approx(100)
    assert dealt("physical", 100, 0) == pytest.approx(100)
    assert dealt("true", 100, 100) == pytest.approx(200)


def test_crit_is_expected_value_and_deterministic():
    def dealt(crit):
        a = cu(0, 0, atk=100, crit_avg=1 + crit * 0.4)
        b = cu(1, 0, hp=10_000, atk=0)
        simulate([a], [b], max_ticks=10)
        return 10_000 - b.hp
    assert dealt(0.0) == pytest.approx(100) and dealt(0.5) == pytest.approx(120)


def test_buff_as_and_stun_skills():
    def dealt(**kw):
        a = cu(0, 0, atk=100, max_mana=1000, mana=995, **kw)  # exactly one cast, on the first attack
        b = cu(1, 0, hp=1_000_000, atk=0)
        simulate([a], [b], max_ticks=100)
        return 1_000_000 - b.hp, a
    plain, _ = dealt(skill_type="damage", skill_value=0)
    buffed, unit = dealt(skill_type="buff_as", skill_extra=1.0)
    assert buffed > plain * 1.3               # faster attacks while buffed (50 ticks at double speed)
    assert unit.interval == unit.base_interval  # buff wore off
    s = cu(0, 0, atk=0, max_mana=10, mana=10, skill_type="stun", skill_value=0, skill_extra=30)
    t = cu(1, 0, hp=1000, atk=100)
    simulate([s], [t], max_ticks=12)
    assert t.cd >= 30 - 2


def test_trait_and_item_effects_include_mr_and_crit():
    d = GameData()
    book = ItemBook({"Cloak": {"mr": 20}, "Gem": {"crit_chance": 0.2}}, {"Shield+": {"from": ["Cloak", "Gem"], "stats": {"mr": 50}}})
    d.items = book
    base = combat.build_units(0, {0: Unit("Aria")}, d)[0]
    geared = combat.build_units(0, {0: Unit("Aria", items=["Cloak", "Gem"])}, d)[0]
    assert geared.mr == base.mr + 20 and geared.crit_avg == pytest.approx(1 + 0.2 * 0.4)
    assert combat.build_units(0, {0: Unit("Aria", items=["Shield+"])}, d)[0].mr == base.mr + 50


def test_item_recipes_and_fallback():
    b = ItemBook({"A": {"atk": 1}, "B": {"hp": 5}}, {"AB_Special": {"from": ["A", "B"], "stats": {"atk": 9}}})
    assert b.combine("B", "A") == "AB_Special" and b.stats("AB_Special") == {"atk": 9}
    b2 = ItemBook({"A": {"atk": 1}, "B": {"hp": 5}})
    assert b2.combine("A", "B") == "A+B" and b2.stats("A+B") == {"atk": 1, "hp": 5}


def test_rules_from_data():
    r = DEFAULT_RULES
    assert r.streak_bonus(5) == 3 and r.streak_bonus(3) == 1 and r.streak_bonus(1) == 0
    assert r.stage_damage(3) == 3 and r.stage_damage(9) == 15
    assert r.odds(12) == r.odds(9)  # levels above the table reuse the last row
    custom = Rules.from_dict({**json.loads(open("tft_sim/data/rules.json").read()), "interest_cap": 2, "reroll_cost": 1})
    assert custom.interest_cap == 2 and custom.reroll_cost == 1


def test_dataset_identity_mapping_for_imported_ids(tmp_path):
    from imitation import dataset as ds
    recs = [ds.normalize_record({"placement": 1, "units": ["TFT99_Champ0", {"id": "TFT99_Champ1", "tier": 2}, "Other"]})]
    meta = ds.build_dataset(recs, out_dir=tmp_path, name="d", sim_names=["TFT99_Champ0", "TFT99_Champ1"])
    z = np.load(tmp_path / "d.npz")
    assert list(z["sim_units"][0]) == [1.0, 2.0] and meta["sim_unit_coverage"] == pytest.approx(2 / 3, abs=1e-3)
