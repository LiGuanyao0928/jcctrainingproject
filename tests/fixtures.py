"""A hand-written dump shaped like Community Dragon's TFT en_us.json (NOT real data; names are invented)."""
import json


def make_cdragon_dump(set_no=99):
    p = f"TFT{set_no}_"
    traits = ["Bruiser", "Mystic", "Gunner", "Shade", "Warden", "Sage"]
    champs = []
    i = 0
    for cost, n in ((1, 6), (2, 6), (3, 5), (4, 4), (5, 3)):
        for _ in range(n):
            t = [traits[i % 6], traits[(i + 2) % 6]]
            ab = {"name": "Skill", "variables": [
                {"name": "MagicDamage", "value": [0, 200 + 50 * cost, 300, 450, 0, 0, 0]},
                {"name": "Duration", "value": [0, 4, 4, 4, 0, 0, 0]}]}
            if i % 5 == 1:
                ab = {"name": "Barrier", "variables": [{"name": "ShieldAmount", "value": [0, 250, 300, 400, 0, 0, 0]}]}
            if i % 7 == 2:
                ab = {"name": "Mystery", "variables": [{"name": "Weird", "value": [0, 1, 1, 1]}]}
            champs.append({"apiName": f"{p}Champ{i}", "name": f"Champ{i}", "cost": cost, "traits": t,
                           "stats": {"hp": 500 + 100 * cost, "damage": 40 + 10 * cost, "armor": 25, "magicResist": 25,
                                     "attackSpeed": 0.7, "range": 1 + 3 * (i % 2), "mana": 80, "initialMana": 20,
                                     "critChance": 0.25, "critMultiplier": 1.4}, "ability": ab})
            i += 1
    champs += [{"apiName": f"{p}Dummy", "name": "Dummy", "cost": 1, "traits": ["Bruiser"], "stats": {"hp": 1}},
               {"apiName": "TFT1_Other", "name": "OtherSet", "cost": 1, "traits": ["Bruiser"], "stats": {"hp": 500}},
               {"apiName": f"{p}NoTraits", "name": "NoTraits", "cost": 2, "traits": [], "stats": {"hp": 500}}]
    trait_defs = [
        {"apiName": p + "Bruiser", "name": "Bruiser", "effects": [{"minUnits": 2, "variables": {"BonusHealth": 0.2}}, {"minUnits": 4, "variables": {"BonusHealth": 0.5}}]},
        {"apiName": p + "Mystic", "name": "Mystic", "effects": [{"minUnits": 2, "variables": {"BonusMR": 20}}, {"minUnits": 4, "variables": {"BonusMR": 50}}]},
        {"apiName": p + "Gunner", "name": "Gunner", "effects": [{"minUnits": 2, "variables": {"AS": 0.2}}, {"minUnits": 4, "variables": {"AS": 0.45}}]},
        {"apiName": p + "Shade", "name": "Shade", "effects": [{"minUnits": 2, "variables": {"Armor": 25, "CritChance": 0.1}}]},
        {"apiName": p + "Warden", "name": "Warden", "effects": [{"minUnits": 2, "variables": {"SpecialThing": 3}}]},
        {"apiName": p + "Sage", "name": "Sage", "effects": [{"minUnits": 3, "variables": {"AP": 25}}]},
    ]
    comps = {"TFT_Item_Sword": {"AD": 10}, "TFT_Item_Bow": {"AS": 0.15}, "TFT_Item_Rod": {"AP": 20},
             "TFT_Item_Vest": {"Armor": 20}, "TFT_Item_Cloak": {"MagicResist": 20}, "TFT_Item_Belt": {"Health": 150},
             "TFT_Item_Tear": {"Mana": 15}, "TFT_Item_Spatula": {}}
    items = [{"apiName": k, "effects": v, "composition": []} for k, v in comps.items()]
    items += [
        {"apiName": "TFT_Item_Edge", "effects": {"AD": 20, "AS": 0.3}, "composition": ["TFT_Item_Sword", "TFT_Item_Bow"]},
        {"apiName": "TFT_Item_Guard", "effects": {"Armor": 40, "Health": 300}, "composition": ["TFT_Item_Vest", "TFT_Item_Belt"]},
        {"apiName": "TFT_Item_Bane", "effects": {"MagicResist": 30, "AD": 10}, "composition": ["TFT_Item_Cloak", "TFT_Item_Sword"]},
        {"apiName": "TFT_Item_Weird", "effects": {"Thing": 1}, "composition": ["TFT_Item_Rod", "TFT_Item_Tear"]},
        {"apiName": "TFT_Item_Emblem", "effects": {}, "composition": ["TFT_Item_Spatula", "TFT_Item_Sword"]},
    ]
    return {"items": items, "setData": [{"number": set_no, "name": "Fixture", "champions": champs, "traits": trait_defs}]}


def write_set(tmp_path, set_no=99, rules=None):
    from tft_sim.season_import import convert
    out = tmp_path / "set"
    convert(make_cdragon_dump(set_no), set_no, out)
    if rules:
        (out / "rules.json").write_text(json.dumps(rules))
    return out
