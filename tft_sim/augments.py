"""Hextech augments: 3-choice picks at 2-1 (silver), 3-2 (gold), 4-2 (prismatic)."""
AUGMENTS = {
    "silver": [
        {"id": "s_gold", "name": "Pocket Change", "gold": 6},
        {"id": "s_xp", "name": "Quick Study", "xp": 8},
        {"id": "s_atk", "name": "Sharpened Blades", "buff": {"atk_pct": 0.10}},
        {"id": "s_hp", "name": "Thick Skin", "buff": {"hp_pct": 0.12}},
        {"id": "s_comp", "name": "Salvage", "components": 1},
    ],
    "gold": [
        {"id": "g_gold", "name": "Nest Egg", "gold": 12},
        {"id": "g_xp", "name": "Scholar", "xp": 16},
        {"id": "g_reroll", "name": "Free Refills", "free_rerolls": 4},
        {"id": "g_aspd", "name": "Quickdraw", "buff": {"aspd_pct": 0.15}},
        {"id": "g_comp", "name": "Armory", "components": 2},
    ],
    "prismatic": [
        {"id": "p_gold", "name": "Jackpot", "gold": 24},
        {"id": "p_level", "name": "Ascension", "xp": 40},
        {"id": "p_atk", "name": "Warlord", "buff": {"atk_pct": 0.25, "hp_pct": 0.15}},
        {"id": "p_ap", "name": "Arcane Surge", "buff": {"ap": 40, "armor": 10}},
        {"id": "p_comp", "name": "Forge", "components": 3},
    ],
}
AUGMENT_ROUNDS = {(2, 1): "silver", (3, 2): "gold", (4, 2): "prismatic"}
NUM_OFFERS = 3


def offer(rng, tier):
    return rng.sample(AUGMENTS[tier], NUM_OFFERS)
