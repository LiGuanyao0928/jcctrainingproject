"""Components combine pairwise into completed items whose stats are the sum of the parts."""
COMPONENTS = {
    "Sword": {"atk": 15},
    "Bow": {"aspd_pct": 0.15},
    "Rod": {"ap": 20},
    "Vest": {"armor": 20},
    "Belt": {"hp": 150},
    "Tear": {"mana": 15},
}
SEP = "+"


def is_component(item: str) -> bool:
    return item in COMPONENTS


def combine(a: str, b: str) -> str:
    if not (is_component(a) and is_component(b)):
        raise ValueError("can only combine two components")
    return SEP.join(sorted((a, b)))


def item_stats(item: str) -> dict:
    parts = [item] if is_component(item) else item.split(SEP)
    out = {}
    for p in parts:
        for k, v in COMPONENTS[p].items():
            out[k] = out.get(k, 0) + v
    return out
