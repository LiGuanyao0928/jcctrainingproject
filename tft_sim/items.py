"""Items: components combine pairwise into completed items. Data is replaceable (items.json).

items.json: {"components": {name: stats}, "completed": {name: {"from": [a, b], "stats": {...}}}}
A pair with no listed recipe still combines (name "A+B", stats summed), so the default set needs no recipe table.
Stat keys: atk, atk_pct, aspd_pct, ap, armor, mr, hp, hp_pct, mana, crit_chance.
"""
import json
from pathlib import Path

SEP = "+"
DEFAULT_PATH = Path(__file__).parent / "data" / "items.json"


class ItemBook:
    def __init__(self, components, completed=None):
        self.components = dict(components)
        self.completed = dict(completed or {})
        self.recipes = {frozenset(v["from"]): k for k, v in self.completed.items()}

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        d = json.loads(Path(path).read_text())
        return cls(d["components"], d.get("completed"))

    def is_component(self, item):
        return item in self.components

    def combine(self, a, b):
        if not (self.is_component(a) and self.is_component(b)):
            raise ValueError("can only combine two components")
        return self.recipes.get(frozenset((a, b))) or SEP.join(sorted((a, b)))

    def stats(self, item):
        if item in self.components:
            return dict(self.components[item])
        if item in self.completed:
            return dict(self.completed[item]["stats"])
        out = {}
        for part in item.split(SEP):
            for k, v in self.components[part].items():
                out[k] = out.get(k, 0) + v
        return out


_DEFAULT = ItemBook.load()
COMPONENTS = _DEFAULT.components


def is_component(item):
    return _DEFAULT.is_component(item)


def combine(a, b):
    return _DEFAULT.combine(a, b)


def item_stats(item):
    return _DEFAULT.stats(item)
