"""Carousel: 9 champions each holding a component; lowest-HP players pick first."""
CAROUSEL_SIZE = 9
CAROUSEL_COSTS = {1: (1,), 2: (1, 2), 3: (2, 3), 4: (3, 4)}
CAROUSEL_COSTS_LATE = (4, 5)


def carousel_costs(stage):
    return CAROUSEL_COSTS.get(stage, CAROUSEL_COSTS_LATE)


def pick_order(players):
    """Alive players, lowest hp first; ties broken by id."""
    return sorted((p for p in players if p.alive), key=lambda p: (p.hp, p.id))
