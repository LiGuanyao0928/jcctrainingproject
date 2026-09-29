import random
from tft_sim.data import BENCH_SIZE
from tft_sim.game import Agent, board_loc


class RandomAgent(Agent):
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def act(self, game, p):
        for _ in range(12):
            r = self.rng.random()
            if r < 0.45:
                game.buy(p, self.rng.randrange(5))
            elif r < 0.55:
                game.reroll(p)
            elif r < 0.65:
                game.buy_xp(p)
            elif r < 0.75:
                game.sell(p, self.rng.randrange(BENCH_SIZE + 28))
            elif r < 0.9:
                game.move(p, self.rng.randrange(BENCH_SIZE), board_loc(self.rng.randrange(28)))
            else:
                if p.inventory:
                    game.equip(p, 0, self.rng.randrange(BENCH_SIZE + 28))

    def pick_augment(self, game, p, options):
        return self.rng.randrange(len(options))

    def pick_carousel(self, game, p, options):
        return self.rng.randrange(len(options))
