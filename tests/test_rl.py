import numpy as np
import torch

from rl.env import EVAL_LINEUP, N_ACTIONS, END, TFTEnv
from rl.ppo import ActorCritic, gae
from rl.teacher import teacher_action


def rollout(env, policy):
    o, info = env.reset()
    done, n = False, 0
    while not done:
        m = info["action_mask"]
        assert m.any() and m.shape == (N_ACTIONS,)
        o, r, done, trunc, info = env.step(policy(env, m))
        assert o.shape == env.observation_space.shape and np.isfinite(o).all() and np.isfinite(r)
        n += 1
    return info, n


def test_random_legal_actions_never_fail_and_episode_ends():
    rng = np.random.default_rng(0)
    for auto in (True, False):
        env = TFTEnv(seed=3, auto_arrange=auto)
        info, n = rollout(env, lambda e, m: rng.choice(np.flatnonzero(m)))
        assert 1 <= info["placement"] <= 8 and n > 10


def test_illegal_action_rejected():
    env = TFTEnv(seed=1)
    env.reset()
    illegal = int(np.flatnonzero(~env.action_masks())[0])
    try:
        env.step(illegal)
    except AssertionError:
        return
    raise AssertionError("illegal action was accepted")


def test_phases_augment_and_carousel_masks():
    env = TFTEnv(seed=2, lineup=EVAL_LINEUP)
    o, info = env.reset()
    assert env.phase == "car" and info["action_mask"].sum() == len(env.game.carousel)
    seen = set()
    done = False
    while not done and (env.game.stage, env.game.rnd) <= (2, 1):
        seen.add(env.phase)
        o, r, done, _, info = env.step(teacher_action(env, info["action_mask"]))
    assert {"car", "plan", "aug"} <= seen


def test_teacher_beats_random_policy():
    rng = np.random.default_rng(0)
    t = np.mean([rollout(TFTEnv(seed=s, lineup=EVAL_LINEUP), teacher_action)[0]["placement"] for s in range(6)])
    r = np.mean([rollout(TFTEnv(seed=s, lineup=EVAL_LINEUP), lambda e, m: rng.choice(np.flatnonzero(m)))[0]["placement"]
                 for s in range(6)])
    assert t < r - 2


def test_masked_policy_never_samples_illegal():
    net = ActorCritic(10, N_ACTIONS, hidden=16)
    obs = np.random.randn(64, 10).astype(np.float32)
    mask = np.random.rand(64, N_ACTIONS) < 0.2
    mask[:, END] = True
    a, logp, v = net.act(obs, mask)
    assert mask[np.arange(64), a].all() and np.isfinite(logp).all()


def test_gae_terminal_and_bootstrap():
    r = np.array([0.0, 1.0], np.float32)
    v = np.array([0.5, 0.5], np.float32)
    adv, ret = gae(r, v, np.array([0.0, 1.0], np.float32), last_value=9.0, gamma=1.0, lam=1.0)
    assert np.allclose(ret, [1.0, 1.0])  # done at t=1 cuts the bootstrap
    adv, ret = gae(r, v, np.zeros(2, np.float32), last_value=2.0, gamma=1.0, lam=1.0)
    assert np.allclose(ret, [3.0, 3.0])
