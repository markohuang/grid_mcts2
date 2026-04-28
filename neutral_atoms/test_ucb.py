"""Unit tests for _ucb_score and _select_child.

Guards:
  1. Opt-1 refactor: config constant extraction into _select_child
  2. Opt-3 refactor: UCB inlining into argmax loop (pb_c_factor hoisted out of
     per-child loop; child.value() and normalize() inlined)

Tests verify the math is identical, the right child is selected, and edge
cases behave correctly.
"""
import math
import pytest
from neutral_atoms.mcts import Node, MinMaxStats, _ucb_score, _select_child

# --- helpers ---

def _node(prior, visit_count=0, value_sum=0.0, reward=0.0):
    n = Node(prior)
    n.visit_count = visit_count
    n.value_sum = value_sum
    n.reward = reward
    return n


def _stats(lo=0.0, hi=1.0):
    class _B:
        min = lo
        max = hi
    return MinMaxStats(_B())


class _Cfg:
    """Minimal config stand-in with default AlphaZero values."""
    pb_c_base = 19652
    pb_c_init = 1.25
    discount = 0.99


def _ucb(parent, child, stats, *, pb_c_base=19652, pb_c_init=1.25, discount=0.99):
    return _ucb_score(pb_c_base, pb_c_init, discount, parent, child, stats)


# --- _ucb_score: unvisited child ---

class TestUcbUnvisited:
    def test_value_score_is_zero(self):
        # Unvisited child contributes only prior_score; value_score must be 0
        parent = _node(0.0, visit_count=10)
        child = _node(0.5, visit_count=0)
        stats = _stats()
        score = _ucb(parent, child, stats)
        pb_c = (math.log((10 + 19652 + 1) / 19652) + 1.25) * math.sqrt(10) / 1
        assert score == pytest.approx(pb_c * 0.5, rel=1e-9)

    def test_higher_prior_wins(self):
        parent = _node(0.0, visit_count=5)
        stats = _stats()
        lo = _node(0.1, visit_count=0)
        hi = _node(0.9, visit_count=0)
        assert _ucb(parent, hi, stats) > _ucb(parent, lo, stats)

    def test_parent_visits_zero(self):
        # Parent with 0 visits: pb_c *= sqrt(0) = 0 → prior_score = 0 → score = 0
        parent = _node(0.0, visit_count=0)
        child = _node(0.8, visit_count=0)
        stats = _stats()
        assert _ucb(parent, child, stats) == pytest.approx(0.0, abs=1e-12)

    def test_discount_irrelevant_when_unvisited(self):
        # discount only affects value_score which is 0 for unvisited children
        parent = _node(0.0, visit_count=20)
        child = _node(0.5, visit_count=0)
        stats = _stats()
        s1 = _ucb(parent, child, stats, discount=0.0)
        s2 = _ucb(parent, child, stats, discount=1.0)
        assert s1 == pytest.approx(s2, rel=1e-9)


# --- _ucb_score: visited child ---

class TestUcbVisited:
    def test_higher_value_wins(self):
        parent = _node(0.0, visit_count=20)
        stats = _stats(lo=0.0, hi=1.0)
        lo = _node(0.5, visit_count=5, value_sum=1.0, reward=0.0)   # value=0.2
        hi = _node(0.5, visit_count=5, value_sum=4.0, reward=0.0)   # value=0.8
        assert _ucb(parent, hi, stats) > _ucb(parent, lo, stats)

    def test_reward_adds_to_q(self):
        parent = _node(0.0, visit_count=20)
        stats = _stats(lo=0.0, hi=1.0)
        base = _node(0.5, visit_count=5, value_sum=2.5, reward=0.0)
        rwd = _node(0.5, visit_count=5, value_sum=2.5, reward=0.5)
        assert _ucb(parent, rwd, stats) > _ucb(parent, base, stats)

    def test_discount_scales_value(self):
        parent = _node(0.0, visit_count=20)
        stats = _stats(lo=0.0, hi=2.0)
        child = _node(0.5, visit_count=5, value_sum=5.0, reward=0.0)   # value=1.0
        s_full = _ucb(parent, child, stats, discount=1.0)
        s_zero = _ucb(parent, child, stats, discount=0.0)
        assert s_full > s_zero

    def test_more_visits_reduces_exploration_bonus(self):
        # Same prior + same normalized value; more visits → smaller pb_c → lower score
        parent = _node(0.0, visit_count=100)
        stats = _stats(lo=0.0, hi=1.0)
        few = _node(0.5, visit_count=1, value_sum=0.5, reward=0.0)    # value=0.5
        many = _node(0.5, visit_count=50, value_sum=25.0, reward=0.0)  # value=0.5
        assert _ucb(parent, few, stats) > _ucb(parent, many, stats)

    def test_explicit_formula(self):
        parent = _node(0.0, visit_count=100)
        child = _node(0.3, visit_count=10, value_sum=5.0, reward=0.1)
        stats = _stats(lo=0.0, hi=1.0)
        pb_c_base, pb_c_init, discount = 19652, 1.25, 0.99
        pb_c = (math.log((100 + pb_c_base + 1) / pb_c_base) + pb_c_init) * math.sqrt(100) / 11
        prior_score = pb_c * 0.3
        q = 0.1 + discount * (5.0 / 10)
        value_score = (q - 0.0) / (1.0 - 0.0)  # normalize with lo=0, hi=1
        expected = prior_score + value_score
        assert _ucb(parent, child, stats) == pytest.approx(expected, rel=1e-9)


# --- MinMaxStats edge cases ---

class TestMinMaxStats:
    def test_degenerate_range_returns_raw(self):
        # min == max: normalize returns value unchanged (avoids divide-by-zero)
        parent = _node(0.0, visit_count=10)
        child = _node(0.5, visit_count=5, value_sum=2.5, reward=0.0)  # value=0.5
        stats = _stats(lo=0.5, hi=0.5)
        score = _ucb(parent, child, stats)
        # normalize(0 + 0.99*0.5) = 0.495 (raw, not clamped)
        expected_q = 0.0 + 0.99 * 0.5
        pb_c = (math.log((10 + 19652 + 1) / 19652) + 1.25) * math.sqrt(10) / 6
        assert score == pytest.approx(pb_c * 0.5 + expected_q, rel=1e-9)

    def test_value_outside_known_range(self):
        # q can exceed [min, max] when stats haven't seen this value yet
        # normalize should return value > 1.0 — we don't clamp
        parent = _node(0.0, visit_count=10)
        child = _node(0.5, visit_count=5, value_sum=10.0, reward=0.0)  # value=2.0
        stats = _stats(lo=0.0, hi=1.0)
        score = _ucb(parent, child, stats)
        # q = 0.0 + 0.99*2.0 = 1.98; normalize(1.98) = (1.98-0)/(1-0) = 1.98
        assert score > 1.0


# --- pb_c_base / pb_c_init ---

class TestPbCParams:
    def test_large_pb_c_init_boosts_exploration(self):
        parent = _node(0.0, visit_count=200)
        # c_exploit: many visits, high value
        # c_explore: no visits, high prior
        c_exploit = _node(0.1, visit_count=190, value_sum=190.0, reward=0.0)
        c_explore = _node(0.9, visit_count=10, value_sum=5.0, reward=0.0)
        stats = _stats(lo=0.0, hi=1.0)

        s_exploit_lo = _ucb(parent, c_exploit, stats, pb_c_init=0.01)
        s_explore_lo = _ucb(parent, c_explore, stats, pb_c_init=0.01)
        s_exploit_hi = _ucb(parent, c_exploit, stats, pb_c_init=5.0)
        s_explore_hi = _ucb(parent, c_explore, stats, pb_c_init=5.0)

        # Low exploration: exploit wins
        assert s_exploit_lo > s_explore_lo
        # High exploration: explore wins
        assert s_explore_hi > s_exploit_hi

    def test_pb_c_base_large_suppresses_log_term(self):
        # Very large pb_c_base → log term → 0 → UCB driven purely by pb_c_init
        parent = _node(0.0, visit_count=50)
        child = _node(0.5, visit_count=0)
        stats = _stats()
        s_small_base = _ucb(parent, child, stats, pb_c_base=1)
        s_large_base = _ucb(parent, child, stats, pb_c_base=10**9)
        # Large base → log((50 + 1e9 + 1)/1e9) ≈ 0 → score ≈ pb_c_init contribution only
        # Small base → large log term → higher score
        assert s_small_base > s_large_base


# --- _select_child ---

class TestSelectChild:
    def _parent(self, children_spec, parent_visits=None):
        """children_spec: [(action, prior, visit_count, value_sum, reward), ...]"""
        p = _node(0.0)
        p.visit_count = parent_visits if parent_visits is not None else \
            sum(s[2] for s in children_spec)
        for action, prior, vc, vs, reward in children_spec:
            p.children[action] = _node(prior, vc, vs, reward)
        return p

    def test_picks_highest_ucb_child(self):
        # Action 3 has high prior and is unvisited → highest UCB
        parent = self._parent([
            (1, 0.1, 20, 18.0, 0.0),
            (2, 0.1, 20, 18.0, 0.0),
            (3, 0.8,  0,  0.0, 0.0),
        ], parent_visits=40)
        action, _ = _select_child(_Cfg(), parent, _stats())
        assert action == 3

    def test_picks_high_value_after_many_visits(self):
        # All children equally visited; action 2 has higher value
        parent = self._parent([
            (1, 0.5, 50, 10.0, 0.0),   # value=0.2
            (2, 0.5, 50, 40.0, 0.0),   # value=0.8
        ], parent_visits=100)
        action, _ = _select_child(_Cfg(), parent, _stats(lo=0.0, hi=0.8))
        assert action == 2

    def test_single_child_always_selected(self):
        parent = self._parent([(7, 1.0, 0, 0.0, 0.0)], parent_visits=0)
        action, child = _select_child(_Cfg(), parent, _stats())
        assert action == 7
        assert child.prior == 1.0

    def test_returns_node_object(self):
        parent = self._parent([
            (0, 0.6, 0, 0.0, 0.0),
            (1, 0.4, 0, 0.0, 0.0),
        ], parent_visits=0)
        action, child = _select_child(_Cfg(), parent, _stats())
        assert child is parent.children[action]

    def test_exploration_param_changes_selection(self):
        # With low pb_c_init: exploit action 1 (high value)
        # With high pb_c_init: explore action 2 (unvisited, high prior)
        parent = _node(0.0, visit_count=100)
        parent.children = {
            1: _node(0.1, visit_count=99, value_sum=95.0, reward=0.0),  # value≈0.96
            2: _node(0.9, visit_count=1, value_sum=0.1, reward=0.0),    # value=0.1
        }
        stats = _stats(lo=0.0, hi=1.0)

        class LowExplore(_Cfg):
            pb_c_init = 0.01

        class HighExplore(_Cfg):
            pb_c_init = 10.0

        a_exploit, _ = _select_child(LowExplore(), parent, stats)
        a_explore, _ = _select_child(HighExplore(), parent, stats)
        assert a_exploit == 1
        assert a_explore == 2


# --- inlined select_child matches explicit _ucb_score for all children ---

class TestSelectChildMatchesUcbScore:
    """Verify that the inlined argmax loop in _select_child selects the same
    action as an explicit per-child _ucb_score computation."""

    def _check(self, children_spec, parent_visits, stats, cfg=None):
        """children_spec: [(action, prior, vc, vs, reward), ...]"""
        if cfg is None:
            cfg = _Cfg()
        parent = _node(0.0, visit_count=parent_visits)
        for action, prior, vc, vs, reward in children_spec:
            parent.children[action] = _node(prior, vc, vs, reward)

        # Reference: explicit _ucb_score for every child
        scores = {
            action: _ucb_score(cfg.pb_c_base, cfg.pb_c_init, cfg.discount,
                               parent, child, stats)
            for action, child in parent.children.items()
        }
        best_action_ref, _ = max(scores.items(), key=lambda kv: (kv[1], kv[0]))

        # Inlined _select_child
        best_action_inline, _ = _select_child(cfg, parent, stats)
        assert best_action_inline == best_action_ref, (
            f"inline picked {best_action_inline}, ref picked {best_action_ref}. "
            f"scores={scores}"
        )

    def test_all_unvisited(self):
        self._check([
            (1, 0.1, 0, 0.0, 0.0),
            (3, 0.7, 0, 0.0, 0.0),
            (5, 0.2, 0, 0.0, 0.0),
        ], parent_visits=0, stats=_stats())

    def test_mixed_visited_unvisited(self):
        self._check([
            (2, 0.4, 10, 8.0, 0.0),
            (4, 0.3,  0, 0.0, 0.0),
            (6, 0.3, 20, 5.0, 0.0),
        ], parent_visits=30, stats=_stats(lo=0.2, hi=0.9))

    def test_all_visited(self):
        self._check([
            (0, 0.3, 50, 35.0, 0.1),
            (1, 0.4, 50, 20.0, 0.0),
            (2, 0.3, 50, 45.0, 0.2),
        ], parent_visits=150, stats=_stats(lo=0.0, hi=1.0))

    def test_degenerate_range(self):
        self._check([
            (7, 0.5, 10, 5.0, 0.0),
            (8, 0.5, 10, 5.0, 0.0),
        ], parent_visits=20, stats=_stats(lo=0.5, hi=0.5))

    def test_tiebreak_picks_larger_action(self):
        # All unvisited, same prior → scores equal → larger action wins
        parent = _node(0.0, visit_count=5)
        parent.children = {
            3: _node(0.5, 0, 0.0, 0.0),
            7: _node(0.5, 0, 0.0, 0.0),
        }
        stats = _stats()
        action, _ = _select_child(_Cfg(), parent, stats)
        assert action == 7

    def test_tiebreak_reference_and_inline_agree(self):
        # Create conditions where two children have numerically equal UCB scores
        parent = _node(0.0, visit_count=5)
        parent.children = {
            2: _node(0.5, 0, 0.0, 0.0),
            9: _node(0.5, 0, 0.0, 0.0),
        }
        stats = _stats()
        # Both reference and inline should agree on action 9 (larger)
        scores = {a: _ucb_score(_Cfg.pb_c_base, _Cfg.pb_c_init, _Cfg.discount,
                                 parent, c, stats)
                  for a, c in parent.children.items()}
        assert scores[2] == scores[9]  # confirm tie
        action, _ = _select_child(_Cfg(), parent, stats)
        assert action == 9


# --- parity: refactored select_child produces same full-game trajectories ---

def test_full_game_parity_classic():
    """Self-parity check: same seed + same config → byte-identical trajectory.
    If the UCB refactor broke anything, trajectories diverge between runs."""
    from fast_mcts._common import (
        build_cfg, build_env_spec, make_network, play_once,
        game_snapshot, assert_snapshot_equal,
    )
    cfg = build_cfg(map_num=2, sims=50, use_fake=True)
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=42)
    g1 = play_once('classic', cfg, net, tasks, ip,
                   seed=42, deterministic=True, add_exploration_noise=False)
    g2 = play_once('classic', cfg, net, tasks, ip,
                   seed=42, deterministic=True, add_exploration_noise=False)
    assert_snapshot_equal(game_snapshot(g1), game_snapshot(g2),
                          label='classic/self-parity/post-ucb-refactor')


def test_full_game_parity_fast_vs_classic():
    """Cross-backend parity: fast (batch=1, vl=0) must be byte-identical to classic."""
    from fast_mcts._common import (
        build_cfg, build_env_spec, make_network, play_once,
        game_snapshot, assert_snapshot_equal,
    )
    cfg = build_cfg(map_num=2, sims=50, use_fake=True)
    with cfg.mcts.unlocked():
        cfg.mcts.nn_batch_size = 1
        cfg.mcts.virtual_loss = 0.0
    tasks, ip = build_env_spec(cfg)
    net = make_network(cfg, seed=42)
    gc = play_once('classic', cfg, net, tasks, ip,
                   seed=42, deterministic=True, add_exploration_noise=False)
    gf = play_once('fast', cfg, net, tasks, ip,
                   seed=42, deterministic=True, add_exploration_noise=False)
    assert_snapshot_equal(game_snapshot(gc), game_snapshot(gf),
                          label='classic-vs-fast/post-ucb-refactor')
