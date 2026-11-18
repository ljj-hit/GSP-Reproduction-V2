"""GSP-RGNN weighted state-action WA*, cost and return-bound tests."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

import pytest

from baselines import run_astar, run_dijkstra, run_weighted_astar
from config import Config
from environment import (
    MazeInstance,
    TERRAIN_1,
    TERRAIN_3,
    TERRAIN_5,
    bfs_shortest_steps,
    dijkstra_optimal_path,
    path_terrain_cost,
    terrain_map_hash,
    terrain_statistics,
)
from relational_graph import RelationalGraphBuilder
from search import QGuidedWAStar


class ActionPreferenceQ(nn.Module):
    """Fake R-GNN with deterministic per-action Q values."""

    def __init__(self, preferences: Tuple[float, float, float, float]) -> None:
        super().__init__()
        self.register_buffer("preferences", torch.tensor(preferences, dtype=torch.float32))
        self.forward_calls = 0

    def forward(self, batch) -> SimpleNamespace:
        self.forward_calls += 1
        q = self.preferences[batch.action_ids]
        return SimpleNamespace(q_values=q, action_ids=batch.action_ids)


def make_weighted_instance() -> MazeInstance:
    grid = np.full((16, 16), TERRAIN_1, dtype=np.int8)
    start, goal = (0, 0), (0, 2)
    grid[0, 1] = TERRAIN_3
    grid[0, 2] = TERRAIN_5
    shortest = dijkstra_optimal_path(grid, start, goal)
    counts, ratios = terrain_statistics(grid)
    return MazeInstance(
        terrain_grid=grid,
        start=start,
        goal=goal,
        instance_id=0,
        structure="test",
        seed=3,
        map_hash=terrain_map_hash(grid),
        bfs_steps=bfs_shortest_steps(grid, start, goal),
        optimal_cost=shortest.path_cost,
        terrain_counts=counts,
        terrain_ratios=ratios,
    )


def make_searcher(config: Config, preferences=(0.0, 0.0, 0.0, 10.0)):
    instance = make_weighted_instance()
    model = ActionPreferenceQ(preferences)
    builder = RelationalGraphBuilder(config)
    return QGuidedWAStar(model, builder, config, "cpu"), model, builder, instance


def test_state_action_wa_star_prefers_max_q_and_returns_weighted_path() -> None:
    config = Config()
    search, model, builder, instance = make_searcher(config)
    model.eval()
    result = search.search(instance)
    assert result.success
    assert result.actions == [3, 3]  # RIGHT, RIGHT
    assert result.path == [(0, 0), (0, 1), (0, 2)]
    assert result.path_cost == 8.0
    assert result.path_reward == -8.0
    assert instance.optimal_cost == 8.0
    # Root has DOWN/RIGHT (2 evaluations), state (0,1) has 3 legal actions.
    assert result.transition_evaluations == 5
    assert result.node_expansions == 2
    assert result.unique_state_expansions == 2
    assert result.reexpansions == 0
    assert result.q_evaluations == 2
    assert model.forward_calls == 2
    assert result.frontier_size_trace
    assert result.max_frontier_size == 4
    assert result.termination_reason == "goal"
    assert not model.training


def test_success_return_bounds_use_actual_remaining_weighted_reward() -> None:
    config = Config()
    search, _, _, instance = make_searcher(config)
    result = search.search(instance)
    assert result.success
    edge_map: Dict[Tuple[Tuple[int, int], int], object] = {
        (experience.position, experience.action): experience
        for experience in result.experiences
    }
    first = edge_map[((0, 0), 3)]
    second = edge_map[((0, 1), 3)]
    assert first.reward == -3.0
    assert second.reward == -5.0
    assert first.return_bound == -8.0
    assert second.return_bound == -5.0
    assert second.terminal is True
    assert second.dead_end is False
    assert second.goal == instance.goal
    assert second.instance_id == instance.instance_id
    assert all(exp.dead_end is False for exp in result.experiences)


def test_transition_budget_is_strict_and_failure_keeps_experiences() -> None:
    config = Config().copy_with(transition_budget=4)
    search, _, _, instance = make_searcher(config)
    result = search.search(instance)
    assert not result.success
    assert result.transition_evaluations <= 4
    assert result.transition_evaluations == 2
    assert result.termination_reason == "transition_budget"
    assert len(result.experiences) == 2
    assert result.path == []
    assert all(math.isinf(exp.return_bound) for exp in result.experiences)
    assert all(not exp.dead_end for exp in result.experiences)


def test_q_guidance_order_would_be_different_if_minimised() -> None:
    # The goal-directed action has the highest Q.  Choosing the lowest Q first
    # would move DOWN and require more than the direct two-step route.
    config = Config()
    search, _, _, instance = make_searcher(
        config, preferences=(0.0, -5.0, 0.0, 10.0)
    )
    result = search.search(instance)
    assert result.success
    assert result.actions == [3, 3]
    assert result.node_expansions == 2


def test_search_does_not_mark_visual_dead_ends_as_formal_dead_ends() -> None:
    config = Config()
    search, _, _, instance = make_searcher(config)
    result = search.search(instance)
    assert result.success
    assert result.dead_end_transitions == 0
    assert all(not exp.dead_end for exp in result.experiences)


def test_search_does_not_exceed_transition_budget_with_interior_actions() -> None:
    config = Config().copy_with(transition_budget=512)
    search, _, _, instance = make_searcher(config)
    result = search.search(instance)
    assert result.transition_evaluations <= 512
    assert len(result.experiences) == result.node_expansions


def test_traditional_algorithms_optimise_weighted_cost() -> None:
    config = Config()
    instance = make_weighted_instance()
    dijkstra = run_dijkstra(instance.terrain_grid, instance.start, instance.goal)
    astar = run_astar(
        instance.terrain_grid,
        instance.start,
        instance.goal,
        transition_budget=config.transition_budget,
    )
    wastar = run_weighted_astar(
        instance.terrain_grid,
        instance.start,
        instance.goal,
        weight=config.wa_star_weight,
        transition_budget=config.transition_budget,
    )
    assert dijkstra.success and dijkstra.path_cost == 8.0
    assert astar.success and astar.path_cost == 8.0
    assert wastar.success and wastar.path_cost is not None
    assert wastar.path_cost >= 8.0
    for result in (dijkstra, astar, wastar):
        assert path_terrain_cost(instance.terrain_grid, result.path) == pytest.approx(
            result.path_cost
        )
    assert astar.transition_evaluations <= config.transition_budget
    assert wastar.transition_evaluations <= config.transition_budget


def test_astar_budget_is_enforced() -> None:
    instance = make_weighted_instance()
    result = run_astar(
        instance.terrain_grid,
        instance.start,
        instance.goal,
        transition_budget=2,
    )
    assert result.transition_evaluations <= 2


def test_search_experience_reward_matches_entered_terrain() -> None:
    config = Config()
    search, _, _, instance = make_searcher(config)
    result = search.search(instance)
    for experience in result.experiences:
        assert experience.reward == -float(
            instance.terrain_grid[experience.next_position]
        )