"""V2 weighted 16x16 environment, BFS and Dijkstra tests."""

from __future__ import annotations

import numpy as np
import pytest

from environment import (
    ACTION_DELTAS,
    DOWN,
    LEFT,
    RIGHT,
    TERRAIN_1,
    TERRAIN_3,
    TERRAIN_5,
    UP,
    WALL,
    InvalidActionError,
    InvalidTerrainError,
    MazeEnvironment,
    bfs_shortest_path,
    bfs_shortest_steps,
    dijkstra_optimal_path,
    path_terrain_cost,
    terrain_map_hash,
    terrain_statistics,
    valid_actions_for,
)


def base_grid(value: int = TERRAIN_1) -> np.ndarray:
    return np.full((16, 16), value, dtype=np.int8)


def test_required_16x16_terrain_values() -> None:
    grid = base_grid()
    assert grid.shape == (16, 16)
    grid[0, 0] = WALL
    grid[0, 1] = TERRAIN_1
    grid[0, 2] = TERRAIN_3
    grid[0, 3] = TERRAIN_5
    assert set(np.unique(grid)).issubset({0, 1, 3, 5})


def test_invalid_terrain_values_are_rejected() -> None:
    grid = base_grid()
    grid[2, 2] = 2
    with pytest.raises(InvalidTerrainError):
        MazeEnvironment(grid, start=(0, 0), goal=(1, 1))


def test_legal_actions_respect_walls_and_boundaries() -> None:
    grid = base_grid()
    grid[2, 3] = WALL
    env = MazeEnvironment(grid, start=(2, 2), goal=(15, 15))
    actions = env.get_valid_actions((2, 2))
    assert RIGHT not in actions
    assert actions == [UP, DOWN, LEFT]
    assert env.get_valid_actions((0, 0)) == [DOWN, RIGHT]
    assert env.get_valid_actions((15, 15)) == [UP, LEFT]


def test_wall_and_out_of_bounds_steps_are_rejected_without_reward() -> None:
    grid = base_grid()
    grid[0, 1] = WALL
    env = MazeEnvironment(grid, start=(0, 0), goal=(15, 15))
    with pytest.raises(InvalidActionError):
        env.step(RIGHT)
    with pytest.raises(InvalidActionError):
        env.step(UP)
    with pytest.raises(InvalidActionError):
        env.step(99)
    assert env.agent_position == (0, 0)
    assert env.steps == 0
    assert env.cumulative_cost == 0


@pytest.mark.parametrize("terrain, expected_reward", [(1, -1.0), (3, -3.0), (5, -5.0)])
def test_entering_terrain_rewards(terrain: int, expected_reward: float) -> None:
    grid = base_grid()
    grid[0, 1] = terrain
    env = MazeEnvironment(grid, start=(0, 0), goal=(15, 15))
    result = env.step(RIGHT)
    assert result.position == (0, 1)
    assert result.movement_cost == terrain
    assert result.reward == expected_reward
    assert result.cumulative_cost == terrain
    assert result.done is False


def test_start_cost_is_not_charged_and_goal_terminates() -> None:
    grid = base_grid()
    grid[0, 0] = TERRAIN_5
    grid[0, 1] = TERRAIN_3
    grid[0, 2] = TERRAIN_5
    env = MazeEnvironment(grid, start=(0, 0), goal=(0, 2))
    first = env.step(RIGHT)
    assert first.cumulative_cost == 3
    second = env.step(RIGHT)
    assert second.cumulative_cost == 8
    assert second.done is True
    assert env.done is True
    with pytest.raises(RuntimeError):
        env.step(DOWN)
    assert env.reset() == (0, 0)
    assert env.cumulative_cost == 0

    path = [(0, 0), (0, 1), (0, 2)]
    assert path_terrain_cost(grid, path) == 8


def test_goal_cannot_be_wall_start_cannot_equal_goal() -> None:
    grid = base_grid()
    grid[0, 0] = WALL
    with pytest.raises(ValueError):
        MazeEnvironment(grid, start=(0, 0), goal=(1, 1))
    with pytest.raises(ValueError):
        MazeEnvironment(base_grid(), start=(1, 1), goal=(1, 1))


def test_bfs_ignores_terrain_costs_and_finds_fewest_steps() -> None:
    grid = base_grid()
    path = bfs_shortest_path(grid, (0, 0), (0, 5))
    assert path is not None
    assert len(path) - 1 == 5 == bfs_shortest_steps(grid, (0, 0), (0, 5))
    assert bfs_shortest_steps(grid, (0, 0), (0, 5)) == 5


def test_bfs_connectivity_and_barrier() -> None:
    grid = base_grid()
    grid[:, 8] = WALL
    assert bfs_shortest_steps(grid, (0, 0), (0, 15)) is None
    grid[0, 8] = TERRAIN_1
    assert bfs_shortest_steps(grid, (0, 0), (0, 15)) is not None


def test_dijkstra_minimises_weighted_cost_not_step_count() -> None:
    grid = base_grid()
    # Direct two-step route enters terrain 5 twice: cost 10.
    grid[8, 1] = TERRAIN_5
    grid[8, 2] = TERRAIN_5
    # Four-step detour along row 7 costs 1 + 1 + 1 + 5 = 8.
    start, goal = (8, 0), (8, 2)
    assert bfs_shortest_steps(grid, start, goal) == 2
    result = dijkstra_optimal_path(grid, start, goal)
    assert result.success
    assert result.path_cost is not None
    assert result.path_cost == pytest.approx(8.0)
    assert result.path_cost < 10.0
    assert result.plan_length == 4
    assert path_terrain_cost(grid, result.path) == 8


def test_dijkstra_unreachable_failure() -> None:
    grid = base_grid()
    grid[:, 7] = WALL
    result = dijkstra_optimal_path(grid, (0, 0), (0, 15))
    assert result.success is False
    assert result.path_cost is None


def test_trivial_dijkstra_and_bfs_same_cell() -> None:
    result = dijkstra_optimal_path(base_grid(), (2, 2), (2, 2))
    assert result.success
    assert result.path_cost == 0.0
    assert result.path == [(2, 2)]
    assert bfs_shortest_steps(base_grid(), (2, 2), (2, 2)) == 0


def test_terrain_statistics_and_layout_hash() -> None:
    grid = base_grid()
    grid[0, 0] = WALL
    grid[0, 1] = TERRAIN_3
    grid[0, 2] = TERRAIN_5
    counts, ratios = terrain_statistics(grid)
    assert sum(counts) == int(np.sum(grid != WALL))
    assert len(counts) == 3
    assert abs(sum(ratios) - 1.0) < 1e-8
    # The map hash identifies the terrain layout, independently of endpoints.
    assert terrain_map_hash(grid, (0, 0), (15, 15)) == terrain_map_hash(
        grid, (1, 1), (14, 14)
    )


def test_action_deltas_are_cardinal_and_fixed() -> None:
    assert ACTION_DELTAS == ((-1, 0), (1, 0), (0, -1), (0, 1))
    assert (UP, DOWN, LEFT, RIGHT) == (0, 1, 2, 3)
    assert valid_actions_for(base_grid(), (5, 5)) == [UP, DOWN, LEFT, RIGHT]