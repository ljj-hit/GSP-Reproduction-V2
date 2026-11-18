"""Relational object, predicate, action-object and graph-batching tests."""

from __future__ import annotations

import numpy as np
import torch

from config import Config
from environment import (
    MazeInstance,
    TERRAIN_1,
    TERRAIN_3,
    TERRAIN_5,
    WALL,
    bfs_shortest_steps,
    dijkstra_optimal_path,
    terrain_map_hash,
    terrain_statistics,
)
from relational_graph import (
    PREDICATE_ARITIES,
    PREDICATES,
    RelationalGraphBuilder,
    batch_relational_graphs,
    build_relational_graph,
    build_static_map_graph,
    remap_graph_objects,
)


def make_instance(instance_id: int = 0, *, with_walls: bool = True) -> MazeInstance:
    grid = np.full((16, 16), TERRAIN_1, dtype=np.int8)
    if with_walls:
        grid[2, 3] = WALL
        grid[2, 4] = WALL
        grid[10, 10] = TERRAIN_3
        grid[11, 11] = TERRAIN_5
        grid[5, 0:8] = WALL
        grid[5, 8] = TERRAIN_3
    start, goal = (0, 0), (15, 15)
    shortest = dijkstra_optimal_path(grid, start, goal)
    counts, ratios = terrain_statistics(grid)
    return MazeInstance(
        terrain_grid=grid,
        start=start,
        goal=goal,
        instance_id=instance_id,
        structure="test",
        seed=1,
        map_hash=terrain_map_hash(grid),
        bfs_steps=bfs_shortest_steps(grid, start, goal),
        optimal_cost=shortest.path_cost,
        terrain_counts=counts,
        terrain_ratios=ratios,
    )


def test_cell_objects_and_predicate_arities() -> None:
    instance = make_instance()
    graph = build_relational_graph(instance, instance.start)
    assert graph.num_cells == 256
    assert graph.cell_object_ids == tuple(range(256))
    assert graph.num_objects == 256 + len(graph.action_ids)
    assert set(graph.relations) == set(PREDICATES)
    assert PREDICATE_ARITIES["WALL"] == 1
    assert PREDICATE_ARITIES["CURRENT"] == 1
    assert PREDICATE_ARITIES["UP"] == 2
    assert PREDICATE_ARITIES["MOVE_UP"] == 3
    for predicate, facts in graph.relations.items():
        assert facts.shape[1] == PREDICATE_ARITIES[predicate]
    # Every cell receives exactly one terrain/WALL unary predicate.
    all_cells = np.concatenate(
        [
            graph.relations["WALL"].reshape(-1),
            graph.relations["TERRAIN_1"].reshape(-1),
            graph.relations["TERRAIN_3"].reshape(-1),
            graph.relations["TERRAIN_5"].reshape(-1),
        ]
    )
    assert len(all_cells) == 256
    assert set(int(value) for value in all_cells.tolist()) == set(range(256))
    assert len(graph.relations["CURRENT"]) == 1
    assert len(graph.relations["GOAL"]) == 1
    assert graph.current_cell_id == int(graph.relations["CURRENT"][0, 0])
    assert graph.goal_cell_id == 15 * 16 + 15


def test_spatial_relations_are_directed_and_include_walls() -> None:
    instance = make_instance()
    graph = build_relational_graph(instance, instance.start)
    # 16x16 directed cardinal adjacency count = 4*16*15 = 960.
    for predicate in ("UP", "DOWN", "LEFT", "RIGHT"):
        assert len(graph.relations[predicate]) == 15 * 16
    # Both direction facts exist for a normal pair.
    up_facts = {tuple(row) for row in graph.relations["UP"].tolist()}
    down_facts = {tuple(row) for row in graph.relations["DOWN"].tolist()}
    assert (1 * 16 + 1, 0 * 16 + 1) in up_facts
    assert (0 * 16 + 1, 1 * 16 + 1) in down_facts


def test_action_objects_only_for_legal_actions() -> None:
    instance = make_instance()
    builder = RelationalGraphBuilder(Config())
    corner_graph = builder.build(instance, (0, 0))
    assert corner_graph.action_ids == (1, 3)  # DOWN, RIGHT
    assert len(corner_graph.action_object_ids) == 2
    interior_graph = builder.build(instance, (7, 7))
    assert interior_graph.action_ids == (0, 1, 2, 3)
    assert len(interior_graph.action_object_ids) == 4
    wall_adjacent_graph = builder.build(instance, (1, 3))
    assert 1 not in wall_adjacent_graph.action_ids  # down enters a wall

    for action_object_id, action in zip(
        interior_graph.action_object_ids, interior_graph.action_ids
    ):
        predicate = f"MOVE_{('UP', 'DOWN', 'LEFT', 'RIGHT')[action]}"
        facts = interior_graph.relations[predicate]
        assert len(facts) == 1
        assert int(facts[0, 0]) == action_object_id
        assert int(facts[0, 1]) == interior_graph.current_cell_id


def test_current_changes_but_static_relations_are_reused() -> None:
    instance = make_instance()
    builder = RelationalGraphBuilder(Config())
    first = builder.build(instance, (0, 0))
    second = builder.build(instance, (7, 7))
    assert first is not second
    assert first.relations["WALL"] is second.relations["WALL"]
    assert first.relations["UP"] is second.relations["UP"]
    assert first.current_cell_id != second.current_cell_id
    # Dynamic cache returns the exact same graph object for an identical state.
    again = builder.build(instance, (7, 7))
    assert again is second
    stats = builder.cache_stats()
    assert stats["static_maps"] == 1
    assert stats["dynamic_hits"] >= 1


def test_static_goal_and_terrain_predicates() -> None:
    instance = make_instance()
    static = build_static_map_graph(instance)
    assert static.num_cells == 256
    assert int(static.relations["GOAL"][0, 0]) == 15 * 16 + 15
    assert len(static.relations["WALL"]) == int(np.sum(instance.terrain_grid == WALL))
    assert len(static.relations["TERRAIN_3"]) == int(np.sum(instance.terrain_grid == 3))


def test_disjoint_union_batch_has_no_cross_graph_edges() -> None:
    instance_a = make_instance(0)
    instance_b = make_instance(1)
    instance_b.terrain_grid = instance_b.terrain_grid.copy()
    instance_b.terrain_grid[7, 7] = TERRAIN_5
    instance_b.map_hash = terrain_map_hash(instance_b.terrain_grid)
    builder = RelationalGraphBuilder(Config())
    graph_a = builder.build(instance_a, (7, 7))
    graph_b = builder.build(instance_b, (0, 0))
    batch = batch_relational_graphs([graph_a, graph_b], device="cpu")
    counts = batch.graph_object_counts
    offsets = np.asarray([0, counts[0]], dtype=np.int64)
    for predicate, facts in batch.relations.items():
        if facts.numel() == 0:
            continue
        rows = facts.cpu().numpy()
        graph_ids = np.searchsorted(offsets, rows[:, 0], side="right") - 1
        for row, graph_id in zip(rows, graph_ids):
            start = offsets[graph_id]
            end = start + counts[graph_id]
            assert np.all((row >= start) & (row < end)), (predicate, row, start, end)
    # Actions retain per-graph identity and are not pooled into one fixed four.
    assert len(batch.graph_action_ids) == 2
    assert sum(len(v) for v in batch.graph_action_ids) == len(batch.action_ids)


def test_batch_action_lookup_and_max() -> None:
    instance = make_instance()
    builder = RelationalGraphBuilder(Config())
    graph_a = builder.build(instance, (0, 0))
    graph_b = builder.build(instance, (7, 7))
    batch = batch_relational_graphs([graph_a, graph_b], device="cpu")
    flat = batch.select_action_indices([graph_a.action_ids[0], graph_b.action_ids[-1]])
    assert flat.tolist() == [0, len(graph_a.action_ids) + len(graph_b.action_ids) - 1]
    q = torch.arange(len(batch.action_ids), dtype=torch.float32)
    maxima = batch.action_max_per_graph(q)
    assert maxima.tolist() == [
        len(graph_a.action_ids) - 1,
        len(batch.action_ids) - 1,
    ]


def test_variable_action_graphs_do_not_use_fixed_padding() -> None:
    instance = make_instance()
    builder = RelationalGraphBuilder(Config())
    graphs = [
        builder.build(instance, (0, 0)),
        builder.build(instance, (0, 15)),
        builder.build(instance, (7, 7)),
    ]
    counts = [len(graph.action_object_ids) for graph in graphs]
    assert counts == [2, 2, 4]
    batch = batch_relational_graphs(graphs, device="cpu")
    assert batch.graph_action_counts == (2, 2, 4)
    assert len(batch.action_ids) == 8


def test_remapping_object_ids_preserves_relation_structure() -> None:
    instance = make_instance()
    graph = build_relational_graph(instance, (7, 7))
    rng = np.random.default_rng(5)
    permutation = rng.permutation(graph.num_objects)
    remapped = remap_graph_objects(graph, permutation)
    assert remapped.num_objects == graph.num_objects
    assert set(remapped.action_ids) == set(graph.action_ids)
    for predicate in PREDICATES:
        original = graph.relations[predicate]
        expected = permutation[original]
        np.testing.assert_array_equal(remapped.relations[predicate], expected)