"""Five-structure dataset generation and validity tests for V2."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pytest

from config import Config
from environment import (
    ACTION_DELTAS,
    WALL,
    bfs_shortest_steps,
    dijkstra_optimal_path,
    terrain_map_hash,
    terrain_statistics,
)
from generator import (
    GENERATORS,
    STRUCTURES,
    StructuredMaze,
    generate_branching_maze,
    generate_datasets,
    generate_loop_maze,
    generate_random_obstacle,
    generate_rooms_corridors,
    generate_structure_sample,
    generate_u_detour,
    load_dataset,
    load_manifest,
)


def _small_config(root: Path) -> Config:
    return Config().copy_with(
        num_train_instances=20,
        num_validation_instances=10,
        num_test_instances=10,
        data_dir=str(root / "data"),
        checkpoint_dir=str(root / "checkpoints"),
        log_dir=str(root / "logs"),
        results_dir=str(root / "results"),
        figures_dir=str(root / "figures"),
        graph_cache_size=2000,
    )


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("v2_generator")
    config = _small_config(root)
    manifest = generate_datasets(config, force=True)
    splits = {
        name: load_dataset(path)[0]
        for name, path in config.split_paths().items()
    }
    return config, manifest, splits


def _successful_structure(
    generator: Callable,
    seed: int,
    attempts: int = 300,
) -> StructuredMaze:
    config = Config()
    rng = np.random.default_rng(seed)
    for index in range(attempts):
        maze = generator(config, rng)
        if maze is not None:
            return maze
    raise AssertionError(f"Could not generate structure after {attempts} attempts")


def _edge_count(binary: np.ndarray) -> int:
    edges = 0
    h, w = binary.shape
    for r in range(h):
        for c in range(w):
            if binary[r, c] == 0:
                continue
            if r + 1 < h and binary[r + 1, c] == 1:
                edges += 1
            if c + 1 < w and binary[r, c + 1] == 1:
                edges += 1
    return edges


def test_default_balanced_required_counts() -> None:
    config = Config()
    assert config.height == config.width == 16
    assert config.num_train_instances == 1000
    assert config.num_validation_instances == 200
    assert config.num_test_instances == 300
    assert config.num_train_instances // 5 == 200
    assert config.num_validation_instances // 5 == 40
    assert config.num_test_instances // 5 == 60


def test_generated_splits_are_balanced_and_complete(generated) -> None:
    config, manifest, splits = generated
    assert set(splits) == {"train", "validation", "test"}
    expected = {
        "random_obstacle": {"train": 4, "validation": 2, "test": 2},
        "rooms_corridors": {"train": 4, "validation": 2, "test": 2},
        "u_detour": {"train": 4, "validation": 2, "test": 2},
        "branching_maze": {"train": 4, "validation": 2, "test": 2},
        "loop_maze": {"train": 4, "validation": 2, "test": 2},
    }
    for split, instances in splits.items():
        counts = Counter(instance.structure for instance in instances)
        for structure in STRUCTURES:
            assert counts[structure] == expected[structure][split]
            assert manifest["splits"][split]["structure_counts"][structure] == expected[structure][split]


def test_every_instance_satisfies_weighted_grid_validity(generated) -> None:
    config, _, splits = generated
    for instances in splits.values():
        for instance in instances:
            assert instance.terrain_grid.shape == (16, 16)
            assert set(np.unique(instance.terrain_grid)).issubset({0, 1, 3, 5})
            assert instance.start != instance.goal
            assert instance.terrain_grid[instance.start] != WALL
            assert instance.terrain_grid[instance.goal] != WALL
            recomputed_bfs = bfs_shortest_steps(
                instance.terrain_grid, instance.start, instance.goal
            )
            assert recomputed_bfs is not None
            assert recomputed_bfs == instance.bfs_steps
            dijkstra = dijkstra_optimal_path(
                instance.terrain_grid, instance.start, instance.goal
            )
            assert dijkstra.success and dijkstra.path_cost is not None
            assert dijkstra.path_cost == pytest.approx(instance.optimal_cost)
            counts, ratios = terrain_statistics(instance.terrain_grid)
            assert tuple(instance.terrain_counts) == counts
            assert np.allclose(instance.terrain_ratios, ratios, atol=1e-7)
            assert terrain_map_hash(instance.terrain_grid) == instance.map_hash
            assert instance.seed is not None


def test_splits_have_no_duplicate_maps(generated) -> None:
    _, manifest, splits = generated
    hashes = [
        instance.map_hash
        for instances in splits.values()
        for instance in instances
    ]
    assert len(hashes) == len(set(hashes))
    independence = manifest["independence"]
    assert independence["total_duplicate_instances"] == 0
    assert all(overlap == 0 for overlap in independence["pairwise_overlaps"].values())


def test_terrain_randomness_is_not_structure_specific(generated) -> None:
    _, _, splits = generated
    instances = splits["train"]
    for structure in STRUCTURES:
        group = [instance for instance in instances if instance.structure == structure]
        values = set()
        ratios = []
        for instance in group:
            free_values = set(
                int(value)
                for value in np.unique(instance.terrain_grid)
                if int(value) != WALL
            )
            values.update(free_values)
            ratios.append(instance.terrain_ratios)
        # With a 20-map debug split each structure still receives all terrain
        # classes, proving terrain is not fixed per structure.
        assert values == {1, 3, 5}
        average = np.mean(np.asarray(ratios), axis=0)
        assert np.all(average > 0.15) and np.all(average < 0.52)


def test_random_obstacle_wall_ratio() -> None:
    maze = _successful_structure(generate_random_obstacle, seed=11)
    wall_ratio = float(np.mean(maze.binary_grid == 0))
    assert 0.20 - 1e-9 <= wall_ratio <= 0.30 + 1e-9
    assert bfs_shortest_steps(maze.binary_grid, maze.start, maze.goal) is not None


def test_rooms_corridors_has_four_regions_and_cross_room_endpoints() -> None:
    maze = _successful_structure(generate_rooms_corridors, seed=17)
    assert maze.structure == "rooms_corridors"
    assert maze.metadata["room_count"] == 4
    assert maze.metadata["start_room"] != maze.metadata["goal_room"]
    assert maze.metadata["door_count"] >= 4
    # Dividers exist with narrow door openings.
    assert int(np.sum(maze.binary_grid[8, :] == 0)) > 0
    assert int(np.sum(maze.binary_grid[:, 8] == 0)) > 0
    assert bfs_shortest_steps(maze.binary_grid, maze.start, maze.goal) is not None


def test_u_detour_forces_detour_and_is_connected() -> None:
    maze = _successful_structure(generate_u_detour, seed=23)
    manhattan = abs(maze.start[0] - maze.goal[0]) + abs(maze.start[1] - maze.goal[1])
    steps = bfs_shortest_steps(maze.binary_grid, maze.start, maze.goal)
    assert steps is not None
    assert steps - manhattan >= 4
    assert maze.metadata["detour_extra_steps"] == steps - manhattan


def test_u_detour_decorative_walls_never_block_endpoints() -> None:
    # Seed 211 previously raised ValueError because a decorative wall
    # overwrote the goal cell before the BFS connectivity check.
    maze = generate_u_detour(Config(), np.random.default_rng(211))
    assert maze is None or (
        maze.binary_grid[maze.start] == 1 and maze.binary_grid[maze.goal] == 1
    )


def test_branching_maze_has_branches_dead_ends_and_no_cycles() -> None:
    maze = _successful_structure(generate_branching_maze, seed=29)
    vertices = int(np.sum(maze.binary_grid == 1))
    assert _edge_count(maze.binary_grid) == vertices - 1  # perfect tree
    assert maze.metadata["dead_end_cells"] >= 1
    assert bfs_shortest_steps(maze.binary_grid, maze.start, maze.goal) is not None


def test_loop_maze_has_cycles() -> None:
    maze = _successful_structure(generate_loop_maze, seed=31)
    vertices = int(np.sum(maze.binary_grid == 1))
    assert _edge_count(maze.binary_grid) > vertices - 1
    assert maze.metadata["cycle_rank"] >= 1
    assert bfs_shortest_steps(maze.binary_grid, maze.start, maze.goal) is not None


def test_each_structure_varies_across_random_seeds() -> None:
    for structure, generator in GENERATORS.items():
        first = _successful_structure(generator, seed=101)
        second = _successful_structure(generator, seed=202)
        assert not np.array_equal(first.binary_grid, second.binary_grid), structure
        assert (first.start, first.goal) != (second.start, second.goal), structure


def test_generation_is_reproducible(tmp_path: Path) -> None:
    config = _small_config(tmp_path / "first")
    generate_datasets(config, force=True)
    first_hashes = load_dataset(config.train_path)[1]["map_hashes"].copy()
    generate_datasets(config, force=True)
    second_hashes = load_dataset(config.train_path)[1]["map_hashes"].copy()
    np.testing.assert_array_equal(first_hashes, second_hashes)


def test_existing_data_requires_explicit_force(generated) -> None:
    config, _, _ = generated
    with pytest.raises(FileExistsError):
        generate_datasets(config, force=False)


def test_split_specific_seed_and_manifest() -> None:
    config = Config()
    seeds = config.split_seeds()
    assert len(set(seeds.values())) == 3
    manifest = load_manifest(config.manifest_path) if config.manifest_path.exists() else None
    if manifest is not None:
        assert manifest["format_version"] == 2
        assert set(manifest["structures"]) == set(STRUCTURES)


def test_generate_structure_sample_returns_all_terrain_fields() -> None:
    config = Config()
    for structure in STRUCTURES:
        record = generate_structure_sample(
            structure,
            config,
            np.random.default_rng(303),
            max_attempts=500,
        )
        assert record is not None
        assert record["terrain"].shape == (16, 16)
        assert set(int(v) for v in np.unique(record["terrain"])).issubset({0, 1, 3, 5})
        assert sum(record["terrain_counts"]) == int(np.sum(record["terrain"] != WALL))
        assert record["map_hash"] == terrain_map_hash(record["terrain"])
        assert record["optimal_cost"] >= 0
        assert record["bfs_steps"] >= 1