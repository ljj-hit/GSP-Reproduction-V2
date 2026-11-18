"""Procedural generation of five balanced 16 x 16 weighted-maze structures.

The structure label is metadata only and is never provided to the R-GNN.
Terrain values (1, 3, 5) are assigned independently and uniformly to every
walkable cell after the wall layout has been generated.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from config import Config
from environment import (
    ACTION_DELTAS,
    MazeInstance,
    WALL,
    bfs_shortest_steps,
    dijkstra_optimal_path,
    path_terrain_cost,
    terrain_map_hash,
    terrain_statistics,
    validate_terrain,
)
from utils import ensure_dir, save_json, setup_logger, utc_now_iso


STRUCTURES: Tuple[str, ...] = (
    "random_obstacle",
    "rooms_corridors",
    "u_detour",
    "branching_maze",
    "loop_maze",
)


@dataclass
class StructuredMaze:
    binary_grid: np.ndarray
    start: Tuple[int, int]
    goal: Tuple[int, int]
    structure: str
    metadata: Dict[str, Any]


@dataclass
class GenerationStats:
    attempts: int = 0
    accepted: int = 0
    rejected_disconnected: int = 0
    rejected_structure: int = 0
    rejected_duplicate: int = 0


# ---------------------------------------------------------------------------
# Shared map helpers
# ---------------------------------------------------------------------------
def _all_free(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=np.int8)


def _free_positions(binary: np.ndarray) -> List[Tuple[int, int]]:
    rows, cols = np.nonzero(binary == 1)
    return [(int(row), int(col)) for row, col in zip(rows, cols)]


def _connected_path(
    binary: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    from environment import bfs_shortest_path

    return bfs_shortest_path(binary, start, goal)


def _choose_start_goal(
    binary: np.ndarray,
    rng: np.random.Generator,
    *,
    positions: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    candidates = list(positions) if positions is not None else _free_positions(binary)
    candidates = [pos for pos in candidates if binary[pos] == 1]
    if len(candidates) < 2:
        return None
    indices = rng.choice(len(candidates), size=2, replace=False)
    start = candidates[int(indices[0])]
    goal = candidates[int(indices[1])]
    if start == goal:
        return None
    return start, goal


def _transform_grid_and_positions(
    grid: np.ndarray,
    positions: Sequence[Tuple[int, int]],
    rotations: int,
    flip: bool,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Apply dihedral symmetry while remapping coordinates exactly."""
    out = grid
    height, width = grid.shape
    mapped: List[Tuple[float, float]] = [(float(r), float(c)) for r, c in positions]
    for _ in range(rotations % 4):
        out = np.rot90(out, 1)
        # np.rot90 (counter-clockwise): (r,c) -> (W-1-c, r)
        mapped = [(width - 1 - c, r) for r, c in mapped]
        height, width = out.shape
    if flip:
        out = np.fliplr(out)
        mapped = [(r, width - 1 - c) for r, c in mapped]
    int_positions = [(int(r), int(c)) for r, c in mapped]
    return np.ascontiguousarray(out), int_positions


def _degree(binary: np.ndarray, position: Tuple[int, int]) -> int:
    row, col = position
    degree = 0
    for dr, dc in ACTION_DELTAS:
        nr, nc = row + dr, col + dc
        if 0 <= nr < binary.shape[0] and 0 <= nc < binary.shape[1] and binary[nr, nc] == 1:
            degree += 1
    return degree


def _graph_edge_count(binary: np.ndarray) -> int:
    edges = 0
    height, width = binary.shape
    for row in range(height):
        for col in range(width):
            if binary[row, col] == 0:
                continue
            if row + 1 < height and binary[row + 1, col] == 1:
                edges += 1
            if col + 1 < width and binary[row, col + 1] == 1:
                edges += 1
    return edges


# ---------------------------------------------------------------------------
# Five maze structures
# ---------------------------------------------------------------------------
def generate_random_obstacle(config: Config, rng: np.random.Generator) -> Optional[StructuredMaze]:
    height, width = config.height, config.width
    cell_count = height * width
    ratio = float(
        rng.uniform(config.random_obstacle_min_ratio, config.random_obstacle_max_ratio)
    )
    min_walls = int(math.ceil(config.random_obstacle_min_ratio * cell_count))
    max_walls = int(math.floor(config.random_obstacle_max_ratio * cell_count))
    wall_count = max(min_walls, min(max_walls, int(round(ratio * cell_count))))
    flat = np.ones(cell_count, dtype=np.int8)
    wall_indices = rng.choice(cell_count, size=wall_count, replace=False)
    flat[wall_indices] = 0
    binary = flat.reshape(height, width)
    chosen = _choose_start_goal(binary, rng)
    if chosen is None:
        return None
    start, goal = chosen
    if _connected_path(binary, start, goal) is None:
        return None
    actual_ratio = wall_count / cell_count
    if not (config.random_obstacle_min_ratio - 1e-9 <= actual_ratio <= config.random_obstacle_max_ratio + 1e-9):
        return None
    return StructuredMaze(
        binary_grid=binary,
        start=start,
        goal=goal,
        structure="random_obstacle",
        metadata={"wall_count": wall_count, "wall_ratio": actual_ratio},
    )


def generate_rooms_corridors(config: Config, rng: np.random.Generator) -> Optional[StructuredMaze]:
    height, width = config.height, config.width
    if height != 16 or width != 16:
        raise ValueError("rooms_corridors is defined for the required 16x16 grid")
    binary = _all_free(height, width)
    split_row = height // 2
    split_col = width // 2

    # Build four rooms separated by two wall lines.
    binary[split_row, :] = 0
    binary[:, split_col] = 0
    door_cells: List[Tuple[int, int]] = []
    row_door_columns = [
        int(rng.integers(1, split_col)),
        int(rng.integers(split_col + 1, width - 1)),
    ]
    col_door_rows = [
        int(rng.integers(1, split_row)),
        int(rng.integers(split_row + 1, height - 1)),
    ]
    for col in row_door_columns:
        binary[split_row, col] = 1
        door_cells.append((split_row, col))
    for row in col_door_rows:
        binary[row, split_col] = 1
        door_cells.append((row, split_col))

    # Narrow internal wall segments make the open rooms corridor-like.
    door_set = set(door_cells)
    for _ in range(int(rng.integers(4, 8))):
        horizontal = bool(rng.integers(0, 2))
        length = int(rng.integers(3, 7))
        if horizontal:
            room_row = int(rng.integers(1, split_row - 1))
            start_col = int(rng.integers(1, width - length - 1))
            cells = [(room_row, c) for c in range(start_col, start_col + length)]
        else:
            room_col = int(rng.integers(1, split_col - 1))
            start_row = int(rng.integers(1, height - length - 1))
            cells = [(r, room_col) for r in range(start_row, start_row + length)]
        for cell in cells:
            if cell in door_set or cell[0] == split_row or cell[1] == split_col:
                continue
            binary[cell] = 0

    rooms: Dict[int, List[Tuple[int, int]]] = {0: [], 1: [], 2: [], 3: []}
    for row in range(height):
        for col in range(width):
            if binary[row, col] != 1 or (row, col) in door_set:
                continue
            room_id = (1 if row > split_row else 0) * 2 + (1 if col > split_col else 0)
            rooms[room_id].append((row, col))
    room_ids = [room for room, cells in rooms.items() if len(cells) >= 2]
    if len(room_ids) < 2:
        return None
    picked_rooms = rng.choice(len(room_ids), size=2, replace=False)
    first_room = room_ids[int(picked_rooms[0])]
    second_room = room_ids[int(picked_rooms[1])]
    start = rooms[first_room][int(rng.integers(len(rooms[first_room])))]
    goal = rooms[second_room][int(rng.integers(len(rooms[second_room])))]
    if start == goal:
        return None
    if _connected_path(binary, start, goal) is None:
        return None
    return StructuredMaze(
        binary_grid=binary,
        start=start,
        goal=goal,
        structure="rooms_corridors",
        metadata={
            "room_count": 4,
            "start_room": int(first_room),
            "goal_room": int(second_room),
            "door_count": len(door_cells),
        },
    )


def generate_u_detour(config: Config, rng: np.random.Generator) -> Optional[StructuredMaze]:
    height, width = config.height, config.width
    if height != 16 or width != 16:
        raise ValueError("u_detour is defined for the required 16x16 grid")
    binary = _all_free(height, width)
    margin = int(rng.integers(2, 5))
    left = margin
    right = width - 1 - margin
    bar_row = int(rng.integers(6, 9))
    arm_end = int(rng.integers(11, 14))
    center_col = int((left + right) // 2 + rng.integers(-1, 2))
    start_row = int(rng.integers(1, bar_row - 2))
    goal_row = int(rng.integers(bar_row + 2, arm_end))

    # U-shaped barrier: roof and two downward arms, open at the bottom.
    binary[bar_row, left:right + 1] = 0
    binary[bar_row:arm_end + 1, left] = 0
    binary[bar_row:arm_end + 1, right] = 0
    start = (start_row, center_col)
    goal = (goal_row, center_col)
    if binary[start] == 0 or binary[goal] == 0:
        return None

    # Add a few unrelated wall blocks outside the U while preserving an escape.
    for _ in range(int(rng.integers(0, 3))):
        row = int(rng.integers(1, height - 2))
        col = int(rng.integers(1, width - 2))
        if (row, col) == start or (row, col) == goal:
            continue
        if row <= bar_row + 1 and left <= col <= right:
            continue
        binary[row, col] = 0

    # Decorative blocks must never make the endpoint unwalkable.
    if binary[start] == 0 or binary[goal] == 0:
        return None

    path = _connected_path(binary, start, goal)
    if path is None:
        return None
    manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    extra = (len(path) - 1) - manhattan
    if extra < 4:
        return None

    rotations = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))
    binary, transformed = _transform_grid_and_positions(binary, [start, goal], rotations, flip)
    start, goal = transformed
    if binary[start] == 0 or binary[goal] == 0:
        return None
    if _connected_path(binary, start, goal) is None:
        return None
    return StructuredMaze(
        binary_grid=binary,
        start=start,
        goal=goal,
        structure="u_detour",
        metadata={"detour_extra_steps": int(extra), "rotation": rotations, "flip": flip},
    )


def _generate_dfs_maze(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perfect maze on odd-coordinate cells using recursive backtracking."""
    binary = np.zeros((height, width), dtype=np.int8)
    cell_rows = list(range(1, height, 2))
    cell_cols = list(range(1, width, 2))
    for row in cell_rows:
        for col in cell_cols:
            binary[row, col] = 1

    visited: Set[Tuple[int, int]] = set()
    start_cell = (cell_rows[0], cell_cols[0])
    stack: List[Tuple[int, int]] = [start_cell]
    visited.add(start_cell)
    while stack:
        row, col = stack[-1]
        neighbors: List[Tuple[int, int, int, int]] = []
        for dr, dc in ACTION_DELTAS:
            nr, nc = row + 2 * dr, col + 2 * dc
            if (nr, nc) in visited:
                continue
            if 1 <= nr < height and 1 <= nc < width:
                neighbors.append((nr, nc, row + dr, col + dc))
        if not neighbors:
            stack.pop()
            continue
        chosen = neighbors[int(rng.integers(0, len(neighbors)))]
        nr, nc, wall_r, wall_c = chosen
        binary[wall_r, wall_c] = 1
        binary[nr, nc] = 1
        visited.add((nr, nc))
        stack.append((nr, nc))
    return binary


def _choose_odd_start_goal(
    binary: np.ndarray,
    rng: np.random.Generator,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    positions = _free_positions(binary)
    if len(positions) < 2:
        return None
    # Prefer distant pairs to avoid trivial start/goal placement.
    best: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    best_distance = -1
    for _ in range(min(64, len(positions) * 2)):
        sample = rng.choice(len(positions), size=2, replace=False)
        a = positions[int(sample[0])]
        b = positions[int(sample[1])]
        distance = abs(a[0] - b[0]) + abs(a[1] - b[1])
        if distance > best_distance:
            best_distance = distance
            best = (a, b)
    if best is None or best[0] == best[1]:
        return None
    return best


def generate_branching_maze(config: Config, rng: np.random.Generator) -> Optional[StructuredMaze]:
    binary = _generate_dfs_maze(config.height, config.width, rng)
    chosen = _choose_odd_start_goal(binary, rng)
    if chosen is None:
        return None
    start, goal = chosen
    if _connected_path(binary, start, goal) is None:
        return None
    free = _free_positions(binary)
    dead_ends = sum(1 for pos in free if _degree(binary, pos) == 1)
    return StructuredMaze(
        binary_grid=binary,
        start=start,
        goal=goal,
        structure="branching_maze",
        metadata={"dead_end_cells": int(dead_ends), "walkable_cells": len(free)},
    )


def generate_loop_maze(config: Config, rng: np.random.Generator) -> Optional[StructuredMaze]:
    binary = _generate_dfs_maze(config.height, config.width, rng)
    height, width = binary.shape
    candidates: List[Tuple[int, int]] = []
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if binary[row, col] != 0:
                continue
            horizontal = binary[row, col - 1] == 1 and binary[row, col + 1] == 1
            vertical = binary[row - 1, col] == 1 and binary[row + 1, col] == 1
            if horizontal or vertical:
                candidates.append((row, col))
    if not candidates:
        return None
    open_count = max(1, int(round(len(candidates) * float(rng.uniform(0.15, 0.30)))))
    open_indices = rng.choice(len(candidates), size=open_count, replace=False)
    opened: List[Tuple[int, int]] = []
    for index in open_indices:
        cell = candidates[int(index)]
        binary[cell] = 1
        opened.append(cell)
    chosen = _choose_odd_start_goal(binary, rng)
    if chosen is None:
        return None
    start, goal = chosen
    if _connected_path(binary, start, goal) is None:
        return None
    vertices = int(np.sum(binary == 1))
    edges = _graph_edge_count(binary)
    if edges <= vertices - 1:
        return None
    return StructuredMaze(
        binary_grid=binary,
        start=start,
        goal=goal,
        structure="loop_maze",
        metadata={
            "opened_loop_walls": len(opened),
            "cycle_rank": int(edges - vertices + 1),
            "walkable_cells": vertices,
        },
    )


GENERATORS = {
    "random_obstacle": generate_random_obstacle,
    "rooms_corridors": generate_rooms_corridors,
    "u_detour": generate_u_detour,
    "branching_maze": generate_branching_maze,
    "loop_maze": generate_loop_maze,
}


# ---------------------------------------------------------------------------
# Terrain / accepted instance construction
# ---------------------------------------------------------------------------
def assign_terrain(
    binary: np.ndarray,
    config: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    terrain = np.zeros(binary.shape, dtype=np.int8)
    free_mask = binary == 1
    free_count = int(free_mask.sum())
    values = rng.choice(
        np.asarray(config.terrain_values, dtype=np.int8),
        size=free_count,
        p=np.asarray(config.terrain_probabilities, dtype=np.float64),
    )
    terrain[free_mask] = values
    return terrain


def _build_accepted_record(
    structure: str,
    maze: StructuredMaze,
    config: Config,
    rng: np.random.Generator,
    maze_seed: int,
) -> Optional[Dict[str, Any]]:
    if maze.binary_grid.shape != (config.height, config.width):
        return None
    if maze.binary_grid[maze.start] != 1 or maze.binary_grid[maze.goal] != 1:
        return None
    if maze.start == maze.goal:
        return None
    terrain = assign_terrain(maze.binary_grid, config, rng)
    bfs_steps = bfs_shortest_steps(terrain, maze.start, maze.goal)
    if bfs_steps is None:
        return None
    dijkstra = dijkstra_optimal_path(terrain, maze.start, maze.goal)
    if not dijkstra.success or dijkstra.path_cost is None:
        return None
    path_cost = path_terrain_cost(terrain, dijkstra.path)
    if not math.isclose(float(dijkstra.path_cost), float(path_cost), abs_tol=1e-9):
        raise RuntimeError("Dijkstra cost disagrees with terrain path cost")
    counts, ratios = terrain_statistics(terrain)
    if sum(counts) == 0:
        return None
    map_hash = terrain_map_hash(terrain, maze.start, maze.goal)
    return {
        "terrain": terrain,
        "start": maze.start,
        "goal": maze.goal,
        "seed": maze_seed,
        "structure": structure,
        "bfs_steps": int(bfs_steps),
        "optimal_cost": float(dijkstra.path_cost),
        "terrain_counts": tuple(int(v) for v in counts),
        "terrain_ratios": tuple(float(v) for v in ratios),
        "map_hash": map_hash,
        "structure_metadata": maze.metadata,
    }


def generate_structure_sample(
    structure: str,
    config: Config,
    rng: np.random.Generator,
    *,
    max_attempts: int = 200,
) -> Optional[Dict[str, Any]]:
    """Generate one accepted terrain instance of a given structure."""
    if structure not in GENERATORS:
        raise ValueError(f"Unknown structure: {structure}")
    generator = GENERATORS[structure]
    for _ in range(max_attempts):
        maze_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        local_rng = np.random.default_rng(maze_seed)
        structured = generator(config, local_rng)
        if structured is None:
            continue
        record = _build_accepted_record(
            structure, structured, config, local_rng, maze_seed
        )
        if record is not None:
            return record
    return None


# ---------------------------------------------------------------------------
# Split dataset generation
# ---------------------------------------------------------------------------
def _structure_counts(total: int, config: Config) -> Dict[str, int]:
    per_structure = total // config.structure_count
    return {name: per_structure for name in STRUCTURES}


def _records_to_arrays(records: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not records:
        raise ValueError("Cannot serialize an empty dataset")
    return {
        "terrains": np.stack([record["terrain"] for record in records]).astype(np.int8),
        "starts": np.asarray([record["start"] for record in records], dtype=np.int16),
        "goals": np.asarray([record["goal"] for record in records], dtype=np.int16),
        "seeds": np.asarray([record["seed"] for record in records], dtype=np.int64),
        "structures": np.asarray([record["structure"] for record in records], dtype="<U24"),
        "bfs_steps": np.asarray([record["bfs_steps"] for record in records], dtype=np.int16),
        "optimal_costs": np.asarray([record["optimal_cost"] for record in records], dtype=np.float32),
        "map_hashes": np.asarray([record["map_hash"] for record in records], dtype="<U64"),
        "terrain_counts": np.asarray([record["terrain_counts"] for record in records], dtype=np.int32),
        "terrain_ratios": np.asarray([record["terrain_ratios"] for record in records], dtype=np.float32),
    }


def save_split(path: str | Path, records: Sequence[Dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **_records_to_arrays(records))
    return path


def generate_split(
    config: Config,
    split_name: str,
    count: int,
    seed: int,
    used_map_hashes: Set[str],
    logger: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []
    stats = GenerationStats()
    structure_counts = _structure_counts(count, config)
    per_structure_attempts: Dict[str, int] = {}
    structure_metadata: Dict[str, Any] = {}

    for structure in STRUCTURES:
        target = structure_counts[structure]
        accepted_for_structure = 0
        attempts = 0
        max_attempts = max(1, target * config.max_generation_attempts_per_instance)
        while accepted_for_structure < target and attempts < max_attempts:
            attempts += 1
            stats.attempts += 1
            record = generate_structure_sample(
                structure,
                config,
                rng,
                max_attempts=1,
            )
            if record is None:
                stats.rejected_structure += 1
                continue
            if record["map_hash"] in used_map_hashes:
                stats.rejected_duplicate += 1
                continue
            if _connected_path(
                record["terrain"] != WALL,
                record["start"],
                record["goal"],
            ) is None:
                stats.rejected_disconnected += 1
                continue
            used_map_hashes.add(str(record["map_hash"]))
            records.append(record)
            accepted_for_structure += 1
            stats.accepted += 1
        per_structure_attempts[structure] = attempts
        if accepted_for_structure < target:
            raise RuntimeError(
                f"Could not generate {target} valid '{structure}' mazes for split "
                f"'{split_name}' after {attempts} attempts "
                f"({accepted_for_structure} accepted)."
            )
        structure_metadata[structure] = {
            "requested": target,
            "accepted": accepted_for_structure,
            "attempts": attempts,
        }
        logger.info(
            "[%s] %s: %d/%d after %d attempts",
            split_name,
            structure,
            accepted_for_structure,
            target,
            attempts,
        )

    order = rng.permutation(len(records))
    records = [records[int(index)] for index in order]

    def stats_for(values: Sequence[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"min": None, "max": None, "mean": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(array.min()),
            "max": float(array.max()),
            "mean": float(array.mean()),
        }

    terrain_totals = np.asarray(
        [record["terrain_counts"] for record in records], dtype=np.int64
    ).sum(axis=0)
    total_free = int(terrain_totals.sum())
    all_ratio = (terrain_totals / total_free).tolist() if total_free else []
    wall_ratios = [
        float(np.mean(record["terrain"] == WALL)) for record in records
    ]
    metadata = {
        "split": split_name,
        "path": str(config.split_paths()[split_name]),
        "generator_seed": int(seed),
        "requested_count": int(count),
        "actual_count": len(records),
        "structure_counts": {
            structure: int(sum(1 for record in records if record["structure"] == structure))
            for structure in STRUCTURES
        },
        "generation_stats": {
            "attempts": stats.attempts,
            "accepted": stats.accepted,
            "rejected_structure": stats.rejected_structure,
            "rejected_duplicate": stats.rejected_duplicate,
            "rejected_disconnected": stats.rejected_disconnected,
            "per_structure_attempts": per_structure_attempts,
        },
        "terrain": {
            "values": list(config.terrain_values),
            "total_free_cells_by_terrain": [int(v) for v in terrain_totals],
            "overall_ratios_by_terrain": all_ratio,
            "wall_ratio": stats_for(wall_ratios),
        },
        "bfs_steps": stats_for([record["bfs_steps"] for record in records]),
        "optimal_cost": stats_for([record["optimal_cost"] for record in records]),
        "per_structure": structure_metadata,
    }
    return records, metadata


def audit_splits(splits: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    hash_sets = {
        name: {str(record["map_hash"]) for record in records}
        for name, records in splits.items()
    }
    names = list(hash_sets)
    pairwise: Dict[str, int] = {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            pairwise[f"{left}__{right}"] = len(hash_sets[left] & hash_sets[right])
    within = {
        name: len(splits[name]) - len(hash_sets[name]) for name in names
    }
    return {
        "checked": True,
        "pairwise_overlaps": pairwise,
        "duplicate_hashes_within_splits": within,
        "total_duplicate_instances": int(sum(within.values()) + sum(pairwise.values())),
        "unique_maps": {name: len(hash_set) for name, hash_set in hash_sets.items()},
    }


def generate_datasets(
    config: Config,
    *,
    force: bool = False,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate all official splits; refuses to overwrite unless ``force``."""
    ensure_dir(config.data_dir)
    ensure_dir(config.log_dir)
    if logger is None:
        logger = setup_logger(
            "gsp.v2.generator",
            Path(config.log_dir) / "generator.log",
        )
    existing = [path for path in config.split_paths().values() if Path(path).exists()]
    if config.manifest_path.exists():
        existing.append(config.manifest_path)
    if existing and not force:
        raise FileExistsError(
            "Dataset files already exist. Re-run with --force to regenerate: "
            + ", ".join(str(path) for path in existing)
        )

    logger.info("Generating V2 weighted-maze datasets with seed %d", config.generator_seed)
    used_map_hashes: Set[str] = set()
    records_by_split: Dict[str, List[Dict[str, Any]]] = {}
    split_manifests: Dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        records, split_metadata = generate_split(
            config,
            split_name,
            config.split_counts()[split_name],
            config.split_seeds()[split_name],
            used_map_hashes,
            logger,
        )
        path = config.split_paths()[split_name]
        save_split(path, records)
        records_by_split[split_name] = records
        split_manifests[split_name] = split_metadata
        logger.info(
            "Saved %s (%d mazes, structures=%s)",
            path,
            len(records),
            split_metadata["structure_counts"],
        )

    independence = audit_splits(records_by_split)
    if independence["total_duplicate_instances"] != 0:
        raise RuntimeError(f"Dataset independence failed: {independence}")

    manifest = {
        "format_version": 2,
        "created_at_utc": utc_now_iso(),
        "project": "GSP-Reproduction-V2",
        "environment": "16x16 weighted maze; walls=0, terrain=1/3/5",
        "structures": list(STRUCTURES),
        "config": config.to_dict(),
        "splits": split_manifests,
        "independence": independence,
        "labels_note": (
            "BFS steps and Dijkstra optimal costs are metadata/reference labels "
            "only; they are never used as R-GNN inputs or Q-learning targets."
        ),
    }
    save_json(manifest, config.manifest_path)
    logger.info("Saved manifest %s", config.manifest_path)
    return manifest


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_manifest(path: str | Path) -> Dict[str, Any]:
    from utils import load_json

    return load_json(path)


def load_dataset(path: str | Path) -> Tuple[List[MazeInstance], Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Generate it with 'python generator.py --seed 42'."
        )
    with np.load(path, allow_pickle=False) as data:
        terrains = np.asarray(data["terrains"], dtype=np.int8)
        starts = np.asarray(data["starts"], dtype=np.int64)
        goals = np.asarray(data["goals"], dtype=np.int64)
        seeds = np.asarray(data["seeds"], dtype=np.int64)
        structures = np.asarray(data["structures"]).astype(str)
        bfs_steps = np.asarray(data["bfs_steps"], dtype=np.int64)
        optimal_costs = np.asarray(data["optimal_costs"], dtype=np.float64)
        map_hashes = np.asarray(data["map_hashes"]).astype(str)
        terrain_counts = np.asarray(data["terrain_counts"], dtype=np.int64)
        terrain_ratios = np.asarray(data["terrain_ratios"], dtype=np.float64)
    count = len(terrains)
    lengths = {
        "starts": len(starts),
        "goals": len(goals),
        "seeds": len(seeds),
        "structures": len(structures),
        "bfs_steps": len(bfs_steps),
        "optimal_costs": len(optimal_costs),
        "map_hashes": len(map_hashes),
        "terrain_counts": len(terrain_counts),
        "terrain_ratios": len(terrain_ratios),
    }
    if any(length != count for length in lengths.values()):
        raise ValueError(f"Inconsistent dataset array lengths: {lengths}, terrains={count}")
    if count == 0:
        raise ValueError("Empty dataset")
    instances: List[MazeInstance] = []
    for index in range(count):
        terrain = validate_terrain(terrains[index])
        start = (int(starts[index, 0]), int(starts[index, 1]))
        goal = (int(goals[index, 0]), int(goals[index, 1]))
        if terrain[start] == WALL or terrain[goal] == WALL:
            raise ValueError(f"Obstacle start/goal in dataset at index {index}")
        instances.append(
            MazeInstance(
                terrain_grid=terrain.copy(),
                start=start,
                goal=goal,
                instance_id=index,
                structure=str(structures[index]),
                seed=int(seeds[index]),
                map_hash=str(map_hashes[index]),
                bfs_steps=int(bfs_steps[index]),
                optimal_cost=float(optimal_costs[index]),
                terrain_counts=tuple(int(v) for v in terrain_counts[index]),
                terrain_ratios=tuple(float(v) for v in terrain_ratios[index]),
            )
        )
    metadata = {
        "path": str(path),
        "count": count,
        "terrains": terrains,
        "starts": starts,
        "goals": goals,
        "seeds": seeds,
        "structures": structures,
        "bfs_steps": bfs_steps,
        "optimal_costs": optimal_costs,
        "map_hashes": map_hashes,
        "terrain_counts": terrain_counts,
        "terrain_ratios": terrain_ratios,
    }
    return instances, metadata


def load_dataset_instances(path: str | Path) -> List[MazeInstance]:
    return load_dataset(path)[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate V2 16x16 weighted maze datasets")
    parser.add_argument("--seed", type=int, default=42, help="dataset master seed")
    parser.add_argument("--force", action="store_true", help="explicitly overwrite existing data")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--train-size", type=int, default=None, help="debug override; divisible by 5")
    parser.add_argument("--validation-size", type=int, default=None, help="debug override; divisible by 5")
    parser.add_argument("--test-size", type=int, default=None, help="debug override; divisible by 5")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    overrides: Dict[str, Any] = {"generator_seed": int(args.seed)}
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.log_dir is not None:
        overrides["log_dir"] = args.log_dir
    if args.train_size is not None:
        overrides["num_train_instances"] = int(args.train_size)
    if args.validation_size is not None:
        overrides["num_validation_instances"] = int(args.validation_size)
    if args.test_size is not None:
        overrides["num_test_instances"] = int(args.test_size)
    config = Config().copy_with(**overrides)
    manifest = generate_datasets(config, force=args.force)
    print(f"Generated dataset at {config.data_dir}")
    for split_name, metadata in manifest["splits"].items():
        print(
            f"  {split_name}: {metadata['actual_count']} maps, "
            f"structures={metadata['structure_counts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())