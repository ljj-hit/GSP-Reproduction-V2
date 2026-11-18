"""Weighted 16 x 16 grid-world environment for V2.

Terrain encoding:
    0 = wall, 1 = cost-1 terrain, 3 = cost-3 terrain, 5 = cost-5 terrain.

Entering a cell costs that cell's terrain value.  The start cell's cost is not
charged.  Actions are cardinal moves with a global numbering:
    UP=0, DOWN=1, LEFT=2, RIGHT=3
"""

from __future__ import annotations

import hashlib
import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Position = Tuple[int, int]

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_DELTAS: Tuple[Position, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES: Tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT")
ACTION_NAME_TO_ID = {name: index for index, name in enumerate(ACTION_NAMES)}
NUM_ACTIONS = 4

WALL = 0
TERRAIN_1 = 1
TERRAIN_3 = 3
TERRAIN_5 = 5
TERRAIN_VALUES: Tuple[int, ...] = (TERRAIN_1, TERRAIN_3, TERRAIN_5)
WALL_VALUE = WALL


class InvalidActionError(ValueError):
    """Raised when an illegal movement is attempted."""


class InvalidTerrainError(ValueError):
    """Raised when terrain data contains invalid values."""


@dataclass
class MazeInstance:
    """A weighted maze instance with generation/evaluation metadata."""

    terrain_grid: np.ndarray
    start: Position
    goal: Position
    instance_id: int = 0
    structure: str = "unknown"
    seed: Optional[int] = None
    map_hash: str = ""
    bfs_steps: Optional[int] = None
    optimal_cost: Optional[float] = None
    terrain_counts: Tuple[int, int, int] = (0, 0, 0)
    terrain_ratios: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def height(self) -> int:
        return int(self.terrain_grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.terrain_grid.shape[1])

    @property
    def cell_count(self) -> int:
        return self.height * self.width


@dataclass
class StepResult:
    position: Position
    reward: float
    movement_cost: int
    cumulative_cost: int
    done: bool


@dataclass
class WeightedSearchResult:
    """Reference result used by Dijkstra and traditional baselines."""

    success: bool
    path: List[Position] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    path_cost: Optional[float] = None
    state_expansions: int = 0
    unique_state_expansions: int = 0
    reexpansions: int = 0
    transition_evaluations: int = 0
    termination_reason: str = "unknown"
    inference_time_seconds: float = 0.0

    @property
    def plan_length(self) -> int:
        return len(self.actions)


def validate_terrain(terrain_grid: np.ndarray) -> np.ndarray:
    array = np.asarray(terrain_grid)
    if array.ndim != 2:
        raise InvalidTerrainError("Terrain must be a 2D array")
    if array.size == 0:
        raise InvalidTerrainError("Terrain must not be empty")
    if not np.all(np.isin(array, (WALL, TERRAIN_1, TERRAIN_3, TERRAIN_5))):
        raise InvalidTerrainError("Terrain entries must be one of 0, 1, 3, 5")
    return array.astype(np.int8, copy=False)


def in_bounds(terrain_grid: np.ndarray, position: Position) -> bool:
    row, col = position
    return 0 <= row < terrain_grid.shape[0] and 0 <= col < terrain_grid.shape[1]


def is_walkable(terrain_grid: np.ndarray, position: Position) -> bool:
    return in_bounds(terrain_grid, position) and int(terrain_grid[position]) != WALL


def movement_cost(terrain_grid: np.ndarray, position: Position) -> int:
    """Cost of entering ``position``; walls raise ``InvalidActionError``."""
    if not in_bounds(terrain_grid, position):
        raise InvalidActionError(f"Position out of bounds: {position}")
    value = int(terrain_grid[position])
    if value == WALL:
        raise InvalidActionError(f"Position is a wall: {position}")
    return value


def reward_for_terrain(terrain: int) -> float:
    if int(terrain) not in TERRAIN_VALUES:
        raise InvalidActionError(f"Cannot enter terrain value {terrain}")
    return -float(terrain)


def valid_actions_for(terrain_grid: np.ndarray, position: Position) -> List[int]:
    if not in_bounds(terrain_grid, position):
        raise ValueError(f"Position out of bounds: {position}")
    if not is_walkable(terrain_grid, position):
        raise ValueError(f"Position is a wall: {position}")
    actions: List[int] = []
    row, col = position
    for action, (dr, dc) in enumerate(ACTION_DELTAS):
        nr, nc = row + dr, col + dc
        if (
            0 <= nr < terrain_grid.shape[0]
            and 0 <= nc < terrain_grid.shape[1]
            and int(terrain_grid[nr, nc]) != WALL
        ):
            actions.append(action)
    return actions


def valid_action_mask(terrain_grid: np.ndarray, position: Position) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for action in valid_actions_for(terrain_grid, position):
        mask[action] = True
    return mask


def successor_position(position: Position, action: int) -> Position:
    if action not in (UP, DOWN, LEFT, RIGHT):
        raise InvalidActionError(f"Unknown action id: {action}")
    dr, dc = ACTION_DELTAS[action]
    return position[0] + dr, position[1] + dc


def positions_from_actions(start: Position, actions: Sequence[int]) -> List[Position]:
    positions = [start]
    current = start
    for action in actions:
        current = successor_position(current, int(action))
        positions.append(current)
    return positions


def path_to_actions(path: Sequence[Position]) -> List[int]:
    actions: List[int] = []
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        delta = (r1 - r0, c1 - c0)
        if delta not in ACTION_DELTAS:
            raise ValueError(f"Non-cardinal path step: {delta}")
        actions.append(ACTION_DELTAS.index(delta))
    return actions


def path_terrain_cost(
    terrain_grid: np.ndarray,
    path: Sequence[Position],
) -> int:
    """Sum terrain costs of entered cells; the starting cell is free."""
    if not path:
        raise ValueError("Path must not be empty")
    total = 0
    for position in path[1:]:
        total += movement_cost(terrain_grid, position)
    return total


def bfs_shortest_path(
    terrain_grid: np.ndarray,
    start: Position,
    goal: Position,
    max_expansions: Optional[int] = None,
) -> Optional[List[Position]]:
    """Unweighted shortest path in *steps*, treating every non-wall cell as 1."""
    terrain_grid = validate_terrain(terrain_grid)
    if not is_walkable(terrain_grid, start) or not is_walkable(terrain_grid, goal):
        raise ValueError("Start and goal must be walkable cells")
    if start == goal:
        return [start]
    frontier: deque[Position] = deque([start])
    parent: Dict[Position, Optional[Position]] = {start: None}
    expansions = 0
    while frontier:
        if max_expansions is not None and expansions >= max_expansions:
            break
        position = frontier.popleft()
        expansions += 1
        if position == goal:
            path: List[Position] = []
            current: Optional[Position] = goal
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path
        row, col = position
        for dr, dc in ACTION_DELTAS:
            nxt = (row + dr, col + dc)
            if is_walkable(terrain_grid, nxt) and nxt not in parent:
                parent[nxt] = position
                frontier.append(nxt)
    return None


def bfs_shortest_steps(
    terrain_grid: np.ndarray,
    start: Position,
    goal: Position,
) -> Optional[int]:
    path = bfs_shortest_path(terrain_grid, start, goal)
    return None if path is None else len(path) - 1


def dijkstra_optimal_path(
    terrain_grid: np.ndarray,
    start: Position,
    goal: Position,
    *,
    max_transition_evaluations: Optional[int] = None,
) -> WeightedSearchResult:
    """Unbounded-by-default optimal accumulated-terrain-cost path."""
    terrain_grid = validate_terrain(terrain_grid)
    if not is_walkable(terrain_grid, start) or not is_walkable(terrain_grid, goal):
        raise ValueError("Start and goal must be walkable cells")
    if start == goal:
        return WeightedSearchResult(
            success=True,
            path=[start],
            actions=[],
            path_cost=0.0,
            state_expansions=0,
            unique_state_expansions=0,
            reexpansions=0,
            transition_evaluations=0,
            termination_reason="goal",
        )

    dist: Dict[Position, float] = {start: 0.0}
    parent: Dict[Position, Optional[Position]] = {start: None}
    counter = 0
    frontier: List[Tuple[float, int, Position]] = [(0.0, counter, start)]
    counter += 1
    settled: set[Position] = set()
    expansions = 0
    unique_expansions = 0
    reexpansions = 0
    transition_evaluations = 0
    termination_reason = "frontier_exhausted"

    while frontier:
        distance, _, position = heapq.heappop(frontier)
        if position in settled:
            continue
        settled.add(position)
        expansions += 1
        if position == goal:
            path: List[Position] = []
            current: Optional[Position] = goal
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return WeightedSearchResult(
                success=True,
                path=path,
                actions=path_to_actions(path),
                path_cost=float(distance),
                state_expansions=expansions,
                unique_state_expansions=unique_expansions,
                reexpansions=reexpansions,
                transition_evaluations=transition_evaluations,
                termination_reason="goal",
            )
        row, col = position
        outgoing = 0
        for action, (dr, dc) in enumerate(ACTION_DELTAS):
            nxt = (row + dr, col + dc)
            if not is_walkable(terrain_grid, nxt):
                continue
            outgoing += 1
            if (
                max_transition_evaluations is not None
                and transition_evaluations + 1 > max_transition_evaluations
            ):
                termination_reason = "transition_budget"
                return WeightedSearchResult(
                    success=False,
                    state_expansions=expansions,
                    unique_state_expansions=unique_expansions,
                    reexpansions=reexpansions,
                    transition_evaluations=transition_evaluations,
                    termination_reason=termination_reason,
                )
            transition_evaluations += 1
            candidate = distance + movement_cost(terrain_grid, nxt)
            if candidate < dist.get(nxt, float("inf")):
                if nxt in settled:
                    reexpansions += 1
                dist[nxt] = candidate
                parent[nxt] = position
                heapq.heappush(frontier, (candidate, counter, nxt))
                counter += 1
        if outgoing:
            unique_expansions += 1

    return WeightedSearchResult(
        success=False,
        state_expansions=expansions,
        unique_state_expansions=unique_expansions,
        reexpansions=reexpansions,
        transition_evaluations=transition_evaluations,
        termination_reason=termination_reason,
    )


def terrain_statistics(terrain_grid: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[float, float, float]]:
    flat = np.asarray(terrain_grid).reshape(-1)
    free = flat[flat != WALL]
    counts = tuple(int(np.sum(free == value)) for value in TERRAIN_VALUES)
    total = int(counts[0] + counts[1] + counts[2])
    ratios = tuple((count / total) if total else 0.0 for count in counts)
    return counts, ratios  # type: ignore[return-value]


def terrain_map_hash(
    terrain_grid: np.ndarray,
    start: Optional[Position] = None,
    goal: Optional[Position] = None,
) -> str:
    """SHA-256 over the terrain layout only.

    Start/goal are stored separately in the dataset.  Hashing the full terrain
    grid (rather than obstacle classes alone) enforces stronger split
    independence while ignoring endpoint choice.
    """
    del start, goal
    array = validate_terrain(terrain_grid).astype(np.uint8, copy=False)
    header = f"{array.shape[0]}x{array.shape[1]}|V2|terrain|".encode("ascii")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


class MazeEnvironment:
    """Stateful weighted-grid environment for a single maze."""

    def __init__(
        self,
        terrain_grid: np.ndarray,
        start: Position,
        goal: Position,
        *,
        copy_terrain: bool = True,
    ) -> None:
        array = validate_terrain(terrain_grid)
        self.terrain_grid = array.copy() if copy_terrain else array
        self.start = (int(start[0]), int(start[1]))
        self.goal = (int(goal[0]), int(goal[1]))
        if not in_bounds(self.terrain_grid, self.start):
            raise ValueError(f"Start out of bounds: {self.start}")
        if not in_bounds(self.terrain_grid, self.goal):
            raise ValueError(f"Goal out of bounds: {self.goal}")
        if not is_walkable(self.terrain_grid, self.start):
            raise ValueError("Start cannot be a wall")
        if not is_walkable(self.terrain_grid, self.goal):
            raise ValueError("Goal cannot be a wall")
        if self.start == self.goal:
            raise ValueError("Start and goal must be different")
        self.agent_position: Position = self.start
        self.done = False
        self.steps = 0
        self.cumulative_cost = 0

    @property
    def height(self) -> int:
        return int(self.terrain_grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.terrain_grid.shape[1])

    def reset(self) -> Position:
        self.agent_position = self.start
        self.done = False
        self.steps = 0
        self.cumulative_cost = 0
        return self.agent_position

    def get_valid_actions(self, position: Optional[Position] = None) -> List[int]:
        target = self.agent_position if position is None else position
        return valid_actions_for(self.terrain_grid, target)

    def get_valid_action_mask(self, position: Optional[Position] = None) -> np.ndarray:
        target = self.agent_position if position is None else position
        return valid_action_mask(self.terrain_grid, target)

    def get_successor(self, position: Position, action: int) -> Position:
        if action not in (UP, DOWN, LEFT, RIGHT):
            raise InvalidActionError(f"Unknown action id: {action}")
        if not in_bounds(self.terrain_grid, position):
            raise InvalidActionError(f"Position out of bounds: {position}")
        if not is_walkable(self.terrain_grid, position):
            raise InvalidActionError(f"Position is a wall: {position}")
        nxt = successor_position(position, action)
        if not is_walkable(self.terrain_grid, nxt):
            raise InvalidActionError(
                f"Action {ACTION_NAMES[action]} from {position} enters a wall or leaves the map"
            )
        return nxt

    def action_cost(self, position: Position, action: int) -> int:
        return movement_cost(self.terrain_grid, self.get_successor(position, action))

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("Cannot step a finished episode; call reset()")
        next_position = self.get_successor(self.agent_position, action)
        cost = movement_cost(self.terrain_grid, next_position)
        self.agent_position = next_position
        self.steps += 1
        self.cumulative_cost += cost
        self.done = self.agent_position == self.goal
        return StepResult(
            position=next_position,
            reward=-float(cost),
            movement_cost=int(cost),
            cumulative_cost=int(self.cumulative_cost),
            done=self.done,
        )

    def path_cost(self, path: Sequence[Position]) -> int:
        return path_terrain_cost(self.terrain_grid, path)

    @classmethod
    def from_instance(cls, instance: MazeInstance) -> "MazeEnvironment":
        return cls(instance.terrain_grid, instance.start, instance.goal)

    def __repr__(self) -> str:
        return (
            f"MazeEnvironment(start={self.start}, goal={self.goal}, "
            f"position={self.agent_position}, cumulative_cost={self.cumulative_cost})"
        )