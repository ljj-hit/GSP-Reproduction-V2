"""Relational graph construction for the V2 weighted-grid R-GNN.

Each graph contains all 256 cell objects (walls included), then one action
object for every legal action in the current state.  Static map facts are
cached and reused; CURRENT and MOVE_* facts are rebuilt for each state.

Cell IDs are compact object indices only.  Row/column indices and fixed
one-hot cell features are never supplied to the network.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from config import Config
from environment import (
    ACTION_NAMES,
    ACTION_DELTAS,
    MazeInstance,
    Position,
    TERRAIN_VALUES,
    WALL,
    valid_actions_for,
)


TERRAIN_PREDICATES: Dict[int, str] = {
    1: "TERRAIN_1",
    3: "TERRAIN_3",
    5: "TERRAIN_5",
}
UNARY_PREDICATES: Tuple[str, ...] = (
    "WALL",
    "TERRAIN_1",
    "TERRAIN_3",
    "TERRAIN_5",
    "CURRENT",
    "GOAL",
)
SPATIAL_PREDICATES: Tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT")
ACTION_PREDICATES: Tuple[str, ...] = tuple(f"MOVE_{name}" for name in ACTION_NAMES)
PREDICATES: Tuple[str, ...] = UNARY_PREDICATES + SPATIAL_PREDICATES + ACTION_PREDICATES
PREDICATE_ARITIES: Dict[str, int] = {
    **{name: 1 for name in UNARY_PREDICATES},
    **{name: 2 for name in SPATIAL_PREDICATES},
    **{name: 3 for name in ACTION_PREDICATES},
}
ACTION_PREDICATE_BY_ACTION_ID: Dict[int, str] = {
    action: f"MOVE_{ACTION_NAMES[action]}" for action in range(4)
}


def _empty_relations() -> Dict[str, np.ndarray]:
    return {
        predicate: np.empty((0, PREDICATE_ARITIES[predicate]), dtype=np.int64)
        for predicate in PREDICATES
    }


@dataclass
class RelationalGraph:
    """A single relational graph with local object indices."""

    num_objects: int
    relations: Dict[str, np.ndarray]
    cell_object_ids: Tuple[int, ...]
    action_object_ids: Tuple[int, ...]
    action_ids: Tuple[int, ...]
    current_cell_id: int
    goal_cell_id: int
    instance_id: int = -1
    structure: str = "unknown"
    map_hash: str = ""
    position: Position = (-1, -1)
    num_cells: int = 256

    def validate(self) -> None:
        if len(self.relations) != len(PREDICATES):
            raise ValueError("Graph must contain every predicate (possibly empty)")
        for predicate, facts in self.relations.items():
            arity = PREDICATE_ARITIES[predicate]
            if facts.ndim != 2 or facts.shape[1] != arity:
                raise ValueError(f"Invalid relation array shape for {predicate}: {facts.shape}")
            if facts.size and (facts.min() < 0 or facts.max() >= self.num_objects):
                raise ValueError(f"Object index out of range in predicate {predicate}")
        if len(self.action_object_ids) != len(self.action_ids):
            raise ValueError("Action object/action id count mismatch")
        if len(self.relations["CURRENT"]) != 1:
            raise ValueError("Exactly one CURRENT fact is required")
        if len(self.relations["GOAL"]) != 1:
            raise ValueError("Exactly one GOAL fact is required")
        if self.current_cell_id != int(self.relations["CURRENT"][0, 0]):
            raise ValueError("CURRENT cell id mismatch")
        if self.goal_cell_id != int(self.relations["GOAL"][0, 0]):
            raise ValueError("GOAL cell id mismatch")
        if len(self.relations["WALL"]) + sum(
            len(self.relations[TERRAIN_PREDICATES[value]]) for value in TERRAIN_VALUES
        ) != self.num_cells:
            raise ValueError("Each cell must have exactly one terrain/WALL predicate")


@dataclass(frozen=True)
class StaticMapGraph:
    num_cells: int
    goal_cell_id: int
    map_hash: str
    relations: Dict[str, np.ndarray]


def build_static_map_graph(instance: MazeInstance) -> StaticMapGraph:
    """Build the state-independent relation facts for a maze and goal."""
    terrain = instance.terrain_grid
    height, width = terrain.shape
    num_cells = int(height * width)
    relations = _empty_relations()

    wall_ids: List[int] = []
    terrain_ids: Dict[int, List[int]] = {1: [], 3: [], 5: []}
    for row in range(height):
        for col in range(width):
            cell_id = row * width + col
            value = int(terrain[row, col])
            if value == WALL:
                wall_ids.append(cell_id)
            else:
                terrain_ids[value].append(cell_id)
    relations["WALL"] = np.asarray(wall_ids, dtype=np.int64).reshape(-1, 1)
    for value in TERRAIN_VALUES:
        relations[TERRAIN_PREDICATES[value]] = np.asarray(
            terrain_ids[value], dtype=np.int64
        ).reshape(-1, 1)

    for row in range(height):
        for col in range(width):
            source_id = row * width + col
            for direction, (dr, dc) in enumerate(ACTION_DELTAS):
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width:
                    target_id = nr * width + nc
                    predicate = SPATIAL_PREDICATES[direction]
                    facts = relations[predicate]
                    relations[predicate] = np.vstack(
                        (facts, np.asarray([[source_id, target_id]], dtype=np.int64))
                    )

    goal_id = instance.goal[0] * width + instance.goal[1]
    relations["GOAL"] = np.asarray([[goal_id]], dtype=np.int64)
    return StaticMapGraph(
        num_cells=num_cells,
        goal_cell_id=int(goal_id),
        map_hash=instance.map_hash
        or f"{instance.terrain_grid.shape}:{instance.start}:{instance.goal}",
        relations=relations,
    )


def build_relational_graph(
    instance: MazeInstance,
    position: Position,
    static_graph: Optional[StaticMapGraph] = None,
) -> RelationalGraph:
    """Build a dynamic graph for the current position and all legal actions."""
    static = static_graph if static_graph is not None else build_static_map_graph(instance)
    height, width = instance.terrain_grid.shape
    if not (0 <= position[0] < height and 0 <= position[1] < width):
        raise ValueError(f"Position out of bounds: {position}")
    if int(instance.terrain_grid[position]) == WALL:
        raise ValueError(f"Position is a wall: {position}")

    current_id = position[0] * width + position[1]
    legal_actions = valid_actions_for(instance.terrain_grid, position)
    action_object_ids = [static.num_cells + index for index in range(len(legal_actions))]
    action_ids = [int(action) for action in legal_actions]
    num_objects = static.num_cells + len(legal_actions)

    relations = {name: facts for name, facts in static.relations.items()}
    relations["CURRENT"] = np.asarray([[current_id]], dtype=np.int64)
    # Static graph leaves every action predicate empty; dynamic facts overwrite
    # only the predicates that occur in this state.
    for predicate in ACTION_PREDICATES:
        relations[predicate] = np.empty((0, 3), dtype=np.int64)
    for action_object_id, action in zip(action_object_ids, action_ids):
        dr, dc = ACTION_DELTAS[action]
        target = (position[0] + dr, position[1] + dc)
        target_id = target[0] * width + target[1]
        predicate = ACTION_PREDICATE_BY_ACTION_ID[action]
        relations[predicate] = np.vstack(
            (
                relations[predicate],
                np.asarray(
                    [[action_object_id, current_id, target_id]], dtype=np.int64
                ),
            )
        )

    graph = RelationalGraph(
        num_objects=num_objects,
        relations=relations,
        cell_object_ids=tuple(range(static.num_cells)),
        action_object_ids=tuple(action_object_ids),
        action_ids=tuple(action_ids),
        current_cell_id=current_id,
        goal_cell_id=static.goal_cell_id,
        instance_id=int(instance.instance_id),
        structure=instance.structure,
        map_hash=instance.map_hash,
        position=(int(position[0]), int(position[1])),
        num_cells=static.num_cells,
    )
    graph.validate()
    return graph


class RelationalGraphBuilder:
    """Static-map cache plus bounded LRU cache of dynamic state graphs."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._static_cache: Dict[Tuple[Any, ...], StaticMapGraph] = {}
        self._dynamic_cache: "OrderedDict[Tuple[Any, ...], RelationalGraph]" = OrderedDict()
        self.dynamic_cache_hits = 0
        self.dynamic_cache_misses = 0

    def _static_key(self, instance: MazeInstance) -> Tuple[Any, ...]:
        return (
            instance.map_hash,
            instance.terrain_grid.shape,
            instance.goal,
        )

    def get_static(self, instance: MazeInstance) -> StaticMapGraph:
        key = self._static_key(instance)
        cached = self._static_cache.get(key)
        if cached is None:
            cached = build_static_map_graph(instance)
            self._static_cache[key] = cached
        return cached

    def build(self, instance: MazeInstance, position: Position) -> RelationalGraph:
        static_key = self._static_key(instance)
        dynamic_key = (static_key, int(position[0]), int(position[1]))
        cached = self._dynamic_cache.get(dynamic_key)
        if cached is not None:
            self.dynamic_cache_hits += 1
            self._dynamic_cache.move_to_end(dynamic_key)
            return cached
        self.dynamic_cache_misses += 1
        static = self.get_static(instance)
        graph = build_relational_graph(instance, position, static)
        self._dynamic_cache[dynamic_key] = graph
        self._dynamic_cache.move_to_end(dynamic_key)
        while len(self._dynamic_cache) > self.config.graph_cache_size:
            self._dynamic_cache.popitem(last=False)
        return graph

    def clear_dynamic_cache(self) -> None:
        self._dynamic_cache.clear()

    def cache_stats(self) -> Dict[str, int]:
        return {
            "static_maps": len(self._static_cache),
            "dynamic_graphs": len(self._dynamic_cache),
            "dynamic_hits": self.dynamic_cache_hits,
            "dynamic_misses": self.dynamic_cache_misses,
        }


@dataclass
class RelationalGraphBatch:
    """Disjoint-union batch; object offsets prevent cross-graph edges."""

    num_graphs: int
    num_objects: int
    relations: Dict[str, torch.Tensor]
    object_graph_indices: torch.Tensor
    action_object_indices: torch.Tensor
    action_graph_indices: torch.Tensor
    action_ids: torch.Tensor
    action_offsets: Tuple[int, ...]
    graph_action_ids: Tuple[Tuple[int, ...], ...]
    graph_object_counts: Tuple[int, ...]
    graph_action_counts: Tuple[int, ...]
    device: torch.device = torch.device("cpu")

    def to(self, device: str | torch.device) -> "RelationalGraphBatch":
        target = torch.device(device)
        for name in list(self.relations):
            self.relations[name] = self.relations[name].to(target)
        self.object_graph_indices = self.object_graph_indices.to(target)
        self.action_object_indices = self.action_object_indices.to(target)
        self.action_graph_indices = self.action_graph_indices.to(target)
        self.action_ids = self.action_ids.to(target)
        self.device = target
        return self

    def select_action_indices(self, actions: Sequence[int]) -> torch.Tensor:
        """Return flat action-object positions corresponding to given actions."""
        if len(actions) != self.num_graphs:
            raise ValueError(
                f"Expected {self.num_graphs} actions, got {len(actions)}"
            )
        flat: List[int] = []
        for graph_index, action in enumerate(actions):
            try:
                local_index = self.graph_action_ids[graph_index].index(int(action))
            except ValueError as exc:
                raise ValueError(
                    f"Action {action} is illegal in graph {graph_index}"
                ) from exc
            flat.append(self.action_offsets[graph_index] + local_index)
        return torch.as_tensor(flat, dtype=torch.long, device=self.device)

    def action_max_per_graph(self, q_values: torch.Tensor) -> torch.Tensor:
        """Max action Q per graph; invalid/no-action graphs receive -inf."""
        if q_values.ndim != 1 or q_values.shape[0] != len(self.action_ids):
            raise ValueError("Q output must be a flat per-action tensor")
        result = torch.full(
            (self.num_graphs,),
            -float("inf"),
            dtype=q_values.dtype,
            device=q_values.device,
        )
        if q_values.numel():
            result.scatter_reduce_(
                0,
                self.action_graph_indices,
                q_values,
                reduce="amax",
                include_self=True,
            )
        return result

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_graphs": self.num_graphs,
            "num_objects": self.num_objects,
            "relations": {key: len(value) for key, value in self.relations.items()},
            "num_action_objects": int(self.action_object_indices.numel()),
            "graph_object_counts": self.graph_object_counts,
            "graph_action_counts": self.graph_action_counts,
        }


def batch_relational_graphs(
    graphs: Sequence[RelationalGraph],
    *,
    device: str | torch.device = "cpu",
) -> RelationalGraphBatch:
    if len(graphs) == 0:
        raise ValueError("Cannot batch zero graphs")
    target = torch.device(device)
    graph_object_counts = tuple(int(graph.num_objects) for graph in graphs)
    graph_action_counts = tuple(len(graph.action_ids) for graph in graphs)
    object_offsets: List[int] = []
    total_objects = 0
    for graph in graphs:
        object_offsets.append(total_objects)
        total_objects += graph.num_objects

    relation_tensors: Dict[str, torch.Tensor] = {}
    for predicate in PREDICATES:
        arity = PREDICATE_ARITIES[predicate]
        shifted: List[np.ndarray] = []
        for graph, offset in zip(graphs, object_offsets):
            facts = graph.relations[predicate]
            if len(facts):
                shifted.append(facts + offset)
        if shifted:
            array = np.concatenate(shifted, axis=0).astype(np.int64, copy=False)
        else:
            array = np.empty((0, arity), dtype=np.int64)
        relation_tensors[predicate] = torch.as_tensor(array, dtype=torch.long, device=target)

    object_graph_indices = torch.repeat_interleave(
        torch.arange(len(graphs), dtype=torch.long, device=target),
        torch.as_tensor(graph_object_counts, dtype=torch.long, device=target),
    )

    action_object_indices: List[int] = []
    action_graph_indices: List[int] = []
    action_ids: List[int] = []
    action_offsets: List[int] = []
    action_cursor = 0
    for graph_index, (graph, offset) in enumerate(zip(graphs, object_offsets)):
        action_offsets.append(action_cursor)
        for action_object_id, action in zip(graph.action_object_ids, graph.action_ids):
            action_object_indices.append(int(action_object_id) + offset)
            action_graph_indices.append(graph_index)
            action_ids.append(int(action))
            action_cursor += 1

    return RelationalGraphBatch(
        num_graphs=len(graphs),
        num_objects=total_objects,
        relations=relation_tensors,
        object_graph_indices=object_graph_indices,
        action_object_indices=torch.as_tensor(
            action_object_indices, dtype=torch.long, device=target
        ),
        action_graph_indices=torch.as_tensor(
            action_graph_indices, dtype=torch.long, device=target
        ),
        action_ids=torch.as_tensor(action_ids, dtype=torch.long, device=target),
        action_offsets=tuple(action_offsets),
        graph_action_ids=tuple(tuple(int(a) for a in graph.action_ids) for graph in graphs),
        graph_object_counts=graph_object_counts,
        graph_action_counts=graph_action_counts,
        device=target,
    )


def remap_graph_objects(
    graph: RelationalGraph,
    old_to_new: Sequence[int],
) -> RelationalGraph:
    """Remap every object ID while preserving all relation structure.

    This utility exists for permutation-invariance tests and for applications
    that need to compose graphs with externally assigned object IDs.
    """
    mapping = np.asarray(old_to_new, dtype=np.int64)
    if mapping.shape != (graph.num_objects,):
        raise ValueError("old_to_new must assign one new ID per object")
    if len(set(int(value) for value in mapping.tolist())) != graph.num_objects:
        raise ValueError("Object remapping must be a permutation")
    relations = {
        predicate: mapping[facts]
        for predicate, facts in graph.relations.items()
    }
    return RelationalGraph(
        num_objects=graph.num_objects,
        relations=relations,
        cell_object_ids=tuple(int(mapping[i]) for i in graph.cell_object_ids),
        action_object_ids=tuple(int(mapping[i]) for i in graph.action_object_ids),
        action_ids=graph.action_ids,
        current_cell_id=int(mapping[graph.current_cell_id]),
        goal_cell_id=int(mapping[graph.goal_cell_id]),
        instance_id=graph.instance_id,
        structure=graph.structure,
        map_hash=graph.map_hash,
        position=graph.position,
        num_cells=graph.num_cells,
    )