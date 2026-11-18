"""GSP state-action Weighted A* with R-GNN guidance on the weighted grid.

Priority (maximised):
    f(s,a) = g(s) + w * Q_theta(s,a),  g(s) = -C(start,s)

The budget is counted in **transition evaluations**: every legal state-action
candidate generated and evaluated for a state consumes one unit.  The strict
budget is 512 by default.  First-discovery parents are never rewired.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import Config
from environment import (
    MazeEnvironment,
    MazeInstance,
    Position,
    path_terrain_cost,
)
from relational_graph import (
    RelationalGraphBuilder,
    batch_relational_graphs,
)

EdgeKey = Tuple[Position, int]


@dataclass
class TransitionExperience:
    """Compact transition stored by the replay buffer."""

    instance_id: int
    position: Position
    goal: Position
    action: int
    next_position: Position
    reward: float
    return_bound: float
    terminal: bool
    dead_end: bool

    def __post_init__(self) -> None:
        self.instance_id = int(self.instance_id)
        self.position = (int(self.position[0]), int(self.position[1]))
        self.goal = (int(self.goal[0]), int(self.goal[1]))
        self.action = int(self.action)
        self.next_position = (
            int(self.next_position[0]),
            int(self.next_position[1]),
        )
        self.reward = float(self.reward)
        self.return_bound = float(self.return_bound)
        self.terminal = bool(self.terminal)
        self.dead_end = bool(self.dead_end)
        if not 0 <= self.action < 4:
            raise ValueError("Action must be in [0, 3]")


@dataclass
class SearchResult:
    success: bool
    path: List[Position] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    path_cost: Optional[float] = None
    path_reward: Optional[float] = None
    transition_evaluations: int = 0
    node_expansions: int = 0
    unique_state_expansions: int = 0
    reexpansions: int = 0
    frontier_size_trace: List[int] = field(default_factory=list)
    max_frontier_size: int = 0
    termination_reason: str = "unknown"
    inference_time_seconds: float = 0.0
    q_evaluations: int = 0
    dead_end_transitions: int = 0
    experiences: List[TransitionExperience] = field(default_factory=list)
    parent_edges: Dict[Position, Optional[Tuple[Position, int]]] = field(
        default_factory=dict
    )

    @property
    def plan_length(self) -> int:
        return len(self.actions)

    @property
    def state_expansions(self) -> int:
        return self.unique_state_expansions + self.reexpansions

    @property
    def hit_budget(self) -> bool:
        return self.termination_reason == "transition_budget"


class QGuidedWAStar:
    """R-GNN-guided WA* over (state, action) nodes."""

    def __init__(
        self,
        network: nn.Module,
        graph_builder: RelationalGraphBuilder,
        config: Config,
        device: str | torch.device,
        *,
        weight: Optional[float] = None,
        transition_budget: Optional[int] = None,
    ) -> None:
        self.network = network
        self.graph_builder = graph_builder
        self.config = config
        self.device = torch.device(device)
        self.weight = float(config.wa_star_weight if weight is None else weight)
        self.transition_budget = int(
            config.transition_budget if transition_budget is None else transition_budget
        )
        if self.weight <= 0:
            raise ValueError("WA* weight must be positive")
        if self.transition_budget <= 0:
            raise ValueError("transition_budget must be positive")
        if not self._network_on_device():
            raise ValueError(
                f"Network parameters are not on search device {self.device}"
            )

    def _network_on_device(self) -> bool:
        for parameter in self.network.parameters():
            if parameter.device != self.device:
                return False
        for buffer in self.network.buffers():
            if buffer.device != self.device:
                return False
        return True

    def _freeze(self) -> Tuple[bool, List[bool]]:
        was_training = bool(getattr(self.network, "training", False))
        flags = [bool(parameter.requires_grad) for parameter in self.network.parameters()]
        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        return was_training, flags

    def _restore(self, was_training: bool, flags: Sequence[bool]) -> None:
        for parameter, flag in zip(self.network.parameters(), flags):
            parameter.requires_grad_(flag)
        self.network.train(was_training)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _evaluate_actions(
        self,
        instance: MazeInstance,
        position: Position,
    ) -> Dict[int, float]:
        """Build one state graph, evaluate every legal action once."""
        cached = self._q_cache.get(position)
        if cached is not None:
            return cached
        graph_start = time.perf_counter()
        graph = self.graph_builder.build(instance, position)
        batch = batch_relational_graphs([graph], device=self.device)
        graph_build_time = time.perf_counter() - graph_start

        self._sync()
        inference_start = time.perf_counter()
        with torch.inference_mode():
            output = self.network(batch)
        self._sync()
        inference_elapsed = time.perf_counter() - inference_start
        self._inference_time += inference_elapsed
        self._graph_time += graph_build_time
        self._q_evaluations += 1

        q_values = output.q_values.detach().float().cpu().numpy()
        action_ids = output.action_ids.detach().cpu().numpy().tolist()
        if len(action_ids) != len(graph.action_ids):
            raise RuntimeError("R-GNN action output count mismatch")
        if not np.all(np.isfinite(q_values)):
            raise FloatingPointError(
                f"Non-finite R-GNN Q values at position {position}: {q_values}"
            )
        mapping = {
            int(action): float(q_values[index])
            for index, action in enumerate(action_ids)
        }
        self._q_cache[position] = mapping
        return mapping

    def search(
        self,
        instance: MazeInstance,
    ) -> SearchResult:
        environment = MazeEnvironment.from_instance(instance)
        start = environment.start
        goal = environment.goal
        maze = environment.terrain_grid

        self._q_cache: Dict[Position, Dict[int, float]] = {}
        self._inference_time = 0.0
        self._graph_time = 0.0
        self._q_evaluations = 0
        self._counter = 0
        self._frontier_trace: List[int] = []

        was_training, grad_flags = self._freeze()
        try:
            with torch.inference_mode():
                # Frontier stores (-f, unique sequence, source state, action).
                frontier: List[Tuple[float, int, Position, int]] = []
                visited: set[Position] = {start}
                expanded_positions: set[Position] = set()
                state_cost: Dict[Position, int] = {start: 0}
                parent: Dict[Position, Optional[Tuple[Position, int]]] = {
                    start: None
                }
                transitions: Dict[EdgeKey, TransitionExperience] = {}
                transition_evaluations = 0
                node_expansions = 0
                unique_state_expansions = 0
                reexpansions = 0
                dead_end_count = 0
                budget_blocked = False
                success = False
                termination_reason = "frontier_exhausted"
                actions_to_goal: List[int] = []

                def push_state_actions(position: Position) -> bool:
                    """Evaluate one state's legal actions and push all candidates."""
                    nonlocal transition_evaluations
                    nonlocal unique_state_expansions
                    nonlocal reexpansions
                    nonlocal budget_blocked
                    legal_actions = environment.get_valid_actions(position)
                    if not legal_actions:
                        return True
                    already_expanded = position in expanded_positions
                    if transition_evaluations + len(legal_actions) > self.transition_budget:
                        budget_blocked = True
                        return False
                    transition_evaluations += len(legal_actions)
                    q_values = self._evaluate_actions(instance, position)
                    expanded_positions.add(position)
                    if already_expanded:
                        reexpansions += 1
                    else:
                        unique_state_expansions += 1
                    g_value = -float(state_cost[position])
                    for action in legal_actions:
                        priority = g_value + self.weight * q_values[action]
                        heapq.heappush(
                            frontier,
                            (-priority, self._counter, position, action),
                        )
                        self._counter += 1
                    self._frontier_trace.append(len(frontier))
                    return True

                push_state_actions(start)

                while frontier:
                    _, _, position, action = heapq.heappop(frontier)
                    node_expansions += 1
                    next_position = environment.get_successor(position, action)
                    movement_cost = int(environment.terrain_grid[next_position])
                    reward = -float(movement_cost)
                    cumulative_next_cost = state_cost[position] + movement_cost

                    legal_next_actions = environment.get_valid_actions(next_position)
                    is_goal = next_position == goal
                    is_dead_end = (not is_goal) and len(legal_next_actions) == 0

                    transition = TransitionExperience(
                        instance_id=instance.instance_id,
                        position=position,
                        goal=goal,
                        action=action,
                        next_position=next_position,
                        reward=reward,
                        return_bound=(
                            -1.0
                            if is_goal
                            else (self.config.dead_end_return if is_dead_end else -math.inf)
                        ),
                        terminal=is_goal,
                        dead_end=is_dead_end,
                    )
                    transitions[(position, action)] = transition
                    if is_dead_end:
                        dead_end_count += 1

                    if is_goal:
                        parent[goal] = (position, action)
                        state_cost[goal] = cumulative_next_cost
                        success = True
                        termination_reason = "goal"
                        break

                    if is_dead_end:
                        continue

                    if next_position not in visited:
                        visited.add(next_position)
                        state_cost[next_position] = cumulative_next_cost
                        parent[next_position] = (position, action)
                        # This state is discovered, but its outgoing transitions
                        # are only expanded if the strict budget can pay for all
                        # legal candidates.
                        push_state_actions(next_position)

                if success:
                    reversed_actions: List[int] = []
                    current = goal
                    while parent[current] is not None:
                        previous, edge_action = parent[current]  # type: ignore[misc]
                        reversed_actions.append(edge_action)
                        current = previous
                    actions_to_goal = list(reversed(reversed_actions))
                    path = [start]
                    cursor = start
                    for edge_action in actions_to_goal:
                        cursor = environment.get_successor(cursor, edge_action)
                        path.append(cursor)
                    path_cost = float(path_terrain_cost(maze, path))
                    path_reward = -path_cost

                    # Back-propagate actual remaining path rewards.
                    return_value = 0.0
                    current = goal
                    while parent[current] is not None:
                        previous, edge_action = parent[current]  # type: ignore[misc]
                        edge = transitions.get((previous, edge_action))
                        if edge is None:
                            raise RuntimeError("Missing edge on successful path")
                        return_value += edge.reward
                        edge.return_bound = return_value
                        current = previous
                else:
                    path = []
                    path_cost = None
                    path_reward = None
                    if budget_blocked or transition_evaluations >= self.transition_budget:
                        termination_reason = "transition_budget"

                max_frontier = max(self._frontier_trace) if self._frontier_trace else 0
                return SearchResult(
                    success=success,
                    path=path,
                    actions=actions_to_goal,
                    path_cost=path_cost,
                    path_reward=path_reward,
                    transition_evaluations=transition_evaluations,
                    node_expansions=node_expansions,
                    unique_state_expansions=unique_state_expansions,
                    reexpansions=reexpansions,
                    frontier_size_trace=list(self._frontier_trace),
                    max_frontier_size=max_frontier,
                    termination_reason=termination_reason,
                    inference_time_seconds=self._inference_time,
                    q_evaluations=self._q_evaluations,
                    dead_end_transitions=dead_end_count,
                    experiences=list(transitions.values()),
                    parent_edges=dict(parent),
                )
        finally:
            self._restore(was_training, grad_flags)

    def search_from_positions(
        self,
        instance: MazeInstance,
    ) -> SearchResult:
        return self.search(instance)

    @property
    def graph_build_time_seconds(self) -> float:
        """Last-search time spent constructing dynamic graphs (search overhead)."""
        return self._graph_time