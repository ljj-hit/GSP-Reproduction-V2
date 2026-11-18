"""Cost-optimal traditional search baselines for the weighted 16 x 16 grid.

Dijkstra is the unbudgeted optimal-cost reference.  A* and weighted A* use the
same 512 transition-evaluation budget as GSP-RGNN+WA*.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from environment import (
    ACTION_DELTAS,
    MazeEnvironment,
    Position,
    WeightedSearchResult,
    dijkstra_optimal_path,
)


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def run_dijkstra(
    terrain_grid: np.ndarray,
    start: Position,
    goal: Position,
    *,
    max_transition_evaluations: Optional[int] = None,
) -> WeightedSearchResult:
    """Dijkstra reference; by default it is not constrained by the 512 budget."""
    return dijkstra_optimal_path(
        terrain_grid,
        start,
        goal,
        max_transition_evaluations=max_transition_evaluations,
    )


class HeuristicSearch:
    """A* / WA* over state nodes with cumulative terrain cost."""

    def __init__(
        self,
        weight: float,
        transition_budget: int,
        *,
        name: str = "A*",
    ) -> None:
        if weight <= 0:
            raise ValueError("Heuristic weight must be positive")
        if transition_budget <= 0:
            raise ValueError("transition_budget must be positive")
        self.weight = float(weight)
        self.transition_budget = int(transition_budget)
        self.name = name

    def search(
        self,
        terrain_grid: np.ndarray,
        start: Position,
        goal: Position,
    ) -> WeightedSearchResult:
        environment = MazeEnvironment(terrain_grid, start, goal)
        start, goal = environment.start, environment.goal
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

        frontier: List[Tuple[float, int, Position, int]] = []
        counter = 0
        initial_h = _manhattan(start, goal)
        heapq.heappush(frontier, (self.weight * initial_h, counter, start, 0))
        counter += 1
        visited = {start}
        expanded: set[Position] = set()
        parent: Dict[Position, Optional[Position]] = {start: None}
        best_cost: Dict[Position, int] = {start: 0}
        frontier_trace: List[int] = []
        node_expansions = 0
        unique_state_expansions = 0
        reexpansions = 0
        transition_evaluations = 0
        termination_reason = "frontier_exhausted"
        budget_blocked = False

        while frontier:
            _, _, position, cost = heapq.heappop(frontier)
            node_expansions += 1
            if cost > best_cost.get(position, 10**18):
                continue
            if position == goal:
                path: List[Position] = []
                current: Optional[Position] = goal
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                actions = [
                    ACTION_DELTAS.index((b[0] - a[0], b[1] - a[1]))
                    for a, b in zip(path[:-1], path[1:])
                ]
                return WeightedSearchResult(
                    success=True,
                    path=path,
                    actions=actions,
                    path_cost=float(cost),
                    state_expansions=node_expansions,
                    unique_state_expansions=unique_state_expansions,
                    reexpansions=reexpansions,
                    transition_evaluations=transition_evaluations,
                    termination_reason="goal",
                )

            if position in expanded:
                reexpansions += 1
            else:
                expanded.add(position)
                unique_state_expansions += 1

            actions = environment.get_valid_actions(position)
            if transition_evaluations + len(actions) > self.transition_budget:
                budget_blocked = True
                termination_reason = "transition_budget"
                break

            for action in actions:
                transition_evaluations += 1
                nxt = environment.get_successor(position, action)
                step_cost = int(environment.terrain_grid[nxt])
                candidate = cost + step_cost
                if candidate < best_cost.get(nxt, 10**18):
                    best_cost[nxt] = candidate
                    # Standard first-discovery parent selection; no reopening in
                    # this experimental comparison implementation.
                    if nxt not in visited:
                        visited.add(nxt)
                        parent[nxt] = position
                        priority = candidate + self.weight * _manhattan(nxt, goal)
                        heapq.heappush(frontier, (priority, counter, nxt, candidate))
                        counter += 1
            frontier_trace.append(len(frontier))

        max_frontier = max(frontier_trace) if frontier_trace else 0
        del max_frontier
        if budget_blocked:
            termination_reason = "transition_budget"
        return WeightedSearchResult(
            success=False,
            state_expansions=node_expansions,
            unique_state_expansions=unique_state_expansions,
            reexpansions=reexpansions,
            transition_evaluations=transition_evaluations,
            termination_reason=termination_reason,
        )


def run_astar(
    terrain_grid: np.ndarray,
    start: Position,
    goal: Position,
    transition_budget: int = 512,
) -> WeightedSearchResult:
    return HeuristicSearch(1.0, transition_budget, name="A*").search(
        terrain_grid, start, goal
    )


def run_weighted_astar(
    terrain_grid: np.ndarray,
    start: Position,
    goal: Position,
    weight: float = 2.0,
    transition_budget: int = 512,
) -> WeightedSearchResult:
    return HeuristicSearch(weight, transition_budget, name="WA*").search(
        terrain_grid, start, goal
    )