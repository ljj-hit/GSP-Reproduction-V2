"""FIFO replay buffer of compact weighted-grid transitions.

No fixed-length MLP feature vector is stored.  Each transition references its
maze by ``instance_id``; static map facts and dynamic state graphs are rebuilt
or retrieved from the graph builder's cache when a batch is formed.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np

from search import TransitionExperience


class ReplayBuffer:
    def __init__(self, capacity: int, *, seed: Optional[int] = None) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")
        self.capacity = int(capacity)
        self._buffer: Deque[TransitionExperience] = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, experience: TransitionExperience) -> None:
        if not isinstance(experience, TransitionExperience):
            raise TypeError("ReplayBuffer accepts TransitionExperience objects")
        self._buffer.append(experience)

    def add_many(self, experiences: Sequence[TransitionExperience]) -> int:
        count = 0
        for experience in experiences:
            self.add(experience)
            count += 1
        return count

    def clear(self) -> None:
        self._buffer.clear()

    def get_all(self) -> List[TransitionExperience]:
        return list(self._buffer)

    def sample_experiences(self, batch_size: int) -> List[TransitionExperience]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer of size {len(self._buffer)}"
            )
        indices = self._rng.sample(range(len(self._buffer)), batch_size)
        return [self._buffer[index] for index in indices]

    def sample(self, batch_size: int) -> List[TransitionExperience]:
        return self.sample_experiences(batch_size)

    def stats(self) -> Dict[str, int]:
        terminal_count = sum(1 for exp in self._buffer if exp.terminal)
        dead_end_count = sum(1 for exp in self._buffer if exp.dead_end)
        finite_bound_count = sum(
            1
            for exp in self._buffer
            if np.isfinite(exp.return_bound) and not exp.dead_end
        )
        return {
            "size": len(self._buffer),
            "capacity": self.capacity,
            "terminal": terminal_count,
            "dead_end": dead_end_count,
            "finite_solution_bounds": finite_bound_count,
        }