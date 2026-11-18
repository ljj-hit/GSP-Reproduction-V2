"""Central configuration for GSP-Reproduction-V2.

The V2 environment is a 16 x 16 weighted grid world.  Free cells carry
terrain costs 1, 3 or 5, and entering a cell costs its terrain value.  The
Q-function is a predicate-specific relational GNN (R-GNN) rather than an MLP.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Weighted 16 x 16 environment
    # ------------------------------------------------------------------
    height: int = 16
    width: int = 16
    num_actions: int = 4
    terrain_values: Tuple[int, ...] = (1, 3, 5)
    terrain_probabilities: Tuple[float, ...] = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    # ------------------------------------------------------------------
    # Procedural dataset
    # ------------------------------------------------------------------
    num_train_instances: int = 1000
    num_validation_instances: int = 200
    num_test_instances: int = 300
    # Split RNG streams are deterministically separated.
    generator_seed: int = 42
    train_seed_offset: int = 0
    validation_seed_offset: int = 100_003
    test_seed_offset: int = 200_003
    random_obstacle_min_ratio: float = 0.20
    random_obstacle_max_ratio: float = 0.30
    # Generation is bounded; unreachable or structurally invalid maps retry.
    max_generation_attempts_per_instance: int = 200
    # Five structure labels are balanced exactly; this is checked by validate().
    structure_count: int = 5

    # ------------------------------------------------------------------
    # R-GNN architecture
    # ------------------------------------------------------------------
    embedding_dim: int = 32
    gnn_layers: int = 4
    message_hidden_dim: int = 64
    smoothmax_beta: float = 1.0
    q_readout_hidden_dim: int = 64
    graph_cache_size: int = 8192

    # ------------------------------------------------------------------
    # GSP search
    # ------------------------------------------------------------------
    wa_star_weight: float = 2.0
    transition_budget: int = 512
    dead_end_return: float = -2048.0

    # ------------------------------------------------------------------
    # Q-learning and dynamic instance sampling
    # ------------------------------------------------------------------
    training_seed: int = 42
    episodes: int = 3000
    replay_capacity: int = 20000
    batch_size: int = 64
    updates_per_episode: int = 2
    gamma: float = 1.0
    target_update_interval: int = 200
    gradient_clip_norm: float = 1.0
    validation_interval: int = 100
    body_learning_rate: float = 1e-4
    q_readout_learning_rate: float = 1e-3
    use_solution_bounds: bool = True
    use_instance_sampling: bool = True
    # These are project-specific engineering weights, not paper-derived values.
    unsolved_weight: float = 1.0
    solved_weight: float = 2.0
    satisficed_weight: float = 4.0

    # ------------------------------------------------------------------
    # Runtime/output paths
    # ------------------------------------------------------------------
    device: str = "cuda:0"
    data_dir: str = "data/16x16_rgnn"
    checkpoint_dir: str = "checkpoints/16x16_rgnn"
    log_dir: str = "logs/16x16_rgnn"
    results_dir: str = "results/16x16_rgnn"
    figures_dir: str = "results/16x16_rgnn/figures"
    evaluation_split: str = "test"

    def __post_init__(self) -> None:
        self.terrain_values = tuple(int(value) for value in self.terrain_values)
        self.terrain_probabilities = tuple(float(value) for value in self.terrain_probabilities)
        self.validate()

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def cell_count(self) -> int:
        return self.height * self.width

    @property
    def train_path(self) -> Path:
        return Path(self.data_dir) / "train.npz"

    @property
    def validation_path(self) -> Path:
        return Path(self.data_dir) / "validation.npz"

    @property
    def test_path(self) -> Path:
        return Path(self.data_dir) / "test.npz"

    @property
    def manifest_path(self) -> Path:
        return Path(self.data_dir) / "dataset_manifest.json"

    @property
    def initial_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / "initial_model.pt"

    @property
    def best_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / "best_model.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / "last_model.pt"

    @property
    def config_path(self) -> Path:
        return Path(self.checkpoint_dir) / "config.json"

    @property
    def training_summary_path(self) -> Path:
        return Path(self.checkpoint_dir) / "training_summary.json"

    @property
    def training_log_path(self) -> Path:
        return Path(self.log_dir) / "training_log.csv"

    @property
    def validation_log_path(self) -> Path:
        return Path(self.log_dir) / "validation_log.csv"

    @property
    def update_log_path(self) -> Path:
        return Path(self.log_dir) / "update_log.csv"

    @property
    def training_log_file(self) -> Path:
        return Path(self.log_dir) / "training.log"

    @property
    def evaluation_csv_path(self) -> Path:
        return Path(self.results_dir) / "evaluation.csv"

    @property
    def summary_path(self) -> Path:
        return Path(self.results_dir) / "summary.json"

    @property
    def structure_analysis_path(self) -> Path:
        return Path(self.results_dir) / "structure_analysis.csv"

    @property
    def evaluation_log_path(self) -> Path:
        return Path(self.results_dir) / "evaluation.log"

    @property
    def train_split_seed(self) -> int:
        return int(self.generator_seed + self.train_seed_offset)

    @property
    def validation_split_seed(self) -> int:
        return int(self.generator_seed + self.validation_seed_offset)

    @property
    def test_split_seed(self) -> int:
        return int(self.generator_seed + self.test_seed_offset)

    def split_paths(self) -> Dict[str, Path]:
        return {
            "train": self.train_path,
            "validation": self.validation_path,
            "test": self.test_path,
        }

    def split_counts(self) -> Dict[str, int]:
        return {
            "train": self.num_train_instances,
            "validation": self.num_validation_instances,
            "test": self.num_test_instances,
        }

    def split_seeds(self) -> Dict[str, int]:
        return {
            "train": self.train_split_seed,
            "validation": self.validation_split_seed,
            "test": self.test_split_seed,
        }

    # ------------------------------------------------------------------
    # Validation / serialization
    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.num_actions != 4:
            raise ValueError("Four cardinal actions are required")
        if self.terrain_values != (1, 3, 5):
            raise ValueError("The V2 terrain values must be (1, 3, 5)")
        if len(self.terrain_probabilities) != 3 or any(p < 0 for p in self.terrain_probabilities):
            raise ValueError("terrain_probabilities must contain three non-negative values")
        if abs(sum(self.terrain_probabilities) - 1.0) > 1e-8:
            raise ValueError("terrain_probabilities must sum to 1")
        if not (0.0 <= self.random_obstacle_min_ratio <= self.random_obstacle_max_ratio < 1.0):
            raise ValueError("Invalid random-obstacle ratio range")
        if self.structure_count != 5:
            raise ValueError("V2 requires exactly five maze structures")
        for name, count in self.split_counts().items():
            if count <= 0:
                raise ValueError(f"{name} split size must be positive")
            if count % self.structure_count != 0:
                raise ValueError(
                    f"{name} split size ({count}) must be divisible by "
                    f"{self.structure_count} for balanced structure counts"
                )
        if len(set(self.split_seeds().values())) != 3:
            raise ValueError("Dataset split seeds must be distinct")
        if self.max_generation_attempts_per_instance <= 0:
            raise ValueError("max_generation_attempts_per_instance must be positive")
        if self.embedding_dim <= 0 or self.gnn_layers <= 0:
            raise ValueError("Invalid R-GNN dimensions")
        if self.message_hidden_dim <= 0 or self.q_readout_hidden_dim <= 0:
            raise ValueError("Invalid R-GNN hidden dimensions")
        if self.smoothmax_beta <= 0:
            raise ValueError("smoothmax_beta must be positive")
        if self.graph_cache_size <= 0:
            raise ValueError("graph_cache_size must be positive")
        if self.wa_star_weight <= 0 or self.transition_budget <= 0:
            raise ValueError("Invalid search parameters")
        if self.transition_budget < 4:
            raise ValueError("transition_budget must allow at least one root expansion")
        if self.dead_end_return >= -self.cell_count * max(self.terrain_values):
            raise ValueError("dead_end_return must be below any feasible simple-path return")
        if self.episodes <= 0 or self.batch_size <= 0:
            raise ValueError("Invalid episode/batch size")
        if self.replay_capacity < self.batch_size:
            raise ValueError("Replay capacity must be at least batch_size")
        if self.updates_per_episode < 0:
            raise ValueError("updates_per_episode must be non-negative")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive")
        if self.body_learning_rate <= 0 or self.q_readout_learning_rate <= 0:
            raise ValueError("Learning rates must be positive")
        if not self.device.startswith("cuda"):
            raise ValueError("Formal V2 training/evaluation requires CUDA")
        if any(w < 0 for w in (self.unsolved_weight, self.solved_weight, self.satisficed_weight)):
            raise ValueError("Instance-pool weights must be non-negative")
        if sum((self.unsolved_weight, self.solved_weight, self.satisficed_weight)) <= 0:
            raise ValueError("At least one instance-pool weight must be positive")
        if self.evaluation_split not in ("validation", "test"):
            raise ValueError("evaluation_split must be 'validation' or 'test'")

    def to_dict(self) -> Dict[str, Any]:
        result = dataclasses.asdict(self)
        result["terrain_values"] = list(self.terrain_values)
        result["terrain_probabilities"] = list(self.terrain_probabilities)
        return result

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "Config":
        known = {field.name for field in dataclasses.fields(cls)}
        filtered = {key: value for key, value in values.items() if key in known}
        if "terrain_values" in filtered:
            filtered["terrain_values"] = tuple(filtered["terrain_values"])
        if "terrain_probabilities" in filtered:
            filtered["terrain_probabilities"] = tuple(filtered["terrain_probabilities"])
        return cls(**filtered)

    def copy_with(self, **overrides: Any) -> "Config":
        return dataclasses.replace(self, **overrides)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


DEFAULT_CONFIG = Config()