"""Shared utilities: reproducibility, logging, checkpoints, time and drawing."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from config import Config


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch and CUDA RNGs.

    A single helper is used by dataset generation, training and evaluation so
    that all sources of randomness are controlled together.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required before the first CUDA context for deterministic cuBLAS.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # pragma: no cover - compatibility path
            # Older builds may not support this API; seeding is still active,
            # but surface the limitation rather than silently swallowing it.
            warnings.warn(
                f"Could not enable deterministic PyTorch algorithms: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def seed_worker(worker_id: int) -> None:  # pragma: no cover - kept for API parity
    """Seed a NumPy worker (not currently needed, exported for completeness)."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Paths / serialization
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write JSON atomically."""
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logger(
    name: str,
    log_file: Optional[str | Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a non-propagating logger writing to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Avoid duplicate handlers if the same logger is requested twice.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.setLevel(level)
    logger.addHandler(stream)

    if log_file is not None:
        ensure_dir(Path(log_file).parent)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    return logger


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
class Timer:
    """Small context manager/elapsed-time helper."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.end: Optional[float] = None

    @property
    def elapsed(self) -> float:
        if self.end is None:
            return time.perf_counter() - self.start
        return self.end - self.start

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.end = None
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.end = time.perf_counter()

    @staticmethod
    def sync_if_cuda(device: torch.device | str | None) -> None:
        if device is None:
            return
        selected = torch.device(device)
        if selected.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(selected)


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:.1f}s"


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    target_model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    episode: Optional[int] = None,
    config: Optional[Config] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save all state needed to reproduce a trained Q network."""
    path = Path(path)
    ensure_dir(path.parent)
    payload: Dict[str, Any] = {
        "format_version": 1,
        "created_at_utc": utc_now_iso(),
        "model_state_dict": model.state_dict(),
    }
    if target_model is not None:
        payload["target_model_state_dict"] = target_model.state_dict()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if episode is not None:
        payload["episode"] = int(episode)
    if config is not None:
        payload["config"] = config.to_dict()
    if extra:
        payload.update(dict(extra))

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load a locally-created checkpoint.

    ``weights_only=False`` is used because checkpoints intentionally contain
    plain Python metadata (configuration, metrics).  Only load checkpoints
    produced by this project.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint format: {path}")
    return payload


# ---------------------------------------------------------------------------
# Maze helpers
# ---------------------------------------------------------------------------
def maze_to_text(
    maze: np.ndarray,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    path: Optional[Sequence[Tuple[int, int]]] = None,
) -> str:
    """Render a maze as ASCII for logs and debugging.

    ``#`` wall, ``.`` free, ``S`` start, ``G`` goal, ``*`` path.
    """
    height, width = maze.shape
    path_set = set(path or [])
    lines = []
    for row in range(height):
        chars = []
        for col in range(width):
            pos = (row, col)
            if pos == start:
                chars.append("S")
            elif pos == goal:
                chars.append("G")
            elif int(maze[row, col]) == 1:
                chars.append("#")
            elif pos in path_set:
                chars.append("*")
            else:
                chars.append(".")
        lines.append("".join(chars))
    return "\n".join(lines)


def save_maze_figure(
    maze: np.ndarray,
    save_path: str | Path,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    path: Optional[Sequence[Tuple[int, int]]] = None,
    title: str = "Maze",
) -> Path:
    """Save a compact maze figure (used by visualization/debugging)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    image = np.array(maze, dtype=int)
    value_to_class = {0: 0, 1: 1, 3: 2, 5: 3}
    image = np.vectorize(value_to_class.get)(image).astype(int)
    color_map = ListedColormap(["#343a40", "#f1f3f5", "#ffe8cc", "#ffc078"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], color_map.N)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap=color_map, norm=norm)
    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color="#0b7285", linewidth=2, label="path")
    if start is not None:
        ax.scatter(start[1], start[0], c="#2b8a3e", s=70, marker="o", label="start", zorder=3)
    if goal is not None:
        ax.scatter(goal[1], goal[0], c="#c92a2a", s=70, marker="*", label="goal", zorder=3)
    ax.set_xticks(range(maze.shape[1]))
    ax.set_yticks(range(maze.shape[0]))
    ax.set_title(title)
    ax.grid(color="#adb5bd", linewidth=0.3)
    if start is not None or goal is not None or path:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return str(device)


def iter_numpy_arrays(mapping: Mapping[str, Any]) -> Iterator[np.ndarray]:
    """Yield NumPy arrays contained in a mapping (helper for diagnostics)."""
    for value in mapping.values():
        if isinstance(value, np.ndarray):
            yield value