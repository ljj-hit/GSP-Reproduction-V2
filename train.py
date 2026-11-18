"""Independent V2 training entry point for GSP-RGNN.

One episode = choose one 16x16 weighted maze + run one R-GNN-guided state-action
WA* search + add all transitions to the shared replay buffer + perform up to two
Q-learning updates.  The network is frozen during every search.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from check_cuda import require_cuda
from config import Config
from generator import load_dataset
from model import build_rgnn
from relational_graph import RelationalGraphBuilder
from replay_buffer import ReplayBuffer
from search import QGuidedWAStar, SearchResult
from trainer import (
    InstancePoolSampler,
    QTrainer,
    UpdateStats,
    ValidationMetrics,
    validate_network,
)
from utils import (
    ensure_dir,
    format_seconds,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logger,
    utc_now_iso,
)


TRAINING_LOG_COLUMNS = (
    "episode",
    "instance_id",
    "structure",
    "instance_category",
    "success",
    "path_cost",
    "optimal_cost",
    "plan_length",
    "transition_evaluations",
    "unique_state_expansions",
    "node_expansions",
    "buffer_size",
    "loss",
    "q_mean",
    "q_min",
    "q_max",
    "episode_time",
    "gradient_norm",
    "gradient_updates",
    "target_updates",
    "failure_reason",
)

VALIDATION_LOG_COLUMNS = (
    "episode",
    "success_rate",
    "mean_pcr_successful",
    "mean_path_cost_successful",
    "mean_unique_state_expansions",
    "mean_reexpansions",
    "mean_transition_evaluations",
    "num_solved",
    "num_instances",
    "is_best",
    "elapsed_seconds",
)

UPDATE_LOG_COLUMNS = (
    "episode",
    "update_index",
    "loss",
    "q_mean",
    "q_min",
    "q_max",
    "gradient_norm",
    "num_action_objects",
    "target_network_updated",
    "gradient_updates_total",
)


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _aggregate_updates(stats: Sequence[UpdateStats]) -> Dict[str, Optional[float]]:
    if not stats:
        return {
            "loss": None,
            "q_mean": None,
            "q_min": None,
            "q_max": None,
            "gradient_norm": None,
        }
    return {
        "loss": _mean_or_none([entry.loss for entry in stats]),
        "q_mean": _mean_or_none([entry.q_mean for entry in stats]),
        "q_min": min(entry.q_min for entry in stats),
        "q_max": max(entry.q_max for entry in stats),
        "gradient_norm": _mean_or_none([entry.gradient_norm for entry in stats]),
    }


def _check_disjoint(train_hashes: Sequence[str], validation_hashes: Sequence[str]) -> None:
    overlap = set(map(str, train_hashes)) & set(map(str, validation_hashes))
    if overlap:
        raise RuntimeError(
            f"Train/validation map overlap detected: {len(overlap)} maps"
        )
    if len(set(map(str, train_hashes))) != len(train_hashes):
        raise RuntimeError("Duplicate maps inside the training split")
    if len(set(map(str, validation_hashes))) != len(validation_hashes):
        raise RuntimeError("Duplicate maps inside the validation split")


def _save_checkpoint(
    config: Config,
    path: Path,
    trainer: Optional[QTrainer],
    model: torch.nn.Module,
    episode: int,
    validation: Optional[ValidationMetrics],
    *,
    is_best: bool,
    kind: str,
) -> None:
    extra: Dict[str, Any] = {
        "seed": config.training_seed,
        "kind": kind,
        "is_best": bool(is_best),
        "architecture": "relational_predicate_specific_rgnn",
    }
    if validation is not None:
        extra["validation_metrics"] = validation.as_dict()
        extra["selection_key"] = list(validation.selection_key())
    if trainer is not None:
        extra["gradient_updates"] = trainer.gradient_updates
        extra["target_updates"] = trainer.target_updates
        save_checkpoint(
            path,
            trainer.model,
            target_model=trainer.target_model,
            optimizer=trainer.optimizer,
            episode=episode,
            config=config,
            extra=extra,
        )
    else:
        save_checkpoint(
            path,
            model,
            episode=episode,
            config=config,
            extra=extra,
        )


def _reconcile_dataset_sizes(
    config: Config,
    num_train: int,
    num_validation: int,
    logger: Any,
) -> Config:
    if (
        num_train == config.num_train_instances
        and num_validation == config.num_validation_instances
    ):
        return config
    # Debug/smoke datasets may intentionally be smaller.  Record the actual
    # counts in the effective config so checkpoints remain traceable.
    if num_train % config.structure_count != 0 or num_validation % config.structure_count != 0:
        raise ValueError(
            "Dataset split sizes must remain divisible by the five-structure count"
        )
    logger.warning(
        "Dataset sizes (%d train, %d validation) differ from config defaults "
        "(%d, %d); recording actual sizes in the effective config.",
        num_train,
        num_validation,
        config.num_train_instances,
        config.num_validation_instances,
    )
    return config.copy_with(
        num_train_instances=num_train,
        num_validation_instances=num_validation,
    )


def run_training(config: Config) -> Dict[str, Any]:
    """Run one independent V2 training job."""
    config.validate()
    set_seed(config.training_seed, deterministic=True)
    device = require_cuda(config.device, verbose=True)
    for directory in (
        config.data_dir,
        config.checkpoint_dir,
        config.log_dir,
        config.results_dir,
        config.figures_dir,
    ):
        ensure_dir(directory)

    logger = setup_logger("gsp.v2.train", config.training_log_file)
    train_instances, train_metadata = load_dataset(config.train_path)
    validation_instances, validation_metadata = load_dataset(config.validation_path)
    config = _reconcile_dataset_sizes(
        config, len(train_instances), len(validation_instances), logger
    )
    _check_disjoint(
        train_metadata["map_hashes"].tolist(),
        validation_metadata["map_hashes"].tolist(),
    )
    instance_map = {instance.instance_id: instance for instance in train_instances}
    if len(instance_map) != len(train_instances):
        raise RuntimeError("Duplicate training instance IDs")
    config_path = config.save_json(config.config_path)

    set_seed(config.training_seed, deterministic=True)
    model = build_rgnn(config, device=device)
    # Preserve the exact untrained initialization before any training or
    # target-network construction.
    _save_checkpoint(
        config,
        config.initial_checkpoint_path,
        None,
        model,
        episode=0,
        validation=None,
        is_best=False,
        kind="initial",
    )
    graph_builder = RelationalGraphBuilder(config)
    trainer = QTrainer(
        config,
        instance_map,
        graph_builder,
        device=device,
        model=model,
    )
    if trainer.model is not model:
        raise RuntimeError("Trainer did not use the saved initial model object")
    replay = ReplayBuffer(config.replay_capacity, seed=config.training_seed + 10_001)
    sampler = InstancePoolSampler(
        config,
        [instance.instance_id for instance in train_instances],
        seed=config.training_seed + 20_003,
    )
    search = QGuidedWAStar(model, graph_builder, config, device)
    torch.cuda.reset_peak_memory_stats(device)

    logger.info("=" * 78)
    logger.info("GSP-Reproduction-V2 training start")
    logger.info("Seed: %d", config.training_seed)
    logger.info("CUDA: %s (%s)", device, torch.cuda.get_device_name(device))
    logger.info("Online parameter device: %s", next(trainer.model.parameters()).device)
    logger.info("Target parameter device: %s", next(trainer.target_model.parameters()).device)
    logger.info("Train maps: %d | validation maps: %d", len(train_instances), len(validation_instances))
    logger.info("Episodes: %d | transitions/search budget: %d", config.episodes, config.transition_budget)
    logger.info("R-GNN: d=%d, layers=%d, msg_hidden=%d, beta=%.2f", config.embedding_dim, config.gnn_layers, config.message_hidden_dim, config.smoothmax_beta)
    logger.info("Learning rates: body=%.6g, Q-readout=%.6g", config.body_learning_rate, config.q_readout_learning_rate)
    logger.info("Replay: %d | batch=%d | updates/episode=%d", config.replay_capacity, config.batch_size, config.updates_per_episode)
    logger.info("Solution bounds: %s | dynamic sampling: %s", config.use_solution_bounds, config.use_instance_sampling)
    logger.info("Initial model: %s", config.initial_checkpoint_path)
    logger.info("Config: %s", config_path)
    logger.info("=" * 78)

    best_metric: Optional[Tuple[float, float, float]] = None
    best_validation: Optional[ValidationMetrics] = None
    best_episode: Optional[int] = None
    validation_history: List[Dict[str, Any]] = []
    total_gradient_updates = 0
    target_sync_count = 0
    final_loss: Optional[float] = None
    recent: Deque[Tuple[bool, float]] = deque(maxlen=100)
    training_start = time.perf_counter()
    batch_audit_printed = False

    ensure_dir(config.training_log_path.parent)
    with config.training_log_path.open("w", newline="", encoding="utf-8") as train_handle, \
            config.validation_log_path.open("w", newline="", encoding="utf-8") as validation_handle, \
            config.update_log_path.open("w", newline="", encoding="utf-8") as update_handle:
        train_writer = csv.DictWriter(train_handle, fieldnames=TRAINING_LOG_COLUMNS)
        validation_writer = csv.DictWriter(validation_handle, fieldnames=VALIDATION_LOG_COLUMNS)
        update_writer = csv.DictWriter(update_handle, fieldnames=UPDATE_LOG_COLUMNS)
        train_writer.writeheader()
        validation_writer.writeheader()
        update_writer.writeheader()

        progress = tqdm(range(1, config.episodes + 1), desc=f"GSP-RGNN seed {config.training_seed}", unit="ep")
        for episode in progress:
            episode_start = time.perf_counter()
            instance_id = sampler.choose()
            instance = instance_map[instance_id]
            result: SearchResult = search.search(instance)
            category = InstancePoolSampler.classify(result)
            sampler.update(instance.instance_id, category)
            replay.add_many(result.experiences)

            update_stats: List[UpdateStats] = []
            if len(replay) >= config.batch_size:
                for update_index in range(1, config.updates_per_episode + 1):
                    batch = replay.sample(config.batch_size)
                    stats = trainer.train_step(batch)
                    if not batch_audit_printed:
                        logger.info("First graph batch device audit: %s", trainer.last_device_report)
                        logger.info(
                            "CUDA reverse pass verified; online parameter device type=%s",
                            next(trainer.model.parameters()).device.type,
                        )
                        batch_audit_printed = True
                    update_stats.append(stats)
                    update_writer.writerow(
                        {
                            "episode": episode,
                            "update_index": update_index,
                            "loss": f"{stats.loss:.10f}",
                            "q_mean": f"{stats.q_mean:.10f}",
                            "q_min": f"{stats.q_min:.10f}",
                            "q_max": f"{stats.q_max:.10f}",
                            "gradient_norm": f"{stats.gradient_norm:.10f}",
                            "num_action_objects": stats.num_action_objects,
                            "target_network_updated": int(stats.target_network_updated),
                            "gradient_updates_total": stats.gradient_updates,
                        }
                    )
                    update_handle.flush()
                    if stats.target_network_updated:
                        target_sync_count += 1
            total_gradient_updates = trainer.gradient_updates
            aggregate = _aggregate_updates(update_stats)
            if aggregate["loss"] is not None:
                final_loss = aggregate["loss"]
            episode_elapsed = time.perf_counter() - episode_start
            recent.append((result.success, float(result.path_cost) if result.path_cost is not None else 0.0))

            train_writer.writerow(
                {
                    "episode": episode,
                    "instance_id": instance.instance_id,
                    "structure": instance.structure,
                    "instance_category": category,
                    "success": int(result.success),
                    "path_cost": "" if result.path_cost is None else f"{result.path_cost:.6f}",
                    "optimal_cost": "" if instance.optimal_cost is None else f"{instance.optimal_cost:.6f}",
                    "plan_length": result.plan_length if result.success else "",
                    "transition_evaluations": result.transition_evaluations,
                    "unique_state_expansions": result.unique_state_expansions,
                    "node_expansions": result.node_expansions,
                    "buffer_size": len(replay),
                    "loss": "" if aggregate["loss"] is None else f"{aggregate['loss']:.10f}",
                    "q_mean": "" if aggregate["q_mean"] is None else f"{aggregate['q_mean']:.10f}",
                    "q_min": "" if aggregate["q_min"] is None else f"{aggregate['q_min']:.10f}",
                    "q_max": "" if aggregate["q_max"] is None else f"{aggregate['q_max']:.10f}",
                    "episode_time": f"{episode_elapsed:.6f}",
                    "gradient_norm": "" if aggregate["gradient_norm"] is None else f"{aggregate['gradient_norm']:.10f}",
                    "gradient_updates": total_gradient_updates,
                    "target_updates": target_sync_count,
                    "failure_reason": "" if result.success else result.termination_reason,
                }
            )
            train_handle.flush()

            window = list(recent)
            recent_success = 100.0 * sum(1 for solved, _ in window if solved) / len(window)
            recent_eval = sum(exp for _, exp in window) / len(window)
            loss_text = "n/a" if aggregate["loss"] is None else f"{aggregate['loss']:.4f}"
            progress.set_postfix(
                {
                    "succ": f"{recent_success:.1f}%",
                    "exp": f"{recent_eval:.1f}",
                    "loss": loss_text,
                    "buf": len(replay),
                    "upd": total_gradient_updates,
                },
                refresh=False,
            )
            if episode == 1 or episode % 25 == 0:
                logger.info(
                    "Episode %d/%d | recent success=%.1f%% | loss=%s | buffer=%d | "
                    "updates=%d | category=%s | elapsed=%s | GPU=%s",
                    episode,
                    config.episodes,
                    recent_success,
                    loss_text,
                    len(replay),
                    total_gradient_updates,
                    category,
                    format_seconds(time.perf_counter() - training_start),
                    torch.cuda.get_device_name(device),
                )

            should_validate = (
                episode % config.validation_interval == 0 or episode == config.episodes
            )
            if should_validate:
                val_start = time.perf_counter()
                validation = validate_network(
                    trainer.model,
                    validation_instances,
                    config,
                    device,
                    graph_builder=graph_builder,
                )
                key = validation.selection_key()
                is_best = best_metric is None or key > best_metric
                if is_best:
                    best_metric = key
                    best_validation = validation
                    best_episode = episode
                    _save_checkpoint(
                        config,
                        config.best_checkpoint_path,
                        trainer,
                        trainer.model,
                        episode,
                        validation,
                        is_best=True,
                        kind="best",
                    )
                _save_checkpoint(
                    config,
                    config.last_checkpoint_path,
                    trainer,
                    trainer.model,
                    episode,
                    validation,
                    is_best=is_best,
                    kind="last",
                )
                validation_writer.writerow(
                    {
                        "episode": episode,
                        "success_rate": f"{validation.success_rate:.8f}",
                        "mean_pcr_successful": (
                            "" if validation.mean_pcr_successful is None
                            else f"{validation.mean_pcr_successful:.8f}"
                        ),
                        "mean_path_cost_successful": (
                            "" if validation.mean_path_cost_successful is None
                            else f"{validation.mean_path_cost_successful:.8f}"
                        ),
                        "mean_unique_state_expansions": f"{validation.mean_unique_state_expansions:.8f}",
                        "mean_reexpansions": f"{validation.mean_reexpansions:.8f}",
                        "mean_transition_evaluations": f"{validation.mean_transition_evaluations:.8f}",
                        "num_solved": validation.num_solved,
                        "num_instances": validation.num_instances,
                        "is_best": int(is_best),
                        "elapsed_seconds": f"{time.perf_counter() - training_start:.6f}",
                    }
                )
                validation_handle.flush()
                validation_history.append(
                    {
                        "episode": episode,
                        **validation.as_dict(),
                        "is_best": is_best,
                        "validation_time_seconds": time.perf_counter() - val_start,
                    }
                )
                logger.info(
                    "Validation @ %d | success=%.2f%% (%d/%d) | PCR=%s | "
                    "transition_evals=%.2f | unique_state_expansions=%.2f | best=%s",
                    episode,
                    validation.success_rate,
                    validation.num_solved,
                    validation.num_instances,
                    "n/a" if validation.mean_pcr_successful is None
                    else f"{validation.mean_pcr_successful:.4f}",
                    validation.mean_transition_evaluations,
                    validation.mean_unique_state_expansions,
                    is_best,
                )

        if best_validation is None:
            best_validation = validate_network(
                trainer.model, validation_instances, config, device, graph_builder=graph_builder
            )
            best_episode = config.episodes
            _save_checkpoint(
                config,
                config.best_checkpoint_path,
                trainer,
                trainer.model,
                config.episodes,
                best_validation,
                is_best=True,
                kind="best",
            )

    progress.close()
    elapsed = time.perf_counter() - training_start
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    summary = {
        "status": "completed",
        "created_at_utc": utc_now_iso(),
        "seed": config.training_seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "episodes": config.episodes,
        "elapsed_seconds": elapsed,
        "gradient_updates": total_gradient_updates,
        "target_updates": target_sync_count,
        "replay_size": len(replay),
        "last_loss": final_loss,
        "best_episode": best_episode,
        "best_validation": None if best_validation is None else best_validation.as_dict(),
        "validation_history": validation_history,
        "peak_gpu_memory_mb": peak_memory,
        "graph_cache": graph_builder.cache_stats(),
        "initial_checkpoint": str(config.initial_checkpoint_path),
        "best_checkpoint": str(config.best_checkpoint_path),
        "last_checkpoint": str(config.last_checkpoint_path),
        "config_path": str(config.config_path),
        "training_log": str(config.training_log_path),
        "validation_log": str(config.validation_log_path),
        "update_log": str(config.update_log_path),
        "use_solution_bounds": config.use_solution_bounds,
        "use_instance_sampling": config.use_instance_sampling,
        "training_log_columns": list(TRAINING_LOG_COLUMNS),
        "validation_log_columns": list(VALIDATION_LOG_COLUMNS),
        "update_log_columns": list(UPDATE_LOG_COLUMNS),
    }
    save_json(summary, config.training_summary_path)
    logger.info("Training complete in %s", format_seconds(elapsed))
    logger.info("Peak allocated GPU memory: %.2f MiB", peak_memory)
    logger.info("Best validation: %s", None if best_validation is None else best_validation.as_dict())
    logger.info("Initial checkpoint: %s", config.initial_checkpoint_path)
    logger.info("Best checkpoint: %s", config.best_checkpoint_path)
    logger.info("Last checkpoint: %s", config.last_checkpoint_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train V2 GSP-RGNN on CUDA")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--search-budget", type=int, default=None)
    parser.add_argument("--updates-per-episode", type=int, default=None)
    parser.add_argument(
        "--ablation",
        choices=("none", "no_solution_bounds", "uniform_sampling"),
        default="none",
        help="independent ablation mode; outputs go under an ablations/ subdirectory",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    overrides: Dict[str, Any] = {}
    for field in (
        "device",
        "episodes",
        "data_dir",
        "checkpoint_dir",
        "log_dir",
        "validation_interval",
        "batch_size",
        "updates_per_episode",
    ):
        value = getattr(args, field)
        if value is not None:
            overrides[field] = value
    if args.search_budget is not None:
        overrides["transition_budget"] = int(args.search_budget)
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("Seed must be non-negative")
        overrides["training_seed"] = int(args.seed)
    if args.ablation == "no_solution_bounds":
        overrides["use_solution_bounds"] = False
    elif args.ablation == "uniform_sampling":
        overrides["use_instance_sampling"] = False
    if args.ablation != "none":
        if args.checkpoint_dir is None:
            overrides["checkpoint_dir"] = (
                f"checkpoints/16x16_rgnn/ablations/{args.ablation}"
            )
        if args.log_dir is None:
            overrides["log_dir"] = f"logs/16x16_rgnn/ablations/{args.ablation}"
    return Config().copy_with(**overrides)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    config.validate()
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())