"""Independent V2 evaluation entry point.

Runs the fixed comparison set on identical 16x16 weighted test maps:
  Dijkstra (unbudgeted cost reference),
  A* + Manhattan,
  WA* + Manhattan,
  Untrained initial R-GNN + WA*,
  Trained GSP-RGNN + WA*.

All non-Dijkstra algorithms are limited to 512 transition evaluations.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from baselines import run_astar, run_dijkstra, run_weighted_astar
from check_cuda import require_cuda
from config import Config
from environment import MazeInstance
from generator import STRUCTURES, load_dataset
from metrics import (
    PREFERRED_ALGORITHM_ORDER,
    save_summary,
    summarize_by_structure,
    summarize_records,
    write_evaluation_csv,
    write_structure_analysis_csv,
)
from model import RelationalQNetwork, build_rgnn
from relational_graph import RelationalGraphBuilder, batch_relational_graphs
from search import QGuidedWAStar
from utils import (
    ensure_dir,
    format_seconds,
    load_checkpoint,
    set_seed,
    setup_logger,
    utc_now_iso,
)


DIJKSTRA = "Dijkstra"
ASTAR = "A* + Manhattan"
WASTAR = "WA* + Manhattan"
UNTRAINED_RGNN = "Untrained R-GNN + WA*"
TRAINED_GSP = "Trained GSP-RGNN + WA*"
ALL_ALGORITHMS = (DIJKSTRA, ASTAR, WASTAR, UNTRAINED_RGNN, TRAINED_GSP)
CLI_ALGORITHM_KEYS = {
    "all": ALL_ALGORITHMS,
    "dijkstra": (DIJKSTRA,),
    "astar": (ASTAR,),
    "wastar": (WASTAR,),
    "untrained_rgnn": (UNTRAINED_RGNN,),
    "trained_gsp": (TRAINED_GSP,),
}


def _resolve_algorithms(values: Sequence[str]) -> Tuple[str, ...]:
    selected: List[str] = []
    for value in values:
        if value not in CLI_ALGORITHM_KEYS:
            raise ValueError(f"Unknown algorithm selector: {value}")
        for algorithm in CLI_ALGORITHM_KEYS[value]:
            if algorithm not in selected:
                selected.append(algorithm)
    return tuple(selected)


def _load_config_from_checkpoint(path: str | Path) -> Tuple[Config, Dict[str, Any]]:
    payload = load_checkpoint(path, map_location="cpu")
    stored = payload.get("config")
    if not isinstance(stored, dict):
        raise ValueError(f"Checkpoint lacks config metadata: {path}")
    return Config.from_dict(stored), payload


def _load_rgnn_checkpoint(
    path: str | Path,
    device: torch.device,
) -> Tuple[RelationalQNetwork, Config, Dict[str, Any]]:
    config, payload = _load_config_from_checkpoint(path)
    model = build_rgnn(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, config, payload


def _mean_timed(
    function: Callable[[], Any],
    *,
    synchronize: Optional[torch.device],
) -> Tuple[Any, float]:
    if synchronize is not None and synchronize.type == "cuda":
        torch.cuda.synchronize(synchronize)
    start = time.perf_counter()
    result = function()
    if synchronize is not None and synchronize.type == "cuda":
        torch.cuda.synchronize(synchronize)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def _warmup_model(
    model: RelationalQNetwork,
    instance: MazeInstance,
    config: Config,
    device: torch.device,
) -> None:
    builder = RelationalGraphBuilder(config)
    graph = builder.build(instance, instance.start)
    batch = batch_relational_graphs([graph], device=device)
    model.eval()
    with torch.inference_mode():
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _generic_search_fields(result: Any) -> Dict[str, Any]:
    return {
        "success": bool(result.success),
        "path_cost": float(result.path_cost) if result.success and result.path_cost is not None else None,
        "unique_state_expansions": int(result.unique_state_expansions),
        "reexpansions": int(result.reexpansions),
        "transition_evaluations": int(result.transition_evaluations),
        "failure_reason": "" if result.success else str(result.termination_reason),
        "inference_time_ms": float(getattr(result, "inference_time_seconds", 0.0)) * 1000.0,
    }


def _record(
    algorithm: str,
    instance: MazeInstance,
    result: Any,
    *,
    optimal_cost: float,
    search_time_ms: float,
    budget_limited: bool,
) -> Dict[str, Any]:
    fields = _generic_search_fields(result)
    success = fields["success"]
    path_cost = fields["path_cost"]
    pcr = (
        path_cost / optimal_cost
        if success and path_cost is not None and optimal_cost > 0
        else None
    )
    return {
        "algorithm": algorithm,
        "instance_id": instance.instance_id,
        "structure": instance.structure,
        "success": success,
        "path_cost": path_cost,
        "optimal_cost": float(optimal_cost),
        "pcr": pcr,
        "unique_state_expansions": fields["unique_state_expansions"],
        "reexpansions": fields["reexpansions"],
        "transition_evaluations": fields["transition_evaluations"],
        "search_time_ms": float(search_time_ms),
        "gnn_inference_time_ms": fields["inference_time_ms"],
        "failure_reason": fields["failure_reason"],
        "budget_limited": bool(budget_limited),
    }


def evaluate(
    config: Config,
    instances: Sequence[MazeInstance],
    *,
    device: str | torch.device,
    algorithms: Sequence[str],
    checkpoint: Optional[str | Path],
    initial_checkpoint: Optional[str | Path],
    output_dir: str | Path,
) -> Dict[str, Any]:
    if not instances:
        raise ValueError("Evaluation requires at least one instance")
    selected = tuple(algorithms)
    if not selected:
        raise ValueError("At least one algorithm must be selected")
    missing = [name for name in selected if name not in ALL_ALGORITHMS]
    if missing:
        raise ValueError(f"Unknown algorithm display names: {missing}")
    if len({instance.structure for instance in instances}) < len(STRUCTURES):
        raise ValueError("Evaluation split must contain all five structure groups")

    selected_device = require_cuda(device, verbose=True)
    if str(config.device) != str(selected_device):
        config = config.copy_with(device=str(selected_device))
    ensure_dir(output_dir)
    logger = setup_logger("gsp.v2.evaluate", Path(output_dir) / "evaluation.log")
    set_seed(config.training_seed, deterministic=True)

    need_trained = TRAINED_GSP in selected
    need_untrained = UNTRAINED_RGNN in selected
    trained_model = None
    untrained_model = None
    trained_config: Optional[Config] = None
    untrained_config: Optional[Config] = None
    trained_payload: Optional[Dict[str, Any]] = None
    if need_trained:
        if checkpoint is None:
            raise ValueError("--checkpoint is required for Trained GSP-RGNN + WA*")
        trained_model, trained_config, trained_payload = _load_rgnn_checkpoint(
            checkpoint, selected_device
        )
    if need_untrained:
        if initial_checkpoint is None:
            raise ValueError("--initial-checkpoint is required for Untrained R-GNN + WA*")
        untrained_model, untrained_config, _ = _load_rgnn_checkpoint(
            initial_checkpoint, selected_device
        )

    search_config = trained_config or untrained_config or config
    # Use one fresh graph builder per learned algorithm so graph-cache warmth
    # from one method cannot asymmetrically reduce another method's timing.
    trained_builder = RelationalGraphBuilder(search_config) if trained_model is not None else None
    untrained_builder = RelationalGraphBuilder(search_config) if untrained_model is not None else None
    trained_search = (
        QGuidedWAStar(trained_model, trained_builder, search_config, selected_device)
        if trained_model is not None
        else None
    )
    untrained_search = (
        QGuidedWAStar(untrained_model, untrained_builder, search_config, selected_device)
        if untrained_model is not None
        else None
    )
    if trained_model is not None:
        _warmup_model(trained_model, instances[0], search_config, selected_device)
    if untrained_model is not None:
        _warmup_model(untrained_model, instances[0], search_config, selected_device)

    records: List[Dict[str, Any]] = []
    reference_costs: Dict[int, float] = {}
    evaluation_start = time.perf_counter()
    total_runs = len(instances) * len(selected)
    progress = tqdm(total=total_runs, desc="V2 evaluation", unit="run")
    try:
        for instance in instances:
            dijkstra_result, dijkstra_time = _mean_timed(
                lambda inst=instance: run_dijkstra(
                    inst.terrain_grid, inst.start, inst.goal
                ),
                synchronize=None,
            )
            if not dijkstra_result.success or dijkstra_result.path_cost is None:
                raise RuntimeError(f"Dijkstra failed on test instance {instance.instance_id}")
            optimal_cost = float(dijkstra_result.path_cost)
            if (
                instance.optimal_cost is not None
                and abs(float(instance.optimal_cost) - optimal_cost) > 1e-6
            ):
                raise RuntimeError(
                    f"Dijkstra reference mismatch on instance {instance.instance_id}: "
                    f"stored={instance.optimal_cost}, recomputed={optimal_cost}"
                )
            reference_costs[instance.instance_id] = optimal_cost
            if DIJKSTRA in selected:
                records.append(
                    _record(
                        DIJKSTRA,
                        instance,
                        dijkstra_result,
                        optimal_cost=optimal_cost,
                        search_time_ms=dijkstra_time,
                        budget_limited=False,
                    )
                )
                progress.update(1)

            if ASTAR in selected:
                astar_result, astar_time = _mean_timed(
                    lambda inst=instance: run_astar(
                        inst.terrain_grid,
                        inst.start,
                        inst.goal,
                        transition_budget=search_config.transition_budget,
                    ),
                    synchronize=None,
                )
                records.append(
                    _record(
                        ASTAR,
                        instance,
                        astar_result,
                        optimal_cost=optimal_cost,
                        search_time_ms=astar_time,
                        budget_limited=True,
                    )
                )
                progress.update(1)

            if WASTAR in selected:
                wastar_result, wastar_time = _mean_timed(
                    lambda inst=instance: run_weighted_astar(
                        inst.terrain_grid,
                        inst.start,
                        inst.goal,
                        weight=search_config.wa_star_weight,
                        transition_budget=search_config.transition_budget,
                    ),
                    synchronize=None,
                )
                records.append(
                    _record(
                        WASTAR,
                        instance,
                        wastar_result,
                        optimal_cost=optimal_cost,
                        search_time_ms=wastar_time,
                        budget_limited=True,
                    )
                )
                progress.update(1)

            if untrained_search is not None:
                untrained_result, untrained_time = _mean_timed(
                    lambda inst=instance: untrained_search.search(inst),
                    synchronize=selected_device,
                )
                records.append(
                    _record(
                        UNTRAINED_RGNN,
                        instance,
                        untrained_result,
                        optimal_cost=optimal_cost,
                        search_time_ms=untrained_time,
                        budget_limited=True,
                    )
                )
                progress.update(1)

            if trained_search is not None:
                trained_result, trained_time = _mean_timed(
                    lambda inst=instance: trained_search.search(inst),
                    synchronize=selected_device,
                )
                records.append(
                    _record(
                        TRAINED_GSP,
                        instance,
                        trained_result,
                        optimal_cost=optimal_cost,
                        search_time_ms=trained_time,
                        budget_limited=True,
                    )
                )
                progress.update(1)
    finally:
        progress.close()

    elapsed = time.perf_counter() - evaluation_start
    algorithm_summaries = summarize_records(records)
    structure_summaries = summarize_by_structure(records)
    ensure_dir(output_dir)
    evaluation_csv = write_evaluation_csv(records, Path(output_dir) / "evaluation.csv")
    structure_csv = write_structure_analysis_csv(
        records, Path(output_dir) / "structure_analysis.csv"
    )
    summary = {
        "status": "completed",
        "created_at_utc": utc_now_iso(),
        "split": config.evaluation_split,
        "num_instances": len(instances),
        "structures": {
            structure: sum(1 for instance in instances if instance.structure == structure)
            for structure in STRUCTURES
        },
        "device": str(selected_device),
        "gpu_name": torch.cuda.get_device_name(selected_device),
        "budget_transition_evaluations": search_config.transition_budget,
        "budget_note": "Dijkstra is evaluated without the 512-transition budget.",
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "initial_checkpoint": (
            None if initial_checkpoint is None else str(initial_checkpoint)
        ),
        "checkpoint_episode": (
            None if trained_payload is None else trained_payload.get("episode")
        ),
        "selected_algorithms": list(selected),
        "evaluation_elapsed_seconds": elapsed,
        "metrics_definition": {
            "success_rate_percent": "100 * solved / test instances",
            "pcr": "path_cost / Dijkstra optimal cost, solved instances only",
            "mean_path_cost_successful": "absolute weighted cost, solved instances only",
            "unique_state_expansions": "distinct expanded states; failures included",
            "reexpansions": "repeat expansions of already-expanded states",
            "transition_evaluations": "legal candidate transitions evaluated/checked",
            "search_time_ms": "end-to-end wall time including graph/runtime work",
            "gnn_inference_time_ms": "GNN forward component of search time",
        },
        "algorithms": algorithm_summaries,
        "by_structure": structure_summaries,
        "evaluation_csv": str(evaluation_csv),
        "structure_analysis_csv": str(structure_csv),
        "config": search_config.to_dict(),
    }
    summary_path = save_summary(summary, Path(output_dir) / "summary.json")
    logger.info("Evaluation complete: %d maps x %d algorithms in %s", len(instances), len(selected), format_seconds(elapsed))
    logger.info("%-28s %10s %10s %12s %18s", "Algorithm", "Success", "PCR", "UniqExp", "TransEval")
    for algorithm in PREFERRED_ALGORITHM_ORDER:
        entry = algorithm_summaries.get(algorithm)
        if entry is None:
            continue
        logger.info(
            "%-28s %9.2f%% %10s %12.2f %18.2f",
            algorithm,
            entry["success_rate_percent"],
            "n/a" if entry["mean_pcr_successful"] is None
            else f"{entry['mean_pcr_successful']:.4f}",
            entry["mean_unique_state_expansions"] or 0.0,
            entry["mean_transition_evaluations"] or 0.0,
        )
    logger.info("Evaluation CSV: %s", evaluation_csv)
    logger.info("Structure CSV: %s", structure_csv)
    logger.info("Summary: %s", summary_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate V2 GSP-RGNN on held-out weighted mazes")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/16x16_rgnn/best_model.pt",
        help="trained GSP-RGNN checkpoint",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=str,
        default=None,
        help="untrained R-GNN checkpoint; defaults to initial_model.pt beside --checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(CLI_ALGORITHM_KEYS),
        default=["all"],
        help="'all' or one/more of dijkstra astar wastar untrained_rgnn trained_gsp",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    selected = _resolve_algorithms(args.algorithms)
    initial_checkpoint = args.initial_checkpoint
    if UNTRAINED_RGNN in selected and initial_checkpoint is None:
        initial_checkpoint = str(Path(args.checkpoint).parent / "initial_model.pt")
    config = Config().copy_with(evaluation_split=args.split)
    if args.data_dir is not None:
        config = config.copy_with(data_dir=args.data_dir)

    trained_config = None
    if TRAINED_GSP in selected:
        if not Path(args.checkpoint).is_file():
            raise FileNotFoundError(args.checkpoint)
        trained_config, _ = _load_config_from_checkpoint(args.checkpoint)
    if trained_config is not None:
        config = trained_config.copy_with(
            evaluation_split=args.split,
            device=args.device,
            data_dir=args.data_dir or trained_config.data_dir,
            checkpoint_dir=Path(args.checkpoint).parent.as_posix(),
        )
    elif UNTRAINED_RGNN in selected and initial_checkpoint is not None and Path(initial_checkpoint).is_file():
        initial_config, _ = _load_config_from_checkpoint(initial_checkpoint)
        config = initial_config.copy_with(
            evaluation_split=args.split,
            device=args.device,
            data_dir=args.data_dir or initial_config.data_dir,
        )
    elif args.data_dir is not None:
        config = config.copy_with(data_dir=args.data_dir)
    output_dir = args.output_dir or config.results_dir
    split_path = Path(config.data_dir) / f"{args.split}.npz"
    instances, _ = load_dataset(split_path)
    expected_attr = f"num_{args.split}_instances"
    expected_count = int(getattr(config, expected_attr))
    if len(instances) != expected_count:
        if len(instances) % config.structure_count != 0:
            raise ValueError(
                f"Split {args.split} has {len(instances)} instances, not divisible by 5"
            )
        config = config.copy_with(**{expected_attr: len(instances)})
    if not (trained_config is not None or UNTRAINED_RGNN in selected):
        config = config.copy_with(results_dir=output_dir)

    summary = evaluate(
        config,
        instances,
        device=args.device,
        algorithms=selected,
        checkpoint=args.checkpoint if TRAINED_GSP in selected else None,
        initial_checkpoint=initial_checkpoint if UNTRAINED_RGNN in selected else None,
        output_dir=output_dir,
    )
    print(f"Evaluation summary: {Path(output_dir) / 'summary.json'}")
    print(f"Per-instance results: {Path(output_dir) / 'evaluation.csv'}")
    print(f"Structure analysis: {Path(output_dir) / 'structure_analysis.csv'}")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())