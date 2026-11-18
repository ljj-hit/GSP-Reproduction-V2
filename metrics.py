"""Fixed V2 evaluation metrics and CSV/JSON helpers.

The evaluation set is fixed to five metrics (with specified sub-components):
success rate, weighted path-cost ratio plus absolute successful path cost,
unique state expansions (and separately re-expansions), transition
evaluations, and end-to-end search time (with GNN inference time recorded as a
component, not as a separate ranking score).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from generator import STRUCTURES
from utils import ensure_dir, save_json


PREFERRED_ALGORITHM_ORDER = (
    "Dijkstra",
    "A* + Manhattan",
    "WA* + Manhattan",
    "Untrained R-GNN + WA*",
    "Trained GSP-RGNN + WA*",
)

EVALUATION_COLUMNS = (
    "algorithm",
    "instance_id",
    "structure",
    "success",
    "path_cost",
    "optimal_cost",
    "pcr",
    "unique_state_expansions",
    "reexpansions",
    "transition_evaluations",
    "search_time_ms",
    "gnn_inference_time_ms",
    "failure_reason",
    "budget_limited",
)

STRUCTURE_ANALYSIS_COLUMNS = (
    "algorithm",
    "structure",
    "num_instances",
    "num_solved",
    "success_rate_percent",
    "mean_path_cost_successful",
    "mean_pcr_successful",
    "mean_unique_state_expansions",
    "mean_reexpansions",
    "mean_transition_evaluations",
    "mean_search_time_ms",
    "mean_gnn_inference_time_ms",
    "budget_limited",
)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_int(value: Any) -> Optional[int]:
    number = _as_float(value)
    return None if number is None else int(number)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def mean_std(values: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not cleaned:
        return None, None
    if len(cleaned) == 1:
        return float(cleaned[0]), 0.0
    return float(np.mean(cleaned)), float(np.std(cleaned, ddof=1))


def algorithm_sort_key(name: str) -> Tuple[int, str]:
    try:
        return PREFERRED_ALGORITHM_ORDER.index(name), name
    except ValueError:
        return len(PREFERRED_ALGORITHM_ORDER), name


def group_records_by_algorithm(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["algorithm"]), []).append(record)
    return dict(sorted(grouped.items(), key=lambda item: algorithm_sort_key(item[0])))


def summarize_algorithm(
    records: Sequence[Mapping[str, Any]],
    algorithm: str,
    *,
    budget_limited: bool,
) -> Dict[str, Any]:
    if not records:
        raise ValueError(f"No records for {algorithm}")
    success = [_as_bool(record["success"]) for record in records]
    solved_records = [record for record, ok in zip(records, success) if ok]
    success_rate = 100.0 * sum(success) / len(records)
    path_costs = [
        _as_float(record.get("path_cost")) for record in solved_records
    ]
    path_costs_clean = [value for value in path_costs if value is not None]
    pcr_values = [_as_float(record.get("pcr")) for record in solved_records]
    pcr_clean = [value for value in pcr_values if value is not None]
    unique_expansions = [
        _as_float(record.get("unique_state_expansions")) for record in records
    ]
    reexpansions = [_as_float(record.get("reexpansions")) for record in records]
    transition_evaluations = [
        _as_float(record.get("transition_evaluations")) for record in records
    ]
    search_times = [_as_float(record.get("search_time_ms")) for record in records]
    inference_times = [
        _as_float(record.get("gnn_inference_time_ms")) for record in records
    ]

    path_mean, path_std = mean_std(path_costs_clean)
    pcr_mean, pcr_std = mean_std(pcr_clean)
    unique_mean, unique_std = mean_std(unique_expansions)
    reexp_mean, _ = mean_std(reexpansions)
    transition_mean, transition_std = mean_std(transition_evaluations)
    time_mean, time_std = mean_std(search_times)
    inference_mean, inference_std = mean_std(inference_times)
    failure_reasons: Dict[str, int] = {}
    for record, ok in zip(records, success):
        if ok:
            continue
        reason = str(record.get("failure_reason") or "unknown")
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return {
        "algorithm": algorithm,
        "budget_limited": bool(budget_limited),
        "num_instances": len(records),
        "num_solved": len(solved_records),
        "num_failed": len(records) - len(solved_records),
        "success_rate_percent": success_rate,
        "success_rate_percent_std": float(
            np.std([1.0 if item else 0.0 for item in success], ddof=1) * 100.0
        ) if len(success) > 1 else 0.0,
        "mean_path_cost_successful": path_mean,
        "std_path_cost_successful": path_std,
        "mean_pcr_successful": pcr_mean,
        "std_pcr_successful": pcr_std,
        "mean_unique_state_expansions": unique_mean,
        "std_unique_state_expansions": unique_std,
        "mean_reexpansions": reexp_mean,
        "mean_transition_evaluations": transition_mean,
        "std_transition_evaluations": transition_std,
        "mean_search_time_ms": time_mean,
        "std_search_time_ms": time_std,
        "mean_gnn_inference_time_ms": inference_mean,
        "std_gnn_inference_time_ms": inference_std,
        "failure_reasons": failure_reasons,
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped = group_records_by_algorithm(records)
    return {
        algorithm: summarize_algorithm(
            group,
            algorithm,
            budget_limited=all(
                _as_bool(record.get("budget_limited", True)) for record in group
            ),
        )
        for algorithm, group in grouped.items()
    }


def summarize_by_structure(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for structure in STRUCTURES:
        subset = [record for record in records if str(record["structure"]) == structure]
        result[structure] = summarize_records(subset) if subset else {}
    return result


def structure_analysis_rows(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_structure = summarize_by_structure(records)
    for structure in STRUCTURES:
        for algorithm, summary in by_structure[structure].items():
            rows.append(
                {
                    "algorithm": algorithm,
                    "structure": structure,
                    "num_instances": summary["num_instances"],
                    "num_solved": summary["num_solved"],
                    "success_rate_percent": summary["success_rate_percent"],
                    "mean_path_cost_successful": summary["mean_path_cost_successful"],
                    "mean_pcr_successful": summary["mean_pcr_successful"],
                    "mean_unique_state_expansions": summary["mean_unique_state_expansions"],
                    "mean_reexpansions": summary["mean_reexpansions"],
                    "mean_transition_evaluations": summary["mean_transition_evaluations"],
                    "mean_search_time_ms": summary["mean_search_time_ms"],
                    "mean_gnn_inference_time_ms": summary["mean_gnn_inference_time_ms"],
                    "budget_limited": summary["budget_limited"],
                }
            )
    return rows


def write_evaluation_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in EVALUATION_COLUMNS})
    return path


def read_evaluation_csv(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                {
                    "algorithm": row["algorithm"],
                    "instance_id": _as_int(row.get("instance_id")),
                    "structure": row.get("structure", "unknown"),
                    "success": _as_bool(row.get("success")),
                    "path_cost": _as_float(row.get("path_cost")),
                    "optimal_cost": _as_float(row.get("optimal_cost")),
                    "pcr": _as_float(row.get("pcr")),
                    "unique_state_expansions": _as_int(row.get("unique_state_expansions")),
                    "reexpansions": _as_int(row.get("reexpansions")),
                    "transition_evaluations": _as_int(row.get("transition_evaluations")),
                    "search_time_ms": _as_float(row.get("search_time_ms")),
                    "gnn_inference_time_ms": _as_float(row.get("gnn_inference_time_ms")),
                    "failure_reason": row.get("failure_reason") or "",
                    "budget_limited": _as_bool(row.get("budget_limited")),
                }
            )
    return records


def write_structure_analysis_csv(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    rows = structure_analysis_rows(records)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=STRUCTURE_ANALYSIS_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def read_structure_analysis_csv(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: Dict[str, Any] = {"algorithm": row["algorithm"], "structure": row["structure"]}
            for key in STRUCTURE_ANALYSIS_COLUMNS:
                if key in ("algorithm", "structure", "budget_limited"):
                    continue
                parsed[key] = _as_float(row.get(key))
            parsed["budget_limited"] = _as_bool(row.get("budget_limited"))
            rows.append(parsed)
    return rows


def save_summary(summary: Mapping[str, Any], path: str | Path) -> Path:
    return save_json(summary, path)