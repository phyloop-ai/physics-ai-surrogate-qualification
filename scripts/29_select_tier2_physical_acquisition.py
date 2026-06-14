#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import PARAMETER_ORDER, ensure_evidence_folders, load_config, read_csv, study_paths, write_csv, write_json


PARAMETER_FIELDS = [
    "case_id",
    "pool",
    *PARAMETER_ORDER,
    "geometry_family",
    "solver_status",
]

SELECTION_FIELDS = [
    *PARAMETER_FIELDS,
    "source_parameter_file",
    "nearest_id_train_distance",
    "ood_axis",
    "distance_bin",
    "stratification_key",
    "temperature_risk_proxy",
    "acquisition_rank_in_pool",
    "target_retained_count",
    "selection_reason",
]

CANONICAL_POOLS = ("id_train", "id_calibration", "id_test", "ood_test")


def resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def base_parameter_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in PARAMETER_FIELDS}


def target_counts(config: dict[str, Any], fresh_target: int) -> dict[str, int]:
    dataset = config["dataset"]
    return {
        "id_train": int(dataset["id_train"]),
        "id_calibration": int(dataset["id_calibration"]),
        "id_test": int(dataset["id_test"]),
        "ood_test": int(dataset["ood_test"]),
        "ood_fresh": int(fresh_target),
    }


def temperature_risk_proxy(row: dict[str, Any]) -> float:
    n_fin = float(row["n_fin"])
    q_w = float(row["q_w"])
    u_in = max(float(row["u_in"]), 0.05)
    effective_inlet_mm = max(min(float(row["d_in"]), float(row["h_ch"])), 0.2)
    return q_w * (n_fin / 8.0) / (u_in * effective_inlet_mm)


def ood_axis(row: dict[str, Any], config: dict[str, Any]) -> str:
    params = config["parameters"]
    axes = []
    n_fin = int(float(row["n_fin"]))
    if n_fin not in {int(value) for value in params["n_fin"]["id_values"]}:
        axes.append(f"n_fin_{n_fin}")
    h_ch = float(row["h_ch"])
    d_in = float(row["d_in"])
    if h_ch > float(params["h_ch"]["id_max"]):
        axes.append("h_ch_high")
    if d_in < float(params["d_in"]["id_min"]):
        axes.append("d_in_low")
    if d_in > float(params["d_in"]["id_max"]):
        axes.append("d_in_high")
    return "+".join(axes) if axes else "id"


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = (len(ordered) - 1) * fraction
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def distance_bin(distance: float | None, edges: list[float]) -> str:
    if distance is None or not edges:
        return ""
    for idx in range(len(edges) - 1):
        upper_inclusive = idx == len(edges) - 2
        if distance >= edges[idx] and (distance <= edges[idx + 1] if upper_inclusive else distance < edges[idx + 1]):
            return str(idx + 1)
    return str(max(1, len(edges) - 1))


def distance_edges(distances: list[float], bins: int) -> list[float]:
    if not distances:
        return []
    return [quantile(distances, idx / bins) for idx in range(bins + 1)]


def interleaved_by_stratum(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stratification_key"])].append(row)
    for group_rows in grouped.values():
        group_rows.sort(
            key=lambda item: (
                float(item["temperature_risk_proxy"]),
                -float(item.get("nearest_id_train_distance") or 0.0),
                str(item["case_id"]),
            )
        )

    ordered: list[dict[str, Any]] = []
    keys = sorted(grouped)
    while keys:
        remaining = []
        for key in keys:
            if grouped[key]:
                ordered.append(grouped[key].pop(0))
            if grouped[key]:
                remaining.append(key)
        keys = remaining
    return ordered


def enriched_rows(
    rows: list[dict[str, str]],
    config: dict[str, Any],
    source_path: Path,
    root: Path,
    target_count_by_pool: dict[str, int],
    distances_by_case: dict[str, float],
    distance_axis_by_case: dict[str, str],
    edges_by_pool: dict[str, list[float]],
) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        pool = row["pool"]
        distance = distances_by_case.get(row["case_id"])
        axis = distance_axis_by_case.get(row["case_id"]) or ood_axis(row, config)
        bin_value = distance_bin(distance, edges_by_pool.get(pool, []))
        if pool in ("id_train", "id_calibration", "id_test"):
            stratum = f"{pool}|n_fin_{int(float(row['n_fin']))}"
        elif pool in ("ood_test", "ood_fresh"):
            stratum = f"{pool}|{axis}|bin_{bin_value or 'na'}"
        else:
            stratum = pool
        enriched.append(
            {
                **base_parameter_row(row),
                "source_parameter_file": str(source_path.relative_to(root)),
                "nearest_id_train_distance": "" if distance is None else f"{distance:.8f}",
                "ood_axis": axis,
                "distance_bin": bin_value,
                "stratification_key": stratum,
                "temperature_risk_proxy": f"{temperature_risk_proxy(row):.8f}",
                "acquisition_rank_in_pool": "",
                "target_retained_count": target_count_by_pool.get(pool, ""),
                "selection_reason": (
                    "physical-acquisition candidate ordered by thermal-risk proxy within a pool/regime stratum"
                ),
            }
        )
    return enriched


def ordered_pool_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pool[row["pool"]].append(row)

    ordered = []
    for pool in (*CANONICAL_POOLS, "ood_fresh"):
        pool_rows = interleaved_by_stratum(by_pool.get(pool, []))
        for idx, row in enumerate(pool_rows, start=1):
            row["acquisition_rank_in_pool"] = idx
            ordered.append(row)
    return ordered


def pool_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row["pool"] for row in rows).items()))


def stratum_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["stratification_key"]) for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-file", default="evidence_pack_tier2_steady/03_data/parameters.csv")
    parser.add_argument("--ood-distance-file", default="evidence_pack_tier2_steady/03_data/ood_distance.csv")
    parser.add_argument("--fresh-parameter-file", default="evidence_pack_tier2_steady/03_data/fresh_ood_parameters.csv")
    parser.add_argument("--fresh-distance-file", default="evidence_pack_tier2_steady/03_data/fresh_ood_distance.csv")
    parser.add_argument("--fresh-retained-target", type=int, default=90)
    parser.add_argument("--distance-bins", type=int, default=5)
    parser.add_argument(
        "--canonical-output-parameters",
        default="evidence_pack_tier2_steady/03_data/physical_acquisition_parameters.csv",
    )
    parser.add_argument(
        "--canonical-selection-file",
        default="evidence_pack_tier2_steady/03_data/physical_acquisition_selection.csv",
    )
    parser.add_argument(
        "--fresh-output-parameters",
        default="evidence_pack_tier2_steady/03_data/physical_fresh_ood_acquisition_parameters.csv",
    )
    parser.add_argument(
        "--fresh-selection-file",
        default="evidence_pack_tier2_steady/03_data/physical_fresh_ood_acquisition_selection.csv",
    )
    parser.add_argument(
        "--combined-output-parameters",
        default="evidence_pack_tier2_steady/03_data/physical_acquisition_all_parameters.csv",
    )
    parser.add_argument(
        "--summary-file",
        default="evidence_pack_tier2_steady/03_data/physical_acquisition_summary.json",
    )
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)

    parameter_path = resolve(paths.root, args.parameter_file)
    ood_distance_path = resolve(paths.root, args.ood_distance_file)
    fresh_parameter_path = resolve(paths.root, args.fresh_parameter_file)
    fresh_distance_path = resolve(paths.root, args.fresh_distance_file)
    required = [parameter_path, ood_distance_path, fresh_parameter_path, fresh_distance_path]
    missing = [str(path.relative_to(paths.root)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required acquisition inputs: {', '.join(missing)}")

    canonical_rows = [row for row in read_csv(parameter_path) if row["pool"] in CANONICAL_POOLS]
    fresh_rows = [row for row in read_csv(fresh_parameter_path) if row["pool"] == "ood_fresh"]
    if args.fresh_retained_target > len(fresh_rows):
        raise SystemExit(
            "Fresh-OOD retained target exceeds available candidates: "
            f"target={args.fresh_retained_target}, candidates={len(fresh_rows)}. "
            "Regenerate the fresh-OOD candidate pool with a larger --candidate-count first."
        )

    ood_distance_rows = read_csv(ood_distance_path)
    fresh_distance_rows = read_csv(fresh_distance_path)
    distances_by_case = {
        **{row["case_id"]: float(row["nearest_id_train_distance"]) for row in ood_distance_rows},
        **{row["case_id"]: float(row["nearest_id_train_distance"]) for row in fresh_distance_rows},
    }
    distance_axis_by_case = {row["case_id"]: row.get("ood_axis", "") for row in fresh_distance_rows}
    edges_by_pool = {
        "ood_test": distance_edges([float(row["nearest_id_train_distance"]) for row in ood_distance_rows], args.distance_bins),
        "ood_fresh": distance_edges(
            [float(row["nearest_id_train_distance"]) for row in fresh_distance_rows],
            args.distance_bins,
        ),
    }
    targets = target_counts(config, args.fresh_retained_target)

    canonical_selection = ordered_pool_rows(
        enriched_rows(
            canonical_rows,
            config,
            parameter_path,
            paths.root,
            targets,
            distances_by_case,
            distance_axis_by_case,
            edges_by_pool,
        )
    )
    fresh_selection = ordered_pool_rows(
        enriched_rows(
            fresh_rows,
            config,
            fresh_parameter_path,
            paths.root,
            targets,
            distances_by_case,
            distance_axis_by_case,
            edges_by_pool,
        )
    )
    combined_selection = [*canonical_selection, *fresh_selection]

    canonical_output = resolve(paths.root, args.canonical_output_parameters)
    canonical_selection_path = resolve(paths.root, args.canonical_selection_file)
    fresh_output = resolve(paths.root, args.fresh_output_parameters)
    fresh_selection_path = resolve(paths.root, args.fresh_selection_file)
    combined_output = resolve(paths.root, args.combined_output_parameters)
    summary_path = resolve(paths.root, args.summary_file)

    write_csv(canonical_output, [base_parameter_row(row) for row in canonical_selection], PARAMETER_FIELDS)
    write_csv(canonical_selection_path, canonical_selection, SELECTION_FIELDS)
    write_csv(fresh_output, [base_parameter_row(row) for row in fresh_selection], PARAMETER_FIELDS)
    write_csv(fresh_selection_path, fresh_selection, SELECTION_FIELDS)
    write_csv(combined_output, [base_parameter_row(row) for row in combined_selection], PARAMETER_FIELDS)
    write_json(
        summary_path,
        {
            "candidate_counts_by_pool": pool_counts(combined_selection),
            "canonical_candidate_counts_by_pool": pool_counts(canonical_selection),
            "fresh_candidate_counts_by_pool": pool_counts(fresh_selection),
            "target_retained_counts": targets,
            "distance_bins": args.distance_bins,
            "ood_test_distance_edges": [round(value, 8) for value in edges_by_pool["ood_test"]],
            "ood_fresh_distance_edges": [round(value, 8) for value in edges_by_pool["ood_fresh"]],
            "canonical_stratum_counts": stratum_counts(canonical_selection),
            "fresh_stratum_counts": stratum_counts(fresh_selection),
            "risk_proxy": "q_w * (n_fin / 8) / (u_in * min(d_in, h_ch)) using mm for inlet width",
            "ordering_policy": "round-robin across pool/regime strata, sorted by lower thermal-risk proxy within each stratum",
            "canonical_parameter_file": str(canonical_output.relative_to(paths.root)),
            "canonical_selection_file": str(canonical_selection_path.relative_to(paths.root)),
            "fresh_parameter_file": str(fresh_output.relative_to(paths.root)),
            "fresh_selection_file": str(fresh_selection_path.relative_to(paths.root)),
            "combined_parameter_file": str(combined_output.relative_to(paths.root)),
        },
    )

    print(f"Wrote canonical acquisition stream: {canonical_output.relative_to(paths.root)}")
    print(f"Wrote fresh-OOD acquisition stream: {fresh_output.relative_to(paths.root)}")
    print(f"Wrote combined extraction/training limit file: {combined_output.relative_to(paths.root)}")


if __name__ == "__main__":
    main()
