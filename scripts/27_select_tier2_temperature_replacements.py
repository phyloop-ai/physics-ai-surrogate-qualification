#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
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
    "validation_pool",
    "selection_group",
    "selection_reason",
    "source_parameter_file",
    "nearest_id_train_distance",
    "target_regime",
    "ood_axis",
    "replaces_case_id",
    "temperature_risk_proxy",
]


def resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def rows_by_case(path: Path) -> dict[str, dict[str, str]]:
    return {row["case_id"]: row for row in read_csv(path)}


def base_parameter_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in PARAMETER_FIELDS}


def temperature_risk_proxy(row: dict[str, str]) -> float:
    n_fin = float(row["n_fin"])
    q_w = float(row["q_w"])
    u_in = max(float(row["u_in"]), 0.05)
    effective_inlet_mm = max(min(float(row["d_in"]), float(row["h_ch"])), 0.2)
    return q_w * (n_fin / 8.0) / (u_in * effective_inlet_mm)


def candidate_sort_key(row: dict[str, str]) -> tuple[float, float]:
    return (temperature_risk_proxy(row), -float(row.get("nearest_id_train_distance", "0") or 0.0))


def failed_temperature_rows(selection_rows: list[dict[str, str]], convergence_by_case: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    failed = []
    for row in selection_rows:
        convergence = convergence_by_case.get(row["case_id"], {})
        if truthy(convergence.get("converged")):
            continue
        reason = convergence.get("drop_reason", "")
        if "raw temperature extrema" in reason:
            failed.append(row)
    return failed


def near_ood_candidates(
    failed_row: dict[str, str],
    base_rows: dict[str, dict[str, str]],
    ood_distance_rows: list[dict[str, str]],
    excluded_ids: set[str],
    min_distance: float,
) -> list[dict[str, str]]:
    n_fin = int(float(failed_row["n_fin"]))
    candidates = []
    for distance_row in ood_distance_rows:
        case = base_rows[distance_row["case_id"]]
        if distance_row["case_id"] in excluded_ids:
            continue
        if int(float(case["n_fin"])) != n_fin:
            continue
        if float(distance_row["nearest_id_train_distance"]) < min_distance:
            continue
        candidates.append({**case, **distance_row})
    return sorted(candidates, key=candidate_sort_key)


def fresh_ood_candidates(
    failed_row: dict[str, str],
    fresh_rows: dict[str, dict[str, str]],
    fresh_distance_rows: list[dict[str, str]],
    excluded_ids: set[str],
    min_distance: float,
) -> list[dict[str, str]]:
    axis = failed_row["ood_axis"]
    candidates = []
    for distance_row in fresh_distance_rows:
        if distance_row["case_id"] in excluded_ids:
            continue
        if distance_row["ood_axis"] != axis:
            continue
        if float(distance_row["nearest_id_train_distance"]) < min_distance:
            continue
        case = fresh_rows[distance_row["case_id"]]
        candidates.append({**case, **distance_row})
    return sorted(candidates, key=candidate_sort_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-selection", default="evidence_pack_tier2_steady/03_data/current_envelope_selection.csv")
    parser.add_argument("--convergence-log", default="evidence_pack_tier2_steady/03_data/convergence_log.csv")
    parser.add_argument("--base-parameter-file", default="evidence_pack_tier2_steady/03_data/parameters.csv")
    parser.add_argument("--ood-distance-file", default="evidence_pack_tier2_steady/03_data/ood_distance.csv")
    parser.add_argument("--fresh-parameter-file", default="evidence_pack_tier2_steady/03_data/fresh_ood_parameters.csv")
    parser.add_argument("--fresh-distance-file", default="evidence_pack_tier2_steady/03_data/fresh_ood_distance.csv")
    parser.add_argument("--candidates-per-failure", type=int, default=2)
    parser.add_argument(
        "--output-parameter-file",
        default="evidence_pack_tier2_steady/03_data/temperature_replacement_parameters.csv",
    )
    parser.add_argument(
        "--output-selection-file",
        default="evidence_pack_tier2_steady/03_data/temperature_replacement_selection.csv",
    )
    parser.add_argument(
        "--summary-file",
        default="evidence_pack_tier2_steady/03_data/temperature_replacement_selection_summary.json",
    )
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)

    current_selection_path = resolve(paths.root, args.current_selection)
    convergence_path = resolve(paths.root, args.convergence_log)
    base_parameter_path = resolve(paths.root, args.base_parameter_file)
    ood_distance_path = resolve(paths.root, args.ood_distance_file)
    fresh_parameter_path = resolve(paths.root, args.fresh_parameter_file)
    fresh_distance_path = resolve(paths.root, args.fresh_distance_file)

    current_selection = read_csv(current_selection_path)
    convergence_by_case = {row["case_id"]: row for row in read_csv(convergence_path)}
    base_rows = rows_by_case(base_parameter_path)
    fresh_rows = rows_by_case(fresh_parameter_path)
    ood_distance_rows = read_csv(ood_distance_path)
    fresh_distance_rows = read_csv(fresh_distance_path)

    selected_ids = {row["case_id"] for row in current_selection}
    used_ids = set(selected_ids)
    replacement_rows: list[dict[str, Any]] = []
    failed_rows = failed_temperature_rows(current_selection, convergence_by_case)

    for failed_row in failed_rows:
        high_distance_replacement = "high_distance" in failed_row.get("selection_group", "")
        if failed_row["validation_pool"] == "near_ood":
            min_distance = 0.75 if high_distance_replacement else 0.0
            candidates = near_ood_candidates(failed_row, base_rows, ood_distance_rows, used_ids, min_distance)
        elif failed_row["validation_pool"] == "fresh_ood":
            min_distance = 1.0 if high_distance_replacement else 0.0
            candidates = fresh_ood_candidates(failed_row, fresh_rows, fresh_distance_rows, used_ids, min_distance)
        else:
            candidates = []

        for idx, candidate in enumerate(candidates[: args.candidates_per_failure], start=1):
            used_ids.add(candidate["case_id"])
            source_path = fresh_parameter_path if candidate["pool"] == "ood_fresh" else base_parameter_path
            replacement_rows.append(
                {
                    **base_parameter_row(candidate),
                    "validation_pool": failed_row["validation_pool"],
                    "selection_group": f"{failed_row['selection_group']}_temperature_replacement_{idx}",
                    "selection_reason": (
                        f"temperature-extrema replacement for {failed_row['case_id']} selected by low heat-risk proxy"
                    ),
                    "source_parameter_file": str(source_path.relative_to(paths.root)),
                    "nearest_id_train_distance": f"{float(candidate.get('nearest_id_train_distance', 0.0)):.8f}",
                    "target_regime": "",
                    "ood_axis": candidate.get("ood_axis", failed_row.get("ood_axis", "")),
                    "replaces_case_id": failed_row["case_id"],
                    "temperature_risk_proxy": f"{temperature_risk_proxy(candidate):.8f}",
                }
            )

    output_parameter_path = resolve(paths.root, args.output_parameter_file)
    output_selection_path = resolve(paths.root, args.output_selection_file)
    summary_path = resolve(paths.root, args.summary_file)

    write_csv(output_selection_path, replacement_rows, SELECTION_FIELDS)
    write_csv(output_parameter_path, [base_parameter_row(row) for row in replacement_rows], PARAMETER_FIELDS)
    write_json(
        summary_path,
        {
            "failed_temperature_cases": [row["case_id"] for row in failed_rows],
            "candidate_count": len(replacement_rows),
            "candidates_per_failure": args.candidates_per_failure,
            "selection_csv": str(output_selection_path.relative_to(paths.root)),
            "parameter_csv": str(output_parameter_path.relative_to(paths.root)),
            "risk_proxy": "q_w * (n_fin / 8) / (u_in * min(d_in, h_ch)) using mm for inlet width",
        },
    )
    print(f"Wrote {len(replacement_rows)} replacement candidate(s): {output_parameter_path.relative_to(paths.root)}")


if __name__ == "__main__":
    main()
