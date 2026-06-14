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

FAILURE_FIELDS = [
    "case_id",
    "pool",
    "validation_pool",
    "selection_group",
    "drop_reason",
    "raw_temperature_extrema_passed",
    "raw_temperature_max_T_K",
    "raw_temperature_reason",
]


def resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def retained(convergence: dict[str, str]) -> bool:
    return (
        truthy(convergence.get("converged"))
        and (convergence.get("raw_temperature_extrema_passed", "") == "" or truthy(convergence.get("raw_temperature_extrema_passed")))
    )


def base_parameter_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in PARAMETER_FIELDS}


def selection_fields(rows: list[dict[str, str]]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key, "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-selection", default="evidence_pack_tier2_steady/03_data/current_envelope_selection.csv")
    parser.add_argument("--replacement-selection", default="evidence_pack_tier2_steady/03_data/temperature_replacement_selection.csv")
    parser.add_argument("--convergence-log", default="evidence_pack_tier2_steady/03_data/convergence_log.csv")
    parser.add_argument("--target-count", type=int, default=19)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--output-parameters", default="evidence_pack_tier2_steady/03_data/physical_envelope_parameters.csv")
    parser.add_argument("--output-selection", default="evidence_pack_tier2_steady/03_data/physical_envelope_selection.csv")
    parser.add_argument("--output-failures", default="evidence_pack_tier2_steady/03_data/physical_envelope_failures.csv")
    parser.add_argument("--output-summary", default="evidence_pack_tier2_steady/03_data/physical_envelope_summary.json")
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)

    current_selection_path = resolve(paths.root, args.current_selection)
    replacement_selection_path = resolve(paths.root, args.replacement_selection)
    convergence_path = resolve(paths.root, args.convergence_log)
    current_rows = read_csv(current_selection_path)
    replacement_rows = read_csv(replacement_selection_path) if replacement_selection_path.exists() else []
    convergence_by_case = {row["case_id"]: row for row in read_csv(convergence_path)}

    accepted: list[dict[str, Any]] = [row for row in current_rows if retained(convergence_by_case.get(row["case_id"], {}))]
    selected_ids = {row["case_id"] for row in accepted}

    replacement_groups: dict[str, list[dict[str, str]]] = {}
    replacement_group_order: list[str] = []
    for row in replacement_rows:
        key = row.get("replaces_case_id", row["case_id"])
        if key not in replacement_groups:
            replacement_group_order.append(key)
            replacement_groups[key] = []
        replacement_groups[key].append(row)

    for key in replacement_group_order:
        if len(accepted) >= args.target_count:
            break
        retained_group = [row for row in replacement_groups[key] if retained(convergence_by_case.get(row["case_id"], {}))]
        if retained_group:
            accepted.append(retained_group[0])
            selected_ids.add(retained_group[0]["case_id"])

    for row in replacement_rows:
        if len(accepted) >= args.target_count:
            break
        if row["case_id"] in selected_ids:
            continue
        if retained(convergence_by_case.get(row["case_id"], {})):
            accepted.append(row)
            selected_ids.add(row["case_id"])

    failures: list[dict[str, Any]] = []
    for row in [*current_rows, *replacement_rows]:
        convergence = convergence_by_case.get(row["case_id"], {})
        if row["case_id"] not in selected_ids:
            failures.append(
                {
                    "case_id": row["case_id"],
                    "pool": row.get("pool", ""),
                    "validation_pool": row.get("validation_pool", ""),
                    "selection_group": row.get("selection_group", ""),
                    "drop_reason": convergence.get("drop_reason", "not selected after target count reached"),
                    "raw_temperature_extrema_passed": convergence.get("raw_temperature_extrema_passed", ""),
                    "raw_temperature_max_T_K": convergence.get("raw_temperature_max_T_K", ""),
                    "raw_temperature_reason": convergence.get("raw_temperature_reason", ""),
                }
            )

    if len(accepted) < args.target_count and not args.allow_short:
        raise SystemExit(f"Only {len(accepted)} retained case(s) available for target {args.target_count}.")

    output_parameters = resolve(paths.root, args.output_parameters)
    output_selection = resolve(paths.root, args.output_selection)
    output_failures = resolve(paths.root, args.output_failures)
    output_summary = resolve(paths.root, args.output_summary)

    fields = selection_fields([*current_rows, *replacement_rows])
    write_csv(output_parameters, [base_parameter_row(row) for row in accepted], PARAMETER_FIELDS)
    write_csv(output_selection, accepted, fields)
    write_csv(output_failures, failures, FAILURE_FIELDS)
    write_json(
        output_summary,
        {
            "target_count": args.target_count,
            "retained_cases": len(accepted),
            "candidate_failures_or_unused": len(failures),
            "case_count_by_pool": count_by(accepted, "pool"),
            "case_count_by_validation_pool": count_by(accepted, "validation_pool"),
            "parameter_file": str(output_parameters.relative_to(paths.root)),
            "selection_file": str(output_selection.relative_to(paths.root)),
            "failures_file": str(output_failures.relative_to(paths.root)),
        },
    )
    print(f"Wrote physical envelope: {output_parameters.relative_to(paths.root)} ({len(accepted)} case(s))")


if __name__ == "__main__":
    main()
