#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import ensure_evidence_folders, load_config, read_csv, study_paths, write_csv, write_json


PARAMETER_FIELDS = [
    "case_id",
    "pool",
    "h_ch",
    "n_fin",
    "h_fin",
    "d_in",
    "u_in",
    "q_w",
    "geometry_family",
    "solver_status",
]

EXCLUSION_FIELDS = [
    "case_id",
    "pool",
    "validation_pool",
    "selection_group",
    "target_regime",
    "reason",
    "drop_reason",
    "converged",
    "scalar_stability_passed",
    "raw_enthalpy_passed",
]


def resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def envelope_passed(summary_row: dict[str, str]) -> bool:
    return (
        truthy(summary_row.get("converged"))
        and truthy(summary_row.get("scalar_stability_passed"))
        and truthy(summary_row.get("raw_enthalpy_passed"))
        and (
            summary_row.get("raw_temperature_extrema_passed", "") == ""
            or truthy(summary_row.get("raw_temperature_extrema_passed"))
        )
    )


def current_envelope_rows(summary_rows: list[dict[str, str]]) -> tuple[set[str], list[dict[str, str]], list[dict[str, str]]]:
    in_envelope_ids: set[str] = set()
    in_envelope_failures: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []

    for row in summary_rows:
        if truthy(row.get("in_current_steady_solver_envelope")):
            in_envelope_ids.add(row["case_id"])
            if not envelope_passed(row):
                in_envelope_failures.append(
                    {
                        "case_id": row["case_id"],
                        "validation_pool": row.get("validation_pool", ""),
                        "drop_reason": row.get("drop_reason", "failed current-envelope solver contract"),
                    }
                )
        else:
            exclusions.append(
                {
                    "case_id": row["case_id"],
                    "pool": row.get("pool", ""),
                    "validation_pool": row.get("validation_pool", ""),
                    "selection_group": row.get("selection_group", ""),
                    "target_regime": row.get("target_regime", ""),
                    "reason": row.get("solver_envelope_reason", ""),
                    "drop_reason": row.get("drop_reason", ""),
                    "converged": row.get("converged", ""),
                    "scalar_stability_passed": row.get("scalar_stability_passed", ""),
                    "raw_enthalpy_passed": row.get("raw_enthalpy_passed", ""),
                }
            )

    return in_envelope_ids, in_envelope_failures, exclusions


def rows_by_case(rows: list[dict[str, str]], source_name: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    duplicates = []
    for row in rows:
        case_id = row["case_id"]
        if case_id in indexed:
            duplicates.append(case_id)
        indexed[case_id] = row
    if duplicates:
        raise SystemExit(f"{source_name} contains duplicate case IDs: {sorted(set(duplicates))}")
    return indexed


def pool_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        pool = row["pool"]
        counts[pool] = counts.get(pool, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        default="evidence_pack_tier2_steady/03_data/solver_validation_summary.csv",
    )
    parser.add_argument(
        "--selection-file",
        default="evidence_pack_tier2_steady/03_data/solver_validation_selection.csv",
    )
    parser.add_argument(
        "--parameter-file",
        default="evidence_pack_tier2_steady/03_data/solver_validation_parameters.csv",
    )
    parser.add_argument(
        "--output-parameters",
        default="evidence_pack_tier2_steady/03_data/current_envelope_parameters.csv",
    )
    parser.add_argument(
        "--output-selection",
        default="evidence_pack_tier2_steady/03_data/current_envelope_selection.csv",
    )
    parser.add_argument(
        "--output-exclusions",
        default="evidence_pack_tier2_steady/03_data/current_envelope_exclusions.csv",
    )
    parser.add_argument(
        "--output-summary",
        default="evidence_pack_tier2_steady/03_data/current_envelope_summary.json",
    )
    parser.add_argument("--require-envelope-pass", action="store_true", default=True)
    parser.add_argument("--allow-envelope-failures", action="store_false", dest="require_envelope_pass")
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)

    summary_path = resolve(paths.root, args.summary_csv)
    selection_path = resolve(paths.root, args.selection_file)
    parameter_path = resolve(paths.root, args.parameter_file)

    missing = [path for path in (summary_path, selection_path, parameter_path) if not path.exists()]
    if missing:
        raise SystemExit("Missing required input(s): " + ", ".join(str(path) for path in missing))

    summary_rows = read_csv(summary_path)
    selection_rows = read_csv(selection_path)
    parameter_rows = read_csv(parameter_path)

    in_envelope_ids, in_envelope_failures, exclusions = current_envelope_rows(summary_rows)
    selection_by_case = rows_by_case(selection_rows, "selection file")
    parameter_by_case = rows_by_case(parameter_rows, "parameter file")

    missing_selection = sorted(in_envelope_ids - set(selection_by_case))
    missing_parameters = sorted(in_envelope_ids - set(parameter_by_case))
    if missing_selection or missing_parameters:
        raise SystemExit(
            "Current-envelope IDs missing from source files: "
            f"selection={missing_selection}, parameters={missing_parameters}"
        )

    ordered_ids = [row["case_id"] for row in selection_rows if row["case_id"] in in_envelope_ids]
    current_parameters = [{field: parameter_by_case[case_id].get(field, "") for field in PARAMETER_FIELDS} for case_id in ordered_ids]
    current_selection = [selection_by_case[case_id] for case_id in ordered_ids]

    output_parameters = resolve(paths.root, args.output_parameters)
    output_selection = resolve(paths.root, args.output_selection)
    output_exclusions = resolve(paths.root, args.output_exclusions)
    output_summary = resolve(paths.root, args.output_summary)

    selection_fields = list(selection_rows[0].keys()) if selection_rows else []
    write_csv(output_parameters, current_parameters, PARAMETER_FIELDS)
    write_csv(output_selection, current_selection, selection_fields)
    write_csv(output_exclusions, exclusions, EXCLUSION_FIELDS)

    payload: dict[str, Any] = {
        "selected_cases": len(summary_rows),
        "current_envelope_cases": len(current_parameters),
        "excluded_cases": len(exclusions),
        "current_envelope_all_passed": not in_envelope_failures,
        "current_envelope_failures": in_envelope_failures,
        "case_count_by_pool": pool_counts(current_parameters),
        "source_summary_csv": str(summary_path.relative_to(paths.root)),
        "source_selection_file": str(selection_path.relative_to(paths.root)),
        "source_parameter_file": str(parameter_path.relative_to(paths.root)),
        "current_envelope_parameter_file": str(output_parameters.relative_to(paths.root)),
        "current_envelope_selection_file": str(output_selection.relative_to(paths.root)),
        "current_envelope_exclusions_file": str(output_exclusions.relative_to(paths.root)),
        "exclusion_rule": "Exclude selected hard-regime cases whose target_regime contains d_in_high until a separate validated steady/transient solver path exists.",
    }
    write_json(output_summary, payload)

    print(f"Wrote {output_parameters.relative_to(paths.root)} ({len(current_parameters)} case(s))")
    print(f"Wrote {output_selection.relative_to(paths.root)}")
    print(f"Wrote {output_exclusions.relative_to(paths.root)} ({len(exclusions)} exclusion(s))")
    print(f"Wrote {output_summary.relative_to(paths.root)}")

    if in_envelope_failures and args.require_envelope_pass:
        raise SystemExit(f"{len(in_envelope_failures)} current-envelope case(s) failed solver-contract requirements")


if __name__ == "__main__":
    main()
