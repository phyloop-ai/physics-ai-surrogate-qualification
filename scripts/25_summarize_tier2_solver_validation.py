#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import ensure_evidence_folders, load_config, read_csv, study_paths, write_csv, write_json


SUMMARY_FIELDS = [
    "case_id",
    "pool",
    "validation_pool",
    "selection_group",
    "target_regime",
    "in_current_steady_solver_envelope",
    "solver_envelope_reason",
    "converged",
    "dropped",
    "drop_reason",
    "scalar_stability_passed",
    "scalar_mean_delta_T_K",
    "scalar_max_delta_T_K",
    "raw_enthalpy_passed",
    "raw_enthalpy_outlet_delta_T_K",
    "raw_enthalpy_expected_delta_T_K",
    "raw_enthalpy_balance_ratio",
    "raw_enthalpy_mass_balance_rel_error",
    "raw_temperature_extrema_passed",
    "raw_temperature_min_T_K",
    "raw_temperature_mean_T_K",
    "raw_temperature_max_T_K",
    "raw_temperature_reason",
    "residual_U_initial",
    "residual_U_final",
    "residual_p_initial",
    "residual_p_final",
    "residual_T_initial",
    "residual_T_final",
    "flow_converged_reported",
    "iterations_flow",
    "iterations_temperature",
    "openfoam_version",
]


def resolve(paths_root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else paths_root / candidate


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def safe_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def current_steady_solver_envelope(selection: dict[str, str]) -> tuple[bool, str]:
    target_regime = selection.get("target_regime", "")
    if selection.get("validation_pool") == "hard_regime" and "d_in_high" in target_regime:
        return False, "hard high-d_in regime requires a separate validated steady/transient solver path"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-file", default="evidence_pack_tier2_steady/03_data/solver_validation_selection.csv")
    parser.add_argument("--convergence-log", default="evidence_pack_tier2_steady/03_data/convergence_log.csv")
    parser.add_argument("--summary-csv", default="evidence_pack_tier2_steady/03_data/solver_validation_summary.csv")
    parser.add_argument("--summary-json", default="evidence_pack_tier2_steady/03_data/solver_validation_summary.json")
    parser.add_argument("--require-all-pass", action="store_true")
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)

    selection_path = resolve(paths.root, args.selection_file)
    convergence_path = resolve(paths.root, args.convergence_log)
    if not selection_path.exists():
        raise SystemExit(f"Missing selection file: {selection_path}")
    if not convergence_path.exists():
        raise SystemExit(f"Missing convergence log: {convergence_path}")

    selections = read_csv(selection_path)
    convergence_by_case = {row["case_id"]: row for row in read_csv(convergence_path)}
    summary_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    raw_ratios: list[float] = []
    scalar_maxima: list[float] = []
    by_validation_pool: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
    envelope_total = 0
    envelope_failures: list[dict[str, str]] = []
    envelope_exclusions: list[dict[str, str]] = []

    for selection in selections:
        case_id = selection["case_id"]
        convergence = convergence_by_case.get(case_id, {})
        in_envelope, envelope_reason = current_steady_solver_envelope(selection)
        passed = (
            truthy(convergence.get("converged"))
            and truthy(convergence.get("scalar_stability_passed"))
            and truthy(convergence.get("raw_enthalpy_passed"))
            and (convergence.get("raw_temperature_extrema_passed", "") == "" or truthy(convergence.get("raw_temperature_extrema_passed")))
        )
        row = {
            "case_id": case_id,
            "pool": selection["pool"],
            "validation_pool": selection["validation_pool"],
            "selection_group": selection["selection_group"],
            "target_regime": selection.get("target_regime", ""),
            "in_current_steady_solver_envelope": str(in_envelope).lower(),
            "solver_envelope_reason": envelope_reason,
        }
        for field in SUMMARY_FIELDS[7:]:
            row[field] = convergence.get(field, "")
        summary_rows.append(row)
        bucket = by_validation_pool[selection["validation_pool"]]
        bucket["total"] += 1
        bucket["passed" if passed else "failed"] += 1
        if not passed:
            failures.append(
                {
                    "case_id": case_id,
                    "validation_pool": selection["validation_pool"],
                    "drop_reason": convergence.get("drop_reason", "missing convergence row"),
                }
            )
        if in_envelope:
            envelope_total += 1
            if not passed:
                envelope_failures.append(
                    {
                        "case_id": case_id,
                        "validation_pool": selection["validation_pool"],
                        "drop_reason": convergence.get("drop_reason", "missing convergence row"),
                    }
                )
        else:
            envelope_exclusions.append(
                {
                    "case_id": case_id,
                    "validation_pool": selection["validation_pool"],
                    "target_regime": selection.get("target_regime", ""),
                    "reason": envelope_reason,
                    "drop_reason": convergence.get("drop_reason", ""),
                    "converged": convergence.get("converged", ""),
                }
            )
        ratio = safe_float(convergence.get("raw_enthalpy_balance_ratio"))
        scalar_max = safe_float(convergence.get("scalar_max_delta_T_K"))
        if ratio is not None:
            raw_ratios.append(ratio)
        if scalar_max is not None:
            scalar_maxima.append(scalar_max)

    summary_csv = resolve(paths.root, args.summary_csv)
    summary_json = resolve(paths.root, args.summary_json)
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    write_json(
        summary_json,
        {
            "selected_cases": len(selections),
            "passed_cases": len(selections) - len(failures),
            "failed_cases": len(failures),
            "all_passed": not failures,
            "case_count_by_validation_pool": dict(sorted(by_validation_pool.items())),
            "failures": failures,
            "current_steady_solver_envelope": {
                "selected_cases": envelope_total,
                "passed_cases": envelope_total - len(envelope_failures),
                "failed_cases": len(envelope_failures),
                "all_passed": not envelope_failures,
                "excluded_cases": len(envelope_exclusions),
                "exclusions": envelope_exclusions,
                "failures": envelope_failures,
            },
            "raw_enthalpy_balance_ratio_min": min(raw_ratios) if raw_ratios else None,
            "raw_enthalpy_balance_ratio_max": max(raw_ratios) if raw_ratios else None,
            "scalar_max_delta_T_K_max": max(scalar_maxima) if scalar_maxima else None,
            "selection_file": str(selection_path.relative_to(paths.root)),
            "summary_csv": str(summary_csv.relative_to(paths.root)),
            "convergence_log": str(convergence_path.relative_to(paths.root)),
        },
    )
    print(f"Wrote solver validation summary: {summary_json.relative_to(paths.root)}")
    if failures and args.require_all_pass:
        raise SystemExit(f"{len(failures)} selected validation case(s) failed")


if __name__ == "__main__":
    main()
