#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import (
    PARAMETER_ORDER,
    ensure_evidence_folders,
    file_sha256,
    load_config,
    read_csv,
    study_paths,
    write_csv,
    write_json,
)


PARAMETER_FIELDS = [
    "case_id",
    "pool",
    *PARAMETER_ORDER,
    "geometry_family",
    "solver_status",
]

STATUS_FIELDS = [
    "combined_position",
    "case_id",
    "pool",
    "selection_status",
    "converged",
    "dropped",
    "has_archive_manifest_row",
    "archive_exists",
    "archive_path",
    "drop_reason",
    "flow_converged_reported",
    "raw_enthalpy_passed",
    "raw_temperature_extrema_passed",
    "raw_temperature_max_T_K",
    "wall_time_sec",
]

RETAINED_MANIFEST_FIELDS = [
    *PARAMETER_FIELDS,
    "combined_position",
    "archive_path",
    "archive_sha256",
    "archive_size_bytes",
    "flow_converged_reported",
    "raw_enthalpy_passed",
    "raw_enthalpy_balance_ratio",
    "raw_temperature_extrema_passed",
    "raw_temperature_max_T_K",
    "wall_time_sec",
]


def resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["case_id"]: row for row in rows}


def bool_text(row: dict[str, str] | None, key: str) -> str:
    if not row:
        return ""
    value = row.get(key, "")
    return value if value in {"true", "false"} else ""


def selection_status(
    parameter_row: dict[str, str],
    convergence_row: dict[str, str] | None,
    archive_row: dict[str, str] | None,
    root: Path,
) -> tuple[str, bool, str]:
    if not convergence_row:
        return "unattempted", False, ""
    archive_path = archive_row.get("archive_path", "") if archive_row else ""
    archive_exists = bool(archive_path and (root / archive_path).exists())
    if convergence_row.get("converged") == "true" and convergence_row.get("dropped") != "true":
        if archive_exists:
            return "retained", True, archive_path
        return "converged_missing_archive", False, archive_path
    if convergence_row.get("dropped") == "true" or convergence_row.get("converged") == "false":
        return "dropped", False, archive_path
    return "attempted_unknown", False, archive_path


def build_final_manifests(
    parameter_rows: list[dict[str, str]],
    convergence_rows: list[dict[str, str]],
    archive_rows: list[dict[str, str]],
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    convergence_by_case = by_case(convergence_rows)
    archive_by_case = by_case(archive_rows)
    retained_parameters: list[dict[str, Any]] = []
    retained_manifest: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for position, parameter_row in enumerate(parameter_rows, start=1):
        case_id = parameter_row["case_id"]
        convergence_row = convergence_by_case.get(case_id)
        archive_row = archive_by_case.get(case_id)
        status, archive_exists, archive_path = selection_status(parameter_row, convergence_row, archive_row, root)
        status_rows.append(
            {
                "combined_position": position,
                "case_id": case_id,
                "pool": parameter_row["pool"],
                "selection_status": status,
                "converged": bool_text(convergence_row, "converged"),
                "dropped": bool_text(convergence_row, "dropped"),
                "has_archive_manifest_row": str(bool(archive_row)).lower(),
                "archive_exists": str(archive_exists).lower(),
                "archive_path": archive_path,
                "drop_reason": convergence_row.get("drop_reason", "") if convergence_row else "",
                "flow_converged_reported": bool_text(convergence_row, "flow_converged_reported"),
                "raw_enthalpy_passed": bool_text(convergence_row, "raw_enthalpy_passed"),
                "raw_temperature_extrema_passed": bool_text(convergence_row, "raw_temperature_extrema_passed"),
                "raw_temperature_max_T_K": convergence_row.get("raw_temperature_max_T_K", "") if convergence_row else "",
                "wall_time_sec": convergence_row.get("wall_time_sec", "") if convergence_row else "",
            }
        )
        if status != "retained":
            continue

        archive_abs = root / archive_path
        retained_parameters.append({field: parameter_row.get(field, "") for field in PARAMETER_FIELDS})
        retained_manifest.append(
            {
                **{field: parameter_row.get(field, "") for field in PARAMETER_FIELDS},
                "combined_position": position,
                "archive_path": archive_path,
                "archive_sha256": file_sha256(archive_abs),
                "archive_size_bytes": archive_row.get("archive_size_bytes", "") if archive_row else "",
                "flow_converged_reported": convergence_row.get("flow_converged_reported", "") if convergence_row else "",
                "raw_enthalpy_passed": convergence_row.get("raw_enthalpy_passed", "") if convergence_row else "",
                "raw_enthalpy_balance_ratio": convergence_row.get("raw_enthalpy_balance_ratio", "") if convergence_row else "",
                "raw_temperature_extrema_passed": convergence_row.get("raw_temperature_extrema_passed", "") if convergence_row else "",
                "raw_temperature_max_T_K": convergence_row.get("raw_temperature_max_T_K", "") if convergence_row else "",
                "wall_time_sec": convergence_row.get("wall_time_sec", "") if convergence_row else "",
            }
        )

    return retained_parameters, retained_manifest, status_rows


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def count_by_pool_and_status(status_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = {}
    for row in status_rows:
        pool = str(row["pool"])
        result.setdefault(pool, Counter())[str(row["selection_status"])] += 1
    return {pool: dict(sorted(counter.items())) for pool, counter in sorted(result.items())}


def canonical_target_failures(config: dict[str, Any], retained_by_pool: Counter[str]) -> list[str]:
    dataset = config.get("dataset", {})
    failures = []
    for pool in ("id_train", "id_calibration", "id_test", "ood_test"):
        target = int(dataset.get(pool, 0))
        actual = int(retained_by_pool.get(pool, 0))
        if actual < target:
            failures.append(f"{pool} retained {actual} below configured target {target}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parameter-file",
        default="evidence_pack_tier2_steady/03_data/physical_acquisition_with_backfill_parameters.csv",
    )
    parser.add_argument("--convergence-log", default="evidence_pack_tier2_steady/03_data/convergence_log.csv")
    parser.add_argument("--solver-output-manifest", default="evidence_pack_tier2_steady/03_data/solver_output_manifest.csv")
    parser.add_argument(
        "--output-retained-parameters",
        default="evidence_pack_tier2_steady/03_data/physical_acquisition_retained_parameters.csv",
    )
    parser.add_argument(
        "--output-retained-manifest",
        default="evidence_pack_tier2_steady/03_data/physical_acquisition_retained_manifest.csv",
    )
    parser.add_argument(
        "--output-status",
        default="evidence_pack_tier2_steady/03_data/physical_acquisition_final_status.csv",
    )
    parser.add_argument(
        "--summary-file",
        default="evidence_pack_tier2_steady/03_data/physical_acquisition_retained_summary.json",
    )
    parser.add_argument("--min-retained-cases", type=int, default=0)
    parser.add_argument("--fresh-retained-target", type=int, default=90)
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)

    parameter_path = resolve(paths.root, args.parameter_file)
    convergence_path = resolve(paths.root, args.convergence_log)
    archive_manifest_path = resolve(paths.root, args.solver_output_manifest)
    retained_parameter_path = resolve(paths.root, args.output_retained_parameters)
    retained_manifest_path = resolve(paths.root, args.output_retained_manifest)
    status_path = resolve(paths.root, args.output_status)
    summary_path = resolve(paths.root, args.summary_file)

    parameter_rows = read_csv(parameter_path)
    convergence_rows = read_csv(convergence_path)
    archive_rows = read_csv(archive_manifest_path)

    retained_parameters, retained_manifest, status_rows = build_final_manifests(
        parameter_rows,
        convergence_rows,
        archive_rows,
        paths.root,
    )

    write_csv(retained_parameter_path, retained_parameters, PARAMETER_FIELDS)
    write_csv(retained_manifest_path, retained_manifest, RETAINED_MANIFEST_FIELDS)
    write_csv(status_path, status_rows, STATUS_FIELDS)

    retained_by_pool = Counter(row["pool"] for row in retained_parameters)
    status_by_status = Counter(row["selection_status"] for row in status_rows)
    drop_reasons = Counter(
        row["drop_reason"]
        for row in status_rows
        if row["selection_status"] == "dropped" and row.get("drop_reason")
    )
    failures = canonical_target_failures(config, retained_by_pool)
    if args.min_retained_cases and len(retained_parameters) < args.min_retained_cases:
        failures.append(
            f"total retained {len(retained_parameters)} below required minimum {args.min_retained_cases}"
        )

    summary = {
        "input_parameter_file": str(parameter_path.relative_to(paths.root)),
        "input_parameter_sha256": file_sha256(parameter_path),
        "convergence_log": str(convergence_path.relative_to(paths.root)),
        "convergence_log_sha256": file_sha256(convergence_path),
        "solver_output_manifest": str(archive_manifest_path.relative_to(paths.root)),
        "solver_output_manifest_sha256": file_sha256(archive_manifest_path),
        "retained_parameter_file": str(retained_parameter_path.relative_to(paths.root)),
        "retained_manifest_file": str(retained_manifest_path.relative_to(paths.root)),
        "status_file": str(status_path.relative_to(paths.root)),
        "policy": (
            "Retain every row present in the combined physical acquisition/backfill parameter file only when "
            "convergence_log marks it converged, it is not dropped, and the solver output archive is present. "
            "Do not cap ID-train rows; report fresh-OOD retained count honestly."
        ),
        "combined_case_count": len(parameter_rows),
        "retained_case_count": len(retained_parameters),
        "retained_by_pool": dict(sorted(retained_by_pool.items())),
        "status_counts": dict(sorted(status_by_status.items())),
        "status_by_pool": count_by_pool_and_status(status_rows),
        "drop_reasons": dict(sorted(drop_reasons.items())),
        "configured_canonical_targets": {
            pool: int(config.get("dataset", {}).get(pool, 0))
            for pool in ("id_train", "id_calibration", "id_test", "ood_test")
        },
        "fresh_retained_target_reported": int(args.fresh_retained_target),
        "fresh_retained_actual": int(retained_by_pool.get("ood_fresh", 0)),
        "min_retained_cases": int(args.min_retained_cases),
        "passed": not failures,
        "failures": failures,
    }
    write_json(summary_path, summary)

    if failures:
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print(f"Wrote {retained_parameter_path.relative_to(paths.root)} ({len(retained_parameters)} retained case(s))")
    print(f"Wrote {retained_manifest_path.relative_to(paths.root)}")
    print(f"Wrote {status_path.relative_to(paths.root)}")
    print(f"Wrote {summary_path.relative_to(paths.root)}")


if __name__ == "__main__":
    main()
