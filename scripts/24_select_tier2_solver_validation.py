#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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
]


def resolve(paths_root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else paths_root / candidate


def rows_by_case(path: Path) -> dict[str, dict[str, str]]:
    return {row["case_id"]: row for row in read_csv(path)}


def base_parameter_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in PARAMETER_FIELDS}


def add_selection(
    selected: list[dict[str, Any]],
    row: dict[str, str],
    validation_pool: str,
    group: str,
    reason: str,
    source_parameter_file: Path,
    paths_root: Path,
    nearest_id_train_distance: str = "",
    target_regime: str = "",
    ood_axis: str = "",
) -> None:
    selected.append(
        {
            **base_parameter_row(row),
            "validation_pool": validation_pool,
            "selection_group": group,
            "selection_reason": reason,
            "source_parameter_file": str(source_parameter_file.relative_to(paths_root)),
            "nearest_id_train_distance": nearest_id_train_distance,
            "target_regime": target_regime,
            "ood_axis": ood_axis,
        }
    )


def require_case(rows: dict[str, dict[str, str]], case_id: str, source: Path) -> dict[str, str]:
    if case_id not in rows:
        raise SystemExit(f"Missing required validation case {case_id} in {source}")
    return rows[case_id]


def distance_extremes(
    distance_rows: list[dict[str, str]],
    group_key: str,
    distance_key: str = "nearest_id_train_distance",
) -> list[tuple[str, str, dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in distance_rows:
        grouped[str(row[group_key])].append(row)
    selected = []
    for group, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: float(item[distance_key]))
        selected.append((group, "low_distance", ordered[0]))
        if ordered[-1]["case_id"] != ordered[0]["case_id"]:
            selected.append((group, "high_distance", ordered[-1]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-file", default="evidence_pack_tier2_steady/03_data/parameters.csv")
    parser.add_argument("--ood-distance-file", default="evidence_pack_tier2_steady/03_data/ood_distance.csv")
    parser.add_argument("--fresh-parameter-file", default="evidence_pack_tier2_steady/03_data/fresh_ood_parameters.csv")
    parser.add_argument("--fresh-distance-file", default="evidence_pack_tier2_steady/03_data/fresh_ood_distance.csv")
    parser.add_argument("--hard-parameter-file", default="evidence_pack_tier2_steady/03_data/hard_regime_parameters.csv")
    parser.add_argument("--hard-distance-file", default="evidence_pack_tier2_steady/03_data/hard_regime_distance.csv")
    parser.add_argument("--output-parameter-file", default="evidence_pack_tier2_steady/03_data/solver_validation_parameters.csv")
    parser.add_argument("--selection-file", default="evidence_pack_tier2_steady/03_data/solver_validation_selection.csv")
    parser.add_argument("--summary-file", default="evidence_pack_tier2_steady/03_data/solver_validation_selection_summary.json")
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)

    parameter_path = resolve(paths.root, args.parameter_file)
    ood_distance_path = resolve(paths.root, args.ood_distance_file)
    fresh_parameter_path = resolve(paths.root, args.fresh_parameter_file)
    fresh_distance_path = resolve(paths.root, args.fresh_distance_file)
    hard_parameter_path = resolve(paths.root, args.hard_parameter_file)
    hard_distance_path = resolve(paths.root, args.hard_distance_file)

    required_paths = [parameter_path, ood_distance_path, fresh_parameter_path, fresh_distance_path, hard_parameter_path, hard_distance_path]
    missing = [str(path.relative_to(paths.root)) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required validation inputs: {', '.join(missing)}")

    base_rows = rows_by_case(parameter_path)
    fresh_rows = rows_by_case(fresh_parameter_path)
    hard_rows = rows_by_case(hard_parameter_path)
    selected: list[dict[str, Any]] = []

    id_cases = [
        ("id_train_0000", "id_smoke", "already-passed smoke baseline"),
        ("id_train_0200", "id_train_mid", "middle ID-train representative"),
        ("id_calibration_0400", "id_calibration_first", "first ID-calibration representative"),
        ("id_test_0500", "id_test_first", "first ID-test representative"),
    ]
    for case_id, group, reason in id_cases:
        add_selection(selected, require_case(base_rows, case_id, parameter_path), "id", group, reason, parameter_path, paths.root)

    ood_distance_rows = []
    for distance_row in read_csv(ood_distance_path):
        case = require_case(base_rows, distance_row["case_id"], parameter_path)
        ood_distance_rows.append({**distance_row, "n_fin": str(int(float(case["n_fin"])))})
    for n_fin, distance_tag, distance_row in distance_extremes(ood_distance_rows, "n_fin"):
        case = require_case(base_rows, distance_row["case_id"], parameter_path)
        add_selection(
            selected,
            case,
            "near_ood",
            f"ood_nfin_{n_fin}_{distance_tag}",
            f"existing OOD n_fin={n_fin} {distance_tag.replace('_', ' ')} representative",
            parameter_path,
            paths.root,
            nearest_id_train_distance=f"{float(distance_row['nearest_id_train_distance']):.8f}",
        )

    fresh_distance_rows = read_csv(fresh_distance_path)
    for axis, distance_tag, distance_row in distance_extremes(fresh_distance_rows, "ood_axis"):
        case = require_case(fresh_rows, distance_row["case_id"], fresh_parameter_path)
        add_selection(
            selected,
            case,
            "fresh_ood",
            f"fresh_{axis}_{distance_tag}",
            f"fresh OOD {axis} {distance_tag.replace('_', ' ')} representative",
            fresh_parameter_path,
            paths.root,
            nearest_id_train_distance=f"{float(distance_row['nearest_id_train_distance']):.8f}",
            ood_axis=axis,
        )

    hard_distance_rows = read_csv(hard_distance_path)
    grouped_hard: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in hard_distance_rows:
        grouped_hard[row["target_regime"]].append(row)
    for regime, rows in sorted(grouped_hard.items()):
        row = max(rows, key=lambda item: float(item["nearest_id_train_distance"]))
        case = require_case(hard_rows, row["case_id"], hard_parameter_path)
        add_selection(
            selected,
            case,
            "hard_regime",
            f"hard_{regime}_high_distance",
            f"hard-regime {regime} highest distance representative",
            hard_parameter_path,
            paths.root,
            nearest_id_train_distance=f"{float(row['nearest_id_train_distance']):.8f}",
            target_regime=regime,
        )

    unique_selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in selected:
        if row["case_id"] in seen:
            continue
        seen.add(row["case_id"])
        unique_selected.append(row)

    output_parameter_path = resolve(paths.root, args.output_parameter_file)
    selection_path = resolve(paths.root, args.selection_file)
    summary_path = resolve(paths.root, args.summary_file)
    parameter_rows = [base_parameter_row(row) for row in unique_selected]

    write_csv(output_parameter_path, parameter_rows, PARAMETER_FIELDS)
    write_csv(selection_path, unique_selected, SELECTION_FIELDS)
    counts: dict[str, int] = defaultdict(int)
    for row in unique_selected:
        counts[row["validation_pool"]] += 1
    write_json(
        summary_path,
        {
            "case_count": len(unique_selected),
            "case_count_by_validation_pool": dict(sorted(counts.items())),
            "case_ids": [row["case_id"] for row in unique_selected],
            "selection_csv": str(selection_path.relative_to(paths.root)),
            "parameter_csv": str(output_parameter_path.relative_to(paths.root)),
            "source_files": {
                "base_parameters": str(parameter_path.relative_to(paths.root)),
                "ood_distance": str(ood_distance_path.relative_to(paths.root)),
                "fresh_parameters": str(fresh_parameter_path.relative_to(paths.root)),
                "fresh_distance": str(fresh_distance_path.relative_to(paths.root)),
                "hard_parameters": str(hard_parameter_path.relative_to(paths.root)),
                "hard_distance": str(hard_distance_path.relative_to(paths.root)),
            },
        },
    )
    print(f"Wrote {len(unique_selected)} validation cases: {output_parameter_path.relative_to(paths.root)}")
    print(f"Wrote selection audit: {selection_path.relative_to(paths.root)}")


if __name__ == "__main__":
    main()
