#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import (
    PARAMETER_ORDER,
    ensure_evidence_folders,
    lhs_values,
    load_config,
    nearest_id_distance,
    pushed_high_value,
    pushed_value,
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
DISTANCE_FIELDS = [
    "case_id",
    "pool",
    "nearest_id_train_distance",
    "ood_axis",
    "n_fin",
    "h_ch",
    "d_in",
]


def parameter_rows_for_training(paths) -> list[dict[str, str]]:
    rows = read_csv(paths.evidence_pack / "03_data" / "parameters.csv")
    backfill = paths.evidence_pack / "03_data" / "backfill_parameters.csv"
    if backfill.exists():
        rows.extend(read_csv(backfill))
    by_case = {row["case_id"]: row for row in rows}

    index_path = paths.evidence_pack / "04_predictions" / "fixed_grid_index.csv"
    if not index_path.exists():
        return [row for row in rows if row["pool"] == "id_train"]
    train_case_ids = [
        row["case_id"]
        for row in read_csv(index_path)
        if row["pool"] == "id_train"
    ]
    return [by_case[case_id] for case_id in train_case_ids if case_id in by_case]


def h_fin_for(h_ch: float, unit: float, config: dict[str, Any]) -> float:
    params = config["parameters"]
    geometry = config["geometry"]
    h_fin_lo = float(params["h_fin"]["id_min"])
    h_fin_hi = min(
        float(params["h_fin"]["id_max"]),
        float(geometry["max_fin_height_fraction_of_channel"]) * h_ch,
    )
    return h_fin_lo + unit * (h_fin_hi - h_fin_lo)


def ood_axis(row: dict[str, Any], config: dict[str, Any]) -> str:
    params = config["parameters"]
    h_ch = float(row["h_ch"])
    d_in = float(row["d_in"])
    if h_ch > float(params["h_ch"]["id_max"]):
        return "h_ch_high"
    if d_in < float(params["d_in"]["id_min"]):
        return "d_in_low"
    if d_in > float(params["d_in"]["id_max"]):
        return "d_in_high"
    return "unknown"


def generate_rows(config: dict[str, Any], pool: str, count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    params = config["parameters"]
    ood = config["ood"]
    continuous_id = {
        name: lhs_values(rng, count, params[name]["id_min"], params[name]["id_max"])
        for name in PARAMETER_ORDER
        if params[name]["kind"] == "continuous" and name != "h_fin"
    }
    h_fin_units = lhs_values(rng, count, 0.0, 1.0)
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        h_ch = continuous_id["h_ch"][idx]
        d_in = continuous_id["d_in"][idx]
        if idx % 2 == 0:
            h_ch = pushed_high_value(
                rng,
                float(params["h_ch"]["id_min"]),
                float(params["h_ch"]["id_max"]),
                float(ood["push_fraction_min"]),
                float(ood["push_fraction_max"]),
            )
        else:
            d_in = pushed_value(
                rng,
                float(params["d_in"]["id_min"]),
                float(params["d_in"]["id_max"]),
                float(ood["push_fraction_min"]),
                float(ood["push_fraction_max"]),
            )
        rows.append(
            {
                "case_id": f"{pool}_{idx:04d}",
                "pool": pool,
                "h_ch": round(h_ch, 6),
                "n_fin": ood["n_fin_values"][idx % len(ood["n_fin_values"])],
                "h_fin": round(h_fin_for(h_ch, h_fin_units[idx], config), 6),
                "d_in": round(d_in, 6),
                "u_in": round(continuous_id["u_in"][idx], 6),
                "q_w": round(continuous_id["q_w"][idx], 6),
                "geometry_family": "uniform_fins",
                "solver_status": "pending",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-count", type=int, default=225)
    parser.add_argument("--pool", default="ood_fresh")
    parser.add_argument("--seed-offset", type=int, default=9000)
    parser.add_argument("--parameter-file", default="evidence_pack/03_data/fresh_ood_parameters.csv")
    parser.add_argument("--distance-file", default="evidence_pack/03_data/fresh_ood_distance.csv")
    parser.add_argument("--summary-file", default="evidence_pack/03_data/fresh_ood_generation_summary.json")
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)
    seed = int(config["study"]["seed"]) + int(args.seed_offset)
    train_rows = parameter_rows_for_training(paths)
    if not train_rows:
        raise SystemExit("No ID-train rows available for fresh OOD distance calculation.")

    rows = generate_rows(config, args.pool, int(args.candidate_count), seed)
    distance_rows = []
    for row in rows:
        distance_rows.append(
            {
                "case_id": row["case_id"],
                "pool": row["pool"],
                "nearest_id_train_distance": f"{nearest_id_distance(row, train_rows, config):.8f}",
                "ood_axis": ood_axis(row, config),
                "n_fin": row["n_fin"],
                "h_ch": row["h_ch"],
                "d_in": row["d_in"],
            }
        )

    parameter_path = Path(args.parameter_file)
    if not parameter_path.is_absolute():
        parameter_path = paths.root / parameter_path
    distance_path = Path(args.distance_file)
    if not distance_path.is_absolute():
        distance_path = paths.root / distance_path
    summary_path = Path(args.summary_file)
    if not summary_path.is_absolute():
        summary_path = paths.root / summary_path

    write_csv(parameter_path, rows, PARAMETER_FIELDS)
    write_csv(distance_path, distance_rows, DISTANCE_FIELDS)
    write_json(
        summary_path,
        {
            "pool": args.pool,
            "candidate_count": len(rows),
            "seed": seed,
            "distance_reference": "retained ID-train cases from fixed_grid_index.csv when available",
            "distance_reference_case_count": len(train_rows),
            "candidate_counts_by_n_fin": dict(sorted(Counter(str(row["n_fin"]) for row in rows).items())),
            "candidate_counts_by_axis": dict(sorted(Counter(row["ood_axis"] for row in distance_rows).items())),
            "parameter_file": str(parameter_path.relative_to(paths.root)),
            "distance_file": str(distance_path.relative_to(paths.root)),
        },
    )
    print(f"Wrote {len(rows)} fresh OOD candidate rows: {parameter_path.relative_to(paths.root)}")
    print(f"Wrote distance audit: {distance_path.relative_to(paths.root)}")


if __name__ == "__main__":
    main()
