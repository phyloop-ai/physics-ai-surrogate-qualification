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
    "target_regime",
    "nearest_id_train_distance",
    "generation_note",
]
DISTANCE_FIELDS = [
    "case_id",
    "pool",
    "target_regime",
    "nearest_id_train_distance",
    "n_fin",
    "h_ch",
    "d_in",
]


REGIMES = (
    "nfin3_id_axes",
    "d_in_high_id_fin",
    "h_ch_high_id_fin",
    "nfin3_d_in_high",
    "nfin3_h_ch_high",
)


def training_reference_rows(paths) -> list[dict[str, str]]:
    rows = read_csv(paths.evidence_pack / "03_data" / "parameters.csv")
    backfill = paths.evidence_pack / "03_data" / "backfill_parameters.csv"
    if backfill.exists():
        rows.extend(read_csv(backfill))
    by_case = {row["case_id"]: row for row in rows}
    index_path = paths.evidence_pack / "04_predictions" / "fixed_grid_index.csv"
    if not index_path.exists():
        return [row for row in rows if row["pool"] == "id_train"]
    return [
        by_case[row["case_id"]]
        for row in read_csv(index_path)
        if row["pool"] == "id_train" and row["case_id"] in by_case
    ]


def h_fin_for(h_ch: float, unit: float, config: dict[str, Any]) -> float:
    params = config["parameters"]
    geometry = config["geometry"]
    lo = float(params["h_fin"]["id_min"])
    hi = min(float(params["h_fin"]["id_max"]), float(geometry["max_fin_height_fraction_of_channel"]) * h_ch)
    return lo + unit * (hi - lo)


def row_for_regime(
    case_id: str,
    regime: str,
    idx: int,
    rng: random.Random,
    config: dict[str, Any],
    samples: dict[str, list[float]],
) -> dict[str, Any]:
    params = config["parameters"]
    ood = config["ood"]
    h_ch = samples["h_ch"][idx]
    d_in = samples["d_in"][idx]
    id_fin_values = [int(value) for value in params["n_fin"]["id_values"]]
    n_fin = id_fin_values[idx % len(id_fin_values)]

    if "nfin3" in regime:
        n_fin = 3
    if "d_in_high" in regime:
        d_in = pushed_high_value(
            rng,
            float(params["d_in"]["id_min"]),
            float(params["d_in"]["id_max"]),
            float(ood["push_fraction_min"]),
            float(ood["push_fraction_max"]),
        )
    if "h_ch_high" in regime:
        h_ch = pushed_high_value(
            rng,
            float(params["h_ch"]["id_min"]),
            float(params["h_ch"]["id_max"]),
            float(ood["push_fraction_min"]),
            float(ood["push_fraction_max"]),
        )

    return {
        "case_id": case_id,
        "pool": "hard_regime",
        "h_ch": round(h_ch, 6),
        "n_fin": n_fin,
        "h_fin": round(h_fin_for(h_ch, samples["h_fin_unit"][idx], config), 6),
        "d_in": round(d_in, 6),
        "u_in": round(samples["u_in"][idx], 6),
        "q_w": round(samples["q_w"][idx], 6),
        "geometry_family": "uniform_fins",
        "solver_status": "pending",
        "target_regime": regime,
        "nearest_id_train_distance": "",
        "generation_note": "Targeted active-learning candidate for hard OOD regime observed in fresh holdout.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-regime", type=int, default=30)
    parser.add_argument("--seed-offset", type=int, default=12000)
    parser.add_argument("--parameter-file", default="evidence_pack/03_data/hard_regime_parameters.csv")
    parser.add_argument("--distance-file", default="evidence_pack/03_data/hard_regime_distance.csv")
    parser.add_argument("--summary-file", default="evidence_pack/03_data/hard_regime_generation_summary.json")
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)
    train_rows = training_reference_rows(paths)
    if not train_rows:
        raise SystemExit("No ID-train reference rows available.")

    seed = int(config["study"]["seed"]) + int(args.seed_offset)
    rows = []
    for regime_index, regime in enumerate(REGIMES):
        rng = random.Random(seed + 101 * regime_index)
        samples = {
            "h_ch": lhs_values(rng, args.per_regime, float(config["parameters"]["h_ch"]["id_min"]), float(config["parameters"]["h_ch"]["id_max"])),
            "d_in": lhs_values(rng, args.per_regime, float(config["parameters"]["d_in"]["id_min"]), float(config["parameters"]["d_in"]["id_max"])),
            "u_in": lhs_values(rng, args.per_regime, float(config["parameters"]["u_in"]["id_min"]), float(config["parameters"]["u_in"]["id_max"])),
            "q_w": lhs_values(rng, args.per_regime, float(config["parameters"]["q_w"]["id_min"]), float(config["parameters"]["q_w"]["id_max"])),
            "h_fin_unit": lhs_values(rng, args.per_regime, 0.0, 1.0),
        }
        for idx in range(args.per_regime):
            short = regime.replace("_", "")
            case_id = f"hard_{short}_{idx:04d}"
            row = row_for_regime(case_id, regime, idx, rng, config, samples)
            row["nearest_id_train_distance"] = f"{nearest_id_distance(row, train_rows, config):.8f}"
            rows.append(row)

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
    write_csv(
        distance_path,
        [
            {
                "case_id": row["case_id"],
                "pool": row["pool"],
                "target_regime": row["target_regime"],
                "nearest_id_train_distance": row["nearest_id_train_distance"],
                "n_fin": row["n_fin"],
                "h_ch": row["h_ch"],
                "d_in": row["d_in"],
            }
            for row in rows
        ],
        DISTANCE_FIELDS,
    )
    write_json(
        summary_path,
        {
            "seed": seed,
            "per_regime": int(args.per_regime),
            "case_count": len(rows),
            "target_regimes": list(REGIMES),
            "case_count_by_regime": dict(sorted(Counter(row["target_regime"] for row in rows).items())),
            "policy": "Generate targeted OpenFOAM candidates for hard regimes before any retraining. Keep separate from canonical evidence.",
            "parameter_file": str(parameter_path.relative_to(paths.root)),
            "distance_file": str(distance_path.relative_to(paths.root)),
        },
    )
    print(f"Wrote {len(rows)} hard-regime candidate rows: {parameter_path.relative_to(paths.root)}")
    print(f"Wrote distance audit: {distance_path.relative_to(paths.root)}")


if __name__ == "__main__":
    main()
