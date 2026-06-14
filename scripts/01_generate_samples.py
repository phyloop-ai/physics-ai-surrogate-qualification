#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import (
    PARAMETER_ORDER,
    ensure_evidence_folders,
    generate_parameter_tables,
    load_config,
    study_paths,
    write_csv,
    write_json,
)


def main() -> None:
    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)

    parameter_rows, distance_rows = generate_parameter_tables(config)
    parameter_fields = [
        "case_id",
        "pool",
        *PARAMETER_ORDER,
        "geometry_family",
        "solver_status",
    ]
    parameter_path = paths.evidence_pack / "03_data" / "parameters.csv"
    distance_path = paths.evidence_pack / "03_data" / "ood_distance.csv"
    write_csv(parameter_path, parameter_rows, parameter_fields)
    write_csv(distance_path, distance_rows, ["case_id", "pool", "nearest_id_train_distance"])

    counts = {}
    for row in parameter_rows:
        counts[row["pool"]] = counts.get(row["pool"], 0) + 1
    write_json(
        paths.evidence_pack / "03_data" / "sample_generation_summary.json",
        {
            "seed": config["study"]["seed"],
            "counts": counts,
            "parameter_file": str(parameter_path.relative_to(paths.root)),
            "ood_distance_file": str(distance_path.relative_to(paths.root)),
            "note": "These are solver-case parameters, not solver results.",
        },
    )
    print(f"Wrote {len(parameter_rows)} parameter rows and {len(distance_rows)} OOD-distance rows.")


if __name__ == "__main__":
    main()
