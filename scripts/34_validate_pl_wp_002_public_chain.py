#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence_pack_tier2_steady"


REQUIRED_PATHS = [
    EVIDENCE / "03_data" / "physical_acquisition_retained_summary.json",
    EVIDENCE / "04_predictions" / "physical_acquisition_fixed_grid_summary.json",
    EVIDENCE / "02_metrics" / "physical_acquisition_physical_plausibility_summary.json",
    EVIDENCE / "02_metrics" / "tier2_physical_operator" / "run_summary.json",
    EVIDENCE / "02_metrics" / "tier2_physical_operator" / "headlines.json",
    EVIDENCE / "02_metrics" / "tier2_physical_operator" / "ood_summary.csv",
    EVIDENCE / "04_predictions" / "tier2_physical_operator" / "prediction_index.csv",
    EVIDENCE / "02_metrics" / "tier2_gate_diagnostics" / "tier2_gate_diagnostics.json",
    EVIDENCE / "05_figures" / "tier2_physical_operator" / "field_prediction_examples_temperature.png",
    EVIDENCE / "05_figures" / "tier2_physical_operator" / "field_prediction_examples_velocity.png",
    EVIDENCE / "05_figures" / "tier2_physical_operator" / "per_case_coverage_distribution.png",
    EVIDENCE / "05_figures" / "tier2_physical_operator" / "calibration_curve.png",
    EVIDENCE / "05_figures" / "tier2_physical_operator" / "ood_bandwidth_error.png",
    EVIDENCE / "05_figures" / "tier2_gate_diagnostics" / "band_width_mean_norm_90_distribution.png",
    EVIDENCE / "05_figures" / "tier2_gate_diagnostics" / "risk_multiplier_distribution.png",
    EVIDENCE / "05_figures" / "tier2_gate_diagnostics" / "nearest_id_train_distance_distribution.png",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fail(message: str) -> None:
    raise SystemExit(f"PL-WP-002 public-chain validation failed: {message}")


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        fail("missing required paths:\n" + "\n".join(str(path.relative_to(ROOT)) for path in missing))

    retained = read_json(EVIDENCE / "03_data" / "physical_acquisition_retained_summary.json")
    fixed_grid = read_json(EVIDENCE / "04_predictions" / "physical_acquisition_fixed_grid_summary.json")
    plausibility = read_json(EVIDENCE / "02_metrics" / "physical_acquisition_physical_plausibility_summary.json")
    headlines = read_json(EVIDENCE / "02_metrics" / "tier2_physical_operator" / "headlines.json")
    gate = read_json(EVIDENCE / "02_metrics" / "tier2_gate_diagnostics" / "tier2_gate_diagnostics.json")
    predictions = read_csv(EVIDENCE / "04_predictions" / "tier2_physical_operator" / "prediction_index.csv")
    ood_summary = read_csv(EVIDENCE / "02_metrics" / "tier2_physical_operator" / "ood_summary.csv")

    retained_count = int(retained["retained_case_count"])
    fixed_grid_count = int(fixed_grid["case_count"])
    plausibility_count = int(plausibility["case_count"])
    if retained_count != fixed_grid_count or retained_count != plausibility_count:
        fail(
            "retained, fixed-grid, and plausibility case counts disagree: "
            f"{retained_count}, {fixed_grid_count}, {plausibility_count}"
        )
    if int(plausibility["failed_case_count"]) != 0:
        fail("retained fixed-grid physical plausibility has failed cases")

    pools = {row["pool"] for row in predictions}
    required_pools = {"id_test", "ood_test", "ood_fresh"}
    if not required_pools.issubset(pools):
        fail(f"prediction index missing required pools: {sorted(required_pools - pools)}")

    explicit = next((row for row in gate["policies"] if row["policy"] == "explicit_envelope"), None)
    if explicit is None:
        fail("gate diagnostics missing explicit envelope policy")
    if abs(float(explicit["served_coverage_90"]) - float(headlines["id_coverage_90"])) > 1e-12:
        fail("explicit-envelope served coverage does not match headline ID coverage")

    nominal_90 = next((row for row in ood_summary if row["nominal_coverage"] == "0.9"), None)
    if nominal_90 is None:
        fail("OOD summary missing nominal 0.9 row")
    if abs(float(nominal_90["empirical_coverage_id_test"]) - float(headlines["id_coverage_90"])) > 1e-12:
        fail("OOD summary ID coverage does not match headline ID coverage")

    print("PL-WP-002 public-chain validation passed.")
    print(f"retained cases: {retained_count}")
    print(f"ID 90% aggregate coverage: {float(headlines['id_coverage_90']):.6f}")
    print(f"explicit-envelope served coverage: {float(explicit['served_coverage_90']):.6f}")


if __name__ == "__main__":
    main()
