#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import file_sha256, load_config, read_csv, study_paths, write_csv, write_json


FIELDS = [
    "case_id",
    "pool",
    "npz_path",
    "fluid_pixels",
    "T_min_K",
    "T_p01_K",
    "T_mean_K",
    "T_p99_K",
    "T_max_K",
    "U_mag_min_m_per_s",
    "U_mag_mean_m_per_s",
    "U_mag_max_m_per_s",
    "T_mean_delta_K",
    "T_outlet_bulk_K",
    "T_outlet_bulk_delta_K",
    "enthalpy_expected_delta_K",
    "enthalpy_balance_ratio",
    "enthalpy_balance_passed",
    "passed",
    "failure_reasons",
    "npz_sha256",
]


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def case_json_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, Any] | None:
    if "case_json" not in data:
        return None
    raw = data["case_json"]
    return json.loads(str(raw.item() if raw.shape == () else raw))


def heated_area_m2(case_json: dict[str, Any]) -> float:
    geometry = case_json["geometry"]
    thickness_m = float(geometry["thickness_mm"]) * 1e-3
    wetted_length_m = float(geometry["length_mm"]) * 1e-3
    for fin in geometry.get("fin_boxes", []):
        height_m = (float(fin["y1_mm"]) - float(fin["y0_mm"])) * 1e-3
        width_m = (float(fin["x1_mm"]) - float(fin["x0_mm"])) * 1e-3
        wetted_length_m += 2.0 * height_m + width_m
    return wetted_length_m * thickness_m


def outlet_bulk_temperature(
    t_grid: np.ndarray,
    u_grid: np.ndarray,
    fluid_mask: np.ndarray,
    strip_fraction: float = 0.03,
) -> float:
    if t_grid.shape != u_grid.shape or t_grid.shape != fluid_mask.shape or t_grid.ndim != 2:
        return float("nan")
    n_cols = max(1, int(round(t_grid.shape[1] * strip_fraction)))
    strip = np.zeros_like(fluid_mask, dtype=bool)
    strip[:, -n_cols:] = True
    mask = fluid_mask & strip & np.isfinite(t_grid) & np.isfinite(u_grid)
    if not np.any(mask):
        return float("nan")
    temps = t_grid[mask]
    weights = np.maximum(u_grid[mask], 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1.0e-12:
        return float(np.mean(temps))
    return float(np.average(temps, weights=weights))


def enthalpy_balance(
    case_json: dict[str, Any] | None,
    config: dict[str, Any],
    observed_outlet_delta_K: float,
) -> tuple[float, float, bool | None, str]:
    physics = config["physics"]
    if not bool(physics.get("energy_balance_check_enabled", False)):
        return float("nan"), float("nan"), None, ""
    if case_json is None:
        return float("nan"), float("nan"), False, "missing case_json for outlet enthalpy check"
    if not np.isfinite(observed_outlet_delta_K):
        return float("nan"), float("nan"), False, "nonfinite outlet bulk temperature for outlet enthalpy check"

    params = case_json["parameters"]
    geometry = case_json["geometry"]
    q_w = float(params["q_w"]) * 1000.0
    u_in = float(params["u_in"])
    inlet_width_m = float(geometry["effective_inlet_width_mm"]) * 1e-3
    thickness_m = float(geometry["thickness_mm"]) * 1e-3
    flow_area_m2 = inlet_width_m * thickness_m
    volumetric_flow_m3_s = max(u_in * flow_area_m2, 1.0e-15)
    rho_cp = float(
        physics.get(
            "volumetric_heat_capacity_J_m3K",
            float(physics["thermal_conductivity_W_mK"]) / float(physics["thermal_diffusivity_m2_s"]),
        )
    )
    heat_input_W = q_w * heated_area_m2(case_json)
    expected_delta_K = heat_input_W / max(rho_cp * volumetric_flow_m3_s, 1.0e-15)
    ratio = observed_outlet_delta_K / expected_delta_K if expected_delta_K > 1.0e-12 else float("nan")
    min_ratio = float(physics.get("energy_balance_min_ratio", -0.05))
    max_ratio = float(physics.get("energy_balance_max_ratio", 10.0))
    margin = float(physics.get("energy_balance_abs_margin_K", 0.0))
    lower_bound = min_ratio * expected_delta_K - margin
    upper_bound = max_ratio * expected_delta_K + margin
    passed = lower_bound <= observed_outlet_delta_K <= upper_bound
    reason = "" if passed else (
        f"outlet bulk temperature rise {observed_outlet_delta_K:.6g} K outside broad enthalpy-balance bounds "
        f"[{lower_bound:.6g}, {upper_bound:.6g}] K"
    )
    return expected_delta_K, ratio, passed, reason


def check_case(row: dict[str, str], config: dict[str, Any], root: Path) -> dict[str, Any]:
    physics = config["physics"]
    npz_path = Path(row["npz_path"])
    if not npz_path.is_absolute():
        npz_path = root / npz_path

    reasons: list[str] = []
    with np.load(npz_path) as data:
        targets = data["targets"]
        fluid_mask = data["fluid_mask"].astype(bool)
        case_json = case_json_from_npz(data)

    if targets.shape[0] < 2:
        reasons.append(f"targets has {targets.shape[0]} channels; expected at least 2")

    t_grid = targets[0]
    u_grid = targets[1]
    fluid_t = t_grid[fluid_mask]
    fluid_u = u_grid[fluid_mask]

    if fluid_t.size == 0:
        reasons.append("no fluid pixels")

    if not np.all(np.isfinite(fluid_t)):
        reasons.append("nonfinite temperature values")
    if not np.all(np.isfinite(fluid_u)):
        reasons.append("nonfinite velocity values")

    inlet = float(physics["inlet_temperature_K"])
    min_allowed = max(
        float(physics["temperature_hard_min_K"]),
        inlet - float(physics["max_negative_delta_from_inlet_K"]),
    )
    max_allowed = min(
        float(physics["temperature_hard_max_K"]),
        inlet + float(physics["max_positive_delta_from_inlet_K"]),
    )
    velocity_max = float(physics["max_velocity_mag_m_per_s"])

    if fluid_t.size:
        t_min = float(np.min(fluid_t))
        t_max = float(np.max(fluid_t))
        t_mean = float(np.mean(fluid_t))
        t_mean_delta = t_mean - inlet
        t_outlet_bulk = outlet_bulk_temperature(t_grid, u_grid, fluid_mask)
        t_outlet_bulk_delta = t_outlet_bulk - inlet if np.isfinite(t_outlet_bulk) else float("nan")
        if t_min < min_allowed:
            reasons.append(f"T_min_K {t_min:.6g} below allowed {min_allowed:.6g}")
        if t_max > max_allowed:
            reasons.append(f"T_max_K {t_max:.6g} above allowed {max_allowed:.6g}")
        if t_mean < inlet - float(physics["max_negative_delta_from_inlet_K"]):
            reasons.append("mean fluid temperature is lower than heated-inlet expectation")
    else:
        t_min = t_max = t_mean = t_outlet_bulk = float("nan")
        t_mean_delta = t_outlet_bulk_delta = float("nan")

    if fluid_u.size:
        u_min = float(np.min(fluid_u))
        u_max = float(np.max(fluid_u))
        u_mean = float(np.mean(fluid_u))
        if u_min < -1.0e-7:
            reasons.append(f"U_mag_min_m_per_s {u_min:.6g} is negative")
        if u_max > velocity_max:
            reasons.append(f"U_mag_max_m_per_s {u_max:.6g} above allowed {velocity_max:.6g}")
    else:
        u_min = u_max = u_mean = float("nan")

    expected_delta, enthalpy_ratio, enthalpy_passed, enthalpy_reason = enthalpy_balance(
        case_json,
        config,
        t_outlet_bulk_delta,
    )
    if enthalpy_passed is False:
        reasons.append(enthalpy_reason)

    return {
        "case_id": row["case_id"],
        "pool": row["pool"],
        "npz_path": row["npz_path"],
        "fluid_pixels": int(fluid_t.size),
        "T_min_K": t_min,
        "T_p01_K": percentile(fluid_t, 1),
        "T_mean_K": t_mean,
        "T_p99_K": percentile(fluid_t, 99),
        "T_max_K": t_max,
        "U_mag_min_m_per_s": u_min,
        "U_mag_mean_m_per_s": u_mean,
        "U_mag_max_m_per_s": u_max,
        "T_mean_delta_K": t_mean_delta,
        "T_outlet_bulk_K": t_outlet_bulk,
        "T_outlet_bulk_delta_K": t_outlet_bulk_delta,
        "enthalpy_expected_delta_K": expected_delta,
        "enthalpy_balance_ratio": enthalpy_ratio,
        "enthalpy_balance_passed": "" if enthalpy_passed is None else str(bool(enthalpy_passed)).lower(),
        "passed": str(not reasons).lower(),
        "failure_reasons": "; ".join(reasons),
        "npz_sha256": file_sha256(npz_path),
    }


def summarize(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    index_path: Path,
    output_csv: Path,
    min_cases: int,
) -> dict[str, Any]:
    by_pool: dict[str, dict[str, int]] = {}
    failures = []
    enthalpy_checked = 0
    enthalpy_failed = 0
    for row in rows:
        pool = row["pool"]
        by_pool.setdefault(pool, {"cases": 0, "passed": 0, "failed": 0})
        by_pool[pool]["cases"] += 1
        if row.get("enthalpy_balance_passed") != "":
            enthalpy_checked += 1
            if row.get("enthalpy_balance_passed") != "true":
                enthalpy_failed += 1
        if row["passed"] == "true":
            by_pool[pool]["passed"] += 1
        else:
            by_pool[pool]["failed"] += 1
            failures.append(
                {
                    "case_id": row["case_id"],
                    "pool": row["pool"],
                    "failure_reasons": row["failure_reasons"],
                }
            )

    physics = config["physics"]
    empty_failure = []
    if len(rows) < min_cases:
        empty_failure.append(
            {
                "case_id": "",
                "pool": "",
                "failure_reasons": f"validated case count {len(rows)} below required minimum {min_cases}",
            }
        )

    enthalpy_summary = {
        "enabled": bool(physics.get("energy_balance_check_enabled", False)),
        "checked_case_count": enthalpy_checked,
        "failed_case_count": enthalpy_failed,
        "rule": (
            "Broad outlet-strip bulk enthalpy sanity check using q_w, estimated heated area, inlet flow area, "
            "volumetric heat capacity, and velocity-magnitude-weighted outlet temperature."
        ),
    }

    return {
        "passed": not failures and not empty_failure,
        "case_count": len(rows),
        "failed_case_count": len(failures) + len(empty_failure),
        "by_pool": by_pool,
        "enthalpy_balance": enthalpy_summary,
        "energy_balance": enthalpy_summary,
        "failures": (empty_failure + failures)[:50],
        "index_file": str(index_path.relative_to(ROOT)),
        "by_case_csv": str(output_csv.relative_to(ROOT)),
        "scalar_semantics": physics["scalar_semantics"],
        "inlet_temperature_K": physics["inlet_temperature_K"],
        "temperature_allowed_K": [
            max(
                float(physics["temperature_hard_min_K"]),
                float(physics["inlet_temperature_K"]) - float(physics["max_negative_delta_from_inlet_K"]),
            ),
            min(
                float(physics["temperature_hard_max_K"]),
                float(physics["inlet_temperature_K"]) + float(physics["max_positive_delta_from_inlet_K"]),
            ),
        ],
        "velocity_mag_max_m_per_s": physics["max_velocity_mag_m_per_s"],
        "rule": "Strict verification fails unless every retained fixed-grid case passes these physics checks.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-file", default="evidence_pack/04_predictions/fixed_grid_index.csv")
    parser.add_argument("--summary-json", default="evidence_pack/02_metrics/physical_plausibility_summary.json")
    parser.add_argument("--summary-csv", default="evidence_pack/02_metrics/physical_plausibility_by_case.csv")
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    index_path = Path(args.index_file)
    if not index_path.is_absolute():
        index_path = paths.root / index_path
    output_json = Path(args.summary_json)
    if not output_json.is_absolute():
        output_json = paths.root / output_json
    output_csv = Path(args.summary_csv)
    if not output_csv.is_absolute():
        output_csv = paths.root / output_csv

    rows = [check_case(row, config, paths.root) for row in read_csv(index_path)]
    write_csv(output_csv, rows, FIELDS)
    summary = summarize(rows, config, index_path, output_csv, args.min_cases)
    write_json(output_json, summary)

    if not summary["passed"] and not args.allow_failures:
        print(f"Physical plausibility failed. Report: {output_json}")
        for failure in summary["failures"][:10]:
            print(f"- {failure['case_id']}: {failure['failure_reasons']}")
        raise SystemExit(1)

    print(f"Physical plausibility passed for {summary['case_count']} case(s). Report: {output_json}")


if __name__ == "__main__":
    main()
