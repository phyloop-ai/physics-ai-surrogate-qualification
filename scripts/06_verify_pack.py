#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import (
    EVIDENCE_FOLDERS,
    file_sha256,
    load_config,
    read_csv,
    read_json,
    study_paths,
    write_json,
)


REQUIRED_HEADLINES = {
    "id_coverage_90",
    "id_sharpness_ratio_90",
    "ood_coverage_90",
    "ood_band_width_inflation_ratio_90",
    "ood_spearman_bandwidth_error",
    "gate_threshold_tau",
    "gate_id_served_at_coverage90",
    "gate_ood_routed_at_threshold",
}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def verify_structure(config: dict, paths) -> tuple[list[str], dict]:
    errors: list[str] = []
    report: dict = {"checks": {}}

    physics = config.get("physics", {})
    report["checks"]["physics_contract_present"] = bool(physics)
    for key in (
        "scalar_semantics",
        "inlet_temperature_K",
        "thermal_conductivity_W_mK",
        "fixed_gradient_sign",
        "temperature_hard_min_K",
        "temperature_hard_max_K",
    ):
        if key not in physics:
            fail(errors, f"Missing physics config key: physics.{key}")
    allowed_scalar_semantics = {
        "temperature_K",
        "temperature_K_transient_snapshot_t0.05s",
        "temperature_K_steady_transport_stability_window",
    }
    if physics.get("scalar_semantics") not in allowed_scalar_semantics:
        fail(
            errors,
            "physics.scalar_semantics must be temperature_K or temperature_K_transient_snapshot_t0.05s for this study.",
        )
    if physics.get("fixed_gradient_sign") != "positive_q_over_k":
        fail(errors, "physics.fixed_gradient_sign must document the corrected heat-flux sign convention.")

    for folder in EVIDENCE_FOLDERS:
        exists = (paths.evidence_pack / folder).is_dir()
        report["checks"][f"folder:{folder}"] = exists
        if not exists:
            fail(errors, f"Missing evidence folder: {folder}")

    parameter_path = paths.evidence_pack / "03_data" / "parameters.csv"
    distance_path = paths.evidence_pack / "03_data" / "ood_distance.csv"

    if parameter_path.exists():
        rows = read_csv(parameter_path)
        seen_case_ids: set[str] = set()
        duplicate_case_ids: set[str] = set()
        expected_counts = {
            "id_train": config["dataset"]["id_train"],
            "id_calibration": config["dataset"]["id_calibration"],
            "id_test": config["dataset"]["id_test"],
            "ood_test": config["dataset"]["ood_test"],
        }
        actual_counts: dict[str, int] = {}
        clearance_violations = []
        max_fin_fraction = float(config["geometry"]["max_fin_height_fraction_of_channel"])
        for row in rows:
            case_id = row["case_id"]
            if case_id in seen_case_ids:
                duplicate_case_ids.add(case_id)
            seen_case_ids.add(case_id)
            actual_counts[row["pool"]] = actual_counts.get(row["pool"], 0) + 1
            if float(row["h_fin"]) > max_fin_fraction * float(row["h_ch"]) + 1e-9:
                clearance_violations.append(case_id)
        report["checks"]["parameter_counts"] = actual_counts
        report["checks"]["duplicate_case_ids"] = sorted(duplicate_case_ids)
        report["checks"]["geometry_clearance_violations"] = clearance_violations[:20]
        if duplicate_case_ids:
            fail(errors, f"Duplicate case IDs: {', '.join(sorted(duplicate_case_ids)[:10])}")
        if clearance_violations:
            fail(errors, f"{len(clearance_violations)} rows violate the configured fin-clearance rule.")
        for pool, expected in expected_counts.items():
            actual = actual_counts.get(pool, 0)
            if actual != expected:
                fail(errors, f"Pool {pool} has {actual} rows; expected {expected}.")
        report["parameters_sha256"] = file_sha256(parameter_path)
    else:
        fail(errors, "Missing 03_data/parameters.csv")

    if distance_path.exists():
        distance_rows = read_csv(distance_path)
        report["checks"]["ood_distance_rows"] = len(distance_rows)
        if len(distance_rows) != config["dataset"]["ood_test"]:
            fail(errors, f"OOD distance table has {len(distance_rows)} rows; expected {config['dataset']['ood_test']}.")
        report["ood_distance_sha256"] = file_sha256(distance_path)
    else:
        fail(errors, "Missing 03_data/ood_distance.csv")

    manifest_path = paths.evidence_pack / "01_config" / "environment_manifest.json"
    report["checks"]["environment_manifest_exists"] = manifest_path.exists()
    if manifest_path.exists():
        report["environment_manifest_sha256"] = file_sha256(manifest_path)

    return errors, report


def verify_strict(config: dict, paths) -> tuple[list[str], dict]:
    errors, report = verify_structure(config, paths)

    required_files = [
        paths.evidence_pack / "02_metrics" / "run_summary.json",
        paths.evidence_pack / "02_metrics" / "headlines.json",
        paths.evidence_pack / "02_metrics" / "calibration_curve.csv",
        paths.evidence_pack / "02_metrics" / "ood_summary.csv",
        paths.evidence_pack / "02_metrics" / "risk_coverage_curve_id.csv",
        paths.evidence_pack / "02_metrics" / "risk_coverage_curve_mixed.csv",
        paths.evidence_pack / "02_metrics" / "gate_diagnostics.json",
        paths.evidence_pack / "02_metrics" / "gate_score_distribution_summary.csv",
        paths.evidence_pack / "02_metrics" / "regime_gate_decisions.csv",
        paths.evidence_pack / "02_metrics" / "regime_gate_summary.csv",
        paths.evidence_pack / "02_metrics" / "regime_gate_summary.json",
        paths.evidence_pack / "02_metrics" / "solver_execution_summary.csv",
        paths.evidence_pack / "02_metrics" / "solver_execution_summary.json",
        paths.evidence_pack / "02_metrics" / "inference_runtime_summary.json",
        paths.evidence_pack / "02_metrics" / "physical_plausibility_summary.json",
        paths.evidence_pack / "02_metrics" / "physical_plausibility_by_case.csv",
        paths.evidence_pack / "03_data" / "convergence_log.csv",
        paths.evidence_pack / "04_predictions" / "fixed_grid_index.csv",
        paths.evidence_pack / "04_predictions" / "prediction_index.csv",
        paths.evidence_pack / "05_figures" / "field_prediction_examples_temperature.png",
        paths.evidence_pack / "05_figures" / "field_prediction_examples_velocity.png",
        paths.evidence_pack / "05_figures" / "per_case_coverage_distribution.png",
        paths.evidence_pack / "05_figures" / "gate_score_distributions.png",
        paths.evidence_pack / "05_figures" / "gate_score_ecdf.png",
        paths.evidence_pack / "05_figures" / "regime_gate_comparison.png",
    ]
    hard_batch_path = paths.evidence_pack / "03_data" / "hard_regime_batch_status.json"
    if hard_batch_path.exists():
        required_files.extend(
            [
                paths.evidence_pack / "02_metrics" / "hard_regime_solver_summary.csv",
                paths.evidence_pack / "02_metrics" / "hard_regime_solver_summary.json",
                paths.evidence_pack / "03_data" / "hard_regime_solver_decisions.csv",
                paths.evidence_pack / "04_predictions" / "hard_regime_fixed_grid_index.csv",
                paths.evidence_pack / "04_predictions" / "hard_regime_fixed_grid_summary.json",
                paths.evidence_pack / "05_figures" / "hard_regime_solver_retention.png",
                paths.evidence_pack / "05_figures" / "hard_regime_drop_bias_distance.png",
            ]
        )
    hard_branch_checkpoint = paths.evidence_pack / "04_predictions" / "hard_regime_branch" / "model" / "compact_fno.pt"
    if hard_branch_checkpoint.exists():
        required_files.extend(
            [
                hard_branch_checkpoint,
                paths.evidence_pack / "02_metrics" / "hard_regime_branch" / "run_summary.json",
                paths.evidence_pack / "02_metrics" / "hard_regime_branch" / "headlines.json",
                paths.evidence_pack / "02_metrics" / "hard_regime_branch" / "calibration_curve.csv",
                paths.evidence_pack / "02_metrics" / "hard_regime_branch" / "ood_summary.csv",
                paths.evidence_pack / "02_metrics" / "hard_regime_branch" / "risk_coverage_curve_id.csv",
                paths.evidence_pack / "02_metrics" / "hard_regime_branch" / "risk_coverage_curve_mixed.csv",
                paths.evidence_pack / "02_metrics" / "hard_regime_branch" / "risk_head_summary.json",
                paths.evidence_pack / "02_metrics" / "hard_regime_branch" / "fresh_ood_summary.csv",
                paths.evidence_pack / "02_metrics" / "hard_regime_branch" / "fresh_ood_headlines.json",
                paths.evidence_pack / "02_metrics" / "hard_regime_branch" / "branch_comparison_summary.csv",
                paths.evidence_pack / "02_metrics" / "hard_regime_branch" / "branch_comparison_summary.json",
                paths.evidence_pack / "04_predictions" / "hard_regime_branch" / "prediction_index.csv",
                paths.evidence_pack / "04_predictions" / "hard_regime_branch" / "fresh_ood_prediction_index.csv",
                paths.evidence_pack / "05_figures" / "hard_regime_branch" / "calibration_curve.png",
                paths.evidence_pack / "05_figures" / "hard_regime_branch" / "risk_coverage_curve_id.png",
                paths.evidence_pack / "05_figures" / "hard_regime_branch" / "risk_coverage_curve_mixed.png",
                paths.evidence_pack / "05_figures" / "hard_regime_branch" / "ood_bandwidth_error.png",
                paths.evidence_pack / "05_figures" / "hard_regime_branch" / "fresh_ood_bandwidth_error.png",
                paths.evidence_pack / "05_figures" / "hard_regime_branch" / "fresh_ood_drop_distance.png",
                paths.evidence_pack / "05_figures" / "hard_regime_branch" / "branch_comparison.png",
            ]
        )
    for path in required_files:
        exists = path.exists()
        report["checks"][f"required_file:{path.relative_to(paths.evidence_pack)}"] = exists
        if not exists:
            fail(errors, f"Missing required final artifact: {path.relative_to(paths.evidence_pack)}")

    headlines_path = paths.evidence_pack / "02_metrics" / "headlines.json"
    if headlines_path.exists():
        headlines = read_json(headlines_path)
        missing = sorted(REQUIRED_HEADLINES - set(headlines))
        report["checks"]["headline_keys"] = sorted(headlines)
        if missing:
            fail(errors, f"Missing headline keys: {', '.join(missing)}")

        gates = config["acceptance_gates"]
        if "id_coverage_90" in headlines:
            coverage = float(headlines["id_coverage_90"])
            if not (gates["id_coverage90_min"] <= coverage <= gates["id_coverage90_max"]):
                fail(errors, f"id_coverage_90={coverage} outside gate.")
        if "id_sharpness_ratio_90" in headlines:
            sharpness = float(headlines["id_sharpness_ratio_90"])
            if sharpness > gates["sharpness_ratio_max"]:
                fail(errors, f"id_sharpness_ratio_90={sharpness} exceeds gate.")
        if "ood_spearman_bandwidth_error" in headlines:
            spearman = float(headlines["ood_spearman_bandwidth_error"])
            if spearman < gates["ood_spearman_target"]:
                fail(errors, f"ood_spearman_bandwidth_error={spearman} below target.")

        report["headlines_sha256"] = file_sha256(headlines_path)

    gate_diagnostics_path = paths.evidence_pack / "02_metrics" / "gate_diagnostics.json"
    if gate_diagnostics_path.exists():
        gate_diagnostics = read_json(gate_diagnostics_path)
        id_only = gate_diagnostics.get("id_only", {})
        mixed_original = gate_diagnostics.get("mixed_original", {})
        report["gate_diagnostics"] = {
            "target_accepted_coverage": gate_diagnostics.get("target_accepted_coverage"),
            "id_only_has_target_coverage_point": id_only.get("has_target_coverage_point"),
            "id_only_best_at_target": id_only.get("best_at_target"),
            "mixed_has_target_coverage_point": mixed_original.get("has_target_coverage_point"),
            "mixed_best_at_target": mixed_original.get("best_at_target"),
            "mixed_best_available": mixed_original.get("best_available"),
            "regime_aware_prototype": gate_diagnostics.get("regime_aware_prototype"),
        }
        report["gate_diagnostics_sha256"] = file_sha256(gate_diagnostics_path)

    regime_gate_path = paths.evidence_pack / "02_metrics" / "regime_gate_summary.json"
    if regime_gate_path.exists():
        regime_gate = read_json(regime_gate_path)
        report["regime_gate"] = {
            "policy": regime_gate.get("policy"),
            "id_threshold_tau": regime_gate.get("id_threshold_tau"),
            "summaries": regime_gate.get("summaries"),
            "caveat": regime_gate.get("caveat"),
        }
        report["regime_gate_sha256"] = file_sha256(regime_gate_path)

    solver_execution_path = paths.evidence_pack / "02_metrics" / "solver_execution_summary.json"
    if solver_execution_path.exists():
        solver_execution = read_json(solver_execution_path)
        report["solver_execution_summary"] = {
            "solver": solver_execution.get("solver"),
            "environment": solver_execution.get("environment"),
            "summary_by_pool": solver_execution.get("summary_by_pool"),
        }
        report["solver_execution_summary_sha256"] = file_sha256(solver_execution_path)

    inference_runtime_path = paths.evidence_pack / "02_metrics" / "inference_runtime_summary.json"
    if inference_runtime_path.exists():
        inference_runtime = read_json(inference_runtime_path)
        report["inference_runtime_summary"] = {
            "device": inference_runtime.get("device"),
            "case_count": inference_runtime.get("case_count"),
            "ensemble_members": inference_runtime.get("ensemble_members"),
            "median_inference_ms_per_case_for_full_ensemble": inference_runtime.get(
                "median_inference_ms_per_case_for_full_ensemble"
            ),
            "openfoam_median_wall_time_sec_converged_canonical": inference_runtime.get(
                "openfoam_median_wall_time_sec_converged_canonical"
            ),
            "note": inference_runtime.get("note"),
        }
        report["inference_runtime_summary_sha256"] = file_sha256(inference_runtime_path)

    physical_plausibility_path = paths.evidence_pack / "02_metrics" / "physical_plausibility_summary.json"
    if physical_plausibility_path.exists():
        physical_plausibility = read_json(physical_plausibility_path)
        report["physical_plausibility"] = {
            "passed": physical_plausibility.get("passed"),
            "case_count": physical_plausibility.get("case_count"),
            "failed_case_count": physical_plausibility.get("failed_case_count"),
            "temperature_allowed_K": physical_plausibility.get("temperature_allowed_K"),
        }
        report["physical_plausibility_sha256"] = file_sha256(physical_plausibility_path)
        if physical_plausibility.get("passed") is not True:
            fail(errors, "Physical plausibility checks did not pass.")

    hard_summary_path = paths.evidence_pack / "02_metrics" / "hard_regime_solver_summary.json"
    if hard_summary_path.exists():
        hard_summary = read_json(hard_summary_path)
        report["hard_regime_solver_summary"] = {
            "total_generated": hard_summary.get("total_generated"),
            "total_converged": hard_summary.get("total_converged"),
            "total_dropped": hard_summary.get("total_dropped"),
            "retained_fraction": hard_summary.get("retained_fraction"),
            "summary_by_regime": hard_summary.get("summary_by_regime"),
        }
        report["hard_regime_solver_summary_sha256"] = file_sha256(hard_summary_path)

    branch_comparison_path = paths.evidence_pack / "02_metrics" / "hard_regime_branch" / "branch_comparison_summary.json"
    if branch_comparison_path.exists():
        branch_comparison = read_json(branch_comparison_path)
        report["hard_regime_branch_comparison"] = {
            "decision": branch_comparison.get("decision"),
            "reason": branch_comparison.get("reason"),
            "important_read": branch_comparison.get("important_read"),
        }
        report["hard_regime_branch_comparison_sha256"] = file_sha256(branch_comparison_path)

    return errors, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-only", action="store_true", help="Verify scaffold and sample tables only.")
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    errors, report = verify_structure(config, paths) if args.structure_only else verify_strict(config, paths)
    report["mode"] = "structure-only" if args.structure_only else "strict"
    report["passed"] = not errors
    report["errors"] = errors

    target = paths.evidence_pack / "06_verification" / "verification_report.json"
    write_json(target, report)

    if errors:
        print(f"Verification failed. Report: {target}")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"Verification passed. Report: {target}")


if __name__ == "__main__":
    main()
