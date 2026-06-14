from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit = load_script("audit_module", "src/pl_wp_002/audit.py")
train = load_script("train_operator_module", "scripts/09_train_field_operator.py")
gate = load_script("gate_module", "scripts/13_apply_regime_gate.py")
case_factory = load_script("case_factory_module", "scripts/02_create_openfoam_cases.py")
vm_runner = load_script("vm_runner_module", "scripts/03_run_openfoam_case_vm.py")
batch_runner = load_script("batch_runner_module", "scripts/04_run_openfoam_batch_vm.py")
solver_validation_summary = load_script(
    "solver_validation_summary_module",
    "scripts/25_summarize_tier2_solver_validation.py",
)
current_envelope_builder = load_script(
    "current_envelope_builder_module",
    "scripts/26_build_tier2_current_envelope.py",
)
physical_acquisition = load_script(
    "physical_acquisition_module",
    "scripts/29_select_tier2_physical_acquisition.py",
)
physical_backfill = load_script(
    "physical_backfill_module",
    "scripts/30_generate_tier2_physical_backfill.py",
)
physical_manifest = load_script(
    "physical_manifest_module",
    "scripts/31_finalize_tier2_physical_acquisition_manifest.py",
)
tier2_gate = load_script(
    "tier2_gate_module",
    "scripts/32_tier2_gate_diagnostics.py",
)


class CoreMathTests(unittest.TestCase):
    def test_qhat_uses_finite_sample_correction(self) -> None:
        scores = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        self.assertEqual(train.qhat(scores, 0.80), 5.0)
        self.assertEqual(train.qhat(scores, 0.50), 4.0)

    def test_spearman_handles_ties(self) -> None:
        self.assertAlmostEqual(audit.spearman([1.0, 2.0, 2.0, 4.0], [1.0, 3.0, 3.0, 7.0]), 1.0)

    def test_masked_mse_ignores_solid_pixels(self) -> None:
        pred = torch.tensor([[[[1.0, 10.0], [3.0, 10.0]]]])
        target = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]])
        mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        self.assertAlmostEqual(float(train.masked_mse(pred, target, mask)), 2.5)

    def test_spectral_conv_preserves_shape(self) -> None:
        model = train.CompactFNO(in_channels=3, out_channels=2, width=4, modes=2, layers=1)
        out = model(torch.zeros(2, 3, 8, 8))
        self.assertEqual(tuple(out.shape), (2, 2, 8, 8))

    def test_regime_gate_does_not_require_pool_label_for_serving(self) -> None:
        config = {
            "parameters": {
                "h_ch": {"kind": "continuous", "id_min": 4.0, "id_max": 10.0},
                "n_fin": {"kind": "discrete", "id_values": [4, 5, 6, 7, 8]},
                "h_fin": {"kind": "continuous", "id_min": 2.0, "id_max": 6.0},
                "d_in": {"kind": "continuous", "id_min": 2.0, "id_max": 6.0},
                "u_in": {"kind": "continuous", "id_min": 0.2, "id_max": 1.5},
                "q_w": {"kind": "continuous", "id_min": 5.0, "id_max": 50.0},
            }
        }
        predictions = [
            {
                "case_id": "hidden_shift_0000",
                "pool": "hidden_shift",
                "band_width_mean_norm_90": "0.1",
                "covered_pairs_90": "9",
                "total_pairs": "10",
                "coverage_90": "0.9",
            }
        ]
        parameters = {
            "hidden_shift_0000": {
                "h_ch": "6.0",
                "n_fin": "5",
                "h_fin": "3.0",
                "d_in": "3.0",
                "u_in": "0.5",
                "q_w": "10.0",
            }
        }
        decisions, summary = gate.apply_gate(
            predictions,
            parameters,
            config,
            scenario="unit",
            ood_pool="hidden_shift",
            id_tau=0.2,
        )
        self.assertEqual(decisions[0]["decision"], "serve_surrogate_id_like_low_score")
        self.assertEqual(decisions[0]["served_by_surrogate"], "true")
        self.assertEqual(summary["accepted_cases"], 1)


class CaseFactoryThermalContractTests(unittest.TestCase):
    def config(self, mode: str) -> dict:
        return {
            "physics": {
                "kinematic_viscosity_m2_s": 1e-6,
                "thermal_diffusivity_m2_s": 1e-7,
                "thermal_conductivity_W_mK": 0.6,
                "inlet_temperature_K": 293.15,
                "temperature_hard_min_K": 273.15,
                "temperature_hard_max_K": 500.0,
                "max_negative_delta_from_inlet_K": 2.0,
                "max_positive_delta_from_inlet_K": 200.0,
            },
            "solver": {
                "thermal_solver_mode": mode,
                "scalar_transport_n_correctors": 50,
                "scalar_transport_delta_t": 1e-4,
                "scalar_transport_duration_s": 0.05,
                "scalar_transport_write_interval_s": 0.005,
                "scalar_steady_check_enabled": True,
                "raw_enthalpy_check_enabled": True,
                "raw_temperature_extrema_check_enabled": True,
                "thermal_energy_iterations": 500,
                "thermal_energy_write_interval": 10,
                "thermal_equation_relaxation": 0.7,
                "thermal_residual_target": 1e-6,
                "convergence_residual_target": 1e-6,
            },
        }

    def test_dedicated_thermal_cases_do_not_load_scalar_transport_functions(self) -> None:
        config = self.config("dedicated_energy_foam")
        files = case_factory.common_system_files(config)
        allrun = case_factory.allrun_script(config)
        self.assertNotIn("system/functions", files)
        self.assertIn("thermalEnergyFoam -noFunctionObjects", allrun)
        self.assertIn("checkRawTemperatureExtrema.py", allrun)
        self.assertNotIn("foamRun -solver functions", allrun)

    def test_passive_scalar_diagnostic_route_keeps_function_object(self) -> None:
        config = self.config("passive_scalar_function")
        files = case_factory.common_system_files(config)
        allrun = case_factory.allrun_script(config)
        self.assertIn("system/functions", files)
        self.assertIn("#includeFunc scalarTransport", files["system/functions"])
        self.assertIn("foamRun -solver functions", allrun)

    def test_raw_enthalpy_lower_bound_does_not_hide_outlet_cooling(self) -> None:
        script = case_factory.raw_outlet_enthalpy_script(self.config("dedicated_energy_foam"))
        self.assertIn("lower_bound = max(MIN_RATIO * expected_delta, -ABS_MARGIN_K)", script)

    def test_raw_temperature_extrema_uses_physical_temperature_bounds(self) -> None:
        script = case_factory.raw_temperature_extrema_script(self.config("dedicated_energy_foam"))
        self.assertIn("MIN_ALLOWED_K = 291.15", script)
        self.assertIn("MAX_ALLOWED_K = 493.15", script)


class VmRunnerParsingTests(unittest.TestCase):
    def test_combined_log_keeps_flow_and_thermal_convergence_separate(self) -> None:
        log_text = """
Time = 2s

smoothSolver:  Solving for Ux, Initial residual = 2e-04, Final residual = 1e-08, No Iterations 2
smoothSolver:  Solving for Uy, Initial residual = 3e-04, Final residual = 2e-08, No Iterations 2
smoothSolver:  Solving for Uz, Initial residual = 4e-04, Final residual = 3e-08, No Iterations 2
GAMG:  Solving for p, Initial residual = 5e-03, Final residual = 4e-08, No Iterations 3
End

Exec   : thermalEnergyFoam -noFunctionObjects
Time = 3s

DILUPBiCGStab:  Solving for T, Initial residual = 5e-07, Final residual = 7e-10, No Iterations 5
SIMPLE solution converged in 3 iterations
End
"""
        row = vm_runner.parse_run(
            "unit_case",
            "unit_pool",
            "openfoam_cases/unit_case",
            "simpleFoam+thermalEnergyFoam",
            log_text,
            "    cells:           10\n",
            1.0,
            "OpenFOAM-13",
            1e-6,
            {"passed": True},
            {"passed": True},
        )
        self.assertEqual(row["flow_converged_reported"], "false")
        self.assertEqual(row["converged"], "false")
        self.assertEqual(row["iterations_flow"], 2)
        self.assertEqual(row["iterations_temperature"], 1)
        self.assertEqual(row["residual_U_initial"], "0.0004")
        self.assertEqual(row["residual_p_initial"], "0.005")
        self.assertEqual(row["residual_T_initial"], "5e-07")
        self.assertIn("flow solver did not report convergence", row["drop_reason"])
        self.assertIn("flow residuals did not meet convergence contract", row["drop_reason"])

    def test_raw_temperature_extrema_failure_drops_case(self) -> None:
        row = vm_runner.parse_run(
            "unit_case",
            "unit_pool",
            "openfoam_cases/unit_case",
            "simpleFoam+thermalEnergyFoam",
            """
Time = 2s
smoothSolver:  Solving for Ux, Initial residual = 2e-08, Final residual = 1e-09, No Iterations 1
smoothSolver:  Solving for Uy, Initial residual = 3e-08, Final residual = 2e-09, No Iterations 1
smoothSolver:  Solving for Uz, Initial residual = 4e-08, Final residual = 3e-09, No Iterations 1
GAMG:  Solving for p, Initial residual = 5e-08, Final residual = 4e-09, No Iterations 1
SIMPLE solution converged in 2 iterations
End

Exec   : thermalEnergyFoam -noFunctionObjects
Time = 3s
DILUPBiCGStab:  Solving for T, Initial residual = 5e-07, Final residual = 7e-10, No Iterations 5
End
""",
            "    cells:           10\n",
            1.0,
            "OpenFOAM-13",
            1e-6,
            {"passed": True},
            {"passed": True},
            {"passed": False, "min_T_K": 293.15, "mean_T_K": 400.0, "max_T_K": 900.0, "reason": "raw T_max_K above allowed"},
        )
        self.assertEqual(row["raw_temperature_extrema_passed"], "false")
        self.assertEqual(row["converged"], "false")
        self.assertIn("raw temperature extrema check failed", row["drop_reason"])


class BatchRunnerAccountingTests(unittest.TestCase):
    def test_retained_counts_are_scoped_to_selected_ids_and_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            convergence_path = root / "convergence_log.csv"
            manifest_path = root / "solver_output_manifest.csv"
            archive_path = root / "archives" / "selected_ok.tar.gz"
            archive_path.parent.mkdir()
            archive_path.write_bytes(b"archive")
            audit.write_csv(
                convergence_path,
                [
                    {"case_id": "selected_ok", "pool": "id_train", "converged": "true"},
                    {"case_id": "selected_missing_archive", "pool": "id_train", "converged": "true"},
                    {"case_id": "outside_ok", "pool": "id_train", "converged": "true"},
                ],
                ["case_id", "pool", "converged"],
            )
            audit.write_csv(
                manifest_path,
                [
                    {"case_id": "selected_ok", "archive_path": "archives/selected_ok.tar.gz"},
                    {"case_id": "outside_ok", "archive_path": "archives/outside_ok.tar.gz"},
                ],
                ["case_id", "archive_path"],
            )
            counts = batch_runner.retained_by_pool(
                convergence_path,
                manifest_path,
                root,
                allowed_ids={"selected_ok", "selected_missing_archive"},
                require_archive=True,
            )
            self.assertEqual(counts, {"id_train": 1})


class SolverValidationEnvelopeTests(unittest.TestCase):
    def test_current_steady_envelope_excludes_only_hard_high_din(self) -> None:
        in_envelope, reason = solver_validation_summary.current_steady_solver_envelope(
            {"validation_pool": "hard_regime", "target_regime": "nfin3_d_in_high"}
        )
        self.assertFalse(in_envelope)
        self.assertIn("hard high-d_in", reason)

        in_envelope, reason = solver_validation_summary.current_steady_solver_envelope(
            {"validation_pool": "fresh_ood", "target_regime": "d_in_high"}
        )
        self.assertTrue(in_envelope)
        self.assertEqual(reason, "")

    def test_current_envelope_builder_separates_exclusions_from_failures(self) -> None:
        in_envelope_ids, failures, exclusions = current_envelope_builder.current_envelope_rows(
            [
                {
                    "case_id": "id_train_0000",
                    "pool": "id_train",
                    "validation_pool": "id",
                    "in_current_steady_solver_envelope": "true",
                    "converged": "true",
                    "scalar_stability_passed": "true",
                    "raw_enthalpy_passed": "true",
                },
                {
                    "case_id": "near_ood_0000",
                    "pool": "ood_test",
                    "validation_pool": "near_ood",
                    "in_current_steady_solver_envelope": "true",
                    "converged": "false",
                    "scalar_stability_passed": "true",
                    "raw_enthalpy_passed": "true",
                    "drop_reason": "flow residuals did not meet convergence contract",
                },
                {
                    "case_id": "hard_dinhighidfin_0005",
                    "pool": "hard_regime",
                    "validation_pool": "hard_regime",
                    "selection_group": "unit",
                    "target_regime": "d_in_high_id_fin",
                    "in_current_steady_solver_envelope": "false",
                    "solver_envelope_reason": "hard high-d_in regime requires a separate validated steady/transient solver path",
                    "converged": "false",
                    "scalar_stability_passed": "true",
                    "raw_enthalpy_passed": "true",
                },
            ]
        )
        self.assertEqual(in_envelope_ids, {"id_train_0000", "near_ood_0000"})
        self.assertEqual(failures[0]["case_id"], "near_ood_0000")
        self.assertEqual(exclusions[0]["case_id"], "hard_dinhighidfin_0005")


class Tier2PhysicalAcquisitionTests(unittest.TestCase):
    def test_temperature_risk_proxy_increases_with_heat_load(self) -> None:
        low = {
            "n_fin": "4",
            "q_w": "10",
            "u_in": "1.0",
            "d_in": "4.0",
            "h_ch": "8.0",
        }
        high = {**low, "q_w": "40"}
        self.assertLess(
            physical_acquisition.temperature_risk_proxy(low),
            physical_acquisition.temperature_risk_proxy(high),
        )

    def test_interleaved_order_preserves_regime_strata(self) -> None:
        rows = [
            {
                "case_id": "a_hot",
                "stratification_key": "axis_a",
                "temperature_risk_proxy": "10.0",
                "nearest_id_train_distance": "0.1",
            },
            {
                "case_id": "a_cool",
                "stratification_key": "axis_a",
                "temperature_risk_proxy": "1.0",
                "nearest_id_train_distance": "0.1",
            },
            {
                "case_id": "b_cool",
                "stratification_key": "axis_b",
                "temperature_risk_proxy": "2.0",
                "nearest_id_train_distance": "0.1",
            },
            {
                "case_id": "b_hot",
                "stratification_key": "axis_b",
                "temperature_risk_proxy": "20.0",
                "nearest_id_train_distance": "0.1",
            },
        ]
        ordered = physical_acquisition.interleaved_by_stratum(rows)
        self.assertEqual([row["case_id"] for row in ordered], ["a_cool", "b_cool", "a_hot", "b_hot"])


class Tier2PhysicalBackfillTests(unittest.TestCase):
    def test_dropped_seed_rows_are_only_canonical_acquisition_rows(self) -> None:
        acquisition_rows = [
            {"case_id": "id_train_0001", "pool": "id_train"},
            {"case_id": "ood_test_0001", "pool": "ood_test"},
        ]
        convergence_by_case = {
            "id_train_0001": {"dropped": "true"},
            "ood_test_0001": {"dropped": "false"},
            "id_train_pbf01_0001": {"dropped": "true"},
            "ood_test_pbf01_0001": {"dropped": "true"},
        }

        dropped = physical_backfill.dropped_canonical_acquisition_rows(acquisition_rows, convergence_by_case)

        self.assertEqual([row["case_id"] for row in dropped], ["id_train_0001"])

    def test_id_backfill_preserves_original_fin_stratum(self) -> None:
        config = {
            "parameters": {
                "h_ch": {"kind": "continuous", "id_min": 4.0, "id_max": 10.0},
                "n_fin": {"kind": "discrete", "id_values": [4, 5, 6, 7, 8]},
                "h_fin": {"kind": "continuous", "id_min": 2.0, "id_max": 6.0},
                "d_in": {"kind": "continuous", "id_min": 2.0, "id_max": 6.0},
                "u_in": {"kind": "continuous", "id_min": 0.2, "id_max": 1.5},
                "q_w": {"kind": "continuous", "id_min": 5.0, "id_max": 50.0},
            },
            "geometry": {
                "max_fin_height_fraction_of_channel": 0.75,
            },
        }
        original = {
            "case_id": "id_train_drop",
            "pool": "id_train",
            "n_fin": "7",
        }
        row = physical_backfill.id_candidate(
            "id_train_pbf01_0000",
            original,
            physical_backfill.random.Random(123),
            config,
            1,
            0,
        )
        self.assertEqual(row["pool"], "id_train")
        self.assertEqual(row["n_fin"], 7)
        self.assertEqual(row["replacement_for"], "id_train_drop")
        self.assertGreaterEqual(float(row["h_ch"]), 4.0)
        self.assertLessEqual(float(row["h_ch"]), 10.0)


class Tier2PhysicalManifestTests(unittest.TestCase):
    def test_final_manifest_requires_convergence_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "archives" / "kept.tar.gz"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"archive")
            parameter_rows = [
                {
                    "case_id": "kept",
                    "pool": "id_train",
                    "h_ch": "5",
                    "n_fin": "4",
                    "h_fin": "2",
                    "d_in": "3",
                    "u_in": "0.5",
                    "q_w": "10",
                    "geometry_family": "uniform_fins",
                    "solver_status": "pending",
                },
                {
                    "case_id": "missing_archive",
                    "pool": "id_train",
                    "h_ch": "5",
                    "n_fin": "4",
                    "h_fin": "2",
                    "d_in": "3",
                    "u_in": "0.5",
                    "q_w": "10",
                    "geometry_family": "uniform_fins",
                    "solver_status": "pending",
                },
                {
                    "case_id": "dropped",
                    "pool": "ood_test",
                    "h_ch": "5",
                    "n_fin": "3",
                    "h_fin": "2",
                    "d_in": "7",
                    "u_in": "0.5",
                    "q_w": "10",
                    "geometry_family": "uniform_fins",
                    "solver_status": "pending",
                },
            ]
            convergence_rows = [
                {"case_id": "kept", "converged": "true", "dropped": "false"},
                {"case_id": "missing_archive", "converged": "true", "dropped": "false"},
                {"case_id": "dropped", "converged": "false", "dropped": "true", "drop_reason": "raw temperature extrema check failed"},
            ]
            archive_rows = [
                {"case_id": "kept", "archive_path": "archives/kept.tar.gz", "archive_size_bytes": "7"},
                {"case_id": "missing_archive", "archive_path": "archives/missing.tar.gz", "archive_size_bytes": "7"},
            ]

            retained, manifest, status = physical_manifest.build_final_manifests(
                parameter_rows,
                convergence_rows,
                archive_rows,
                root,
            )

            self.assertEqual([row["case_id"] for row in retained], ["kept"])
            self.assertEqual([row["case_id"] for row in manifest], ["kept"])
            status_by_case = {row["case_id"]: row["selection_status"] for row in status}
            self.assertEqual(status_by_case["kept"], "retained")
            self.assertEqual(status_by_case["missing_archive"], "converged_missing_archive")
            self.assertEqual(status_by_case["dropped"], "dropped")

    def test_canonical_target_failures_report_short_pool(self) -> None:
        config = {"dataset": {"id_train": 2, "id_calibration": 1, "id_test": 1, "ood_test": 1}}
        failures = physical_manifest.canonical_target_failures(
            config,
            Counter({"id_train": 2, "id_calibration": 1, "id_test": 1}),
        )
        self.assertEqual(failures, ["ood_test retained 0 below configured target 1"])


class Tier2GateDiagnosticsTests(unittest.TestCase):
    def test_explicit_out_of_envelope_detects_discrete_and_continuous_shifts(self) -> None:
        config = {
            "parameters": {
                "h_ch": {"kind": "continuous", "id_min": 4.0, "id_max": 10.0},
                "n_fin": {"kind": "discrete", "id_values": [4, 5, 6, 7, 8]},
            }
        }
        self.assertFalse(tier2_gate.explicit_out_of_envelope({"h_ch": "6.0", "n_fin": "5"}, config))
        self.assertTrue(tier2_gate.explicit_out_of_envelope({"h_ch": "11.0", "n_fin": "5"}, config))
        self.assertTrue(tier2_gate.explicit_out_of_envelope({"h_ch": "6.0", "n_fin": "3"}, config))

    def test_learned_score_policy_routes_high_score_ood(self) -> None:
        config = {
            "parameters": {
                "h_ch": {"kind": "continuous", "id_min": 4.0, "id_max": 10.0},
                "n_fin": {"kind": "discrete", "id_values": [4, 5, 6, 7, 8]},
            }
        }
        rows = [
            {
                "case_id": "id_low",
                "pool": "id_test",
                "band_width_mean_norm_90": "0.1",
                "covered_pairs_90": "9",
                "total_pairs": "10",
            },
            {
                "case_id": "id_high",
                "pool": "id_test",
                "band_width_mean_norm_90": "0.4",
                "covered_pairs_90": "9",
                "total_pairs": "10",
            },
            {
                "case_id": "ood_high",
                "pool": "ood_test",
                "band_width_mean_norm_90": "1.0",
                "covered_pairs_90": "8",
                "total_pairs": "10",
            },
        ]
        params = {
            "id_low": {"h_ch": "6.0", "n_fin": "5"},
            "id_high": {"h_ch": "6.0", "n_fin": "5"},
            "ood_high": {"h_ch": "11.0", "n_fin": "5"},
        }

        auc = tier2_gate.auroc_higher_score_is_more_ood(rows, "band_width_mean_norm_90", "id_test", {"ood_test"})
        summary = tier2_gate.evaluate_policy(
            rows,
            params,
            config,
            {"id_test", "ood_test"},
            {"ood_test"},
            "learned_score",
            "band_width_mean_norm_90",
            0.5,
        )

        self.assertEqual(auc, 1.0)
        self.assertEqual(summary["id_served_fraction"], 1.0)
        self.assertEqual(summary["ood_routed_fraction"], 1.0)


class PublicReleaseHygieneTests(unittest.TestCase):
    def test_public_files_do_not_expose_internal_release_labels(self) -> None:
        forbidden_tokens = (
            "P" + "X-",
            "p" + "x-",
            "P" + "X_",
            "p" + "x_",
            "scope" + "_id",
            "S" + "OW",
        )
        explicit_files = [
            ROOT / "README.md",
            ROOT / "Makefile",
            ROOT / "pyproject.toml",
            ROOT / "CITATION.cff",
            ROOT / ".github" / "workflows" / "tests.yml",
            ROOT / "configs" / "study_tier2_steady.toml",
            ROOT / "evidence_pack_tier2_steady" / "01_config" / "study.toml",
            ROOT / "templates" / "openfoam" / "fluid_side_fixed_flux" / "README.md",
        ]
        scanned = list(explicit_files)
        text_suffixes = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".csv", ".cff", ".C"}
        for directory in ["scripts", "src", "openfoam_solvers"]:
            scanned.extend(
                path
                for path in (ROOT / directory).rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and (path.suffix in text_suffixes or path.name in {"files", "options"})
            )

        for path in scanned:
            text = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                self.assertNotIn(token, text, f"{token!r} leaked in {path.relative_to(ROOT)}")


if __name__ == "__main__":
    unittest.main()
