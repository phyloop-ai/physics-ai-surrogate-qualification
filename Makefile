PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
DEPS_STAMP := $(VENV)/.deps.stamp

TIER2_CONFIG ?= configs/study_tier2_steady.toml
TIER2_EVIDENCE ?= evidence_pack_tier2_steady
TIER2_FRESH_OOD_RETAINED ?= 90
TIER2_PHYSICAL_ACQUISITION_MIN_CASES ?= 840
TIER2_PHYSICAL_ACQUISITION_COMBINED_PARAMETER_FILE ?= $(TIER2_EVIDENCE)/03_data/physical_acquisition_with_backfill_parameters.csv
TIER2_PHYSICAL_ACQUISITION_PARAMETER_FILE ?= $(TIER2_EVIDENCE)/03_data/physical_acquisition_retained_parameters.csv
TIER2_PHYSICAL_ACQUISITION_LIMIT ?=
TIER2_PHYSICAL_FRESH_OOD_ACQUISITION_LIMIT ?=
TIER2_PHYSICAL_BACKFILL_ROUND ?= 1
TIER2_PHYSICAL_BACKFILL_PREFIX ?= physical_backfill_round01
TIER2_PHYSICAL_BACKFILL_PARAMETERS ?= $(TIER2_EVIDENCE)/03_data/$(TIER2_PHYSICAL_BACKFILL_PREFIX)_parameters.csv
TIER2_PHYSICAL_PREVIOUS_BACKFILL_PARAMETERS ?=
TIER2_PHYSICAL_BACKFILL_ID_CANDIDATES_PER_DROP ?= 3
TIER2_PHYSICAL_BACKFILL_OOD_CANDIDATES_PER_DROP ?= 4
TIER2_PHYSICAL_BACKFILL_LIMIT ?=

.PHONY: deps test public-check pl-wp-002-public-check \
	tier2-bootstrap tier2-samples tier2-verify-structure \
	tier2-cases-smoke tier2-run-smoke-vm tier2-extract-smoke tier2-validate-smoke tier2-smoke \
	tier2-fresh-ood-candidates tier2-hard-regime-candidates \
	tier2-select-solver-validation tier2-cases-solver-validation tier2-run-solver-validation-vm \
	tier2-summarize-solver-validation tier2-solver-validation \
	tier2-current-envelope tier2-cases-current-envelope tier2-run-current-envelope-vm \
	tier2-extract-current-envelope tier2-validate-current-envelope tier2-current-envelope-archive \
	tier2-temperature-replacements tier2-cases-temperature-replacements tier2-run-temperature-replacements-vm \
	tier2-physical-envelope tier2-extract-physical-envelope tier2-validate-physical-envelope \
	tier2-train-physical-smoke tier2-physical-acquisition-candidates tier2-cases-physical-acquisition \
	tier2-run-physical-acquisition-vm tier2-physical-backfill-candidates tier2-cases-physical-backfill \
	tier2-run-physical-backfill-vm tier2-physical-backfill \
	tier2-cases-physical-fresh-ood-acquisition tier2-run-physical-fresh-ood-acquisition-vm \
	tier2-finalize-physical-acquisition tier2-extract-physical-acquisition \
	tier2-validate-physical-acquisition tier2-train-physical-operator tier2-gate-diagnostics \
	tier2-physical-acquisition-evidence

deps: $(DEPS_STAMP)

test: deps
	$(VENV_PYTHON) -m pytest -q tests

public-check: test pl-wp-002-public-check

$(DEPS_STAMP): requirements.txt requirements.lock
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.lock
	touch $(DEPS_STAMP)

pl-wp-002-public-check: deps
	$(VENV_PYTHON) scripts/34_validate_pl_wp_002_public_chain.py

tier2-bootstrap:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/00_bootstrap_evidence_pack.py

tier2-samples:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/01_generate_samples.py

tier2-verify-structure:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/06_verify_pack.py --structure-only

tier2-cases-smoke:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/02_create_openfoam_cases.py --limit 3 --force \
		--parameter-file $(TIER2_EVIDENCE)/03_data/parameters.csv \
		--manifest-file $(TIER2_EVIDENCE)/03_data/case_factory_manifest.csv \
		--summary-file $(TIER2_EVIDENCE)/03_data/case_factory_summary.json

tier2-run-smoke-vm:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/03_run_openfoam_case_vm.py \
		--case-id id_train_0000 \
		--force \
		--archive-output \
		--cleanup-remote \
		--parameter-file $(TIER2_EVIDENCE)/03_data/parameters.csv

tier2-extract-smoke: deps
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(VENV_PYTHON) scripts/08_extract_fixed_grid_fields.py \
		--case-id id_train_0000 \
		--output-dir $(TIER2_EVIDENCE)/04_predictions/fixed_grid_fields \
		--index-file $(TIER2_EVIDENCE)/04_predictions/fixed_grid_index.csv \
		--summary-file $(TIER2_EVIDENCE)/04_predictions/fixed_grid_summary.json \
		--force

tier2-validate-smoke: deps
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(VENV_PYTHON) scripts/18_validate_physical_plausibility.py \
		--index-file $(TIER2_EVIDENCE)/04_predictions/fixed_grid_index.csv \
		--summary-json $(TIER2_EVIDENCE)/02_metrics/physical_plausibility_summary.json \
		--summary-csv $(TIER2_EVIDENCE)/02_metrics/physical_plausibility_by_case.csv

tier2-smoke: tier2-bootstrap tier2-samples tier2-verify-structure tier2-cases-smoke tier2-run-smoke-vm tier2-extract-smoke tier2-validate-smoke

tier2-fresh-ood-candidates:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/10_generate_fresh_ood_holdout.py \
		--candidate-count 90 \
		--parameter-file $(TIER2_EVIDENCE)/03_data/fresh_ood_parameters.csv \
		--distance-file $(TIER2_EVIDENCE)/03_data/fresh_ood_distance.csv \
		--summary-file $(TIER2_EVIDENCE)/03_data/fresh_ood_generation_summary.json

tier2-hard-regime-candidates:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/14_generate_hard_regime_samples.py \
		--per-regime 6 \
		--parameter-file $(TIER2_EVIDENCE)/03_data/hard_regime_parameters.csv \
		--distance-file $(TIER2_EVIDENCE)/03_data/hard_regime_distance.csv \
		--summary-file $(TIER2_EVIDENCE)/03_data/hard_regime_generation_summary.json

tier2-select-solver-validation:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/24_select_tier2_solver_validation.py

tier2-cases-solver-validation:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/02_create_openfoam_cases.py --all --force \
		--parameter-file $(TIER2_EVIDENCE)/03_data/solver_validation_parameters.csv \
		--manifest-file $(TIER2_EVIDENCE)/03_data/solver_validation_case_factory_manifest.csv \
		--summary-file $(TIER2_EVIDENCE)/03_data/solver_validation_case_factory_summary.json

tier2-run-solver-validation-vm:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/04_run_openfoam_batch_vm.py \
		--parameter-file $(TIER2_EVIDENCE)/03_data/solver_validation_parameters.csv \
		--status-file $(TIER2_EVIDENCE)/03_data/solver_validation_batch_status.json \
		--no-archive-output \
		--rerun-converged \
		--rerun-dropped \
		--allow-drops

tier2-summarize-solver-validation:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/25_summarize_tier2_solver_validation.py

tier2-solver-validation: tier2-bootstrap tier2-fresh-ood-candidates tier2-hard-regime-candidates tier2-select-solver-validation tier2-cases-solver-validation tier2-run-solver-validation-vm tier2-summarize-solver-validation

tier2-current-envelope:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/26_build_tier2_current_envelope.py

tier2-cases-current-envelope:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/02_create_openfoam_cases.py --all --force \
		--parameter-file $(TIER2_EVIDENCE)/03_data/current_envelope_parameters.csv \
		--manifest-file $(TIER2_EVIDENCE)/03_data/current_envelope_case_factory_manifest.csv \
		--summary-file $(TIER2_EVIDENCE)/03_data/current_envelope_case_factory_summary.json

tier2-run-current-envelope-vm:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/04_run_openfoam_batch_vm.py \
		--parameter-file $(TIER2_EVIDENCE)/03_data/current_envelope_parameters.csv \
		--status-file $(TIER2_EVIDENCE)/03_data/current_envelope_batch_status.json \
		--archive-output

tier2-extract-current-envelope: deps
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(VENV_PYTHON) scripts/08_extract_fixed_grid_fields.py \
		--parameter-file $(TIER2_EVIDENCE)/03_data/current_envelope_parameters.csv \
		--include-extra-pools \
		--output-dir $(TIER2_EVIDENCE)/04_predictions/current_envelope_fixed_grid_fields \
		--index-file $(TIER2_EVIDENCE)/04_predictions/current_envelope_fixed_grid_index.csv \
		--summary-file $(TIER2_EVIDENCE)/04_predictions/current_envelope_fixed_grid_summary.json \
		--force

tier2-validate-current-envelope: deps
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(VENV_PYTHON) scripts/18_validate_physical_plausibility.py \
		--index-file $(TIER2_EVIDENCE)/04_predictions/current_envelope_fixed_grid_index.csv \
		--summary-json $(TIER2_EVIDENCE)/02_metrics/current_envelope_physical_plausibility_summary.json \
		--summary-csv $(TIER2_EVIDENCE)/02_metrics/current_envelope_physical_plausibility_by_case.csv \
		--min-cases 19

tier2-current-envelope-archive: tier2-current-envelope tier2-cases-current-envelope tier2-run-current-envelope-vm tier2-extract-current-envelope tier2-validate-current-envelope

tier2-temperature-replacements:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/27_select_tier2_temperature_replacements.py

tier2-cases-temperature-replacements:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/02_create_openfoam_cases.py --all --force \
		--parameter-file $(TIER2_EVIDENCE)/03_data/temperature_replacement_parameters.csv \
		--manifest-file $(TIER2_EVIDENCE)/03_data/temperature_replacement_case_factory_manifest.csv \
		--summary-file $(TIER2_EVIDENCE)/03_data/temperature_replacement_case_factory_summary.json

tier2-run-temperature-replacements-vm:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/04_run_openfoam_batch_vm.py \
		--parameter-file $(TIER2_EVIDENCE)/03_data/temperature_replacement_parameters.csv \
		--status-file $(TIER2_EVIDENCE)/03_data/temperature_replacement_batch_status.json \
		--archive-output \
		--rerun-converged \
		--rerun-dropped \
		--allow-drops

tier2-physical-envelope:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/28_build_tier2_physical_envelope.py

tier2-extract-physical-envelope: deps
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(VENV_PYTHON) scripts/08_extract_fixed_grid_fields.py \
		--parameter-file $(TIER2_EVIDENCE)/03_data/physical_envelope_parameters.csv \
		--include-extra-pools \
		--output-dir $(TIER2_EVIDENCE)/04_predictions/physical_envelope_fixed_grid_fields \
		--index-file $(TIER2_EVIDENCE)/04_predictions/physical_envelope_fixed_grid_index.csv \
		--summary-file $(TIER2_EVIDENCE)/04_predictions/physical_envelope_fixed_grid_summary.json \
		--force

tier2-validate-physical-envelope: deps
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(VENV_PYTHON) scripts/18_validate_physical_plausibility.py \
		--index-file $(TIER2_EVIDENCE)/04_predictions/physical_envelope_fixed_grid_index.csv \
		--summary-json $(TIER2_EVIDENCE)/02_metrics/physical_envelope_physical_plausibility_summary.json \
		--summary-csv $(TIER2_EVIDENCE)/02_metrics/physical_envelope_physical_plausibility_by_case.csv \
		--min-cases 19

tier2-train-physical-smoke: deps
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(VENV_PYTHON) scripts/09_train_field_operator.py \
		--fixed-grid-index $(TIER2_EVIDENCE)/04_predictions/physical_envelope_fixed_grid_index.csv \
		--extra-parameter-file $(TIER2_EVIDENCE)/03_data/physical_envelope_parameters.csv \
		--epochs 3 \
		--batch-size 2 \
		--predict-batch-size 4 \
		--width 8 \
		--modes 6 \
		--layers 2 \
		--ensemble-size 2 \
		--ensemble-strategy full \
		--distance-gain 0 \
		--ensemble-std-gain 1 \
		--risk-head none \
		--model-dir $(TIER2_EVIDENCE)/04_predictions/tier2_physical_smoke/model \
		--prediction-dir $(TIER2_EVIDENCE)/04_predictions/tier2_physical_smoke/eval_cases \
		--prediction-index $(TIER2_EVIDENCE)/04_predictions/tier2_physical_smoke/prediction_index.csv \
		--metrics-dir $(TIER2_EVIDENCE)/02_metrics/tier2_physical_smoke \
		--figures-dir $(TIER2_EVIDENCE)/05_figures/tier2_physical_smoke

tier2-physical-acquisition-candidates:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/29_select_tier2_physical_acquisition.py \
		--fresh-retained-target $(TIER2_FRESH_OOD_RETAINED)

tier2-cases-physical-acquisition:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/02_create_openfoam_cases.py --all --force \
		--parameter-file $(TIER2_EVIDENCE)/03_data/physical_acquisition_parameters.csv \
		--manifest-file $(TIER2_EVIDENCE)/03_data/physical_acquisition_case_factory_manifest.csv \
		--summary-file $(TIER2_EVIDENCE)/03_data/physical_acquisition_case_factory_summary.json

tier2-run-physical-acquisition-vm:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/04_run_openfoam_batch_vm.py \
		--parameter-file $(TIER2_EVIDENCE)/03_data/physical_acquisition_parameters.csv \
		--status-file $(TIER2_EVIDENCE)/03_data/physical_acquisition_batch_status.json \
		$(if $(TIER2_PHYSICAL_ACQUISITION_LIMIT),--limit $(TIER2_PHYSICAL_ACQUISITION_LIMIT),) \
		--archive-output \
		--stop-at-config-targets \
		--allow-drops

tier2-physical-backfill-candidates:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/30_generate_tier2_physical_backfill.py \
		--round $(TIER2_PHYSICAL_BACKFILL_ROUND) \
		--id-candidates-per-drop $(TIER2_PHYSICAL_BACKFILL_ID_CANDIDATES_PER_DROP) \
		--ood-candidates-per-drop $(TIER2_PHYSICAL_BACKFILL_OOD_CANDIDATES_PER_DROP) \
		$(foreach value,$(TIER2_PHYSICAL_PREVIOUS_BACKFILL_PARAMETERS),--previous-backfill-parameter-file $(value)) \
		--output-parameter-file $(TIER2_PHYSICAL_BACKFILL_PARAMETERS) \
		--output-ood-distance-file $(TIER2_EVIDENCE)/03_data/$(TIER2_PHYSICAL_BACKFILL_PREFIX)_ood_distance.csv \
		--summary-file $(TIER2_EVIDENCE)/03_data/$(TIER2_PHYSICAL_BACKFILL_PREFIX)_summary.json

tier2-cases-physical-backfill:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/02_create_openfoam_cases.py --all --force \
		--parameter-file $(TIER2_PHYSICAL_BACKFILL_PARAMETERS) \
		--manifest-file $(TIER2_EVIDENCE)/03_data/$(TIER2_PHYSICAL_BACKFILL_PREFIX)_case_factory_manifest.csv \
		--summary-file $(TIER2_EVIDENCE)/03_data/$(TIER2_PHYSICAL_BACKFILL_PREFIX)_case_factory_summary.json

tier2-run-physical-backfill-vm:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/04_run_openfoam_batch_vm.py \
		--parameter-file $(TIER2_PHYSICAL_BACKFILL_PARAMETERS) \
		--status-file $(TIER2_EVIDENCE)/03_data/$(TIER2_PHYSICAL_BACKFILL_PREFIX)_batch_status.json \
		$(if $(TIER2_PHYSICAL_BACKFILL_LIMIT),--limit $(TIER2_PHYSICAL_BACKFILL_LIMIT),) \
		--archive-output \
		--allow-drops

tier2-physical-backfill: tier2-physical-backfill-candidates tier2-cases-physical-backfill tier2-run-physical-backfill-vm

tier2-cases-physical-fresh-ood-acquisition:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/02_create_openfoam_cases.py --all --force \
		--parameter-file $(TIER2_EVIDENCE)/03_data/physical_fresh_ood_acquisition_parameters.csv \
		--manifest-file $(TIER2_EVIDENCE)/03_data/physical_fresh_ood_acquisition_case_factory_manifest.csv \
		--summary-file $(TIER2_EVIDENCE)/03_data/physical_fresh_ood_acquisition_case_factory_summary.json

tier2-run-physical-fresh-ood-acquisition-vm:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/04_run_openfoam_batch_vm.py \
		--parameter-file $(TIER2_EVIDENCE)/03_data/physical_fresh_ood_acquisition_parameters.csv \
		--status-file $(TIER2_EVIDENCE)/03_data/physical_fresh_ood_acquisition_batch_status.json \
		$(if $(TIER2_PHYSICAL_FRESH_OOD_ACQUISITION_LIMIT),--limit $(TIER2_PHYSICAL_FRESH_OOD_ACQUISITION_LIMIT),) \
		--archive-output \
		--target-retained-pool ood_fresh \
		--target-retained-count $(TIER2_FRESH_OOD_RETAINED) \
		--allow-drops

tier2-finalize-physical-acquisition:
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(PYTHON) scripts/31_finalize_tier2_physical_acquisition_manifest.py \
		--parameter-file $(TIER2_PHYSICAL_ACQUISITION_COMBINED_PARAMETER_FILE) \
		--output-retained-parameters $(TIER2_PHYSICAL_ACQUISITION_PARAMETER_FILE) \
		--min-retained-cases $(TIER2_PHYSICAL_ACQUISITION_MIN_CASES) \
		--fresh-retained-target $(TIER2_FRESH_OOD_RETAINED)

tier2-extract-physical-acquisition: deps
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(VENV_PYTHON) scripts/08_extract_fixed_grid_fields.py \
		--parameter-file $(TIER2_PHYSICAL_ACQUISITION_PARAMETER_FILE) \
		--include-extra-pools \
		--output-dir $(TIER2_EVIDENCE)/04_predictions/physical_acquisition_fixed_grid_fields \
		--index-file $(TIER2_EVIDENCE)/04_predictions/physical_acquisition_fixed_grid_index.csv \
		--summary-file $(TIER2_EVIDENCE)/04_predictions/physical_acquisition_fixed_grid_summary.json \
		--force

tier2-validate-physical-acquisition: deps
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(VENV_PYTHON) scripts/18_validate_physical_plausibility.py \
		--index-file $(TIER2_EVIDENCE)/04_predictions/physical_acquisition_fixed_grid_index.csv \
		--summary-json $(TIER2_EVIDENCE)/02_metrics/physical_acquisition_physical_plausibility_summary.json \
		--summary-csv $(TIER2_EVIDENCE)/02_metrics/physical_acquisition_physical_plausibility_by_case.csv \
		--min-cases $(TIER2_PHYSICAL_ACQUISITION_MIN_CASES)

tier2-train-physical-operator: deps
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(VENV_PYTHON) scripts/09_train_field_operator.py \
		--fixed-grid-index $(TIER2_EVIDENCE)/04_predictions/physical_acquisition_fixed_grid_index.csv \
		--extra-parameter-file $(TIER2_PHYSICAL_ACQUISITION_PARAMETER_FILE) \
		--epochs 30 \
		--batch-size 8 \
		--predict-batch-size 16 \
		--width 24 \
		--modes 12 \
		--layers 4 \
		--ensemble-size 5 \
		--ensemble-strategy fin_holdout \
		--distance-gain 0 \
		--ensemble-std-gain 1 \
		--risk-head pseudo_quadratic \
		--risk-head-lambda 0.1 \
		--risk-gain 0.5 \
		--risk-multiplier-cap 8 \
		--model-dir $(TIER2_EVIDENCE)/04_predictions/tier2_physical_operator/model \
		--prediction-dir $(TIER2_EVIDENCE)/04_predictions/tier2_physical_operator/eval_cases \
		--prediction-index $(TIER2_EVIDENCE)/04_predictions/tier2_physical_operator/prediction_index.csv \
		--metrics-dir $(TIER2_EVIDENCE)/02_metrics/tier2_physical_operator \
		--figures-dir $(TIER2_EVIDENCE)/05_figures/tier2_physical_operator

tier2-gate-diagnostics: deps
	CRL_MPE_CONFIG=$(TIER2_CONFIG) $(VENV_PYTHON) scripts/32_tier2_gate_diagnostics.py

tier2-physical-acquisition-evidence: tier2-physical-acquisition-candidates tier2-cases-physical-acquisition tier2-run-physical-acquisition-vm tier2-cases-physical-fresh-ood-acquisition tier2-run-physical-fresh-ood-acquisition-vm tier2-finalize-physical-acquisition tier2-extract-physical-acquisition tier2-validate-physical-acquisition
