#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import load_config, read_csv, read_json, study_paths, write_csv, write_json


DECISION_FIELDS = [
    "scenario",
    "case_id",
    "pool",
    "explicit_parameter_ood",
    "score_name",
    "score",
    "threshold_tau",
    "decision",
    "served_by_surrogate",
    "routed_to_solver",
    "covered_pairs_90",
    "total_pairs",
    "coverage_90",
]
SUMMARY_FIELDS = [
    "scenario",
    "policy",
    "threshold_tau",
    "accepted_cases",
    "id_total",
    "ood_total",
    "accepted_coverage_90",
    "id_served_fraction",
    "ood_routed_fraction",
]


def load_parameter_rows(paths) -> dict[str, dict[str, str]]:
    rows = read_csv(paths.evidence_pack / "03_data" / "parameters.csv")
    backfill = paths.evidence_pack / "03_data" / "backfill_parameters.csv"
    if backfill.exists():
        rows.extend(read_csv(backfill))
    fresh = paths.evidence_pack / "03_data" / "fresh_ood_parameters.csv"
    if fresh.exists():
        rows.extend(read_csv(fresh))
    return {row["case_id"]: row for row in rows}


def load_prediction_rows(paths) -> list[dict[str, str]]:
    rows = read_csv(paths.evidence_pack / "04_predictions" / "prediction_index.csv")
    fresh = paths.evidence_pack / "04_predictions" / "fresh_ood_prediction_index.csv"
    if fresh.exists():
        rows.extend(read_csv(fresh))
    return rows


def explicit_out_of_envelope(row: dict[str, str], config: dict[str, Any]) -> bool:
    for name, spec in config["parameters"].items():
        value = float(row[name])
        if spec["kind"] == "discrete":
            allowed = {float(item) for item in spec["id_values"]}
            if value not in allowed:
                return True
            continue
        lo = float(spec["id_min"])
        hi = float(spec["id_max"])
        if value < lo or value > hi:
            return True
    return False


def apply_gate(
    rows: list[dict[str, str]],
    parameters: dict[str, dict[str, str]],
    config: dict[str, Any],
    scenario: str,
    ood_pool: str,
    id_tau: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [row for row in rows if row["pool"] in {"id_test", ood_pool}]
    decisions = []
    accepted = []
    routed_ood = 0
    id_total = sum(row["pool"] == "id_test" for row in selected)
    ood_total = sum(row["pool"] == ood_pool for row in selected)

    for row in selected:
        params = parameters[row["case_id"]]
        explicit_ood = explicit_out_of_envelope(params, config)
        score = float(row["band_width_mean_norm_90"])
        served = False
        if explicit_ood:
            decision = "route_solver_explicit_parameter_ood"
        elif not explicit_ood and score <= id_tau:
            decision = "serve_surrogate_id_like_low_score"
            served = True
        else:
            decision = "route_solver_high_score"

        routed = not served
        if row["pool"] == ood_pool and routed:
            routed_ood += 1
        if served:
            accepted.append(row)

        decisions.append(
            {
                "scenario": scenario,
                "case_id": row["case_id"],
                "pool": row["pool"],
                "explicit_parameter_ood": str(explicit_ood).lower(),
                "score_name": "band_width_mean_norm_90",
                "score": score,
                "threshold_tau": id_tau,
                "decision": decision,
                "served_by_surrogate": str(served).lower(),
                "routed_to_solver": str(routed).lower(),
                "covered_pairs_90": row["covered_pairs_90"] if served else "",
                "total_pairs": row["total_pairs"] if served else "",
                "coverage_90": row["coverage_90"] if served else "",
            }
        )

    covered = sum(int(row["covered_pairs_90"]) for row in accepted)
    total = sum(int(row["total_pairs"]) for row in accepted)
    summary = {
        "scenario": scenario,
        "policy": "route explicit parameter-envelope OOD; otherwise serve ID-like cases only if score <= ID-only tau",
        "threshold_tau": id_tau,
        "accepted_cases": len(accepted),
        "id_total": id_total,
        "ood_total": ood_total,
        "accepted_coverage_90": covered / total if total else float("nan"),
        "id_served_fraction": len(accepted) / id_total if id_total else float("nan"),
        "ood_routed_fraction": routed_ood / ood_total if ood_total else float("nan"),
    }
    return decisions, summary


def single_mixed_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    row = diagnostics["mixed_original"]["best_available"]
    return {
        "scenario": "single_mixed_threshold_original",
        "policy": "single score threshold on ID test + original OOD",
        "threshold_tau": float(row["threshold_tau"]),
        "accepted_cases": int(row["accepted_cases"]),
        "id_total": 100,
        "ood_total": 150,
        "accepted_coverage_90": float(row["empirical_coverage_90"]),
        "id_served_fraction": float(row["id_served_fraction"]),
        "ood_routed_fraction": float(row["ood_routed_fraction"]),
    }


def plot_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [
        "single\nmixed",
        "regime\noriginal",
        "regime\nfresh",
    ]
    metrics = [
        ("accepted_coverage_90", "Accepted coverage", "#2166ac"),
        ("id_served_fraction", "ID served", "#4d9221"),
        ("ood_routed_fraction", "OOD routed", "#b2182b"),
    ]
    x = range(len(rows))
    width = 0.24
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=180, layout="constrained")
    for offset, (key, label, color) in enumerate(metrics):
        positions = [idx + (offset - 1) * width for idx in x]
        ax.bar(positions, [float(row[key]) for row in rows], width=width, color=color, label=label)
    ax.axhline(0.90, color="#52606d", linestyle="--", linewidth=1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction")
    ax.set_title("Regime-Aware Gate vs Single Threshold", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    config = load_config()
    paths = study_paths(config)
    diagnostics = read_json(paths.evidence_pack / "02_metrics" / "gate_diagnostics.json")
    id_tau = float(diagnostics["id_only"]["best_at_target"]["threshold_tau"])
    parameters = load_parameter_rows(paths)
    predictions = load_prediction_rows(paths)

    original_decisions, original_summary = apply_gate(
        predictions,
        parameters,
        config,
        scenario="regime_original_id_plus_ood",
        ood_pool="ood_test",
        id_tau=id_tau,
    )
    fresh_decisions, fresh_summary = apply_gate(
        predictions,
        parameters,
        config,
        scenario="regime_id_plus_fresh_ood",
        ood_pool="ood_fresh",
        id_tau=id_tau,
    )
    summaries = [single_mixed_summary(diagnostics), original_summary, fresh_summary]

    metrics_dir = paths.evidence_pack / "02_metrics"
    figures_dir = paths.evidence_pack / "05_figures"
    decision_path = metrics_dir / "regime_gate_decisions.csv"
    summary_path = metrics_dir / "regime_gate_summary.csv"
    json_path = metrics_dir / "regime_gate_summary.json"
    figure_path = figures_dir / "regime_gate_comparison.png"
    write_csv(decision_path, original_decisions + fresh_decisions, DECISION_FIELDS)
    write_csv(summary_path, summaries, SUMMARY_FIELDS)
    write_json(
        json_path,
        {
            "score_name": "band_width_mean_norm_90",
            "policy": "route explicit parameter-envelope OOD; otherwise serve ID-like cases only if score <= ID-only tau",
            "threshold_source": "ID-only risk coverage curve at >=90% accepted coverage",
            "id_threshold_tau": id_tau,
            "summaries": summaries,
            "artifacts": {
                "decisions": str(decision_path.relative_to(paths.root)),
                "summary_csv": str(summary_path.relative_to(paths.root)),
                "comparison_png": str(figure_path.relative_to(paths.root)),
            },
            "caveat": "This is a deployment rule for known parameter-envelope OOD. It is not a detector for unknown geometry-family shifts.",
        },
    )
    plot_comparison(figure_path, summaries)
    print(f"Wrote {decision_path.relative_to(paths.root)}")
    print(f"Wrote {summary_path.relative_to(paths.root)}")
    print(f"Wrote {json_path.relative_to(paths.root)}")
    print(f"Wrote {figure_path.relative_to(paths.root)}")


if __name__ == "__main__":
    main()
