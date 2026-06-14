#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import load_config, read_csv, study_paths, write_csv, write_json


SCORE_NAMES = (
    "band_width_mean_norm_90",
    "risk_score",
    "risk_multiplier",
    "nearest_id_train_distance",
)
SCORE_SUMMARY_FIELDS = [
    "score_name",
    "pool",
    "count",
    "min",
    "p10",
    "median",
    "p90",
    "max",
]
THRESHOLD_FIELDS = [
    "score_name",
    "calibration_pool",
    "id_accept_fraction",
    "threshold",
]
POLICY_FIELDS = [
    "policy",
    "score_name",
    "threshold",
    "selected_pools",
    "total_cases",
    "id_cases",
    "ood_cases",
    "served_cases",
    "routed_cases",
    "id_served_fraction",
    "ood_routed_fraction",
    "served_coverage_90",
]
AUROC_FIELDS = [
    "score_name",
    "id_pool",
    "ood_pools",
    "id_cases",
    "ood_cases",
    "auroc_higher_score_is_more_ood",
]


def resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["case_id"]: row for row in rows}


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


def finite_score(row: dict[str, str], score_name: str) -> float:
    value = float(row[score_name])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {score_name} for {row['case_id']}")
    return value


def quantile_threshold(rows: list[dict[str, str]], pool: str, score_name: str, fraction: float) -> float:
    values = [finite_score(row, score_name) for row in rows if row["pool"] == pool]
    if not values:
        raise ValueError(f"No rows for calibration pool {pool}")
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def score_summary(rows: list[dict[str, str]], score_name: str, pool: str) -> dict[str, Any]:
    values = np.asarray([finite_score(row, score_name) for row in rows if row["pool"] == pool], dtype=np.float64)
    if values.size == 0:
        return {
            "score_name": score_name,
            "pool": pool,
            "count": 0,
            "min": "",
            "p10": "",
            "median": "",
            "p90": "",
            "max": "",
        }
    return {
        "score_name": score_name,
        "pool": pool,
        "count": int(values.size),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def auroc_higher_score_is_more_ood(
    rows: list[dict[str, str]],
    score_name: str,
    id_pool: str,
    ood_pools: set[str],
) -> float:
    id_values = [finite_score(row, score_name) for row in rows if row["pool"] == id_pool]
    ood_values = [finite_score(row, score_name) for row in rows if row["pool"] in ood_pools]
    if not id_values or not ood_values:
        return float("nan")
    wins = 0.0
    total = 0
    for ood in ood_values:
        for item in id_values:
            if ood > item:
                wins += 1.0
            elif ood == item:
                wins += 0.5
            total += 1
    return wins / total


def evaluate_policy(
    rows: list[dict[str, str]],
    parameter_rows: dict[str, dict[str, str]],
    config: dict[str, Any],
    selected_pools: set[str],
    ood_pools: set[str],
    policy: str,
    score_name: str = "",
    threshold: float | None = None,
) -> dict[str, Any]:
    selected = [row for row in rows if row["pool"] in selected_pools]
    served = []
    routed = []
    for row in selected:
        params = parameter_rows[row["case_id"]]
        explicit_ood = explicit_out_of_envelope(params, config)
        if policy == "explicit_envelope":
            serve = not explicit_ood
        elif policy == "learned_score":
            if threshold is None or not score_name:
                raise ValueError("learned_score policy requires score_name and threshold")
            serve = finite_score(row, score_name) <= threshold
        elif policy == "explicit_then_learned_score":
            if threshold is None or not score_name:
                raise ValueError("explicit_then_learned_score policy requires score_name and threshold")
            serve = (not explicit_ood) and finite_score(row, score_name) <= threshold
        else:
            raise ValueError(f"Unknown policy {policy}")
        (served if serve else routed).append(row)

    id_cases = [row for row in selected if row["pool"] not in ood_pools]
    ood_cases = [row for row in selected if row["pool"] in ood_pools]
    id_served = [row for row in served if row["pool"] not in ood_pools]
    ood_routed = [row for row in routed if row["pool"] in ood_pools]
    covered = sum(int(row["covered_pairs_90"]) for row in served)
    total = sum(int(row["total_pairs"]) for row in served)
    return {
        "policy": policy,
        "score_name": score_name,
        "threshold": "" if threshold is None else threshold,
        "selected_pools": ",".join(sorted(selected_pools)),
        "total_cases": len(selected),
        "id_cases": len(id_cases),
        "ood_cases": len(ood_cases),
        "served_cases": len(served),
        "routed_cases": len(routed),
        "id_served_fraction": len(id_served) / len(id_cases) if id_cases else float("nan"),
        "ood_routed_fraction": len(ood_routed) / len(ood_cases) if ood_cases else float("nan"),
        "served_coverage_90": covered / total if total else float("nan"),
    }


def plot_score_histograms(
    rows: list[dict[str, str]],
    thresholds: dict[str, float],
    figures_dir: Path,
) -> None:
    pools = [
        ("id_test", "#2166ac"),
        ("ood_test", "#b2182b"),
        ("ood_fresh", "#4d9221"),
    ]
    for score_name in ("band_width_mean_norm_90", "risk_multiplier", "nearest_id_train_distance"):
        values_by_pool = {
            pool: np.asarray([finite_score(row, score_name) for row in rows if row["pool"] == pool], dtype=np.float64)
            for pool, _ in pools
        }
        values = np.concatenate([value for value in values_by_pool.values() if value.size])
        upper = float(np.quantile(values, 0.98))
        bins = np.linspace(float(values.min()), upper, 36)
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
        for pool, color in pools:
            if values_by_pool[pool].size:
                ax.hist(
                    values_by_pool[pool],
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=2.0,
                    color=color,
                    label=pool,
                )
        ax.axvline(thresholds[score_name], color="#7f3b08", linestyle="--", linewidth=1.3, label="ID-cal p90")
        ax.set_xlabel(score_name)
        ax.set_ylabel("Density")
        ax.set_title(f"Tier 2 Gate Score: {score_name}", loc="left", fontweight="bold")
        ax.grid(color="#d9e2ec", linewidth=0.8)
        ax.legend(frameon=False)
        fig.savefig(figures_dir / f"{score_name}_distribution.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-index",
        default="evidence_pack_tier2_steady/04_predictions/tier2_physical_operator/prediction_index.csv",
    )
    parser.add_argument(
        "--parameter-file",
        default="evidence_pack_tier2_steady/03_data/physical_acquisition_retained_parameters.csv",
    )
    parser.add_argument("--metrics-dir", default="evidence_pack_tier2_steady/02_metrics/tier2_gate_diagnostics")
    parser.add_argument("--figures-dir", default="evidence_pack_tier2_steady/05_figures/tier2_gate_diagnostics")
    parser.add_argument("--id-accept-fraction", type=float, default=0.90)
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    prediction_path = resolve(paths.root, args.prediction_index)
    parameter_path = resolve(paths.root, args.parameter_file)
    metrics_dir = resolve(paths.root, args.metrics_dir)
    figures_dir = resolve(paths.root, args.figures_dir)

    rows = read_csv(prediction_path)
    parameters = by_case(read_csv(parameter_path))
    pools = sorted({row["pool"] for row in rows})
    ood_pools = {"ood_test", "ood_fresh"}
    selected_pools = {"id_test", *ood_pools}
    thresholds = {
        score_name: quantile_threshold(rows, "id_calibration", score_name, float(args.id_accept_fraction))
        for score_name in SCORE_NAMES
    }

    score_summary_rows = [score_summary(rows, score_name, pool) for score_name in SCORE_NAMES for pool in pools]
    threshold_rows = [
        {
            "score_name": score_name,
            "calibration_pool": "id_calibration",
            "id_accept_fraction": float(args.id_accept_fraction),
            "threshold": threshold,
        }
        for score_name, threshold in thresholds.items()
    ]
    policy_rows = [
        evaluate_policy(rows, parameters, config, selected_pools, ood_pools, "explicit_envelope"),
    ]
    for score_name, threshold in thresholds.items():
        policy_rows.append(
            evaluate_policy(
                rows,
                parameters,
                config,
                selected_pools,
                ood_pools,
                "learned_score",
                score_name,
                threshold,
            )
        )
        policy_rows.append(
            evaluate_policy(
                rows,
                parameters,
                config,
                selected_pools,
                ood_pools,
                "explicit_then_learned_score",
                score_name,
                threshold,
            )
        )

    auc_rows = []
    for score_name in SCORE_NAMES:
        for ood_group_name, group in (
            ("ood_test", {"ood_test"}),
            ("ood_fresh", {"ood_fresh"}),
            ("ood_test,ood_fresh", ood_pools),
        ):
            auc_rows.append(
                {
                    "score_name": score_name,
                    "id_pool": "id_test",
                    "ood_pools": ood_group_name,
                    "id_cases": sum(row["pool"] == "id_test" for row in rows),
                    "ood_cases": sum(row["pool"] in group for row in rows),
                    "auroc_higher_score_is_more_ood": auroc_higher_score_is_more_ood(rows, score_name, "id_test", group),
                }
            )

    explicit_counts = Counter()
    for row in rows:
        explicit_counts[row["pool"]] += int(explicit_out_of_envelope(parameters[row["case_id"]], config))

    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_dir / "tier2_gate_score_summary.csv", score_summary_rows, SCORE_SUMMARY_FIELDS)
    write_csv(metrics_dir / "tier2_gate_thresholds.csv", threshold_rows, THRESHOLD_FIELDS)
    write_csv(metrics_dir / "tier2_gate_policy_summary.csv", policy_rows, POLICY_FIELDS)
    write_csv(metrics_dir / "tier2_gate_score_auroc.csv", auc_rows, AUROC_FIELDS)
    plot_score_histograms(rows, thresholds, figures_dir)
    write_json(
        metrics_dir / "tier2_gate_diagnostics.json",
        {
            "prediction_index": str(prediction_path.relative_to(paths.root)),
            "parameter_file": str(parameter_path.relative_to(paths.root)),
            "case_counts_by_pool": dict(sorted(Counter(row["pool"] for row in rows).items())),
            "explicit_out_of_envelope_counts_by_pool": dict(sorted(explicit_counts.items())),
            "id_accept_fraction_for_learned_thresholds": float(args.id_accept_fraction),
            "thresholds_from_id_calibration": dict(sorted(thresholds.items())),
            "policies": policy_rows,
            "auroc": auc_rows,
            "interpretation": (
                "Explicit envelope gating is the honest baseline for this dataset because OOD pools are constructed "
                "by known parameter-envelope shifts. Learned score routing remains experimental and should not be "
                "presented as a solved hidden-OOD detector."
            ),
        },
    )
    print(f"Wrote {(metrics_dir / 'tier2_gate_diagnostics.json').relative_to(paths.root)}")
    print(f"Wrote {(metrics_dir / 'tier2_gate_policy_summary.csv').relative_to(paths.root)}")
    print(f"Wrote {(metrics_dir / 'tier2_gate_score_auroc.csv').relative_to(paths.root)}")


if __name__ == "__main__":
    main()
