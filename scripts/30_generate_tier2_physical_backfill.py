#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import (
    PARAMETER_ORDER,
    ensure_evidence_folders,
    load_config,
    nearest_id_distance,
    read_csv,
    study_paths,
    write_csv,
    write_json,
)


CORE_PARAMETER_FIELDS = [
    "case_id",
    "pool",
    *PARAMETER_ORDER,
    "geometry_family",
    "solver_status",
]

BACKFILL_PARAMETER_FIELDS = [
    *CORE_PARAMETER_FIELDS,
    "backfill_round",
    "replacement_for",
    "candidate_index_for_replacement",
    "target_ood_axis",
    "target_distance_bin",
    "target_nearest_id_train_distance",
    "nearest_id_train_distance",
    "temperature_risk_proxy",
    "generation_note",
]

CANONICAL_POOLS = ("id_train", "id_calibration", "id_test", "ood_test")


def resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def base_parameter_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in CORE_PARAMETER_FIELDS}


def as_int(value: str | int | float) -> int:
    return int(float(value))


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    idx = (len(ordered) - 1) * fraction
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def distance_edges(values: list[float], bins: int) -> list[float]:
    if not values:
        return []
    return [quantile(values, idx / bins) for idx in range(bins + 1)]


def distance_bin(distance: float, edges: list[float]) -> int:
    if not edges:
        return 0
    for idx in range(len(edges) - 1):
        upper_inclusive = idx == len(edges) - 2
        if distance >= edges[idx] and (distance <= edges[idx + 1] if upper_inclusive else distance < edges[idx + 1]):
            return idx + 1
    return max(1, len(edges) - 1)


def ood_axis(row: dict[str, str], config: dict[str, Any]) -> str:
    h_ch = float(row["h_ch"])
    d_in = float(row["d_in"])
    params = config["parameters"]
    if h_ch > float(params["h_ch"]["id_max"]):
        return "h_ch_high"
    if d_in < float(params["d_in"]["id_min"]):
        return "d_in_low"
    if d_in > float(params["d_in"]["id_max"]):
        return "d_in_high"
    return "unknown"


def h_fin_for(rng: random.Random, h_ch: float, config: dict[str, Any]) -> float:
    params = config["parameters"]
    geometry = config["geometry"]
    h_fin_lo = float(params["h_fin"]["id_min"])
    h_fin_hi = min(float(params["h_fin"]["id_max"]), float(geometry["max_fin_height_fraction_of_channel"]) * h_ch)
    return rng.uniform(h_fin_lo, h_fin_hi)


def pushed_value(rng: random.Random, lo: float, hi: float, direction: str, min_frac: float, max_frac: float) -> float:
    span = hi - lo
    frac = rng.uniform(min_frac, max_frac)
    if direction == "low":
        return lo - frac * span
    if direction == "high":
        return hi + frac * span
    raise ValueError(direction)


def temperature_risk_proxy(row: dict[str, Any]) -> float:
    n_fin = float(row["n_fin"])
    q_w = float(row["q_w"])
    u_in = max(float(row["u_in"]), 0.05)
    effective_inlet_mm = max(min(float(row["d_in"]), float(row["h_ch"])), 0.2)
    return q_w * (n_fin / 8.0) / (u_in * effective_inlet_mm)


def make_backfill_row(
    *,
    case_id: str,
    pool: str,
    h_ch: float,
    n_fin: int,
    h_fin: float,
    d_in: float,
    u_in: float,
    q_w: float,
    backfill_round: int,
    replacement_for: str,
    candidate_index_for_replacement: int,
    target_ood_axis: str = "",
    target_distance_bin: str = "",
    target_nearest_id_train_distance: str = "",
    nearest_id_train_distance: str = "",
    generation_note: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case_id,
        "pool": pool,
        "h_ch": round(h_ch, 6),
        "n_fin": n_fin,
        "h_fin": round(h_fin, 6),
        "d_in": round(d_in, 6),
        "u_in": round(u_in, 6),
        "q_w": round(q_w, 6),
        "geometry_family": "uniform_fins",
        "solver_status": "pending",
        "backfill_round": backfill_round,
        "replacement_for": replacement_for,
        "candidate_index_for_replacement": candidate_index_for_replacement,
        "target_ood_axis": target_ood_axis,
        "target_distance_bin": target_distance_bin,
        "target_nearest_id_train_distance": target_nearest_id_train_distance,
        "nearest_id_train_distance": nearest_id_train_distance,
        "generation_note": generation_note,
    }
    row["temperature_risk_proxy"] = f"{temperature_risk_proxy(row):.8f}"
    return row


def id_candidate(
    case_id: str,
    original: dict[str, str],
    rng: random.Random,
    config: dict[str, Any],
    backfill_round: int,
    candidate_index_for_replacement: int,
) -> dict[str, Any]:
    params = config["parameters"]
    h_ch = rng.uniform(float(params["h_ch"]["id_min"]), float(params["h_ch"]["id_max"]))
    n_fin = as_int(original["n_fin"])
    h_fin = h_fin_for(rng, h_ch, config)
    d_in = rng.uniform(float(params["d_in"]["id_min"]), float(params["d_in"]["id_max"]))
    u_in = rng.uniform(float(params["u_in"]["id_min"]), float(params["u_in"]["id_max"]))
    q_w = rng.uniform(float(params["q_w"]["id_min"]), float(params["q_w"]["id_max"]))
    return make_backfill_row(
        case_id=case_id,
        pool=original["pool"],
        h_ch=h_ch,
        n_fin=n_fin,
        h_fin=h_fin,
        d_in=d_in,
        u_in=u_in,
        q_w=q_w,
        backfill_round=backfill_round,
        replacement_for=original["case_id"],
        candidate_index_for_replacement=candidate_index_for_replacement,
        generation_note="ID physical-acquisition backfill sampled inside configured ID envelope with original n_fin stratum preserved",
    )


def random_ood_candidate(
    case_id: str,
    original: dict[str, str],
    rng: random.Random,
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    backfill_round: int,
    candidate_index_for_replacement: int,
    target_bin: int,
    target_distance: float,
) -> dict[str, Any]:
    params = config["parameters"]
    ood = config["ood"]
    axis = ood_axis(original, config)
    min_frac = float(ood["push_fraction_min"])
    max_frac = float(ood["push_fraction_max"])
    h_ch = rng.uniform(float(params["h_ch"]["id_min"]), float(params["h_ch"]["id_max"]))
    d_in = rng.uniform(float(params["d_in"]["id_min"]), float(params["d_in"]["id_max"]))
    if axis == "h_ch_high":
        h_ch = pushed_value(rng, float(params["h_ch"]["id_min"]), float(params["h_ch"]["id_max"]), "high", min_frac, max_frac)
    elif axis == "d_in_low":
        d_in = pushed_value(rng, float(params["d_in"]["id_min"]), float(params["d_in"]["id_max"]), "low", min_frac, max_frac)
    elif axis == "d_in_high":
        d_in = pushed_value(rng, float(params["d_in"]["id_min"]), float(params["d_in"]["id_max"]), "high", min_frac, max_frac)
    else:
        raise ValueError(f"Unknown OOD axis for {original['case_id']}: {axis}")
    h_fin = h_fin_for(rng, h_ch, config)
    row = make_backfill_row(
        case_id=case_id,
        pool="ood_test",
        h_ch=h_ch,
        n_fin=as_int(original["n_fin"]),
        h_fin=h_fin,
        d_in=d_in,
        u_in=rng.uniform(float(params["u_in"]["id_min"]), float(params["u_in"]["id_max"])),
        q_w=rng.uniform(float(params["q_w"]["id_min"]), float(params["q_w"]["id_max"])),
        backfill_round=backfill_round,
        replacement_for=original["case_id"],
        candidate_index_for_replacement=candidate_index_for_replacement,
        target_ood_axis=axis,
        target_distance_bin=str(target_bin),
        target_nearest_id_train_distance=f"{target_distance:.8f}",
        generation_note="OOD physical-acquisition backfill matched to dropped case n_fin, OOD axis, and distance quintile",
    )
    row["nearest_id_train_distance"] = f"{nearest_id_distance(row, train_rows, config):.8f}"
    row["temperature_risk_proxy"] = f"{temperature_risk_proxy(row):.8f}"
    return row


def ood_candidate_matching_bin(
    case_id: str,
    original: dict[str, str],
    rng: random.Random,
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    backfill_round: int,
    candidate_index_for_replacement: int,
    target_bin: int,
    target_distance: float,
    edges: list[float],
) -> dict[str, Any]:
    lo = edges[target_bin - 1]
    hi = edges[target_bin]
    best: dict[str, Any] | None = None
    best_gap = math.inf
    for _ in range(5000):
        row = random_ood_candidate(
            case_id,
            original,
            rng,
            config,
            train_rows,
            backfill_round,
            candidate_index_for_replacement,
            target_bin,
            target_distance,
        )
        distance = float(row["nearest_id_train_distance"])
        if distance >= lo and (distance <= hi if target_bin == len(edges) - 1 else distance < hi):
            return row
        gap = min(abs(distance - lo), abs(distance - hi))
        if gap < best_gap:
            best_gap = gap
            best = row
    assert best is not None
    best["generation_note"] = (
        "OOD physical-acquisition backfill matched n_fin and axis; nearest candidate was outside requested distance bin"
    )
    return best


def rows_by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["case_id"]: row for row in rows}


def unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = []
    seen: set[str] = set()
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in seen:
            continue
        seen.add(case_id)
        ordered.append(row)
    return ordered


def target_counts(config: dict[str, Any]) -> dict[str, int]:
    dataset = config["dataset"]
    return {
        "id_train": int(dataset["id_train"]),
        "id_calibration": int(dataset["id_calibration"]),
        "id_test": int(dataset["id_test"]),
        "ood_test": int(dataset["ood_test"]),
    }


def dropped_canonical_acquisition_rows(
    acquisition_rows: list[dict[str, str]],
    convergence_by_case: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    return [
        row
        for row in acquisition_rows
        if row["pool"] in CANONICAL_POOLS and convergence_by_case.get(row["case_id"], {}).get("dropped") == "true"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1, dest="backfill_round")
    parser.add_argument("--id-candidates-per-drop", type=int, default=3)
    parser.add_argument("--ood-candidates-per-drop", type=int, default=4)
    parser.add_argument("--distance-bins", type=int, default=5)
    parser.add_argument("--acquisition-parameter-file", default="evidence_pack_tier2_steady/03_data/physical_acquisition_parameters.csv")
    parser.add_argument("--base-parameter-file", default="evidence_pack_tier2_steady/03_data/parameters.csv")
    parser.add_argument("--base-combined-parameter-file", default="evidence_pack_tier2_steady/03_data/physical_acquisition_all_parameters.csv")
    parser.add_argument("--ood-distance-file", default="evidence_pack_tier2_steady/03_data/ood_distance.csv")
    parser.add_argument("--convergence-log", default="evidence_pack_tier2_steady/03_data/convergence_log.csv")
    parser.add_argument("--previous-backfill-parameter-file", action="append", default=[])
    parser.add_argument("--output-parameter-file", default="")
    parser.add_argument("--output-ood-distance-file", default="")
    parser.add_argument("--output-combined-parameter-file", default="evidence_pack_tier2_steady/03_data/physical_acquisition_with_backfill_parameters.csv")
    parser.add_argument("--summary-file", default="")
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)
    data_dir = paths.evidence_pack / "03_data"
    round_tag = f"round{args.backfill_round:02d}"

    acquisition_path = resolve(paths.root, args.acquisition_parameter_file)
    base_parameter_path = resolve(paths.root, args.base_parameter_file)
    base_combined_path = resolve(paths.root, args.base_combined_parameter_file)
    ood_distance_path = resolve(paths.root, args.ood_distance_file)
    convergence_path = resolve(paths.root, args.convergence_log)
    output_parameter_path = resolve(
        paths.root,
        args.output_parameter_file or str(data_dir / f"physical_backfill_{round_tag}_parameters.csv"),
    )
    output_distance_path = resolve(
        paths.root,
        args.output_ood_distance_file or str(data_dir / f"physical_backfill_{round_tag}_ood_distance.csv"),
    )
    output_combined_path = resolve(paths.root, args.output_combined_parameter_file)
    summary_path = resolve(
        paths.root,
        args.summary_file or str(data_dir / f"physical_backfill_{round_tag}_summary.json"),
    )

    required = [acquisition_path, base_parameter_path, base_combined_path, ood_distance_path, convergence_path]
    missing = [str(path.relative_to(paths.root)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required backfill inputs: {', '.join(missing)}")

    acquisition_rows = read_csv(acquisition_path)
    base_rows = read_csv(base_parameter_path)
    convergence_by_case = rows_by_case(read_csv(convergence_path))
    ood_distance_rows = read_csv(ood_distance_path)
    distances = {row["case_id"]: float(row["nearest_id_train_distance"]) for row in ood_distance_rows}
    edges = distance_edges(list(distances.values()), int(args.distance_bins))
    train_rows = [row for row in base_rows if row["pool"] == "id_train"]
    targets = target_counts(config)

    previous_backfill_paths = [resolve(paths.root, value) for value in args.previous_backfill_parameter_file]
    previous_backfill_rows: list[dict[str, str]] = []
    for path in previous_backfill_paths:
        if path.exists():
            previous_backfill_rows.extend(read_csv(path))

    active_rows = [*acquisition_rows, *previous_backfill_rows]
    active_ids = {row["case_id"] for row in active_rows}
    retained = Counter(
        row["pool"]
        for row in active_rows
        if convergence_by_case.get(row["case_id"], {}).get("converged") == "true"
    )
    dropped_rows = dropped_canonical_acquisition_rows(acquisition_rows, convergence_by_case)
    deficits = {pool: max(0, target - retained.get(pool, 0)) for pool, target in targets.items()}
    dropped_by_pool = Counter(row["pool"] for row in dropped_rows)

    existing_ids = {row["case_id"] for row in read_csv(base_combined_path)}
    existing_ids.update(active_ids)
    existing_ids.update(row["case_id"] for row in previous_backfill_rows)
    rng = random.Random(int(config["study"]["seed"]) + 7000 + args.backfill_round)
    case_counters: dict[str, int] = defaultdict(int)
    candidates: list[dict[str, Any]] = []

    for dropped in dropped_rows:
        pool = dropped["pool"]
        copies = int(args.ood_candidates_per_drop) if pool == "ood_test" else int(args.id_candidates_per_drop)
        if deficits.get(pool, 0) <= 0:
            continue
        for copy_idx in range(copies):
            while True:
                local_index = case_counters[pool]
                case_counters[pool] += 1
                case_id = f"{pool}_pbf{args.backfill_round:02d}_{local_index:04d}"
                if case_id not in existing_ids:
                    existing_ids.add(case_id)
                    break
            if pool == "ood_test":
                target_distance = distances[dropped["case_id"]]
                target_bin = distance_bin(target_distance, edges)
                row = ood_candidate_matching_bin(
                    case_id,
                    dropped,
                    rng,
                    config,
                    train_rows,
                    args.backfill_round,
                    copy_idx,
                    target_bin,
                    target_distance,
                    edges,
                )
            else:
                row = id_candidate(case_id, dropped, rng, config, args.backfill_round, copy_idx)
            candidates.append(row)

    candidates.sort(
        key=lambda row: (
            row["pool"],
            row["replacement_for"],
            float(row["temperature_risk_proxy"]),
            int(row["candidate_index_for_replacement"]),
        )
    )
    base_combined_rows = read_csv(base_combined_path)
    combined_rows = unique_rows([*base_combined_rows, *previous_backfill_rows, *candidates])
    ood_candidates = [row for row in candidates if row["pool"] == "ood_test"]

    write_csv(output_parameter_path, candidates, BACKFILL_PARAMETER_FIELDS)
    write_csv(
        output_distance_path,
        [
            {
                "case_id": row["case_id"],
                "pool": row["pool"],
                "nearest_id_train_distance": row["nearest_id_train_distance"],
                "replacement_for": row["replacement_for"],
                "target_distance_bin": row["target_distance_bin"],
                "target_ood_axis": row["target_ood_axis"],
            }
            for row in ood_candidates
        ],
        ["case_id", "pool", "nearest_id_train_distance", "replacement_for", "target_distance_bin", "target_ood_axis"],
    )
    write_csv(output_combined_path, [base_parameter_row(row) for row in combined_rows], CORE_PARAMETER_FIELDS)
    write_json(
        summary_path,
        {
            "backfill_round": args.backfill_round,
            "seed": int(config["study"]["seed"]) + 7000 + args.backfill_round,
            "targets": targets,
            "active_acquisition_cases": len(active_rows),
            "active_retained_by_pool": dict(sorted(retained.items())),
            "active_dropped_by_pool": dict(sorted(dropped_by_pool.items())),
            "deficits_after_active_rows": deficits,
            "candidate_counts_by_pool": dict(sorted(Counter(row["pool"] for row in candidates).items())),
            "id_candidates_per_drop": int(args.id_candidates_per_drop),
            "ood_candidates_per_drop": int(args.ood_candidates_per_drop),
            "policy": (
                "Generate physical-acquisition replacement candidates for dropped active rows; "
                "ID replacements preserve original n_fin stratum; OOD replacements preserve n_fin, axis, and distance bin."
            ),
            "parameter_file": str(output_parameter_path.relative_to(paths.root)),
            "ood_distance_file": str(output_distance_path.relative_to(paths.root)),
            "combined_parameter_file": str(output_combined_path.relative_to(paths.root)),
            "previous_backfill_parameter_files": [
                str(path.relative_to(paths.root)) for path in previous_backfill_paths if path.exists()
            ],
        },
    )
    print(f"Wrote {len(candidates)} physical backfill candidate(s): {output_parameter_path.relative_to(paths.root)}")
    print(f"Wrote combined acquisition file: {output_combined_path.relative_to(paths.root)}")
    print(f"Deficits: {deficits}")


if __name__ == "__main__":
    main()
