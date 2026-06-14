#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import ensure_evidence_folders, load_config, read_csv, study_paths, write_json


def read_existing(path: Path, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def run_case(case_id: str, parameter_file: Path, archive_output: bool, cleanup_remote: bool) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/03_run_openfoam_case_vm.py",
        "--case-id",
        case_id,
        "--force",
        "--parameter-file",
        str(parameter_file),
    ]
    if archive_output:
        command.append("--archive-output")
    if cleanup_remote:
        command.append("--cleanup-remote")
    return subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def selected_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.pool:
        rows = [row for row in rows if row["pool"] == args.pool]
    if args.case_id:
        rows = [row for row in rows if row["case_id"] == args.case_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def retained_by_pool(
    convergence_path: Path,
    output_manifest_path: Path,
    root: Path,
    allowed_ids: set[str] | None = None,
    require_archive: bool = False,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not convergence_path.exists():
        return counts
    outputs = read_existing(output_manifest_path, "case_id") if require_archive else {}
    for row in read_existing(convergence_path, "case_id").values():
        if allowed_ids is not None and row["case_id"] not in allowed_ids:
            continue
        if row.get("converged") == "true":
            if require_archive:
                output = outputs.get(row["case_id"])
                if output is None or not (root / output["archive_path"]).exists():
                    continue
            counts[row["pool"]] = counts.get(row["pool"], 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool")
    parser.add_argument("--case-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rerun-converged", action="store_true")
    parser.add_argument("--rerun-dropped", action="store_true")
    parser.add_argument("--archive-output", action="store_true", default=True)
    parser.add_argument("--no-archive-output", action="store_false", dest="archive_output")
    parser.add_argument("--cleanup-remote", action="store_true", default=True)
    parser.add_argument("--no-cleanup-remote", action="store_false", dest="cleanup_remote")
    parser.add_argument(
        "--parameter-file",
        default="evidence_pack/03_data/parameters.csv",
        help="CSV of case parameters, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--status-file",
        default="evidence_pack/03_data/batch_status.json",
        help="Batch status JSON path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--stop-at-config-targets",
        action="store_true",
        help="Stop once each pool has reached the configured retained count, counting existing convergence_log rows.",
    )
    parser.add_argument(
        "--target-retained-pool",
        help="Optional pool to stop on once --target-retained-count converged cases exist.",
    )
    parser.add_argument(
        "--target-retained-count",
        type=int,
        help="Optional retained-count target for --target-retained-pool.",
    )
    parser.add_argument(
        "--allow-drops",
        action="store_true",
        help="Finish with exit code 0 even if selected solver cases drop. Tooling failures are still logged.",
    )
    args = parser.parse_args()
    if (args.target_retained_pool is None) != (args.target_retained_count is None):
        raise SystemExit("--target-retained-pool and --target-retained-count must be provided together.")

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)
    parameter_path = Path(args.parameter_file)
    if not parameter_path.is_absolute():
        parameter_path = paths.root / parameter_path
    status_path = Path(args.status_file)
    if not status_path.is_absolute():
        status_path = paths.root / status_path
    rows = selected_rows(read_csv(parameter_path), args)
    convergence_path = paths.evidence_pack / "03_data" / "convergence_log.csv"
    output_manifest_path = paths.evidence_pack / "03_data" / "solver_output_manifest.csv"
    selected_case_ids = {row["case_id"] for row in rows}
    targets = {
        "id_train": int(config["dataset"]["id_train"]),
        "id_calibration": int(config["dataset"]["id_calibration"]),
        "id_test": int(config["dataset"]["id_test"]),
        "ood_test": int(config["dataset"]["ood_test"]),
    }

    started = time.time()
    failures: list[dict[str, Any]] = []
    attempted = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        case_id = row["case_id"]
        if args.target_retained_pool:
            current_counts = retained_by_pool(
                convergence_path,
                output_manifest_path,
                paths.root,
                selected_case_ids,
                require_archive=args.archive_output,
            )
            if current_counts.get(args.target_retained_pool, 0) >= int(args.target_retained_count):
                print(
                    f"[{index}/{len(rows)}] STOP {args.target_retained_pool} retained target reached",
                    flush=True,
                )
                break
        if args.stop_at_config_targets:
            current_counts = retained_by_pool(
                convergence_path,
                output_manifest_path,
                paths.root,
                selected_case_ids,
                require_archive=args.archive_output,
            )
            if all(current_counts.get(pool, 0) >= target for pool, target in targets.items()):
                print(f"[{index}/{len(rows)}] STOP configured retained targets reached", flush=True)
                break
            if row["pool"] in targets and current_counts.get(row["pool"], 0) >= targets[row["pool"]]:
                skipped += 1
                print(f"[{index}/{len(rows)}] SKIP {case_id} pool target already reached", flush=True)
                continue
        convergence = read_existing(convergence_path, "case_id")
        outputs = read_existing(output_manifest_path, "case_id")
        archive_exists = False
        if case_id in outputs:
            archive_exists = (paths.root / outputs[case_id]["archive_path"]).exists()
        already_done = convergence.get(case_id, {}).get("converged") == "true" and (archive_exists or not args.archive_output)
        if already_done and not args.rerun_converged:
            skipped += 1
            print(f"[{index}/{len(rows)}] SKIP {case_id} already converged with archive", flush=True)
            continue
        if convergence.get(case_id, {}).get("dropped") == "true" and not args.rerun_dropped:
            skipped += 1
            print(f"[{index}/{len(rows)}] SKIP {case_id} already dropped", flush=True)
            continue

        attempted += 1
        print(f"[{index}/{len(rows)}] RUN {case_id}", flush=True)
        result = run_case(case_id, parameter_path, args.archive_output, args.cleanup_remote)
        print(result.stdout, flush=True)
        if result.returncode != 0:
            failures.append({"case_id": case_id, "returncode": result.returncode, "output_tail": result.stdout[-4000:]})
            print(f"[{index}/{len(rows)}] FAIL {case_id}", flush=True)
        else:
            current_convergence = read_existing(convergence_path, "case_id")
            if current_convergence.get(case_id, {}).get("converged") != "true":
                failures.append({"case_id": case_id, "returncode": 0, "output_tail": "completed but converged=false"})
                print(f"[{index}/{len(rows)}] DROPPED {case_id}", flush=True)
            else:
                print(f"[{index}/{len(rows)}] OK {case_id}", flush=True)

        write_json(
            status_path,
            {
                "selected_cases": len(rows),
                "attempted": attempted,
                "skipped": skipped,
                "failures": failures,
                "elapsed_sec": round(time.time() - started, 3),
                "last_case_id": case_id,
                "parameter_file": str(parameter_path.relative_to(paths.root)),
                "targets": targets if args.stop_at_config_targets else None,
                "target_retained_pool": args.target_retained_pool,
                "target_retained_count": args.target_retained_count,
                "allow_drops": args.allow_drops,
                "current_converged_by_pool": retained_by_pool(
                    convergence_path,
                    output_manifest_path,
                    paths.root,
                    selected_case_ids,
                    require_archive=args.archive_output,
                ),
                "retained_count_scope": "selected parameter file; archive required when --archive-output is active",
            },
        )

    final_counts = retained_by_pool(
        convergence_path,
        output_manifest_path,
        paths.root,
        selected_case_ids,
        require_archive=args.archive_output,
    )
    targets_met = all(final_counts.get(pool, 0) >= target for pool, target in targets.items())
    target_met = (
        args.target_retained_pool is not None
        and final_counts.get(args.target_retained_pool, 0) >= int(args.target_retained_count)
    )
    if failures and not args.allow_drops and not (args.stop_at_config_targets and targets_met) and not target_met:
        raise SystemExit(f"{len(failures)} case(s) failed or were dropped. See evidence_pack/03_data/batch_status.json")
    print(f"Batch complete: selected={len(rows)} attempted={attempted} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
