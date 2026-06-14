#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import (
    configured_multipass_instance,
    ensure_evidence_folders,
    load_config,
    read_csv,
    study_paths,
    write_csv,
    write_json,
)


CONVERGENCE_FIELDS = [
    "case_id",
    "pool",
    "case_dir",
    "solver",
    "mesh_cells",
    "residual_U_initial",
    "residual_U_final",
    "residual_p_initial",
    "residual_p_final",
    "residual_T_initial",
    "residual_T_final",
    "flow_converged_reported",
    "iterations_flow",
    "iterations_temperature",
    "scalar_stability_passed",
    "scalar_mean_delta_T_K",
    "scalar_max_delta_T_K",
    "scalar_stability_window_s",
    "scalar_stability_reason",
    "raw_enthalpy_passed",
    "raw_enthalpy_outlet_delta_T_K",
    "raw_enthalpy_expected_delta_T_K",
    "raw_enthalpy_balance_ratio",
    "raw_enthalpy_mass_balance_rel_error",
    "raw_enthalpy_reason",
    "raw_temperature_extrema_passed",
    "raw_temperature_min_T_K",
    "raw_temperature_mean_T_K",
    "raw_temperature_max_T_K",
    "raw_temperature_reason",
    "wall_time_sec",
    "converged",
    "dropped",
    "drop_reason",
    "openfoam_version",
]

OUTPUT_MANIFEST_FIELDS = [
    "case_id",
    "pool",
    "archive_path",
    "flow_time",
    "temperature_time",
    "archive_size_bytes",
    "remote_case_dir",
]


def run(command: list[str], cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and result.returncode != 0:
        raise SystemExit(result.stdout)
    return result


def find_case_row(case_id: str, rows: list[dict[str, str]], parameter_path: Path) -> dict[str, str]:
    for row in rows:
        if row["case_id"] == case_id:
            return row
    raise SystemExit(f"Unknown case_id in {parameter_path}: {case_id}")


def latest_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    return float(value)


def latest_int(pattern: str, text: str) -> int | None:
    value = latest_float(pattern, text)
    return None if value is None else int(value)


def latest_pair(pattern: str, text: str) -> tuple[float, float] | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    initial, final = matches[-1]
    return float(initial), float(final)


def split_solver_phases(log_text: str) -> tuple[str, str]:
    match = re.search(r"(?m)^Exec\s+:\s+thermalEnergyFoam\b", log_text)
    if match:
        return log_text[: match.start()], log_text[match.start() :]
    marker = log_text.find("Solving steady forced-convection energy equation")
    if marker >= 0:
        return log_text[:marker], log_text[marker:]
    return log_text, ""


def parse_run(
    case_id: str,
    pool: str,
    case_dir: str,
    solver_label: str,
    log_text: str,
    mesh_text: str,
    wall_time: float,
    openfoam_version: str | None,
    threshold: float,
    thermal_stability: dict[str, Any] | None = None,
    raw_enthalpy: dict[str, Any] | None = None,
    raw_temperature_extrema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    flow_text, thermal_text = split_solver_phases(log_text)
    thermal_residual_text = thermal_text or log_text
    u_residual_pairs = [
        (float(initial), float(final))
        for initial, final in re.findall(
            r"Solving for U[xyz], Initial residual = ([0-9.eE+-]+), Final residual = ([0-9.eE+-]+)",
            flow_text,
        )
    ]
    residual_u_initial = max(pair[0] for pair in u_residual_pairs[-3:]) if len(u_residual_pairs) >= 3 else None
    residual_u_final = max(pair[1] for pair in u_residual_pairs[-3:]) if len(u_residual_pairs) >= 3 else None
    residual_p_pair = latest_pair(
        r"Solving for p, Initial residual = ([0-9.eE+-]+), Final residual = ([0-9.eE+-]+)",
        flow_text,
    )
    residual_p_initial = None if residual_p_pair is None else residual_p_pair[0]
    residual_p_final = None if residual_p_pair is None else residual_p_pair[1]
    residual_t_pair = latest_pair(
        r"Solving for T, Initial residual = ([0-9.eE+-]+), Final residual = ([0-9.eE+-]+)",
        thermal_residual_text,
    )
    residual_t_initial = None if residual_t_pair is None else residual_t_pair[0]
    residual_t_final = None if residual_t_pair is None else residual_t_pair[1]
    mesh_cells = latest_int(r"^\s*cells:\s+([0-9]+)", mesh_text)
    flow_converged_reported = re.search(r"(?:SIMPLE|PIMPLE) solution converged in ([0-9]+) iterations", flow_text) is not None
    iterations_flow = latest_int(r"(?:SIMPLE|PIMPLE) solution converged in ([0-9]+) iterations", flow_text)
    if iterations_flow is None:
        iterations_flow = latest_int(r"^Time = ([0-9]+)s$", flow_text)
    iterations_temperature = len(re.findall(r"Solving for T,", thermal_residual_text))
    status_ok = "End" in flow_text and (not thermal_text or "End" in thermal_text)
    flow_residuals_ok = all(
        value is not None and value <= threshold
        for value in (
            residual_u_initial,
            residual_u_final,
            residual_p_initial,
            residual_p_final,
        )
    )
    thermal_residual_ok = residual_t_final is not None and residual_t_final <= threshold
    scalar_stability_ok = True if thermal_stability is None else bool(thermal_stability.get("passed"))
    raw_enthalpy_ok = True if raw_enthalpy is None else bool(raw_enthalpy.get("passed"))
    raw_temperature_extrema_ok = True if raw_temperature_extrema is None else bool(raw_temperature_extrema.get("passed"))
    converged = bool(
        status_ok
        and flow_converged_reported
        and flow_residuals_ok
        and thermal_residual_ok
        and iterations_flow is not None
        and iterations_temperature > 0
        and scalar_stability_ok
        and raw_enthalpy_ok
        and raw_temperature_extrema_ok
    )
    drop_reasons = []
    if not converged:
        if not status_ok:
            drop_reasons.append("OpenFOAM run did not reach End")
        if not flow_converged_reported:
            drop_reasons.append("flow solver did not report convergence")
        if not flow_residuals_ok:
            drop_reasons.append("flow residuals did not meet convergence contract")
        if not thermal_residual_ok:
            drop_reasons.append("temperature solver residual did not meet convergence contract")
        if iterations_flow is None or iterations_temperature <= 0:
            drop_reasons.append("solver did not complete required flow/temperature phases")
        if thermal_stability is not None and not scalar_stability_ok:
            drop_reasons.append("scalar field did not meet steady stability contract")
        if raw_enthalpy is not None and not raw_enthalpy_ok:
            drop_reasons.append("raw outlet enthalpy check failed")
        if raw_temperature_extrema is not None and not raw_temperature_extrema_ok:
            drop_reasons.append("raw temperature extrema check failed")
    drop_reason = "; ".join(dict.fromkeys(drop_reasons))
    return {
        "case_id": case_id,
        "pool": pool,
        "case_dir": case_dir,
        "solver": solver_label,
        "mesh_cells": "" if mesh_cells is None else mesh_cells,
        "residual_U_initial": "" if residual_u_initial is None else f"{residual_u_initial:.12g}",
        "residual_U_final": "" if residual_u_final is None else f"{residual_u_final:.12g}",
        "residual_p_initial": "" if residual_p_initial is None else f"{residual_p_initial:.12g}",
        "residual_p_final": "" if residual_p_final is None else f"{residual_p_final:.12g}",
        "residual_T_initial": "" if residual_t_initial is None else f"{residual_t_initial:.12g}",
        "residual_T_final": "" if residual_t_final is None else f"{residual_t_final:.12g}",
        "flow_converged_reported": str(flow_converged_reported).lower(),
        "iterations_flow": "" if iterations_flow is None else iterations_flow,
        "iterations_temperature": iterations_temperature,
        "scalar_stability_passed": "" if thermal_stability is None else str(scalar_stability_ok).lower(),
        "scalar_mean_delta_T_K": "" if thermal_stability is None else thermal_stability.get("mean_delta_T_K", ""),
        "scalar_max_delta_T_K": "" if thermal_stability is None else thermal_stability.get("max_delta_T_K", ""),
        "scalar_stability_window_s": "" if thermal_stability is None else thermal_stability.get("window_s", ""),
        "scalar_stability_reason": "" if thermal_stability is None else thermal_stability.get("reason", ""),
        "raw_enthalpy_passed": "" if raw_enthalpy is None else str(raw_enthalpy_ok).lower(),
        "raw_enthalpy_outlet_delta_T_K": "" if raw_enthalpy is None else raw_enthalpy.get("outlet_bulk_delta_K", ""),
        "raw_enthalpy_expected_delta_T_K": "" if raw_enthalpy is None else raw_enthalpy.get("expected_delta_K", ""),
        "raw_enthalpy_balance_ratio": "" if raw_enthalpy is None else raw_enthalpy.get("balance_ratio", ""),
        "raw_enthalpy_mass_balance_rel_error": "" if raw_enthalpy is None else raw_enthalpy.get("mass_balance_relative_error", ""),
        "raw_enthalpy_reason": "" if raw_enthalpy is None else raw_enthalpy.get("reason", ""),
        "raw_temperature_extrema_passed": "" if raw_temperature_extrema is None else str(raw_temperature_extrema_ok).lower(),
        "raw_temperature_min_T_K": "" if raw_temperature_extrema is None else raw_temperature_extrema.get("min_T_K", ""),
        "raw_temperature_mean_T_K": "" if raw_temperature_extrema is None else raw_temperature_extrema.get("mean_T_K", ""),
        "raw_temperature_max_T_K": "" if raw_temperature_extrema is None else raw_temperature_extrema.get("max_T_K", ""),
        "raw_temperature_reason": "" if raw_temperature_extrema is None else raw_temperature_extrema.get("reason", ""),
        "wall_time_sec": f"{wall_time:.3f}",
        "converged": str(converged).lower(),
        "dropped": "false" if converged else "true",
        "drop_reason": drop_reason,
        "openfoam_version": openfoam_version or "",
    }


def update_convergence_log(path: Path, row: dict[str, Any]) -> None:
    rows = read_csv(path) if path.exists() else []
    kept = [existing for existing in rows if existing["case_id"] != row["case_id"]]
    kept.append(row)
    kept.sort(key=lambda item: item["case_id"])
    write_csv(path, kept, CONVERGENCE_FIELDS)


def update_output_manifest(path: Path, row: dict[str, Any]) -> None:
    rows = read_csv(path) if path.exists() else []
    kept = [existing for existing in rows if existing["case_id"] != row["case_id"]]
    kept.append(row)
    kept.sort(key=lambda item: item["case_id"])
    write_csv(path, kept, OUTPUT_MANIFEST_FIELDS)


def ensure_thermal_energy_solver(instance: str, vm_workdir: str, bashrc: str) -> None:
    source_dir = ROOT / "openfoam_solvers" / "thermalEnergyFoam"
    if not source_dir.exists():
        raise SystemExit(f"Missing local thermalEnergyFoam source: {source_dir}")
    remote_parent = f"{vm_workdir}/openfoam_solvers"
    remote_solver_dir = f"{remote_parent}/thermalEnergyFoam"
    run(["multipass", "exec", instance, "--", "bash", "-lc", f"mkdir -p {remote_parent} && rm -rf {remote_solver_dir}"])
    run(["multipass", "transfer", "-r", str(source_dir), f"{instance}:{remote_parent}/"])
    build_command = f"""
set -e
source {bashrc} >/tmp/openfoam_build_source.log 2>&1 || true
cd {remote_solver_dir}
wmake > wmake.log 2>&1
"""
    run(["multipass", "exec", instance, "--", "bash", "-lc", build_command])


def archive_remote_output(
    instance: str,
    remote_case_dir: str,
    case_id: str,
    pool: str,
    flow_time: str,
    local_archive: Path,
) -> dict[str, Any]:
    local_archive.parent.mkdir(parents=True, exist_ok=True)
    remote_archive = f"/tmp/{case_id}_solver_output.tar.gz"
    command = f"""
set -eu
cd {remote_case_dir}
temperature_time="$(cat log.latestTime | tail -n 1)"
rm -rf /tmp/{case_id}_solver_output
mkdir -p /tmp/{case_id}_solver_output/{case_id}/constant
mkdir -p /tmp/{case_id}_solver_output/{case_id}/flow
mkdir -p /tmp/{case_id}_solver_output/{case_id}/temperature
cp case.json /tmp/{case_id}_solver_output/{case_id}/case.json
cp -r constant/polyMesh /tmp/{case_id}_solver_output/{case_id}/constant/polyMesh
cp "{flow_time}/U" /tmp/{case_id}_solver_output/{case_id}/flow/U
cp "{flow_time}/p" /tmp/{case_id}_solver_output/{case_id}/flow/p
cp "{flow_time}/phi" /tmp/{case_id}_solver_output/{case_id}/flow/phi
cp "$temperature_time/T" /tmp/{case_id}_solver_output/{case_id}/temperature/T
if [ -f postProcessing/thermal_stability.json ]; then
    cp postProcessing/thermal_stability.json /tmp/{case_id}_solver_output/{case_id}/temperature/thermal_stability.json
fi
if [ -f postProcessing/raw_outlet_enthalpy.json ]; then
    cp postProcessing/raw_outlet_enthalpy.json /tmp/{case_id}_solver_output/{case_id}/temperature/raw_outlet_enthalpy.json
fi
if [ -f postProcessing/raw_temperature_extrema.json ]; then
    cp postProcessing/raw_temperature_extrema.json /tmp/{case_id}_solver_output/{case_id}/temperature/raw_temperature_extrema.json
fi
cat > /tmp/{case_id}_solver_output/{case_id}/solver_output.json <<EOF
{{"case_id":"{case_id}","flow_time":"{flow_time}","temperature_time":"$temperature_time","source_case_dir":"{remote_case_dir}"}}
EOF
tar -C /tmp/{case_id}_solver_output -czf {remote_archive} {case_id}
echo "$temperature_time"
"""
    result = run(["multipass", "exec", instance, "--", "bash", "-lc", command])
    temperature_time = result.stdout.strip().splitlines()[-1]
    run(["multipass", "transfer", f"{instance}:{remote_archive}", str(local_archive)])
    size_bytes = local_archive.stat().st_size
    return {
        "case_id": case_id,
        "pool": pool,
        "archive_path": str(local_archive.relative_to(ROOT)),
        "flow_time": flow_time,
        "temperature_time": temperature_time,
        "archive_size_bytes": size_bytes,
        "remote_case_dir": remote_case_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--force", action="store_true", help="Replace the remote VM case directory before running.")
    parser.add_argument("--archive-output", action="store_true", help="Copy a minimal solved-output archive back to the evidence pack.")
    parser.add_argument("--cleanup-remote", action="store_true", help="Remove the remote VM case folder after logs/output are captured.")
    parser.add_argument(
        "--parameter-file",
        default="evidence_pack/03_data/parameters.csv",
        help="CSV of case parameters, relative to repo root unless absolute.",
    )
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    ensure_evidence_folders(paths)
    solver = config["solver"]
    instance, instance_source = configured_multipass_instance(config)
    if not instance:
        raise SystemExit("OpenFOAM Multipass instance is not configured.")
    bashrc = solver["openfoam_bashrc"]
    vm_workdir = solver["vm_workdir"].rstrip("/")
    threshold = float(solver["convergence_residual_target"])
    thermal_solver_mode = str(solver.get("thermal_solver_mode", "passive_scalar_function"))
    solver_label = (
        "simpleFoam+thermalEnergyFoam"
        if thermal_solver_mode == "dedicated_energy_foam"
        else "simpleFoam+foamRun(functions:scalarTransport)"
    )

    parameter_path = Path(args.parameter_file)
    if not parameter_path.is_absolute():
        parameter_path = paths.root / parameter_path
    parameter_rows = read_csv(parameter_path)
    case_row = find_case_row(args.case_id, parameter_rows, parameter_path)
    local_case_dir = paths.root / "openfoam_cases" / args.case_id
    if not local_case_dir.exists():
        raise SystemExit(f"Missing local case directory: {local_case_dir}. Run scripts/02_create_openfoam_cases.py first.")

    remote_parent = f"{vm_workdir}/openfoam_cases"
    remote_case_dir = f"{remote_parent}/{args.case_id}"
    if thermal_solver_mode == "dedicated_energy_foam":
        ensure_thermal_energy_solver(instance, vm_workdir, bashrc)
    run(["multipass", "exec", instance, "--", "bash", "-lc", f"mkdir -p {remote_parent}"])
    if args.force:
        run(["multipass", "exec", instance, "--", "bash", "-lc", f"rm -rf {remote_case_dir}"])
    run(["multipass", "transfer", "-r", str(local_case_dir), f"{instance}:{remote_parent}/"])

    command = f"""
source {bashrc} >/tmp/openfoam_source.log 2>&1 || true
cd {remote_case_dir}
./Allrun > log.Allrun 2>&1
run_status=$?
checkMesh > log.checkMesh.final 2>&1 || true
foamListTimes -latestTime > log.latestTime 2>&1 || true
foamVersion > log.foamVersion 2>&1 || true
exit $run_status
"""
    started = time.monotonic()
    result = run(["multipass", "exec", instance, "--", "bash", "-lc", command], check=False)
    wall_time = time.monotonic() - started

    log_dir = paths.evidence_pack / "03_data" / "solver_logs" / args.case_id
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "log.Allrun",
        "log.checkMesh.final",
        "log.latestTime",
        "log.foamVersion",
        "log.rawOutletEnthalpy",
        "log.rawTemperatureExtrema",
        "log.thermalStability",
    ):
        run(["multipass", "transfer", f"{instance}:{remote_case_dir}/{name}", str(log_dir / name)], check=False)
    run(
        [
            "multipass",
            "transfer",
            f"{instance}:{remote_case_dir}/postProcessing/thermal_stability.json",
            str(log_dir / "thermal_stability.json"),
        ],
        check=False,
    )
    run(
        [
            "multipass",
            "transfer",
            f"{instance}:{remote_case_dir}/postProcessing/raw_outlet_enthalpy.json",
            str(log_dir / "raw_outlet_enthalpy.json"),
        ],
        check=False,
    )
    run(
        [
            "multipass",
            "transfer",
            f"{instance}:{remote_case_dir}/postProcessing/raw_temperature_extrema.json",
            str(log_dir / "raw_temperature_extrema.json"),
        ],
        check=False,
    )

    log_text = (log_dir / "log.Allrun").read_text(encoding="utf-8", errors="replace")
    mesh_text = (log_dir / "log.checkMesh.final").read_text(encoding="utf-8", errors="replace") if (log_dir / "log.checkMesh.final").exists() else ""
    openfoam_version = (log_dir / "log.foamVersion").read_text(encoding="utf-8", errors="replace").strip() if (log_dir / "log.foamVersion").exists() else None
    thermal_stability = None
    thermal_stability_path = log_dir / "thermal_stability.json"
    if thermal_stability_path.exists():
        thermal_stability = json.loads(thermal_stability_path.read_text(encoding="utf-8"))
    raw_enthalpy = None
    raw_enthalpy_path = log_dir / "raw_outlet_enthalpy.json"
    if raw_enthalpy_path.exists():
        raw_enthalpy = json.loads(raw_enthalpy_path.read_text(encoding="utf-8"))
    raw_temperature_extrema = None
    raw_temperature_extrema_path = log_dir / "raw_temperature_extrema.json"
    if raw_temperature_extrema_path.exists():
        raw_temperature_extrema = json.loads(raw_temperature_extrema_path.read_text(encoding="utf-8"))
    convergence_row = parse_run(
        args.case_id,
        case_row["pool"],
        f"openfoam_cases/{args.case_id}",
        solver_label,
        log_text,
        mesh_text,
        wall_time,
        openfoam_version,
        threshold,
        thermal_stability,
        raw_enthalpy,
        raw_temperature_extrema,
    )
    if result.returncode != 0:
        convergence_row["converged"] = "false"
        convergence_row["dropped"] = "true"
        if not convergence_row.get("drop_reason"):
            convergence_row["drop_reason"] = "OpenFOAM run exited nonzero"

    update_convergence_log(paths.evidence_pack / "03_data" / "convergence_log.csv", convergence_row)
    archive_row = None
    if args.archive_output and convergence_row["converged"] == "true":
        archive_row = archive_remote_output(
            instance=instance,
            remote_case_dir=remote_case_dir,
            case_id=args.case_id,
            pool=case_row["pool"],
            flow_time=str(convergence_row["iterations_flow"]),
            local_archive=paths.evidence_pack / "03_data" / "solver_outputs" / f"{args.case_id}.tar.gz",
        )
        update_output_manifest(paths.evidence_pack / "03_data" / "solver_output_manifest.csv", archive_row)

    if args.cleanup_remote:
        run(["multipass", "exec", instance, "--", "bash", "-lc", f"rm -rf {remote_case_dir}"], check=False)

    write_json(
        paths.evidence_pack / "03_data" / "last_vm_run.json",
        {
            "case_id": args.case_id,
            "multipass_instance": instance,
            "multipass_instance_source": instance_source,
            "remote_case_dir": remote_case_dir,
            "returncode": result.returncode,
            "convergence_row": convergence_row,
            "archive_row": archive_row,
        },
    )
    print(f"VM run complete for {args.case_id}: converged={convergence_row['converged']}")
    if convergence_row["converged"] != "true":
        raise SystemExit(2)
    if result.returncode != 0:
        print(result.stdout)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
