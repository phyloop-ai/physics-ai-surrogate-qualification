from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "study.toml"
CONFIG_ENV = "CRL_MPE_CONFIG"

EVIDENCE_FOLDERS = (
    "01_config",
    "02_metrics",
    "03_data",
    "04_predictions",
    "05_figures",
    "06_verification",
)

PARAMETER_ORDER = ("h_ch", "n_fin", "h_fin", "d_in", "u_in", "q_w")
POOL_ORDER = ("id_train", "id_calibration", "id_test", "ood_test")


@dataclass(frozen=True)
class StudyPaths:
    root: Path
    config_path: Path
    evidence_pack: Path


def resolve_config_path(path: str | Path | None = None) -> Path:
    value = path or os.environ.get(CONFIG_ENV) or CONFIG_PATH
    resolved = Path(value)
    return resolved if resolved.is_absolute() else ROOT / resolved


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_config_path(path)
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def study_paths(config: dict[str, Any] | None = None) -> StudyPaths:
    config = config or load_config()
    evidence_pack = ROOT / config["paths"]["evidence_pack"]
    return StudyPaths(root=ROOT, config_path=resolve_config_path(), evidence_pack=evidence_pack)


def configured_multipass_instance(config: dict[str, Any]) -> tuple[str | None, str]:
    solver = config.get("solver", {})
    env_name = solver.get("multipass_instance_env")
    if env_name:
        env_value = os.environ.get(str(env_name))
        if env_value:
            return env_value, f"env:{env_name}"
    instance = solver.get("multipass_instance")
    return (str(instance), "config:multipass_instance") if instance else (None, "unset")


def ensure_evidence_folders(paths: StudyPaths) -> None:
    for name in EVIDENCE_FOLDERS:
        directory = paths.evidence_pack / name
        directory.mkdir(parents=True, exist_ok=True)
        keep = directory / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")


def copy_config_to_pack(paths: StudyPaths) -> None:
    target = paths.evidence_pack / "01_config" / "study.toml"
    target.write_text(paths.config_path.read_text(encoding="utf-8"), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_optional(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def environment_manifest() -> dict[str, Any]:
    config = load_config()
    solver = config.get("solver", {})
    multipass_instance, multipass_instance_source = configured_multipass_instance(config)
    openfoam_bashrc = solver.get("openfoam_bashrc")
    vm_openfoam: dict[str, Any] | None = None
    if multipass_instance and openfoam_bashrc:
        vm_openfoam = {
            "instance": multipass_instance,
            "instance_source": multipass_instance_source,
            "info": run_optional(["multipass", "info", str(multipass_instance)]),
            "foamVersion": run_optional(
                [
                    "multipass",
                    "exec",
                    str(multipass_instance),
                    "--",
                    "bash",
                    "-lc",
                    f"source {openfoam_bashrc} >/dev/null 2>&1 || true; foamVersion",
                ]
            ),
            "blockMesh": run_optional(
                [
                    "multipass",
                    "exec",
                    str(multipass_instance),
                    "--",
                    "bash",
                    "-lc",
                    f"source {openfoam_bashrc} >/dev/null 2>&1 || true; command -v blockMesh",
                ]
            ),
            "simpleFoam": run_optional(
                [
                    "multipass",
                    "exec",
                    str(multipass_instance),
                    "--",
                    "bash",
                    "-lc",
                    f"source {openfoam_bashrc} >/dev/null 2>&1 || true; command -v simpleFoam",
                ]
            ),
            "foamRun": run_optional(
                [
                    "multipass",
                    "exec",
                    str(multipass_instance),
                    "--",
                    "bash",
                    "-lc",
                    f"source {openfoam_bashrc} >/dev/null 2>&1 || true; command -v foamRun",
                ]
            ),
        }

    torch_info: dict[str, Any] = {"installed": False}
    try:
        import torch  # type: ignore

        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
        }
        if torch.backends.mps.is_available():
            torch_info["mps_device_count"] = torch.mps.device_count()
            torch_info["mps_recommended_max_memory_bytes"] = torch.mps.recommended_max_memory()
    except Exception as exc:  # pragma: no cover - environment probe
        torch_info = {"installed": False, "error": str(exc)}

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch_info,
        "openfoam_version": run_optional(["foamVersion"]),
        "openfoam_vm": vm_openfoam,
        "git_commit": run_optional(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_optional(["git", "status", "--short"]),
    }


def lhs_values(rng: random.Random, count: int, lo: float, hi: float) -> list[float]:
    values = []
    width = hi - lo
    for idx in range(count):
        unit = (idx + rng.random()) / count
        values.append(lo + unit * width)
    rng.shuffle(values)
    return values


def generate_id_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    total = dataset["id_train"] + dataset["id_calibration"] + dataset["id_test"]
    rng = random.Random(config["study"]["seed"])
    params = config["parameters"]
    geometry = config["geometry"]

    continuous_values = {
        name: lhs_values(rng, total, params[name]["id_min"], params[name]["id_max"])
        for name in PARAMETER_ORDER
        if params[name]["kind"] == "continuous" and name != "h_fin"
    }
    h_fin_units = lhs_values(rng, total, 0.0, 1.0)
    n_fin_values = params["n_fin"]["id_values"]

    pool_counts = [
        ("id_train", dataset["id_train"]),
        ("id_calibration", dataset["id_calibration"]),
        ("id_test", dataset["id_test"]),
    ]

    rows: list[dict[str, Any]] = []
    row_idx = 0
    for pool, count in pool_counts:
        for _ in range(count):
            h_ch = continuous_values["h_ch"][row_idx]
            h_fin_lo = float(params["h_fin"]["id_min"])
            h_fin_hi = min(
                float(params["h_fin"]["id_max"]),
                float(geometry["max_fin_height_fraction_of_channel"]) * h_ch,
            )
            h_fin = h_fin_lo + h_fin_units[row_idx] * (h_fin_hi - h_fin_lo)
            row = {
                "case_id": f"{pool}_{row_idx:04d}",
                "pool": pool,
                "h_ch": round(h_ch, 6),
                "n_fin": n_fin_values[row_idx % len(n_fin_values)],
                "h_fin": round(h_fin, 6),
                "d_in": round(continuous_values["d_in"][row_idx], 6),
                "u_in": round(continuous_values["u_in"][row_idx], 6),
                "q_w": round(continuous_values["q_w"][row_idx], 6),
                "geometry_family": "uniform_fins",
                "solver_status": "pending",
            }
            rows.append(row)
            row_idx += 1
    return rows


def pushed_value(rng: random.Random, lo: float, hi: float, min_frac: float, max_frac: float) -> float:
    span = hi - lo
    frac = rng.uniform(min_frac, max_frac)
    if rng.random() < 0.5:
        return lo - frac * span
    return hi + frac * span


def pushed_high_value(rng: random.Random, lo: float, hi: float, min_frac: float, max_frac: float) -> float:
    span = hi - lo
    frac = rng.uniform(min_frac, max_frac)
    return hi + frac * span


def generate_ood_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = config["dataset"]
    rng = random.Random(config["study"]["seed"] + 17)
    params = config["parameters"]
    ood = config["ood"]
    geometry = config["geometry"]

    total = dataset["ood_test"]
    continuous_id = {
        name: lhs_values(rng, total, params[name]["id_min"], params[name]["id_max"])
        for name in PARAMETER_ORDER
        if params[name]["kind"] == "continuous" and name != "h_fin"
    }
    h_fin_units = lhs_values(rng, total, 0.0, 1.0)
    rows: list[dict[str, Any]] = []
    for idx in range(total):
        n_fin = ood["n_fin_values"][idx % len(ood["n_fin_values"])]
        h_ch = continuous_id["h_ch"][idx]
        d_in = continuous_id["d_in"][idx]

        if idx % 2 == 0:
            h_ch = pushed_high_value(
                rng,
                params["h_ch"]["id_min"],
                params["h_ch"]["id_max"],
                ood["push_fraction_min"],
                ood["push_fraction_max"],
            )
        else:
            d_in = pushed_value(
                rng,
                params["d_in"]["id_min"],
                params["d_in"]["id_max"],
                ood["push_fraction_min"],
                ood["push_fraction_max"],
            )

        h_fin_lo = float(params["h_fin"]["id_min"])
        h_fin_hi = min(
            float(params["h_fin"]["id_max"]),
            float(geometry["max_fin_height_fraction_of_channel"]) * h_ch,
        )
        h_fin = h_fin_lo + h_fin_units[idx] * (h_fin_hi - h_fin_lo)

        rows.append(
            {
                "case_id": f"ood_test_{idx:04d}",
                "pool": "ood_test",
                "h_ch": round(h_ch, 6),
                "n_fin": n_fin,
                "h_fin": round(h_fin, 6),
                "d_in": round(d_in, 6),
                "u_in": round(continuous_id["u_in"][idx], 6),
                "q_w": round(continuous_id["q_w"][idx], 6),
                "geometry_family": "uniform_fins",
                "solver_status": "pending",
            }
        )
    return rows


def normalize_param(row: dict[str, Any], config: dict[str, Any], name: str) -> float:
    spec = config["parameters"][name]
    value = float(row[name])
    if spec["kind"] == "discrete":
        lo = min(spec["id_values"])
        hi = max(spec["id_values"])
    else:
        lo = float(spec["id_min"])
        hi = float(spec["id_max"])
    return (value - lo) / (hi - lo)


def nearest_id_distance(ood_row: dict[str, Any], id_rows: list[dict[str, Any]], config: dict[str, Any]) -> float:
    best = math.inf
    ood_vec = [normalize_param(ood_row, config, name) for name in PARAMETER_ORDER]
    for id_row in id_rows:
        id_vec = [normalize_param(id_row, config, name) for name in PARAMETER_ORDER]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(ood_vec, id_vec, strict=True)))
        best = min(best, dist)
    return best


def generate_parameter_tables(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    id_rows = generate_id_rows(config)
    ood_rows = generate_ood_rows(config)
    train_rows = [row for row in id_rows if row["pool"] == "id_train"]
    distance_rows = []
    for row in ood_rows:
        distance_rows.append(
            {
                "case_id": row["case_id"],
                "pool": row["pool"],
                "nearest_id_train_distance": round(nearest_id_distance(row, train_rows, config), 8),
            }
        )
    return id_rows + ood_rows, distance_rows


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        avg_rank = (idx + end + 1) / 2.0
        for original_idx, _ in indexed[idx:end]:
            ranks[original_idx] = avg_rank
        idx = end
    return ranks


def pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Pearson correlation needs equal-length vectors with at least two values.")
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    x_diffs = [x - x_mean for x in x_values]
    y_diffs = [y - y_mean for y in y_values]
    denom = math.sqrt(sum(x * x for x in x_diffs) * sum(y * y for y in y_diffs))
    if denom == 0:
        raise ValueError("Pearson correlation is undefined for constant input.")
    return sum(x * y for x, y in zip(x_diffs, y_diffs, strict=True)) / denom


def spearman(x_values: list[float], y_values: list[float]) -> float:
    return pearson(rankdata(x_values), rankdata(y_values))
