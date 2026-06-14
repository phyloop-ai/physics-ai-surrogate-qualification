#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import file_sha256, load_config, read_csv, study_paths, write_csv, write_json


INDEX_FIELDS = [
    "case_id",
    "pool",
    "npz_path",
    "grid_resolution",
    "fluid_pixels",
    "solid_pixels",
    "source_cells",
    "nearest_cell_distance_max_m",
    "T_fluid_mean",
    "T_fluid_std",
    "U_mag_fluid_mean",
    "U_mag_fluid_std",
    "npz_sha256",
]
CANONICAL_POOLS = {"id_train", "id_calibration", "id_test", "ood_test"}

FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
INT_LINE_RE = re.compile(r"^\d+$")
VECTOR_RE = re.compile(r"\(([^()]*)\)")
FACE_RE = re.compile(r"^\s*\d+\(([^()]*)\)\s*$")


def read_member(tar: tarfile.TarFile, case_id: str, relative_path: str) -> str:
    member = f"{case_id}/{relative_path}"
    extracted = tar.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(member)
    return extracted.read().decode("utf-8", errors="replace")


def read_json_member(tar: tarfile.TarFile, case_id: str, relative_path: str) -> dict[str, Any]:
    return json.loads(read_member(tar, case_id, relative_path))


def foam_list_lines(text: str) -> list[str]:
    lines = text.splitlines()
    count_idx = None
    count = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if INT_LINE_RE.match(stripped):
            next_nonempty = next((item.strip() for item in lines[idx + 1 :] if item.strip()), "")
            if next_nonempty == "(":
                count_idx = idx
                count = int(stripped)
                break
    if count_idx is None or count is None:
        raise ValueError("Could not find OpenFOAM list count.")
    start_idx = next(idx for idx in range(count_idx + 1, len(lines)) if lines[idx].strip() == "(") + 1
    values: list[str] = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == ")":
            break
        values.append(stripped)
        if len(values) == count:
            break
    if len(values) != count:
        raise ValueError(f"OpenFOAM list count mismatch: expected {count}, found {len(values)}.")
    return values


def parse_points(text: str) -> np.ndarray:
    rows = []
    for line in foam_list_lines(text):
        match = VECTOR_RE.search(line)
        if not match:
            raise ValueError(f"Bad point row: {line}")
        rows.append([float(item) for item in match.group(1).split()])
    return np.asarray(rows, dtype=np.float64)


def parse_faces(text: str) -> list[np.ndarray]:
    faces = []
    for line in foam_list_lines(text):
        match = FACE_RE.match(line)
        if not match:
            raise ValueError(f"Bad face row: {line}")
        faces.append(np.fromiter((int(item) for item in match.group(1).split()), dtype=np.int64))
    return faces


def parse_label_list(text: str) -> np.ndarray:
    return np.asarray([int(line) for line in foam_list_lines(text)], dtype=np.int64)


def parse_scalar_field(text: str, n_cells: int) -> np.ndarray:
    if "internalField   uniform" in text or "internalField uniform" in text:
        match = re.search(r"internalField\s+uniform\s+([^;]+);", text)
        if not match:
            raise ValueError("Could not parse uniform scalar field.")
        return np.full(n_cells, float(match.group(1)), dtype=np.float32)
    values = [float(line) for line in foam_list_lines(text)]
    if len(values) != n_cells:
        raise ValueError(f"Scalar field count mismatch: {len(values)} values for {n_cells} cells.")
    return np.asarray(values, dtype=np.float32)


def parse_vector_field(text: str, n_cells: int) -> np.ndarray:
    if "internalField   uniform" in text or "internalField uniform" in text:
        match = re.search(r"internalField\s+uniform\s+\(([^()]*)\);", text)
        if not match:
            raise ValueError("Could not parse uniform vector field.")
        value = np.asarray([float(item) for item in match.group(1).split()], dtype=np.float32)
        return np.repeat(value[None, :], n_cells, axis=0)
    values = []
    for line in foam_list_lines(text):
        match = VECTOR_RE.search(line)
        if not match:
            raise ValueError(f"Bad vector row: {line}")
        values.append([float(item) for item in match.group(1).split()])
    if len(values) != n_cells:
        raise ValueError(f"Vector field count mismatch: {len(values)} values for {n_cells} cells.")
    return np.asarray(values, dtype=np.float32)


def compute_cell_centers(points: np.ndarray, faces: list[np.ndarray], owner: np.ndarray, neighbour: np.ndarray) -> np.ndarray:
    n_cells = int(max(owner.max(initial=0), neighbour.max(initial=0))) + 1
    sums = np.zeros((n_cells, 3), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.float64)
    face_centers = np.empty((len(faces), 3), dtype=np.float64)
    for idx, face in enumerate(faces):
        face_centers[idx] = points[face].mean(axis=0)
    for face_idx, cell_idx in enumerate(owner):
        sums[cell_idx] += face_centers[face_idx]
        counts[cell_idx] += 1
    for face_idx, cell_idx in enumerate(neighbour):
        sums[cell_idx] += face_centers[face_idx]
        counts[cell_idx] += 1
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0)[:10]
        raise ValueError(f"Cells without faces: {missing.tolist()}")
    return sums / counts[:, None]


def signed_distance_to_rect(x: np.ndarray, y: np.ndarray, rect: dict[str, float]) -> np.ndarray:
    x0 = float(rect["x0_mm"])
    x1 = float(rect["x1_mm"])
    y0 = float(rect["y0_mm"])
    y1 = float(rect["y1_mm"])
    outside_dx = np.maximum(np.maximum(x0 - x, 0.0), x - x1)
    outside_dy = np.maximum(np.maximum(y0 - y, 0.0), y - y1)
    outside = np.hypot(outside_dx, outside_dy)
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    inside_dist = np.minimum.reduce([x - x0, x1 - x, y - y0, y1 - y])
    return np.where(inside, -inside_dist, outside)


def geometry_grid(case_json: dict[str, Any], resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geometry = case_json["geometry"]
    length_mm = float(geometry["length_mm"])
    height_mm = float(geometry["height_mm"])
    x_axis = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution * length_mm
    y_axis = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution * height_mm
    x_mm, y_mm = np.meshgrid(x_axis, y_axis)

    fluid_mask = np.ones((resolution, resolution), dtype=bool)
    sdf_mm = np.minimum(y_mm, height_mm - y_mm)
    for fin in geometry.get("fin_boxes", []):
        rect_distance = signed_distance_to_rect(x_mm, y_mm, fin)
        fluid_mask &= rect_distance >= 0.0
        sdf_mm = np.where(rect_distance < 0.0, rect_distance, np.minimum(sdf_mm, rect_distance))
    return x_mm, y_mm, fluid_mask, sdf_mm.astype(np.float32), np.asarray([length_mm, height_mm], dtype=np.float32)


def normalized_parameter_channels(case_json: dict[str, Any], config: dict[str, Any], resolution: int) -> list[np.ndarray]:
    params = case_json["parameters"]
    pconfig = config["parameters"]
    channels = []
    for name in ("h_ch", "n_fin", "h_fin", "d_in", "u_in", "q_w"):
        value = float(params[name])
        if pconfig[name]["kind"] == "discrete":
            values = [float(item) for item in pconfig[name]["id_values"]]
            lo = min(values)
            hi = max(values)
        else:
            lo = float(pconfig[name]["id_min"])
            hi = float(pconfig[name]["id_max"])
        normalized = (value - lo) / (hi - lo)
        channels.append(np.full((resolution, resolution), normalized, dtype=np.float32))
    return channels


def build_grid_case(
    tar: tarfile.TarFile,
    case_id: str,
    config: dict[str, Any],
    resolution: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, float]]:
    case_json = read_json_member(tar, case_id, "case.json")
    points = parse_points(read_member(tar, case_id, "constant/polyMesh/points"))
    faces = parse_faces(read_member(tar, case_id, "constant/polyMesh/faces"))
    owner = parse_label_list(read_member(tar, case_id, "constant/polyMesh/owner"))
    neighbour = parse_label_list(read_member(tar, case_id, "constant/polyMesh/neighbour"))
    centers = compute_cell_centers(points, faces, owner, neighbour)
    n_cells = centers.shape[0]

    u = parse_vector_field(read_member(tar, case_id, "flow/U"), n_cells)
    t = parse_scalar_field(read_member(tar, case_id, "temperature/T"), n_cells)
    u_mag = np.linalg.norm(u[:, :2], axis=1).astype(np.float32)

    x_mm, y_mm, fluid_mask, sdf_mm, length_height_mm = geometry_grid(case_json, resolution)
    query_xy_m = np.column_stack([(x_mm.ravel() / 1000.0), (y_mm.ravel() / 1000.0)])
    tree = cKDTree(centers[:, :2])
    distances, indices = tree.query(query_xy_m, k=1)

    t_grid = t[indices].reshape(resolution, resolution)
    u_grid = u_mag[indices].reshape(resolution, resolution)
    t_grid = np.where(fluid_mask, t_grid, 0.0).astype(np.float32)
    u_grid = np.where(fluid_mask, u_grid, 0.0).astype(np.float32)

    x_norm = (x_mm / float(length_height_mm[0])).astype(np.float32)
    y_norm = (y_mm / float(length_height_mm[1])).astype(np.float32)
    inputs = np.stack(
        [
            fluid_mask.astype(np.float32),
            (sdf_mm / float(length_height_mm[1])).astype(np.float32),
            x_norm,
            y_norm,
            *normalized_parameter_channels(case_json, config, resolution),
        ],
        axis=0,
    )
    targets = np.stack([t_grid, u_grid], axis=0)

    fluid_t = t_grid[fluid_mask]
    fluid_u = u_grid[fluid_mask]
    stats = {
        "fluid_pixels": int(fluid_mask.sum()),
        "solid_pixels": int((~fluid_mask).sum()),
        "source_cells": int(n_cells),
        "nearest_cell_distance_max_m": float(distances[fluid_mask.ravel()].max(initial=0.0)),
        "T_fluid_mean": float(fluid_t.mean()),
        "T_fluid_std": float(fluid_t.std()),
        "U_mag_fluid_mean": float(fluid_u.mean()),
        "U_mag_fluid_std": float(fluid_u.std()),
    }
    arrays = {
        "inputs": inputs.astype(np.float32),
        "targets": targets.astype(np.float32),
        "fluid_mask": fluid_mask.astype(np.uint8),
    }
    return case_json, arrays, stats


def retained_cases(paths) -> list[dict[str, str]]:
    convergence = {row["case_id"]: row for row in read_csv(paths.evidence_pack / "03_data" / "convergence_log.csv")}
    manifest_rows = read_csv(paths.evidence_pack / "03_data" / "solver_output_manifest.csv")
    rows = []
    for row in manifest_rows:
        conv = convergence.get(row["case_id"])
        if conv and conv.get("converged") == "true":
            rows.append(row)
    pool_rank = {"id_train": 0, "id_calibration": 1, "id_test": 2, "ood_test": 3}
    return sorted(rows, key=lambda row: (pool_rank.get(row["pool"], 99), row["case_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--pool")
    parser.add_argument(
        "--parameter-file",
        help="Optional CSV limiting extraction to case IDs present in that file.",
    )
    parser.add_argument(
        "--include-extra-pools",
        action="store_true",
        help="Include noncanonical pools when no explicit --pool or --case-id filter is given.",
    )
    parser.add_argument("--output-dir", default="evidence_pack/04_predictions/fixed_grid_fields")
    parser.add_argument("--index-file", default="evidence_pack/04_predictions/fixed_grid_index.csv")
    parser.add_argument("--summary-file", default="evidence_pack/04_predictions/fixed_grid_summary.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    resolution = int(config["grid"]["resolution"])
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = paths.root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = retained_cases(paths)
    if args.parameter_file:
        parameter_path = Path(args.parameter_file)
        if not parameter_path.is_absolute():
            parameter_path = paths.root / parameter_path
        allowed_ids = {row["case_id"] for row in read_csv(parameter_path)}
        rows = [row for row in rows if row["case_id"] in allowed_ids]
    if args.pool:
        rows = [row for row in rows if row["pool"] == args.pool]
    elif not args.case_id and not args.include_extra_pools:
        rows = [row for row in rows if row["pool"] in CANONICAL_POOLS]
    if args.case_id:
        wanted = set(args.case_id)
        rows = [row for row in rows if row["case_id"] in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]

    index_rows: list[dict[str, Any]] = []
    started = time.time()
    for idx, row in enumerate(rows, start=1):
        case_id = row["case_id"]
        npz_path = output_dir / f"{case_id}.npz"
        if npz_path.exists() and not args.force:
            npz_rel = npz_path.relative_to(paths.root)
            index_rows.append(
                {
                    "case_id": case_id,
                    "pool": row["pool"],
                    "npz_path": str(npz_rel),
                    "grid_resolution": resolution,
                    "fluid_pixels": "",
                    "solid_pixels": "",
                    "source_cells": "",
                    "nearest_cell_distance_max_m": "",
                    "T_fluid_mean": "",
                    "T_fluid_std": "",
                    "U_mag_fluid_mean": "",
                    "U_mag_fluid_std": "",
                    "npz_sha256": file_sha256(npz_path),
                }
            )
            print(f"[{idx}/{len(rows)}] SKIP {case_id}", flush=True)
            continue

        archive_path = paths.root / row["archive_path"]
        print(f"[{idx}/{len(rows)}] EXTRACT {case_id}", flush=True)
        with tarfile.open(archive_path, "r:gz") as tar:
            case_json, arrays, stats = build_grid_case(tar, case_id, config, resolution)
        np.savez_compressed(
            npz_path,
            inputs=arrays["inputs"],
            targets=arrays["targets"],
            fluid_mask=arrays["fluid_mask"],
            case_json=np.asarray(json.dumps(case_json, sort_keys=True)),
        )
        index_rows.append(
            {
                "case_id": case_id,
                "pool": row["pool"],
                "npz_path": str(npz_path.relative_to(paths.root)),
                "grid_resolution": resolution,
                **stats,
                "npz_sha256": file_sha256(npz_path),
            }
        )

    index_path = Path(args.index_file)
    if not index_path.is_absolute():
        index_path = paths.root / index_path
    summary_path = Path(args.summary_file)
    if not summary_path.is_absolute():
        summary_path = paths.root / summary_path
    write_csv(index_path, index_rows, INDEX_FIELDS)
    by_pool: dict[str, int] = {}
    for row in index_rows:
        by_pool[row["pool"]] = by_pool.get(row["pool"], 0) + 1
    write_json(
        summary_path,
        {
            "grid_resolution": resolution,
            "input_channels": config["grid"]["input_channels"],
            "output_fields": config["grid"]["output_fields"],
            "case_count": len(index_rows),
            "case_count_by_pool": by_pool,
            "selected_pool": args.pool,
            "include_extra_pools": args.include_extra_pools,
            "elapsed_sec": round(time.time() - started, 3),
            "index_file": str(index_path.relative_to(paths.root)),
            "output_dir": str(output_dir.relative_to(paths.root)),
            "resampling": "OpenFOAM cell centers to fixed grid by scipy.spatial.cKDTree nearest neighbor",
            "mask_policy": "metrics and loss use fluid_mask only; solid target pixels are zero-filled",
        },
    )
    print(f"Wrote {index_path.relative_to(paths.root)}")
    print(f"Wrote {summary_path.relative_to(paths.root)}")


if __name__ == "__main__":
    main()
