#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pl_wp_002.audit import file_sha256, load_config, read_csv, read_json, study_paths, write_csv, write_json


FIELD_NAMES = ("T", "U_mag")
PREDICTION_INDEX_FIELDS = [
    "case_id",
    "pool",
    "prediction_path",
    "fluid_pixels",
    "rmse_norm",
    "relative_l2_norm",
    "band_width_mean_norm_90",
    "band_width_p95_norm_90",
    "coverage_90",
    "case_q90_band_width_mean_norm_90",
    "case_q90_coverage_90",
    "case_max_band_width_mean_norm_90",
    "case_max_coverage_90",
    "covered_pairs_90",
    "total_pairs",
    "nearest_id_train_distance",
    "risk_score",
    "risk_multiplier",
    "prediction_sha256",
]
CALIBRATION_FIELDS = [
    "nominal_coverage",
    "empirical_coverage_id_calibration",
    "empirical_coverage_id_test",
    "median_band_width_norm_id_test",
    "median_naive_band_width_norm_id_test",
    "sharpness_ratio_vs_naive",
    "case_q90_empirical_coverage_id_calibration",
    "case_q90_empirical_coverage_id_test",
    "case_max_empirical_coverage_id_test",
    "qhat",
    "qhat_naive",
    "qhat_case_q90",
    "qhat_case_max",
]
OOD_FIELDS = [
    "nominal_coverage",
    "empirical_coverage_id_test",
    "empirical_coverage_ood",
    "median_band_width_norm_id_test",
    "median_band_width_norm_ood",
    "band_width_inflation_ratio",
    "rmse_norm_id_test",
    "rmse_norm_ood",
    "relative_l2_norm_id_test",
    "relative_l2_norm_ood",
    "spearman_bandwidth_error_ood",
    "case_q90_empirical_coverage_ood",
    "case_max_empirical_coverage_ood",
]
RISK_FIELDS = [
    "acceptance_fraction",
    "threshold_tau",
    "accepted_cases",
    "empirical_coverage_90",
    "id_served_fraction",
    "ood_served_fraction",
    "ood_routed_fraction",
]
TRAINING_FIELDS = ["member", "epoch", "train_masked_mse", "epoch_sec"]
PSEUDO_RISK_FIELDS = [
    "case_id",
    "n_fin",
    "held_out_member",
    "pseudo_relative_l2_norm",
    "risk_score",
    "risk_multiplier",
]


class SpectralConv2d(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        scale = 1.0 / (width * width)
        self.weights_pos = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))
        self.weights_neg = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize,
            self.width,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        modes_x = min(self.modes, height)
        modes_y = min(self.modes, width // 2 + 1)
        out_ft[:, :, :modes_x, :modes_y] = self.compl_mul2d(
            x_ft[:, :, :modes_x, :modes_y],
            self.weights_pos[:, :, :modes_x, :modes_y],
        )
        out_ft[:, :, -modes_x:, :modes_y] = self.compl_mul2d(
            x_ft[:, :, -modes_x:, :modes_y],
            self.weights_neg[:, :, :modes_x, :modes_y],
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, modes)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.local = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.pointwise(x) + self.local(x))


class CompactFNO(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int, modes: int, layers: int) -> None:
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(FNOBlock(width, modes) for _ in range(layers))
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


@dataclass
class DatasetBundle:
    rows: list[dict[str, str]]
    case_ids: list[str]
    pools: list[str]
    inputs: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    parameter_vectors: np.ndarray
    nearest_train_distance: np.ndarray


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pool_indices(pools: list[str], pool: str) -> np.ndarray:
    return np.asarray([idx for idx, item in enumerate(pools) if item == pool], dtype=np.int64)


def pool_indices_any(pools: list[str], selected_pools: set[str]) -> np.ndarray:
    return np.asarray([idx for idx, item in enumerate(pools) if item in selected_pools], dtype=np.int64)


def resolve_path(paths, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else paths.root / path


def relpath(paths, path: Path) -> str:
    try:
        return str(path.relative_to(paths.root))
    except ValueError:
        return str(path)


def load_parameter_rows(paths, extra_parameter_files: list[str] | None = None) -> dict[str, dict[str, str]]:
    rows = read_csv(paths.evidence_pack / "03_data" / "parameters.csv")
    backfill = paths.evidence_pack / "03_data" / "backfill_parameters.csv"
    if backfill.exists():
        rows.extend(read_csv(backfill))
    for extra in extra_parameter_files or []:
        rows.extend(read_csv(resolve_path(paths, extra)))
    return {row["case_id"]: row for row in rows}


def parameter_vector(row: dict[str, str], config: dict[str, Any]) -> list[float]:
    values = []
    for name in ("h_ch", "n_fin", "h_fin", "d_in", "u_in", "q_w"):
        pconf = config["parameters"][name]
        value = float(row[name])
        if pconf["kind"] == "discrete":
            domain = [float(item) for item in pconf["id_values"]]
            lo = min(domain)
            hi = max(domain)
        else:
            lo = float(pconf["id_min"])
            hi = float(pconf["id_max"])
        values.append((value - lo) / (hi - lo))
    return values


def load_dataset(
    config: dict[str, Any],
    paths,
    index_file: str | Path = "evidence_pack/04_predictions/fixed_grid_index.csv",
    extra_train_index_files: list[str] | None = None,
    extra_parameter_files: list[str] | None = None,
) -> DatasetBundle:
    index_path = resolve_path(paths, index_file)
    rows = read_csv(index_path)
    for extra in extra_train_index_files or []:
        rows.extend(read_csv(resolve_path(paths, extra)))
    parameters = load_parameter_rows(paths, extra_parameter_files)
    inputs = []
    targets = []
    masks = []
    vectors = []
    case_ids = []
    pools = []
    for row in rows:
        npz_path = paths.root / row["npz_path"]
        with np.load(npz_path) as data:
            inputs.append(data["inputs"].astype(np.float32))
            targets.append(data["targets"].astype(np.float32))
            masks.append(data["fluid_mask"].astype(bool))
        case_ids.append(row["case_id"])
        pools.append(row["pool"])
        vectors.append(parameter_vector(parameters[row["case_id"]], config))
    parameter_vectors = np.asarray(vectors, dtype=np.float32)
    train_idx = np.asarray([idx for idx, pool in enumerate(pools) if pool == "id_train"], dtype=np.int64)
    tree = cKDTree(parameter_vectors[train_idx])
    nearest, _ = tree.query(parameter_vectors, k=1)
    return DatasetBundle(
        rows=rows,
        case_ids=case_ids,
        pools=pools,
        inputs=np.stack(inputs),
        targets=np.stack(targets),
        masks=np.stack(masks),
        parameter_vectors=parameter_vectors,
        nearest_train_distance=nearest.astype(np.float32),
    )


def target_normalization(targets: np.ndarray, masks: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for field_idx in range(targets.shape[1]):
        values = []
        for idx in train_idx:
            values.append(targets[idx, field_idx][masks[idx]])
        joined = np.concatenate(values)
        means.append(float(joined.mean()))
        std = float(joined.std())
        stds.append(std if std > 1e-12 else 1.0)
    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def normalize_targets(targets: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((targets - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask[:, None, :, :].to(pred.dtype)
    denom = weights.sum() * pred.shape[1]
    return (((pred - target) ** 2) * weights).sum() / denom.clamp_min(1.0)


def train_model(
    model: nn.Module,
    bundle: DatasetBundle,
    targets_norm: np.ndarray,
    train_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    member: int,
    ) -> list[dict[str, Any]]:
    x_train = torch.from_numpy(bundle.inputs[train_idx])
    y_train = torch.from_numpy(targets_norm[train_idx])
    m_train = torch.from_numpy(bundle.masks[train_idx].astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_idx))
        total_loss = 0.0
        total_seen = 0
        started = time.time()
        for start in range(0, len(train_idx), args.batch_size):
            batch_ids = order[start : start + args.batch_size]
            xb = x_train[batch_ids].to(device)
            yb = y_train[batch_ids].to(device)
            mb = m_train[batch_ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = masked_mse(pred, yb, mb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_ids)
            total_seen += len(batch_ids)
        row = {
            "member": member,
            "epoch": epoch,
            "train_masked_mse": total_loss / max(total_seen, 1),
            "epoch_sec": round(time.time() - started, 3),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} train_masked_mse={row['train_masked_mse']:.6g} epoch_sec={row['epoch_sec']}",
            flush=True,
        )
    return history


def member_train_indices(
    strategy: str,
    full_train_idx: np.ndarray,
    member: int,
    bundle: DatasetBundle,
    parameter_rows: dict[str, dict[str, str]],
    seed: int,
) -> tuple[np.ndarray, str]:
    if strategy == "full":
        return full_train_idx, "all_id_train"

    if strategy == "bootstrap":
        rng = np.random.default_rng(seed + member)
        sampled = rng.choice(full_train_idx, size=len(full_train_idx), replace=True)
        return np.asarray(sampled, dtype=np.int64), "bootstrap_id_train"

    if strategy == "fin_holdout":
        fin_values = sorted({int(float(parameter_rows[bundle.case_ids[idx]]["n_fin"])) for idx in full_train_idx})
        held_out = fin_values[member % len(fin_values)]
        selected = [
            idx
            for idx in full_train_idx
            if int(float(parameter_rows[bundle.case_ids[idx]]["n_fin"])) != held_out
        ]
        return np.asarray(selected, dtype=np.int64), f"held_out_n_fin_{held_out}"

    raise ValueError(f"Unknown ensemble strategy: {strategy}")


@torch.no_grad()
def predict_all(model: nn.Module, inputs: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    x = torch.from_numpy(inputs)
    for start in range(0, len(inputs), batch_size):
        pred = model(x[start : start + batch_size].to(device)).cpu().numpy().astype(np.float32)
        outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def residual_scale_map(abs_residuals: np.ndarray, masks: np.ndarray, cal_idx: np.ndarray) -> np.ndarray:
    masked = abs_residuals[cal_idx].copy()
    for local_idx, case_idx in enumerate(cal_idx):
        masked[local_idx, :, ~masks[case_idx]] = np.nan
    scale = np.nanmedian(masked, axis=0)
    finite_values = scale[np.isfinite(scale)]
    floor = float(np.nanmedian(finite_values)) * 0.05 + 1e-6 if finite_values.size else 1e-3
    scale = np.where(np.isfinite(scale), scale, floor)
    return np.maximum(scale.astype(np.float32), floor)


def scale_for_cases(
    scale_map: np.ndarray,
    distances: np.ndarray,
    distance_gain: float,
    ensemble_std: np.ndarray,
    ensemble_std_gain: float,
) -> np.ndarray:
    scaled_map = scale_map[None, :, :, :] * (1.0 + distance_gain * distances[:, None, None, None])
    return scaled_map + ensemble_std_gain * ensemble_std


def case_feature_matrix(
    bundle: DatasetBundle,
    parameter_rows: dict[str, dict[str, str]],
    ensemble_std: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    names = [
        "ensemble_width_mean",
        "ensemble_width_p95",
        "nearest_id_train_distance",
        "n_fin",
        "n_fin_below_id",
        "n_fin_above_id",
        "d_in_below_id",
        "d_in_above_id",
        "h_ch_above_id",
        "blockage_ratio",
        "heat_load_per_flow_proxy",
        "fin_crowding_proxy",
        "u_in",
        "q_w",
        "d_in",
        "h_ch",
        "h_fin",
    ]
    rows = []
    for case_idx, case_id in enumerate(bundle.case_ids):
        params = parameter_rows[case_id]
        h_ch = float(params["h_ch"])
        d_in = float(params["d_in"])
        n_fin = float(params["n_fin"])
        h_fin = float(params["h_fin"])
        u_in = float(params["u_in"])
        q_w = float(params["q_w"])
        pitch = 30.0 / n_fin
        fin_width = min(0.8, 0.35 * pitch)
        gap = max(pitch - fin_width, 1e-6)
        blockage = h_fin / h_ch
        heat_proxy = q_w / max(u_in * d_in * h_ch, 1e-6)
        crowding = n_fin * blockage / gap
        mask = bundle.masks[case_idx]
        ensemble_width = 2.0 * ensemble_std[case_idx][:, mask]
        rows.append(
            [
                float(ensemble_width.mean()),
                float(np.quantile(ensemble_width.reshape(-1), 0.95)),
                float(bundle.nearest_train_distance[case_idx]),
                n_fin,
                max(0.0, 4.0 - n_fin),
                max(0.0, n_fin - 8.0),
                max(0.0, 2.0 - d_in),
                max(0.0, d_in - 6.0),
                max(0.0, h_ch - 10.0),
                blockage,
                heat_proxy,
                crowding,
                u_in,
                q_w,
                d_in,
                h_ch,
                h_fin,
            ]
        )
    return np.asarray(rows, dtype=np.float64), names


def quadratic_design(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (features - mean[None, :]) / std[None, :]
    products = [z[:, left] * z[:, right] for left in range(z.shape[1]) for right in range(left, z.shape[1])]
    return np.column_stack([np.ones(len(z)), z, *products])


def fit_pseudo_ood_risk(
    bundle: DatasetBundle,
    parameter_rows: dict[str, dict[str, str]],
    pred_norm_members: np.ndarray,
    pred_norm: np.ndarray,
    targets_norm: np.ndarray,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    features: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    if args.risk_head == "none":
        ones = np.ones(len(bundle.case_ids), dtype=np.float32)
        return np.zeros(len(bundle.case_ids), dtype=np.float32), ones, {"name": "none"}, []

    if args.risk_head != "pseudo_quadratic":
        raise ValueError(f"Unknown risk head: {args.risk_head}")
    if args.ensemble_strategy != "fin_holdout":
        raise ValueError("pseudo_quadratic risk head requires --ensemble-strategy fin_holdout.")

    fin_values = sorted({int(float(parameter_rows[bundle.case_ids[idx]]["n_fin"])) for idx in train_idx})
    held_out_by_member = {
        member: fin_values[member % len(fin_values)]
        for member in range(pred_norm_members.shape[0])
    }
    pseudo_idx = []
    pseudo_member = []
    pseudo_target = []
    for case_idx in train_idx:
        n_fin = int(float(parameter_rows[bundle.case_ids[case_idx]]["n_fin"]))
        matching = [member for member, held_out in held_out_by_member.items() if held_out == n_fin]
        if not matching:
            continue
        member = matching[0]
        pseudo_idx.append(case_idx)
        pseudo_member.append(member)
        pseudo_target.append(case_error(pred_norm_members[member, case_idx], targets_norm[case_idx], bundle.masks[case_idx])[1])

    pseudo_idx_arr = np.asarray(pseudo_idx, dtype=np.int64)
    pseudo_target_arr = np.asarray(pseudo_target, dtype=np.float64)
    if len(pseudo_idx_arr) < features.shape[1] * 2:
        raise RuntimeError("Not enough pseudo-OOD rows to fit risk head.")

    feature_mean = features[pseudo_idx_arr].mean(axis=0)
    feature_std = features[pseudo_idx_arr].std(axis=0) + 1e-9
    design = quadratic_design(features, feature_mean, feature_std)
    regularizer = float(args.risk_head_lambda) * np.eye(design.shape[1])
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design[pseudo_idx_arr].T @ design[pseudo_idx_arr] + regularizer,
        design[pseudo_idx_arr].T @ np.log(pseudo_target_arr + 1e-6),
    )
    risk_score = (design @ coefficients).astype(np.float32)

    cal_scores = risk_score[cal_idx]
    center = float(np.median(cal_scores))
    iqr = float(np.quantile(cal_scores, 0.75) - np.quantile(cal_scores, 0.25) + 1e-9)
    positive_excess = np.maximum(0.0, (risk_score - center) / iqr)
    risk_multiplier = np.clip(
        1.0 + float(args.risk_gain) * positive_excess,
        1.0,
        float(args.risk_multiplier_cap),
    ).astype(np.float32)

    pseudo_rows = []
    for case_idx, member, target in zip(pseudo_idx, pseudo_member, pseudo_target):
        pseudo_rows.append(
            {
                "case_id": bundle.case_ids[case_idx],
                "n_fin": parameter_rows[bundle.case_ids[case_idx]]["n_fin"],
                "held_out_member": member,
                "pseudo_relative_l2_norm": target,
                "risk_score": float(risk_score[case_idx]),
                "risk_multiplier": float(risk_multiplier[case_idx]),
            }
        )

    summary = {
        "name": "pseudo_quadratic",
        "training_source": "Configured training pools only; each label is generated by the ensemble member that held out the case's fin-count group.",
        "target": "log pseudo relative L2 field error",
        "pseudo_rows": len(pseudo_rows),
        "ridge_lambda": float(args.risk_head_lambda),
        "risk_gain": float(args.risk_gain),
        "risk_multiplier_cap": float(args.risk_multiplier_cap),
        "score_center": center,
        "score_iqr": iqr,
        "held_out_n_fin_by_member": held_out_by_member,
    }
    return risk_score, risk_multiplier, summary, pseudo_rows


def masked_values(values: np.ndarray, masks: np.ndarray, indices: np.ndarray) -> np.ndarray:
    chunks = []
    for idx in indices:
        chunks.append(values[idx, :, masks[idx]].reshape(-1))
    return np.concatenate(chunks) if chunks else np.asarray([], dtype=np.float32)


def case_quantile_scores(values: np.ndarray, masks: np.ndarray, indices: np.ndarray, fraction: float) -> np.ndarray:
    scores = []
    for idx in indices:
        case_values = values[idx, :, masks[idx]].reshape(-1)
        if case_values.size:
            scores.append(float(np.quantile(case_values, fraction)))
    return np.asarray(scores, dtype=np.float32)


def case_max_scores(values: np.ndarray, masks: np.ndarray, indices: np.ndarray) -> np.ndarray:
    scores = []
    for idx in indices:
        case_values = values[idx, :, masks[idx]].reshape(-1)
        if case_values.size:
            scores.append(float(np.max(case_values)))
    return np.asarray(scores, dtype=np.float32)


def qhat(scores: np.ndarray, nominal: float) -> float:
    if scores.size == 0:
        return float("nan")
    order = math.ceil((scores.size + 1) * nominal) / scores.size
    return float(np.quantile(scores, min(order, 1.0), method="higher"))


def coverage_counts(abs_residual: np.ndarray, half_width: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    covered = (abs_residual[:, mask] <= half_width[:, mask]).sum()
    total = int(mask.sum()) * abs_residual.shape[0]
    return int(covered), total


def coverage_fraction(abs_residual: np.ndarray, half_width: np.ndarray, mask: np.ndarray) -> float:
    covered, total = coverage_counts(abs_residual, half_width, mask)
    return covered / total if total else float("nan")


def case_error(pred_norm: np.ndarray, target_norm: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    diff = pred_norm[:, mask] - target_norm[:, mask]
    truth = target_norm[:, mask]
    rmse = float(np.sqrt(np.mean(diff**2)))
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(truth), 1e-12))
    return rmse, rel_l2


def aggregate_coverage(case_rows: list[dict[str, Any]], level: str, pools: set[str]) -> float:
    covered = sum(int(row[f"covered_pairs_{level}"]) for row in case_rows if row["pool"] in pools)
    total = sum(int(row["total_pairs"]) for row in case_rows if row["pool"] in pools)
    return covered / total if total else float("nan")


def median_case_value(case_rows: list[dict[str, Any]], key: str, pools: set[str]) -> float:
    values = [float(row[key]) for row in case_rows if row["pool"] in pools]
    return float(np.median(values)) if values else float("nan")


def mean_case_value(case_rows: list[dict[str, Any]], key: str, pools: set[str]) -> float:
    values = [float(row[key]) for row in case_rows if row["pool"] in pools]
    return float(np.mean(values)) if values else float("nan")


def fraction_cases_at_or_above(
    rows: list[dict[str, Any]],
    key: str,
    pools: set[str],
    threshold: float,
) -> float:
    values = [float(row[key]) for row in rows if row["pool"] in pools and row.get(key) not in ("", None)]
    return float(np.mean([value >= threshold for value in values])) if values else float("nan")


def risk_curve(case_rows: list[dict[str, Any]], pools: set[str]) -> list[dict[str, Any]]:
    selected = [row for row in case_rows if row["pool"] in pools]
    selected = sorted(selected, key=lambda row: float(row["band_width_mean_norm_90"]))
    total_cases = len(selected)
    id_total = sum(row["pool"] == "id_test" for row in selected)
    ood_total = sum(row["pool"] == "ood_test" for row in selected)
    rows: list[dict[str, Any]] = [
        {
            "acceptance_fraction": 0.0,
            "threshold_tau": "",
            "accepted_cases": 0,
            "empirical_coverage_90": "",
            "id_served_fraction": 0.0 if id_total else "",
            "ood_served_fraction": 0.0 if ood_total else "",
            "ood_routed_fraction": 1.0 if ood_total else "",
        }
    ]
    covered = 0
    pairs = 0
    id_served = 0
    ood_served = 0
    for idx, row in enumerate(selected, start=1):
        covered += int(row["covered_pairs_90"])
        pairs += int(row["total_pairs"])
        id_served += row["pool"] == "id_test"
        ood_served += row["pool"] == "ood_test"
        rows.append(
            {
                "acceptance_fraction": idx / total_cases,
                "threshold_tau": float(row["band_width_mean_norm_90"]),
                "accepted_cases": idx,
                "empirical_coverage_90": covered / pairs if pairs else "",
                "id_served_fraction": id_served / id_total if id_total else "",
                "ood_served_fraction": ood_served / ood_total if ood_total else "",
                "ood_routed_fraction": 1.0 - (ood_served / ood_total) if ood_total else "",
            }
        )
    return rows


def choose_gate(curve: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        row
        for row in curve
        if row["accepted_cases"] and row["empirical_coverage_90"] != "" and float(row["empirical_coverage_90"]) >= 0.90
    ]
    if not candidates:
        nonempty = [row for row in curve if row["accepted_cases"]]
        return max(nonempty, key=lambda row: float(row["empirical_coverage_90"])) if nonempty else curve[0]
    return max(candidates, key=lambda row: float(row["acceptance_fraction"]))


def write_prediction_npz(
    path: Path,
    target_phys: np.ndarray,
    pred_phys: np.ndarray,
    half_width_phys_90: np.ndarray,
    residual_phys: np.ndarray,
    mask: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        target=target_phys.astype(np.float32),
        prediction=pred_phys.astype(np.float32),
        band_half_width_90=half_width_phys_90.astype(np.float32),
        residual=residual_phys.astype(np.float32),
        fluid_mask=mask.astype(np.uint8),
        field_names=np.asarray(FIELD_NAMES),
    )


def plot_calibration(path: Path, rows: list[dict[str, Any]]) -> None:
    nominal = [float(row["nominal_coverage"]) for row in rows]
    empirical = [float(row["empirical_coverage_id_test"]) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=180)
    ax.plot([0, 1], [0, 1], color="#52606d", linewidth=1.0, linestyle="--", label="ideal")
    ax.plot(nominal, empirical, marker="o", color="#2166ac", label="ID test")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Split-Conformal Field Coverage", loc="left", fontweight="bold")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.grid(color="#d9e2ec", linewidth=0.8)
    ax.legend(frameon=False)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_risk(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    valid = [row for row in rows if row["empirical_coverage_90"] != ""]
    x = [float(row["acceptance_fraction"]) for row in valid]
    y = [float(row["empirical_coverage_90"]) for row in valid]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=180)
    ax.plot(x, y, color="#2166ac", linewidth=1.8)
    ax.axhline(0.90, color="#b2182b", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Fraction served by surrogate")
    ax.set_ylabel("Accepted-set empirical coverage at 90% band")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="#d9e2ec", linewidth=0.8)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_ood_bandwidth_error(path: Path, case_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in case_rows if row["pool"] == "ood_test"]
    x = [float(row["band_width_mean_norm_90"]) for row in rows]
    y = [float(row["relative_l2_norm"]) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=180)
    ax.scatter(x, y, s=24, color="#b2182b", alpha=0.72, edgecolors="none")
    ax.set_xlabel("Mean predicted 90% band width, normalized")
    ax.set_ylabel("Per-case relative L2 error, normalized")
    ax.set_title("OOD Uncertainty vs Error", loc="left", fontweight="bold")
    ax.grid(color="#d9e2ec", linewidth=0.8)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--ensemble-strategy", choices=("full", "bootstrap", "fin_holdout"), default="full")
    parser.add_argument("--distance-gain", type=float, default=0.0)
    parser.add_argument("--ensemble-std-gain", type=float, default=1.0)
    parser.add_argument("--risk-head", choices=("none", "pseudo_quadratic"), default="none")
    parser.add_argument("--risk-head-lambda", type=float, default=0.1)
    parser.add_argument("--risk-gain", type=float, default=0.5)
    parser.add_argument("--risk-multiplier-cap", type=float, default=8.0)
    parser.add_argument("--reuse-checkpoint", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--fixed-grid-index", default="evidence_pack/04_predictions/fixed_grid_index.csv")
    parser.add_argument("--extra-train-index-file", action="append", default=[])
    parser.add_argument("--extra-parameter-file", action="append", default=[])
    parser.add_argument("--train-pool", action="append", default=None)
    parser.add_argument("--model-dir", default="evidence_pack/04_predictions/model")
    parser.add_argument("--prediction-dir", default="evidence_pack/04_predictions/eval_cases")
    parser.add_argument("--prediction-index", default="evidence_pack/04_predictions/prediction_index.csv")
    parser.add_argument("--metrics-dir", default="evidence_pack/02_metrics")
    parser.add_argument("--figures-dir", default="evidence_pack/05_figures")
    args = parser.parse_args()

    config = load_config()
    paths = study_paths(config)
    train_pools = set(args.train_pool or ["id_train"])
    seed = int(config["study"]["seed"])
    set_seeds(seed)
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}", flush=True)

    started = time.time()
    bundle = load_dataset(
        config,
        paths,
        index_file=args.fixed_grid_index,
        extra_train_index_files=args.extra_train_index_file,
        extra_parameter_files=args.extra_parameter_file,
    )
    parameter_rows = load_parameter_rows(paths, args.extra_parameter_file)
    train_idx = pool_indices_any(bundle.pools, train_pools)
    train_idx_set = set(int(idx) for idx in train_idx)
    cal_idx = pool_indices(bundle.pools, "id_calibration")
    id_test_idx = pool_indices(bundle.pools, "id_test")
    ood_idx = pool_indices(bundle.pools, "ood_test")
    target_mean, target_std = target_normalization(bundle.targets, bundle.masks, train_idx)
    targets_norm = normalize_targets(bundle.targets, target_mean, target_std)

    template_model = CompactFNO(
        in_channels=bundle.inputs.shape[1],
        out_channels=bundle.targets.shape[1],
        width=args.width,
        modes=args.modes,
        layers=args.layers,
    )
    parameter_count = sum(param.numel() for param in template_model.parameters())
    del template_model
    print(f"Model parameters per member: {parameter_count}", flush=True)
    train_started = time.time()
    history: list[dict[str, Any]] = []
    member_predictions = []
    member_states = []
    model_dir = resolve_path(paths, args.model_dir)
    checkpoint_path = model_dir / "compact_fno.pt"
    member_training_sets: list[dict[str, Any]] = []
    if args.reuse_checkpoint and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        member_states = checkpoint["member_state_dicts"]
        args.ensemble_size = len(member_states)
        print(f"Reusing checkpoint with {args.ensemble_size} ensemble members", flush=True)
        for member, state in enumerate(member_states):
            model = CompactFNO(
                in_channels=bundle.inputs.shape[1],
                out_channels=bundle.targets.shape[1],
                width=args.width,
                modes=args.modes,
                layers=args.layers,
            ).to(device)
            model.load_state_dict(state)
            member_predictions.append(predict_all(model, bundle.inputs, args.predict_batch_size, device))
            del model
    else:
        for member in range(args.ensemble_size):
            print(f"Training ensemble member {member + 1}/{args.ensemble_size}", flush=True)
            set_seeds(seed + member)
            member_train_idx, member_policy = member_train_indices(
                args.ensemble_strategy,
                train_idx,
                member,
                bundle,
                parameter_rows,
                seed,
            )
            member_training_sets.append(
                {
                    "member": member,
                    "policy": member_policy,
                    "train_case_count": int(len(member_train_idx)),
                }
            )
            model = CompactFNO(
                in_channels=bundle.inputs.shape[1],
                out_channels=bundle.targets.shape[1],
                width=args.width,
                modes=args.modes,
                layers=args.layers,
            ).to(device)
            history.extend(train_model(model, bundle, targets_norm, member_train_idx, args, device, member))
            member_predictions.append(predict_all(model, bundle.inputs, args.predict_batch_size, device))
            member_states.append({key: value.cpu() for key, value in model.cpu().state_dict().items()})
            del model
    train_elapsed = time.time() - train_started
    pred_norm_members = np.stack(member_predictions, axis=0)
    pred_norm = pred_norm_members.mean(axis=0).astype(np.float32)
    ensemble_std = pred_norm_members.std(axis=0).astype(np.float32)
    pred_phys = pred_norm * target_std[None, :, None, None] + target_mean[None, :, None, None]
    abs_residual_norm = np.abs(pred_norm - targets_norm)

    risk_features, risk_feature_names = case_feature_matrix(bundle, parameter_rows, ensemble_std)
    risk_score, risk_multiplier, risk_summary, pseudo_risk_rows = fit_pseudo_ood_risk(
        bundle,
        parameter_rows,
        pred_norm_members,
        pred_norm,
        targets_norm,
        train_idx,
        cal_idx,
        risk_features,
        args,
    )

    scale_map = residual_scale_map(abs_residual_norm, bundle.masks, cal_idx)
    scales = scale_for_cases(
        scale_map,
        bundle.nearest_train_distance,
        args.distance_gain,
        ensemble_std,
        args.ensemble_std_gain,
    ) * risk_multiplier[:, None, None, None]
    normalized_scores = abs_residual_norm / np.maximum(scales, 1e-8)
    score_values = masked_values(normalized_scores, bundle.masks, cal_idx)
    naive_score_values = masked_values(abs_residual_norm, bundle.masks, cal_idx)

    levels = [float(item) for item in config["uq"]["nominal_coverage_levels"]]
    q_by_level = {level: qhat(score_values, level) for level in levels}
    q_naive_by_level = {level: qhat(naive_score_values, level) for level in levels}
    q_case_quantile_by_level = {
        level: qhat(case_quantile_scores(normalized_scores, bundle.masks, cal_idx, level), level)
        for level in levels
    }
    calibration_case_max_scores = case_max_scores(normalized_scores, bundle.masks, cal_idx)
    q_case_max_by_level = {level: qhat(calibration_case_max_scores, level) for level in levels}

    prediction_dir = resolve_path(paths, args.prediction_dir)
    metrics_dir = resolve_path(paths, args.metrics_dir)
    figures_dir = resolve_path(paths, args.figures_dir)
    case_metric_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    primary_level = float(config["uq"]["primary_level"])

    half_width_norm_by_level = {level: q_by_level[level] * scales for level in levels}
    naive_width_norm_by_level = {level: np.full_like(scales, q_naive_by_level[level]) for level in levels}
    case_quantile_width_norm_by_level = {level: q_case_quantile_by_level[level] * scales for level in levels}
    case_max_width_norm_by_level = {level: q_case_max_by_level[level] * scales for level in levels}

    for case_idx, case_id in enumerate(bundle.case_ids):
        if case_idx in train_idx_set:
            continue
        mask = bundle.masks[case_idx]
        rmse, rel_l2 = case_error(pred_norm[case_idx], targets_norm[case_idx], mask)
        level_rows: dict[str, tuple[int, int, float, float]] = {}
        for level in levels:
            half_width = half_width_norm_by_level[level][case_idx]
            covered, total = coverage_counts(abs_residual_norm[case_idx], half_width, mask)
            band_width_mean = float((2.0 * half_width[:, mask]).mean())
            band_width_p95 = float(np.quantile((2.0 * half_width[:, mask]).reshape(-1), 0.95))
            level_rows[f"{level:.2f}"] = (covered, total, band_width_mean, band_width_p95)
        covered90, total90, width90, width95_90 = level_rows[f"{primary_level:.2f}"]
        case_q90_half_width = case_quantile_width_norm_by_level[primary_level][case_idx]
        case_max_half_width = case_max_width_norm_by_level[primary_level][case_idx]
        case_q90_coverage = coverage_fraction(abs_residual_norm[case_idx], case_q90_half_width, mask)
        case_max_coverage = coverage_fraction(abs_residual_norm[case_idx], case_max_half_width, mask)
        case_q90_width = float((2.0 * case_q90_half_width[:, mask]).mean())
        case_max_width = float((2.0 * case_max_half_width[:, mask]).mean())
        target_phys = bundle.targets[case_idx]
        residual_phys = pred_phys[case_idx] - target_phys
        half_width_phys_90 = half_width_norm_by_level[primary_level][case_idx] * target_std[:, None, None]
        prediction_path = prediction_dir / f"{case_id}.npz"
        write_prediction_npz(prediction_path, target_phys, pred_phys[case_idx], half_width_phys_90, residual_phys, mask)
        case_metric_rows.append(
            {
                "case_id": case_id,
                "pool": bundle.pools[case_idx],
                "prediction_path": str(prediction_path.relative_to(paths.root)),
                "fluid_pixels": int(mask.sum()),
                "rmse_norm": rmse,
                "relative_l2_norm": rel_l2,
                "band_width_mean_norm_90": width90,
                "band_width_p95_norm_90": width95_90,
                "coverage_90": covered90 / total90,
                "case_q90_band_width_mean_norm_90": case_q90_width,
                "case_q90_coverage_90": case_q90_coverage,
                "case_max_band_width_mean_norm_90": case_max_width,
                "case_max_coverage_90": case_max_coverage,
                "covered_pairs_90": covered90,
                "total_pairs": total90,
                "nearest_id_train_distance": float(bundle.nearest_train_distance[case_idx]),
                "risk_score": float(risk_score[case_idx]),
                "risk_multiplier": float(risk_multiplier[case_idx]),
                "prediction_sha256": file_sha256(prediction_path),
            }
        )

    for level in levels:
        level_key = f"{level:.2f}"
        for case_idx, case_id in enumerate(bundle.case_ids):
            if case_idx in train_idx_set:
                continue
        temp_rows = []
        for case_idx, case_id in enumerate(bundle.case_ids):
            if case_idx in train_idx_set:
                continue
            mask = bundle.masks[case_idx]
            half_width = half_width_norm_by_level[level][case_idx]
            case_q_half_width = case_quantile_width_norm_by_level[level][case_idx]
            case_max_half_width = case_max_width_norm_by_level[level][case_idx]
            naive_half_width = naive_width_norm_by_level[level][case_idx]
            covered, total = coverage_counts(abs_residual_norm[case_idx], half_width, mask)
            naive_width = float((2.0 * naive_half_width[:, mask]).mean())
            temp_rows.append(
                {
                    "case_id": case_id,
                    "pool": bundle.pools[case_idx],
                    "covered": covered,
                    "total": total,
                    "case_q_coverage": coverage_fraction(abs_residual_norm[case_idx], case_q_half_width, mask),
                    "case_max_coverage": coverage_fraction(abs_residual_norm[case_idx], case_max_half_width, mask),
                    "band_width": float((2.0 * half_width[:, mask]).mean()),
                    "naive_band_width": naive_width,
                }
            )
        id_cal_covered = sum(row["covered"] for row in temp_rows if row["pool"] == "id_calibration")
        id_cal_total = sum(row["total"] for row in temp_rows if row["pool"] == "id_calibration")
        id_test_covered = sum(row["covered"] for row in temp_rows if row["pool"] == "id_test")
        id_test_total = sum(row["total"] for row in temp_rows if row["pool"] == "id_test")
        id_width = [row["band_width"] for row in temp_rows if row["pool"] == "id_test"]
        naive_width = [row["naive_band_width"] for row in temp_rows if row["pool"] == "id_test"]
        calibration_rows.append(
            {
                "nominal_coverage": level,
                "empirical_coverage_id_calibration": id_cal_covered / id_cal_total,
                "empirical_coverage_id_test": id_test_covered / id_test_total,
                "median_band_width_norm_id_test": float(np.median(id_width)),
                "median_naive_band_width_norm_id_test": float(np.median(naive_width)),
                "sharpness_ratio_vs_naive": float(np.median(id_width) / max(np.median(naive_width), 1e-12)),
                "case_q90_empirical_coverage_id_calibration": fraction_cases_at_or_above(
                    temp_rows,
                    "case_q_coverage",
                    {"id_calibration"},
                    level,
                ),
                "case_q90_empirical_coverage_id_test": fraction_cases_at_or_above(
                    temp_rows,
                    "case_q_coverage",
                    {"id_test"},
                    level,
                ),
                "case_max_empirical_coverage_id_test": fraction_cases_at_or_above(
                    temp_rows,
                    "case_max_coverage",
                    {"id_test"},
                    1.0,
                ),
                "qhat": q_by_level[level],
                "qhat_naive": q_naive_by_level[level],
                "qhat_case_q90": q_case_quantile_by_level[level],
                "qhat_case_max": q_case_max_by_level[level],
            }
        )

    ood_spearman = float(
        spearmanr(
            [float(row["band_width_mean_norm_90"]) for row in case_metric_rows if row["pool"] == "ood_test"],
            [float(row["relative_l2_norm"]) for row in case_metric_rows if row["pool"] == "ood_test"],
        ).statistic
    )
    if math.isnan(ood_spearman):
        ood_spearman = 0.0

    ood_rows = []
    for cal_row in calibration_rows:
        level = float(cal_row["nominal_coverage"])
        temp_rows = []
        for case_idx, case_id in enumerate(bundle.case_ids):
            if bundle.pools[case_idx] not in {"id_test", "ood_test"}:
                continue
            mask = bundle.masks[case_idx]
            half_width = half_width_norm_by_level[level][case_idx]
            case_q_half_width = case_quantile_width_norm_by_level[level][case_idx]
            case_max_half_width = case_max_width_norm_by_level[level][case_idx]
            covered, total = coverage_counts(abs_residual_norm[case_idx], half_width, mask)
            temp_rows.append(
                {
                    "pool": bundle.pools[case_idx],
                    "covered": covered,
                    "total": total,
                    "band_width": float((2.0 * half_width[:, mask]).mean()),
                    "case_q_coverage": coverage_fraction(abs_residual_norm[case_idx], case_q_half_width, mask),
                    "case_max_coverage": coverage_fraction(abs_residual_norm[case_idx], case_max_half_width, mask),
                }
            )
        id_covered = sum(row["covered"] for row in temp_rows if row["pool"] == "id_test")
        id_total = sum(row["total"] for row in temp_rows if row["pool"] == "id_test")
        ood_covered = sum(row["covered"] for row in temp_rows if row["pool"] == "ood_test")
        ood_total = sum(row["total"] for row in temp_rows if row["pool"] == "ood_test")
        id_width = [row["band_width"] for row in temp_rows if row["pool"] == "id_test"]
        ood_width = [row["band_width"] for row in temp_rows if row["pool"] == "ood_test"]
        ood_rows.append(
            {
                "nominal_coverage": level,
                "empirical_coverage_id_test": id_covered / id_total,
                "empirical_coverage_ood": ood_covered / ood_total,
                "median_band_width_norm_id_test": float(np.median(id_width)),
                "median_band_width_norm_ood": float(np.median(ood_width)),
                "band_width_inflation_ratio": float(np.median(ood_width) / max(np.median(id_width), 1e-12)),
                "rmse_norm_id_test": mean_case_value(case_metric_rows, "rmse_norm", {"id_test"}),
                "rmse_norm_ood": mean_case_value(case_metric_rows, "rmse_norm", {"ood_test"}),
                "relative_l2_norm_id_test": mean_case_value(case_metric_rows, "relative_l2_norm", {"id_test"}),
                "relative_l2_norm_ood": mean_case_value(case_metric_rows, "relative_l2_norm", {"ood_test"}),
                "spearman_bandwidth_error_ood": ood_spearman,
                "case_q90_empirical_coverage_ood": fraction_cases_at_or_above(
                    temp_rows,
                    "case_q_coverage",
                    {"ood_test"},
                    level,
                ),
                "case_max_empirical_coverage_ood": fraction_cases_at_or_above(
                    temp_rows,
                    "case_max_coverage",
                    {"ood_test"},
                    1.0,
                ),
            }
        )

    risk_id = risk_curve(case_metric_rows, {"id_test"})
    risk_mixed = risk_curve(case_metric_rows, {"id_test", "ood_test"})
    gate = choose_gate(risk_mixed)
    tau = float(gate["threshold_tau"]) if gate["threshold_tau"] != "" else float("inf")
    id_served_at_tau = sum(
        row["pool"] == "id_test" and float(row["band_width_mean_norm_90"]) <= tau for row in case_metric_rows
    ) / max(sum(row["pool"] == "id_test" for row in case_metric_rows), 1)
    ood_routed_at_tau = sum(
        row["pool"] == "ood_test" and float(row["band_width_mean_norm_90"]) > tau for row in case_metric_rows
    ) / max(sum(row["pool"] == "ood_test" for row in case_metric_rows), 1)

    primary_cal = next(row for row in calibration_rows if abs(float(row["nominal_coverage"]) - primary_level) < 1e-9)
    primary_ood = next(row for row in ood_rows if abs(float(row["nominal_coverage"]) - primary_level) < 1e-9)
    headlines = {
        "id_coverage_90": float(primary_cal["empirical_coverage_id_test"]),
        "id_sharpness_ratio_90": float(primary_cal["sharpness_ratio_vs_naive"]),
        "ood_coverage_90": float(primary_ood["empirical_coverage_ood"]),
        "ood_band_width_inflation_ratio_90": float(primary_ood["band_width_inflation_ratio"]),
        "ood_spearman_bandwidth_error": ood_spearman,
        "gate_threshold_tau": tau,
        "gate_id_served_at_coverage90": id_served_at_tau,
        "gate_ood_routed_at_threshold": ood_routed_at_tau,
    }

    if args.reuse_checkpoint:
        existing_history_path = metrics_dir / "training_history.csv"
        if existing_history_path.exists() and not history:
            history = read_csv(existing_history_path)
        existing_summary_path = metrics_dir / "run_summary.json"
        if existing_summary_path.exists() and not member_training_sets:
            existing_summary = read_json(existing_summary_path)
            member_training_sets = existing_summary.get("model", {}).get("member_training_sets", [])

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "member_state_dicts": member_states,
            "target_mean": target_mean,
            "target_std": target_std,
            "args": vars(args),
            "input_channels": config["grid"]["input_channels"],
            "output_fields": config["grid"]["output_fields"],
        },
        model_dir / "compact_fno.pt",
    )

    training_history_path = metrics_dir / "training_history.csv"
    write_csv(training_history_path, history, TRAINING_FIELDS)
    pseudo_risk_path = metrics_dir / "pseudo_ood_risk_training.csv"
    write_csv(pseudo_risk_path, pseudo_risk_rows, PSEUDO_RISK_FIELDS)
    write_json(metrics_dir / "risk_head_summary.json", risk_summary | {"feature_names": risk_feature_names})
    prediction_index_path = resolve_path(paths, args.prediction_index)
    write_csv(prediction_index_path, case_metric_rows, PREDICTION_INDEX_FIELDS)
    calibration_path = metrics_dir / "calibration_curve.csv"
    ood_path = metrics_dir / "ood_summary.csv"
    risk_id_path = metrics_dir / "risk_coverage_curve_id.csv"
    risk_mixed_path = metrics_dir / "risk_coverage_curve_mixed.csv"
    write_csv(calibration_path, calibration_rows, CALIBRATION_FIELDS)
    write_csv(ood_path, ood_rows, OOD_FIELDS)
    write_csv(risk_id_path, risk_id, RISK_FIELDS)
    write_csv(risk_mixed_path, risk_mixed, RISK_FIELDS)
    write_json(metrics_dir / "headlines.json", headlines)

    run_summary = {
        "model": {
            "name": "CompactFNO",
            "description": "Small Fourier neural operator with local convolution skip paths; predicts full fixed-grid T and |U| fields from geometry/mask/SDF/parameter channels.",
            "parameter_count_per_member": parameter_count,
            "ensemble_size": args.ensemble_size,
            "ensemble_strategy": args.ensemble_strategy,
            "member_training_sets": member_training_sets,
            "ensemble_parameter_count": parameter_count * args.ensemble_size,
            "width": args.width,
            "modes": args.modes,
            "layers": args.layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": str(device),
            "torch_version": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
            "training_pools": sorted(train_pools),
        },
        "data": {
            "fixed_grid_index": relpath(paths, resolve_path(paths, args.fixed_grid_index)),
            "fixed_grid_index_sha256": file_sha256(resolve_path(paths, args.fixed_grid_index)),
            "extra_train_index_files": [relpath(paths, resolve_path(paths, item)) for item in args.extra_train_index_file],
            "extra_train_index_sha256": {
                relpath(paths, resolve_path(paths, item)): file_sha256(resolve_path(paths, item))
                for item in args.extra_train_index_file
            },
            "extra_parameter_files": [relpath(paths, resolve_path(paths, item)) for item in args.extra_parameter_file],
            "extra_parameter_sha256": {
                relpath(paths, resolve_path(paths, item)): file_sha256(resolve_path(paths, item))
                for item in args.extra_parameter_file
            },
            "case_counts": {
                "train_total": int(len(train_idx)),
                "id_calibration": int(len(cal_idx)),
                "id_test": int(len(id_test_idx)),
                "ood_test": int(len(ood_idx)),
                "by_pool": {pool: int(sum(item == pool for item in bundle.pools)) for pool in sorted(set(bundle.pools))},
            },
            "target_mean": {FIELD_NAMES[idx]: float(target_mean[idx]) for idx in range(len(FIELD_NAMES))},
            "target_std": {FIELD_NAMES[idx]: float(target_std[idx]) for idx in range(len(FIELD_NAMES))},
        },
        "uq": {
            "method": "split conformal over normalized fluid-pixel residuals with spatial calibration residual map, ensemble disagreement, and optional nearest-ID train distance inflation",
            "levels": levels,
            "primary_level": primary_level,
            "distance_gain": args.distance_gain,
            "ensemble_std_gain": args.ensemble_std_gain,
            "risk_head": risk_summary,
            "qhat": {str(level): q_by_level[level] for level in levels},
            "qhat_naive": {str(level): q_naive_by_level[level] for level in levels},
            "qhat_case_q90": {str(level): q_case_quantile_by_level[level] for level in levels},
            "qhat_case_max": {str(level): q_case_max_by_level[level] for level in levels},
            "naive_baseline": "constant-width split-conformal residual band in normalized target units",
            "case_level_interpretation": (
                "case_q90 bands calibrate a per-case residual quantile, so the reported case-level metric is the "
                "fraction of cases with at least the nominal fraction of fluid pixels covered. case_max bands use "
                "the maximum calibration residual per case and approximate simultaneous field coverage."
            ),
            "epistemic_signal": "per-pixel standard deviation across independently seeded FNO ensemble members",
        },
        "artifacts": {
            "model_checkpoint": relpath(paths, model_dir / "compact_fno.pt"),
            "prediction_index": relpath(paths, prediction_index_path),
            "training_history": relpath(paths, training_history_path),
            "pseudo_ood_risk_training": relpath(paths, pseudo_risk_path),
            "risk_head_summary": relpath(paths, metrics_dir / "risk_head_summary.json"),
            "headlines": relpath(paths, metrics_dir / "headlines.json"),
            "calibration_curve": relpath(paths, calibration_path),
            "ood_summary": relpath(paths, ood_path),
            "risk_coverage_curve_id": relpath(paths, risk_id_path),
            "risk_coverage_curve_mixed": relpath(paths, risk_mixed_path),
        },
        "timing": {
            "training_wall_time_sec": round(train_elapsed, 3),
            "total_wall_time_sec": round(time.time() - started, 3),
        },
        "headlines": headlines,
    }
    write_json(metrics_dir / "run_summary.json", run_summary)

    plot_calibration(figures_dir / "calibration_curve.png", calibration_rows)
    plot_risk(figures_dir / "risk_coverage_curve_id.png", risk_id, "ID Risk-Coverage Curve")
    plot_risk(figures_dir / "risk_coverage_curve_mixed.png", risk_mixed, "Mixed ID+OOD Risk-Coverage Curve")
    plot_ood_bandwidth_error(figures_dir / "ood_bandwidth_error.png", case_metric_rows)

    print(f"Wrote {relpath(paths, metrics_dir / 'run_summary.json')}")
    print(f"Wrote {relpath(paths, metrics_dir / 'headlines.json')}")
    print(f"Wrote {relpath(paths, prediction_index_path)}")


if __name__ == "__main__":
    main()
