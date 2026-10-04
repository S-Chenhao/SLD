"""Evaluate one SLD checkpoint on an explicitly configured test split.

The supplied checkpoints do not store split IDs. Task 1 defaults to held-out
IC 1, as requested for the supplied CaseB_1 checkpoints. Task 2 defaults to
the PDE-specific partitions in the training notebooks, rather than the
seed-2 partition in the original visualization notebook.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import scipy.io as scio
import torch

from .model import build_model


PDE_CONFIG = {
    "gray_scott": {
        "label": "Gray-Scott",
        "lr_name": "2DGS_IC{ic}_ri10_2x751x32x32.mat",
        "hr_name": "2DGS_IC{ic}_ri10_2x3001x256x256.mat",
        "lr_shape": (2, 751, 32, 32),
        "hr_shape": (2, 3001, 256, 256),
        "lr_offset": 50,
        "hr_offset": 200,
        "stride": 2,
        "task2_seed": 4,
        "task2_validation_ics": [4, 5, 2, 6, 10],
        "task2_test_ics": [1, 3, 7, 8, 9],
    },
    "ginzburg_landau": {
        "label": "Ginzburg-Landau",
        "lr_name": "2DGL_IC{ic}_ri100_2x300x32x32.mat",
        "hr_name": "2DGL_IC{ic}_ri100_2x1200x256x256.mat",
        "lr_shape": (2, 300, 32, 32),
        "hr_shape": (2, 1200, 256, 256),
        "lr_offset": 0,
        "hr_offset": 0,
        "stride": 1,
        "task2_seed": 48,
        "task2_validation_ics": [9, 6, 3, 5, 10],
        "task2_test_ics": [1, 2, 4, 7, 8],
    },
    "burgers": {
        "label": "Burgers",
        "lr_name": "2DBurgers_IC{ic}_ri100_sig040_indIC_2x300x32x32.mat",
        "hr_name": "2DBurgers_IC{ic}_ri100_sig040_indIC_2x1200x256x256.mat",
        "lr_shape": (2, 300, 32, 32),
        "hr_shape": (2, 1200, 256, 256),
        "lr_offset": 0,
        "hr_offset": 0,
        "stride": 1,
        "task2_seed": 1,
        "task2_validation_ics": [3, 2, 5, 1, 4],
        "task2_test_ics": [6, 7, 8, 9, 10],
    },
}

_PDE_ALIASES = {
    "gs": "gray_scott", "grayscott": "gray_scott",
    "gray_scott": "gray_scott", "gray-scott": "gray_scott",
    "gl": "ginzburg_landau", "ginzburglandau": "ginzburg_landau",
    "ginzburg_landau": "ginzburg_landau", "ginzburg-landau": "ginzburg_landau",
    "complex_ginzburg_landau": "ginzburg_landau", "burgers": "burgers",
}
_TASK_ALIASES = {
    "1": "task1_unseen_ic", "task1": "task1_unseen_ic",
    "task1_unseen_ic": "task1_unseen_ic", "caseb": "task1_unseen_ic",
    "2": "task2_later_time", "task2": "task2_later_time",
    "task2_later_time": "task2_later_time", "casec": "task2_later_time",
}
_SUM_KEYS = ("squared_error", "truth_squared", "absolute_error", "count")


def metric_sums(pred, truth) -> dict:
    """Sufficient statistics over every element, without averaging batches."""
    pred, truth = torch.as_tensor(pred), torch.as_tensor(truth)
    if pred.shape != truth.shape or pred.numel() == 0:
        raise ValueError(f"Nonempty matching shapes required: {pred.shape}, {truth.shape}")
    if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
        raise ValueError("Prediction or target contains NaN/Inf.")
    # Float64 arithmetic avoids float32 sum overflow and gives stable pooling.
    p = pred.to(dtype=torch.float64)
    y = truth.to(device=p.device, dtype=torch.float64)
    d = p - y
    result = {
        "squared_error": float(torch.sum(d * d).item()),
        "truth_squared": float(torch.sum(y * y).item()),
        "absolute_error": float(torch.sum(torch.abs(d)).item()),
        "count": int(y.numel()),
    }
    if any(not math.isfinite(result[k]) for k in _SUM_KEYS):
        raise ValueError("Metric statistics are not finite.")
    return result


def metrics_from_sums(sums: Mapping) -> dict:
    """Pooled relative L2 and MAE, over all supplied elements."""
    if any(k not in sums for k in _SUM_KEYS):
        raise ValueError(f"Metric sums require {_SUM_KEYS}.")
    if sums["count"] <= 0 or any(
        not math.isfinite(float(sums[k])) or sums[k] < 0 for k in _SUM_KEYS
    ):
        raise ValueError("Metric sums must be finite and nonnegative with positive count.")
    return {
        "relative_l2": math.sqrt(sums["squared_error"] / (sums["truth_squared"] + 1e-12)),
        "mae": sums["absolute_error"] / (sums["count"] + 1e-12),
    }


def _empty_sums():
    return {"squared_error": 0.0, "truth_squared": 0.0, "absolute_error": 0.0, "count": 0}


def _add_sums(total, extra):
    for key in _SUM_KEYS:
        total[key] += extra[key]


def _indices(values, name, *, maximum=None):
    values = list(values)
    if not values or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        for v in values
    ):
        raise ValueError(f"{name} must be a nonempty list of integer indices.")
    values = [int(v) for v in values]
    minimum = 1 if name == "test_ics" else 0
    if len(set(values)) != len(values) or any(
        v < minimum or (maximum is not None and v > maximum) for v in values
    ):
        raise ValueError(f"{name} contains duplicates or out-of-range indices.")
    return values


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path, device):
    """Load tensor data only, then restore parameters and normalization strictly."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise ValueError("Checkpoint must contain a model state dictionary.")
    state = loaded.get("model_state_dict", loaded.get("state_dict", loaded))
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("Expected tensor model_state_dict; history-only files cannot be evaluated.")
    norm_keys = ("shift_in.mean", "shift_in.std", "shift_out.mean", "shift_out.std")
    missing_norm = [key for key in norm_keys if key not in state]
    if missing_norm:
        raise ValueError(f"Checkpoint is missing saved normalization buffers: {missing_norm}")
    for key, value in state.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint tensor is not finite: {key}")
    for key in ("shift_in.std", "shift_out.std"):
        if not torch.all(state[key] > 0):
            raise ValueError(f"Normalization standard deviations must be positive: {key}")
    model = build_model(device=device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint does not match the published SLD architecture: {exc}") from exc
    model.eval()
    metadata = {}
    # These are recorded values from training, not test results recomputed here.
    for key in ("epoch", "best_error"):
        value = loaded.get(key)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.item()
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metadata[key] = value
    normalization = {key: state[key].detach().cpu().reshape(-1).tolist() for key in norm_keys}
    return model, metadata, normalization


def _load_uv(path, expected_shape):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    content = scio.loadmat(path, variable_names=["uv"])
    if "uv" not in content:
        raise ValueError(f"MAT file has no 'uv' array: {path}")
    uv = content["uv"]
    if uv.shape != expected_shape or uv.dtype.kind not in "fiu":
        raise ValueError(f"Expected real numeric uv{expected_shape}, got {uv.shape}/{uv.dtype}: {path}")
    return uv


def _raw_starts(cfg, lr_start):
    stride = cfg["stride"]
    return {
        "postprocess_lr_start": lr_start,
        "postprocess_hr_start": 4 * lr_start,
        "raw_lr_start": cfg["lr_offset"] + stride * lr_start,
        "raw_hr_start": cfg["hr_offset"] + stride * 4 * lr_start,
        "raw_lr_last": cfg["lr_offset"] + stride * (lr_start + 24),
        "raw_hr_last": cfg["hr_offset"] + stride * (4 * lr_start + 99),
        "raw_sampling_stride": stride,
    }


def _load_windows(dataset_dir, cfg, ic, segments):
    """Release the raw MAT pair before returning independent float32 windows."""
    lr_path = dataset_dir / cfg["lr_name"].format(ic=ic)
    hr_path = dataset_dir / cfg["hr_name"].format(ic=ic)
    lr = _load_uv(lr_path, cfg["lr_shape"])
    hr = _load_uv(hr_path, cfg["hr_shape"])
    stride = cfg["stride"]
    lr = lr[:, cfg["lr_offset"]::stride]
    hr = hr[:, cfg["hr_offset"]::stride]
    windows = []
    for start in segments:
        hs = 4 * start
        if start + 25 > lr.shape[1] or hs + 100 > hr.shape[1]:
            raise ValueError(f"Window LR start {start} exceeds available frames for IC {ic}.")
        # Copy after slicing: even float32 source files cannot retain a raw-array view.
        lwin = np.array(lr[:, start:start + 25].transpose(1, 0, 2, 3),
                        dtype=np.float32, order="C", copy=True)
        hwin = np.array(hr[:, hs:hs + 100].transpose(1, 0, 2, 3),
                        dtype=np.float32, order="C", copy=True)
        if not np.isfinite(lwin).all() or not np.isfinite(hwin).all():
            raise ValueError(f"Selected dataset window contains NaN/Inf: IC {ic}, LR start {start}.")
        windows.append((start, torch.from_numpy(lwin), torch.from_numpy(hwin)))
    # No raw MAT arrays or views are returned. Windows own their CPU float32 storage.
    del lr, hr
    return windows, [lr_path.name, hr_path.name]


def evaluate(pde, task, data_root=None, checkpoint=None, device=None,
             output_path=None, test_ics=None, segments=None) -> dict:
    """Evaluate one explicit checkpoint; return small JSON-compatible results.

    ``data_root`` contains the canonical PDE subfolders. By default it is the
    repository's ``data`` folder. ``output_path`` is optional; when supplied,
    the same result is saved as JSON. No predictions or figures are exported.
    ``test_ics`` and ``segments`` override the documented defaults explicitly.
    """
    pde_key = _PDE_ALIASES.get(str(pde).strip().lower().replace(" ", "_"))
    task_key = _TASK_ALIASES.get(str(task).strip().lower())
    if pde_key is None or task_key is None:
        raise ValueError(f"Unknown PDE/task: {pde!r}/{task!r}")
    if checkpoint is None:
        raise ValueError("Supply one explicit checkpoint path.")
    cfg = PDE_CONFIG[pde_key]
    later = task_key == "task2_later_time"
    default_ics = cfg["task2_test_ics"] if later else [1]
    default_segments = [225, 250] if later else [100, 150, 200, 250]
    ids = _indices(default_ics if test_ics is None else test_ics, "test_ics", maximum=10)
    starts = _indices(default_segments if segments is None else segments, "segments")
    chosen_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if chosen_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; choose device='cpu'.")
    repo = Path(__file__).resolve().parents[1]
    dataset_dir = Path(data_root or repo / "data").expanduser().resolve() / pde_key
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    model, recorded_metadata, normalization = _load_checkpoint(checkpoint_path, chosen_device)
    total = _empty_sums()
    per_ic, per_window, input_files = [], [], []
    with torch.inference_mode():
        for ic in ids:
            windows, filenames = _load_windows(dataset_dir, cfg, ic, starts)
            input_files.extend(filenames)
            ic_sums = _empty_sums()
            for start, lr, hr in windows:
                prediction = model(lr.unsqueeze(1).to(chosen_device))
                truth = hr.unsqueeze(1).to(chosen_device)
                if prediction.shape != truth.shape:
                    raise ValueError(f"Unexpected SLD output shape: {prediction.shape}, expected {truth.shape}")
                sums = metric_sums(prediction, truth)
                _add_sums(total, sums)
                _add_sums(ic_sums, sums)
                per_window.append({"ic": ic, "lr_start": start,
                                   "metrics": metrics_from_sums(sums), "sums": sums})
                del prediction, truth
            per_ic.append({"ic": ic, "metrics": metrics_from_sums(ic_sums), "sums": ic_sums})
            del windows
    result = {
        "pde": pde_key, "task": task_key, "model": "SLD",
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path),
                       "recorded_training_metadata": recorded_metadata,
                       "metadata_note": "best_error is saved validation metadata, not evaluated test error.",
                       "normalization_buffers": normalization},
        "configuration": {
            "device": str(chosen_device), "data_directory": str(dataset_dir),
            "test_ics": ids, "lr_starts": starts,
            "test_ics_overridden": test_ics is not None,
            "segments_overridden": segments is not None,
            "default_split_seed": cfg["task2_seed"] if later else None,
            "default_training_ics": list(range(1, 11)) if later else None,
            "default_validation_ics": cfg["task2_validation_ics"] if later else None,
            "split_note": ("PDE-specific default training partition; checkpoint does not store split IDs."
                           if later else "Checkpoint CaseB_1 uses held-out IC 1; train/validation IDs are not stored."),
            "target_frames_per_window": 100,
            "raw_frame_mapping": [_raw_starts(cfg, start) for start in starts],
            "preprocessing": ("LR[:,50::2], HR[:,200::2]; starts apply after slicing."
                              if pde_key == "gray_scott" else "Stored arrays without temporal trimming."),
            "input_files": input_files,
        },
        "aggregation": "Pooled over all 100 target frames, both channels, all pixels and configured test windows; one checkpoint.",
        "metrics": metrics_from_sums(total), "sums": total,
        "per_ic": per_ic, "per_window": per_window,
    }
    if output_path is not None:
        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result
