"""
src/data/dataset.py
===================
Shared dataset loader for the PDEBench 2D reaction-diffusion / shallow-water
HDF5 file (``2D_rdb_NA_NA.h5``).

The file contains groups keyed by seed string.  Each group has a ``data``
array of shape ``[T, H, W, C]`` where:
    T  = total time steps  (typically 101)
    H  = grid height       (typically 128)
    W  = grid width        (typically 128)
    C  = number of channels (typically 1 for the scalar SWE dataset)

The dataset is split deterministically via a fixed seed:
    • Training   – first (1 - test_ratio - val_ratio) fraction of seeds
    • Validation – next val_ratio fraction
    • Test        – last test_ratio fraction

Each sample returned by __getitem__ is a tuple:
    xx : [H, W, initial_step, C]  – autoregressive context window
    yy : [H, W, T, C]             – full trajectory (ground truth)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SWEDataset(Dataset):
    """
    Torch Dataset wrapping the PDEBench 2D HDF5 file.

    Parameters
    ----------
    filename : str
        HDF5 file stem (e.g. ``"2D_rdb_NA_NA"``).
    saved_folder : str
        Directory that contains ``<filename>.h5``.
    initial_step : int
        Number of time steps used as the autoregressive input context.
    if_test : bool
        If True, return the test split; otherwise return train or val split.
    if_val : bool
        If True (and if_test is False), return the validation split.
    test_ratio : float
        Fraction of seeds reserved for the test split.
    val_ratio : float
        Fraction of seeds reserved for the validation split.
    max_samples : int, optional
        If > 0, cap the number of samples (for quick-mode runs).
    """

    def __init__(
        self,
        filename: str,
        saved_folder: str = "../../data/",
        initial_step: int = 10,
        if_test: bool = False,
        if_val: bool = False,
        test_ratio: float = 0.10,
        val_ratio: float = 0.10,
        max_samples: int = -1,
    ) -> None:
        data_file = f"{filename}.h5"
        workspace_root = Path(__file__).resolve().parents[3]
        project_root = Path(__file__).resolve().parents[2]

        search_dirs: list[Path] = []
        raw_saved = Path(saved_folder)
        if raw_saved.is_absolute():
            search_dirs.append(raw_saved)
        else:
            # Resolve relative to current working directory and project root.
            search_dirs.append(raw_saved.resolve())
            search_dirs.append((project_root / raw_saved).resolve())

        # Hard fallback: always try the workspace data folder.
        search_dirs.append((workspace_root / "data").resolve())

        self.file_path = search_dirs[0] / data_file
        for base_dir in search_dirs:
            candidate = base_dir / data_file
            if candidate.exists():
                self.file_path = candidate
                break

        if not self.file_path.exists():
            tried = "\n  - " + "\n  - ".join(str(d / data_file) for d in search_dirs)
            raise FileNotFoundError(
                "Dataset not found. Tried:" + tried
            )
        self.initial_step = initial_step

        # Discover all seed keys and perform a deterministic split
        with h5py.File(self.file_path, "r") as f:
            all_keys = sorted(f.keys())

        n_total = len(all_keys)

        # Keep historical behavior (at least one sample) for positive ratios,
        # but allow explicit 0.0 ratio to disable a split (e.g., true 90/10).
        n_test = 0 if test_ratio <= 0 else max(1, math.floor(n_total * test_ratio))
        n_val = 0 if val_ratio <= 0 else max(1, math.floor(n_total * val_ratio))

        # Guard against over-allocation when small datasets and large ratios collide.
        if n_test + n_val >= n_total:
            overflow = (n_test + n_val) - (n_total - 1)
            if n_val >= overflow:
                n_val -= overflow
            else:
                overflow -= n_val
                n_val = 0
                n_test = max(0, n_test - overflow)

        n_train = n_total - n_test - n_val

        if if_test:
            keys = all_keys[n_train + n_val :]
        elif if_val:
            keys = all_keys[n_train : n_train + n_val]
        else:
            keys = all_keys[:n_train]

        if max_samples > 0:
            keys = keys[: max_samples]

        self.data_list = np.array(keys)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        xx : FloatTensor [H, W, initial_step, C]
        yy : FloatTensor [H, W, T, C]
        """
        with h5py.File(self.file_path, "r") as f:
            seed_group = f[self.data_list[idx]]
            # HDF5 shape: [T, H, W, C]
            data = np.array(seed_group["data"], dtype=np.float32)

        data = torch.tensor(data)  # [T, H, W, C]

        # Rearrange to [H, W, T, C] — spatial dims first, then time, then channel
        permute_idx = list(range(1, len(data.shape) - 1))  # [1, 2]
        permute_idx.extend([0, len(data.shape) - 1])       # [1, 2, 0, 3]
        data = data.permute(permute_idx)                    # [H, W, T, C]

        xx = data[..., : self.initial_step, :]  # context
        return xx, data


# ---------------------------------------------------------------------------
# FNO-style dataset: includes spatial grid tensor
# ---------------------------------------------------------------------------

class SWEDatasetWithGrid(SWEDataset):
    """
    Extends SWEDataset to also return a normalized spatial grid tensor, as
    expected by FNO2d.  The grid has shape [H, W, 2] with x and y coordinates
    linearly spaced in [0, 1].
    """

    def __getitem__(self, idx: int):
        xx, yy = super().__getitem__(idx)
        H, W = xx.shape[0], xx.shape[1]

        # Build a spatial grid [H, W, 2]
        xs = torch.linspace(0.0, 1.0, H)
        ys = torch.linspace(0.0, 1.0, W)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

        return xx, yy, grid


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: dict,
    *,
    with_grid: bool = False,
    max_samples: int = -1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a config dict.

    Parameters
    ----------
    cfg : dict
        Configuration dict (typically loaded from configs/default.yaml).
    with_grid : bool
        If True, return SWEDatasetWithGrid loaders (needed for FNO).
    max_samples : int
        Cap on samples per split (quick mode).

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    data_cfg = cfg["data"]
    ds_cls = SWEDatasetWithGrid if with_grid else SWEDataset

    common_kwargs = dict(
        filename=data_cfg["filename"],
        saved_folder=data_cfg["base_path"],
        initial_step=data_cfg["initial_step"],
        test_ratio=data_cfg.get("test_ratio", 0.10),
        val_ratio=data_cfg.get("val_ratio", 0.10),
        max_samples=max_samples,
    )

    train_ds = ds_cls(if_test=False, if_val=False, **common_kwargs)
    val_ds = ds_cls(if_test=False, if_val=True, **common_kwargs)
    test_ds = ds_cls(if_test=True, **common_kwargs)

    tr_cfg = cfg["training"]
    num_workers = data_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
