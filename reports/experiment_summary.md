# Experiment Summary

**Project:** ML for CFD – Comparative Study of Surrogate Models  
**Author:** Dipesh Shrestha  
**Date:** March 2026

---

## 1. Dataset Assumptions

| Assumption | Detail |
|---|---|
| File | `data/2D_rdb_NA_NA.h5` |
| PDE type | 2D reaction-diffusion (PDEBench `rdb` class) |
| HDF5 structure | Groups keyed by seed string; `data` key, shape `[T, H, W, C]` |
| T (timesteps) | 101 per trajectory |
| H × W (grid) | 128 × 128 (inferred from data at runtime; dataset is read-once) |
| C (channels) | 1 (scalar field u) |
| Physical domain | x, y ∈ [0, 1] normalized; t ∈ [0, 1] normalized |
| PDE coefficients | ν (diffusion), ρ (reaction) treated as **unknown** — **ASSUMPTION**: a constant approximation ν = 0.5, ρ = 1.0 is used for the PINN. Actual coefficients may differ per seed. |
| Boundary conditions | Not explicitly known; assumed periodic or Dirichlet (consistent with PDEBench `rdb` class documentation) |

---

## 2. Input/Output Tensor Conventions

All models follow a consistent spatial-first, channels-last internal convention for data loading (`SWEDataset.__getitem__`):

```
Full trajectory:    [H, W, T, C]   = [128, 128, 101, 1]
Context window xx:  [H, W, T0, C]  = [128, 128, 10,  1]  (initial_step=10)
Target yy:          [H, W, T,  C]  = [128, 128, 101, 1]
```

Each model reformats its input internally:

| Model | Input to forward | Output of forward |
|---|---|---|
| MLP | `[B, H*W*T0*C]` (flattened) | `[B, H, W, 1, C]` |
| CNN | `[B, T0*C, H, W]` (channels-first) | `[B, C, H, W]` |
| U-Net | `[B, T0*C, H, W]` (channels-first) | `[B, C, H, W]` |
| FNO | `[B, H, W, T0*C]` + `[B, H, W, 2]` grid | `[B, H, W, 1, C]` |
| PINN | `[N, 3]` (x, y, t) coordinates | `[N, C]` |

---

## 3. Train/Validation/Test Split

```
All seeds → sorted alphabetically → deterministic index split
  Train:      seeds[0 : n_train]
  Validation: seeds[n_train : n_train + n_val]
  Test:       seeds[n_train + n_val :]

Default ratios: test=10%, val=10%, train=80%
Fixed: no shuffling before split (fully deterministic from data order)
```

---

## 4. Preprocessing Choices

| Step | Choice | Rationale |
|---|---|---|
| Normalization | Channel-wise z-score (mean/std from train set) | Ensures unit-variance inputs; improves optimizer conditioning |
| Time axis convention | `[H, W, T, C]` spatial-first | Consistent with FNO paper data convention |
| Context window | `initial_step = 10` | Matches existing U-Net training configuration |
| Autoregressive rollout | Pushforward trick (Brandstetter et al., 2022) | More stable than full AR; computes loss only on last `unroll_step=20` steps |

---

## 5. Implemented vs. Future Work

### ✅ Implemented in this project

| Component | Status |
|---|---|
| `SWEDataset` / `SWEDatasetWithGrid` | Complete |
| `Normalizer` utility | Complete |
| MLP model | Complete |
| CNN2d (residual) | Complete |
| UNet2d (refactored, GELU) | Complete |
| UNet2dLoMix (multi-scale fusion) | Complete |
| FNO2d (4 Fourier layers) | Complete |
| PINN (pure-PyTorch, autograd PDE residual) | Complete |
| Shared `Trainer` (AR + single-step) | Complete |
| `ExperimentLogger` (CSV + JSON) | Complete |
| `evaluate_all.py` (unified metrics) | Complete |
| `generate_figures.py` (6 publication figures) | Complete |
| README.md | Complete |
| Academic project report | Complete |

### 🔲 Identified Future Work

| Item | Reason Not Implemented |
|---|---|
| Multi-channel SWE (h, u, v) | Dataset uses scalar field; multi-channel requires dataset with vector outputs |
| Cross-resolution FNO evaluation | Would require training on one resolution and testing on another; out of dataset scope |
| PINN with known per-scenario ν, ρ | Coefficients not provided in HDF5 metadata |
| Graph Neural Network surrogate | Requires unstructured mesh data; dataset is on structured grid |
| Uncertainty quantification (dropout / ensemble) | Out of scope for initial comparison |
| CFD solver runtime baseline | No CFD solver is bundled; speedup comparisons are relative to PINN inference |
| Hyperparameter optimization | Fixed hyperparameters used for fair comparison; tuning would favor each model differently |

---

## 6. Limitations and Caveats

1. **PINN physical parameters:** The PINN assumes constant ν = 0.5, ρ = 1.0. If the dataset was generated with different parameter values (PDEBench rdb uses varying ν ∈ [0.5, 2.0] and ρ ∈ [0.5, 2.0] in some configurations), this assumption introduces modeling error. PINN residuals should be interpreted as approximate physics enforcement, not exact.

2. **Illustrative metrics:** If training scripts are not run before generating figures, `evaluate_all.py` uses template values from the ML-for-CFD literature. These are clearly labeled `ILLUSTRATIVE_TEMPLATE` in all output files and figures. They reflect the expected relative ordering and approximate magnitudes, not measured results.

3. **Hardware dependence:** Inference times reported are absolute wall-clock values on the machine where evaluation is run. GPU availability significantly affects measured speedups.

4. **Single-channel evaluation:** All comparisons use channel 0 of the scalar field. For future vector-valued datasets, metrics should be averaged across channels.

5. **Autoregressive error accumulation:** For U-Net and FNO, errors accumulate over the autoregressive rollout. The reported RMSE is computed over the full rollout (steps 10–100), which penalizes models whose errors grow over time. Single-step models (MLP, CNN) are evaluated at step 10 only, making direct RMSE comparison across modes non-trivial.

6. **No physics-based solver baseline:** The project compares ML models against each other. A reference CFD solver runtime would allow absolute speedup computation; this was not implemented as no solver is bundled with the dataset.
