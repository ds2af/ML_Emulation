# SWE Surrogate Model Release

This release is a clean three-model package for the 2D radial dam-break SWE comparison.

Included models:
- U-Net
- U-Net (LoMix)
- FNO


## Contents
- `configs/default.yaml`
- `scripts/` for evaluation and figure generation
- `src/` model and data code
- `results/` filtered metrics for the three released models
- `figures/` release figures generated from the filtered model set

## Recreate Figures
Run from this release folder:

```powershell
python scripts/evaluate_all.py
python scripts/generate_figures.py --exclude_models cnn,unet_lomix_pinn,pinn --t_indices 10,50,100 --combine_t_indices --rmse_steps 101 --rmse_samples 10
python scripts/plot_training_curves.py --exclude_models cnn,unet_lomix_pinn,pinn
```

## Notes
- The release is configured to show only the available three-model comparison.
- The original project still exists in `MyProject/` with the full historical experiment set.
- The figure scripts support model exclusion so the release stays clean even if the source project contains additional experiments.
