# MyProject 3-Model Release

## Summary
This release packages the SWE surrogate comparison with only the three models requested for the clean release set:
- U-Net
- U-Net (LoMix)
- FNO

CNN, U-Net (LoMix + PINN), and PINN are excluded from the release artifacts and regenerated figures.

## What Is Included
- Source code for data loading, models, metrics, and figure generation
- Configuration file for the main experiment setup
- Model checkpoints and logs for the released models
- Filtered metrics summary for the released models
- Release figures generated from the three-model subset

## Key Results
- U-Net: RMSE 0.7881, Relative L2 0.7614
- U-Net (LoMix): RMSE 0.08536, Relative L2 0.08213
- FNO: RMSE 0.07476, Relative L2 0.07193

## Generated Figures
- fig1_workflow.png
- fig2_architectures.png
- fig3_field_comparison.png
- fig3_field_comparison_multistep.png
- fig4_error_comparison.png
- fig4_error_comparison_bar.png
- fig5_speedup.png
- training_loss_curves.png
- val_loss_curves.png
- rmse_bar.png
- rel_l2_bar.png
- metrics_combined.png
- training_loss_200ep.png

## Reproducibility
The release is reproducible from the release folder by running:

```powershell
python scripts/evaluate_all.py
python scripts/generate_figures.py --exclude_models cnn,unet_lomix_pinn,pinn --t_indices 10,50,100 --combine_t_indices --rmse_steps 101 --rmse_samples 10
python scripts/plot_training_curves.py --exclude_models mlp,cnn,unet_lomix_pinn,pinn
```

## Notes
- The release is intentionally scoped to the three-model comparison only.
- The source project in MyProject/ remains intact for the full historical experiment set.
- Optional figure 6 is skipped because the alternate OOD dataset is not present in this workspace.
