# GitHub Release Package Guide

## Recommended Release Name
`MyProject-3model-swe-comparison`

## Suggested Tag
`v3models-release`

## What to Attach
- The entire `MyProject_release_3models/` folder as a zip archive
- `RELEASE_NOTES.md`
- `results/metrics_summary.json`
- `figures/` PNG outputs

## Suggested GitHub Release Body
Use the text in `RELEASE_NOTES.md` as the release description.

## If You Want a Local Git Tag
From the repository root:

```powershell
git tag v3models-release
```

## If You Want to Push the Tag
```powershell
git push origin v3models-release
```

## If You Want a GitHub Release Draft
1. Create the tag locally.
2. Push the tag to GitHub.
3. Create a release on GitHub and upload the release folder archive.
4. Paste the contents of `RELEASE_NOTES.md` into the release body.

## Verification Checklist
- Release figures only show U-Net, U-Net (LoMix), and FNO.
- `results/metrics_summary.json` contains only the three released models.
- The release folder can be opened and run from its own root.
