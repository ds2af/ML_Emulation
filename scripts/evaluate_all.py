"""
Aggregate measured metrics from trained models into results/metrics_summary.json.

This script is strict by design:
- No illustrative/template substitutions.
- Missing model files are reported as source="missing" with NaN metrics.

Usage
-----
    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --results_dir results
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_ORDER = ["cnn", "cnn_lstm", "unet", "unet_lomix", "unet_lomix_pinn", "fno", "pinn"]
MODEL_DISPLAY_NAMES = {
    "cnn": "CNN",
    "cnn_lstm": "CNN + LSTM",
    "unet": "U-Net",
    "unet_lomix": "U-Net (LoMix)",
    "unet_lomix_pinn": "U-Net (LoMix + PINN)",
    "fno": "FNO",
    "pinn": "PINN",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    return p.parse_args()


def _nan() -> float:
    return float("nan")


def load_metric(results_dir: Path, model_key: str) -> dict:
    path = results_dir / f"{model_key}_metrics.json"
    if not path.exists():
        print(f"  [missing] {path.name}")
        return {
            "model": model_key,
            "rmse": _nan(),
            "rel_l2": _nan(),
            "max_error": _nan(),
            "inference_time_s": _nan(),
            "n_params": 0,
            "source": "missing",
        }

    with path.open() as f:
        data = json.load(f)

    print(f"  [ok] {path.name}")
    data["source"] = "measured"
    return data


def _is_finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v)


def main():
    args = parse_args()
    res_dir = ROOT / args.results_dir
    res_dir.mkdir(parents=True, exist_ok=True)

    print("\n[evaluate_all] Collecting measured model metrics ...")
    summary: dict[str, dict] = {}

    for key in MODEL_ORDER:
        metrics = load_metric(res_dir, key)
        summary[key] = {
            "display_name": MODEL_DISPLAY_NAMES[key],
            "rmse": metrics.get("rmse", _nan()),
            "rel_l2": metrics.get("rel_l2", _nan()),
            "max_error": metrics.get("max_error", _nan()),
            "inference_time_s": metrics.get("inference_time_s", _nan()),
            "n_params": metrics.get("n_params", 0),
            "source": metrics.get("source", "missing"),
        }

    pinn_time = summary["pinn"]["inference_time_s"]
    for key in MODEL_ORDER:
        t = summary[key]["inference_time_s"]
        if _is_finite(pinn_time) and _is_finite(t) and t > 0:
            summary[key]["speedup_vs_pinn"] = round(float(pinn_time) / float(t), 2)
        else:
            summary[key]["speedup_vs_pinn"] = _nan()

    out_path = res_dir / "metrics_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n[evaluate_all] ----- Results Summary -----")
    print(f"{'Model':<16}{'RMSE':>12}{'Rel-L2':>12}{'Inf.time(s)':>14}{'Speedup':>10}{'Source':>12}")
    print("-" * 76)
    for key in MODEL_ORDER:
        row = summary[key]
        rmse = row["rmse"]
        rel = row["rel_l2"]
        t = row["inference_time_s"]
        sp = row["speedup_vs_pinn"]
        rmse_s = f"{rmse:.4e}" if _is_finite(rmse) else "nan"
        rel_s = f"{rel:.4e}" if _is_finite(rel) else "nan"
        t_s = f"{t:.3f}" if _is_finite(t) else "nan"
        sp_s = f"{sp:.2f}" if _is_finite(sp) else "nan"
        print(f"{row['display_name']:<16}{rmse_s:>12}{rel_s:>12}{t_s:>14}{sp_s:>10}{row['source']:>12}")

    print(f"\n[evaluate_all] Summary saved -> {out_path}")


if __name__ == "__main__":
    main()
