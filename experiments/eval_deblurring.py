"""
Evaluation script for Poisson deblurring: PSNR / SSIM / LPIPS table and scatter plot.

Reproduces the state-of-the-art comparison from:
    Klatzer et al., "Efficient Bayesian Computation Using Plug-and-Play Priors
    for Poisson Inverse Problems", 2026.

Data is auto-downloaded from Google Drive on first run (see GDRIVE_FOLDER_ID below).
Expected layout under --data_dir (default: <repo>/data/eval/):

    alpha_{a}_kernel_{k}/
        label/          # ground-truth PNGs (00000.png – 00009.png)
        BM3D/recon_n/
        BPnP/recon_n/
        SKROCK_c20/recon_n/
        SKROCK_c300/recon_n/
        Unrolled/recon_n/
        MLA/recon_n/

Usage examples:
    python experiments/eval_deblurring.py                       # all conditions
    python experiments/eval_deblurring.py --alpha 20 --kernel 4
    python experiments/eval_deblurring.py --download            # force re-download
"""

import argparse
import os
from pathlib import Path

import cv2
import gdown
import lpips
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from natsort import os_sorted
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "eval"
RESULTS_DIR = ROOT_DIR / "results" / "eval"

# GDrive folder containing all conditions.
# Set this to the folder ID after uploading the data.
# Share the folder as "Anyone with the link can view" to allow gdown access.
GDRIVE_FOLDER_ID = "https://drive.google.com/drive/folders/1sipEFGynrffO-VfxN-XpGeiOoEbyqiRQ?usp=drive_link"

# Experimental conditions present in the data
ALL_ALPHAS = [5, 10, 20]
ALL_KERNELS = [4, 9]

# Methods present per condition. Conditions not listed here use ALL_METHODS.
METHODS_BY_CONDITION = {
    (2, 4):  ["SKROCK_c20", "MLA"],
    (2, 9):  ["SKROCK_c20", "MLA"],
}
ALL_METHODS = ["BM3D", "BPnP", "SKROCK_c20", "SKROCK_c300", "Unrolled", "MLA"]

# Display names for the paper
METHOD_LABELS = {
    "BM3D":      "PIP",
    "BPnP":      "PnP-BPG",
    "SKROCK_c20":  "RPnP-SKROCK (c=20)",
    "SKROCK_c300": "RPnP-SKROCK (c=300)",
    "Unrolled":  "PhD-Net",
    "MLA":       "PnP-MLA",
}

# Scatter plot visual style
SCATTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
SCATTER_MARKERS = ["+", "*", "o", "p", "D", "x"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def imread_float(path: str, n_channels: int = 3) -> np.ndarray:
    """Read image as float32 array in [0, 1], shape (1, C, H, W)."""
    if n_channels == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img[None, None]  # (1, 1, H, W)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)[None]  # (1, C, H, W)
    return img.astype(np.float32) / 255.0


def download_data(data_dir: Path, folder_id: str) -> None:
    """Download evaluation data from Google Drive if not already present."""
    if folder_id == "REPLACE_WITH_YOUR_GDRIVE_FOLDER_ID":
        raise RuntimeError(
            "Set GDRIVE_FOLDER_ID in eval_deblurring.py before downloading.\n"
            "Upload the data folder to Google Drive and paste the folder ID."
        )
    print(f"Downloading evaluation data to {data_dir} ...")
    data_dir.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(id=folder_id, output=str(data_dir), quiet=False)


def get_methods_for_condition(alpha: int, kernel: int) -> list[str]:
    return METHODS_BY_CONDITION.get((alpha, kernel), ALL_METHODS)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(data_dir: Path, alpha: int, kernel: int) -> pd.DataFrame:
    """Compute PSNR / SSIM / LPIPS for all methods in one condition."""
    loss_fn = lpips.LPIPS(net="alex")

    cond_dir = data_dir / f"alpha_{alpha}_kernel_{kernel}"
    label_dir = cond_dir / "label"

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label directory not found: {label_dir}\n"
            "Run with --download to fetch the data, or check GDRIVE_FOLDER_ID."
        )

    label_paths = os_sorted(glob(str(label_dir / "*.png")))
    if not label_paths:
        raise FileNotFoundError(f"No PNG files found in {label_dir}")

    methods = get_methods_for_condition(alpha, kernel)
    rows = []

    for method in methods:
        recon_dir = cond_dir / method / "recon_n"
        if not recon_dir.exists():
            print(f"  [skip] {method}: recon_n not found at {recon_dir}")
            continue

        recon_paths = os_sorted(glob(str(recon_dir / "*.png")))
        if len(recon_paths) != len(label_paths):
            print(
                f"  [warn] {method}: {len(recon_paths)} recons vs "
                f"{len(label_paths)} labels — skipping"
            )
            continue

        for label_path, recon_path in zip(label_paths, recon_paths):
            img_id = Path(label_path).stem  # e.g. "00003"

            gt = imread_float(label_path)
            rec = imread_float(recon_path)

            psnr_val = psnr_fn(gt, rec, data_range=1.0)
            ssim_val = ssim_fn(gt[0], rec[0], data_range=1.0, channel_axis=0)

            gt_scaled = torch.from_numpy(gt * 2 - 1).float()
            rec_scaled = torch.from_numpy(rec * 2 - 1).float()
            lpips_val = loss_fn(rec_scaled, gt_scaled).item()

            rows.append({
                "alpha": alpha,
                "kernel": kernel,
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "image_id": img_id,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": lpips_val,
            })

            print(
                f"  {method:20s}  img={img_id}  "
                f"PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}  LPIPS={lpips_val:.4f}"
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return per-method mean/std summary."""
    summary = (
        df.groupby("label")[["psnr", "ssim", "lpips"]]
        .agg(["mean", "std"])
        .round(4)
    )
    # Format for display
    display = summary.copy()
    for col in ["psnr", "ssim", "lpips"]:
        fmt = ".2f" if col == "psnr" else ".3f"
        display[(col, "mean")] = summary[(col, "mean")].map(lambda v, f=fmt: format(v, f))
        display[(col, "std")] = summary[(col, "std")].map(lambda v, f=fmt: format(v, f))
    print(display.sort_values(by=("psnr", "mean"), ascending=False).to_string())
    return summary


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def plot_scatter(df: pd.DataFrame, alpha: int, kernel: int, out_dir: Path) -> None:
    """PSNR vs LPIPS scatter.

    Color = method, marker shape = image index.
    Legend shows method colors; a secondary marker legend shows image indices.
    """
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 14
    plt.rcParams["lines.markersize"] = 10

    fig, ax = plt.subplots()

    labels = sorted(df["label"].unique())
    image_ids = sorted(df["image_id"].unique())

    color_map = {label: SCATTER_COLORS[i % len(SCATTER_COLORS)] for i, label in enumerate(labels)}
    marker_map = {img_id: SCATTER_MARKERS[i % len(SCATTER_MARKERS)] for i, img_id in enumerate(image_ids)}

    # One scatter call per (method, image) so each point gets color + marker
    for label in labels:
        sub = df[df["label"] == label]
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.scatter(
                row["psnr"], row["lpips"],
                c=color_map[label],
                marker=marker_map[row["image_id"]],
                label=label if i == 0 else None,
            )

    ax.legend(fontsize=10, loc="upper right", title="Method")

    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("LPIPS")

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"scatter_alpha{alpha}_kernel{kernel}.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Poisson deblurring results (PSNR/SSIM/LPIPS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--alpha", type=int, default=None, choices=ALL_ALPHAS,
                        help="Poisson level to evaluate (default: all)")
    parser.add_argument("--kernel", type=int, default=None, choices=ALL_KERNELS,
                        help="Kernel index to evaluate (default: all)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to eval data (default: <repo>/data/eval)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Output directory (default: <repo>/results/eval)")
    parser.add_argument("--download", action="store_true",
                        help="Download data from Google Drive (re-downloads if already present)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip scatter plot generation")
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    if args.download:
        download_data(data_dir, GDRIVE_FOLDER_ID)

    alphas = [args.alpha] if args.alpha is not None else ALL_ALPHAS
    kernels = [args.kernel] if args.kernel is not None else ALL_KERNELS

    all_dfs = []

    for alpha in alphas:
        for kernel in kernels:
            print(f"\n{'='*60}")
            print(f"alpha={alpha}, kernel={kernel}")
            print(f"{'='*60}")

            try:
                df = compute_metrics(data_dir, alpha, kernel)
            except FileNotFoundError as e:
                print(f"[skip] {e}")
                continue

            all_dfs.append(df)

            print("\n--- Summary ---")
            print_summary_table(df)

            # Save per-condition CSV
            results_dir.mkdir(parents=True, exist_ok=True)
            csv_path = results_dir / f"metrics_alpha{alpha}_kernel{kernel}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

            if not args.no_plots:
                plot_scatter(df, alpha, kernel, results_dir)

    # Save combined CSV if multiple conditions
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = results_dir / "metrics_all.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nSaved combined: {combined_path}")


if __name__ == "__main__":
    main()
