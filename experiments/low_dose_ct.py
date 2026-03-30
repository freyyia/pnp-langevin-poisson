"""
Low-dose CT reconstruction using PnP Langevin sampling.

This experiment demonstrates Bayesian inference for low-dose computed tomography
using Plug-and-Play priors with MCMC sampling methods (SKROCK, MLA, ULA).

Reference:
    Klatzer et al., "Efficient Bayesian Computation Using Plug-and-Play Priors 
    for Poisson Inverse Problems", 2025.
"""

import argparse
import os
from pathlib import Path

import torch
import deepinv as dinv
import gdown
from huggingface_hub import hf_hub_download

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Paths (relative to repository root)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CKPT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"

# Pretrained model checkpoint
CKPT_URL = "https://drive.google.com/uc?id=1WuW6XUuf33P5odf91D_RtFXRrVpYVPhG"
CKPT_FILENAME = "gsdrunet_ct.ckpt"

# LIDC test dataset (preprocessed, hosted on Hugging Face)
DATASET_HF_REPO = "jtachella/equivariant_bootstrap"
DATASET_HF_FILE = "Tomography/dinv_dataset0.h5"

# Default experiment parameters
DEFAULTS = {
    "im_idx": 0,
    "method": "SKROCK",
    "poisson_level": 10.0,
    "regularization": 4.0,
    "step_size": 5e-5,
    "iterations": 2000,
    "sigma": 20 / 255.0,
    "eta": 0.05,
    "inner_iter": 10,
    "thinning": 10,
    "seed": 0,
}


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Low-dose CT reconstruction with PnP Langevin sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument("--im_idx", type=int, default=DEFAULTS["im_idx"],
                        help="Image index from LIDC test set")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to data directory (default: <repo>/data)")
    
    # Method arguments
    parser.add_argument("--method", type=str, default=DEFAULTS["method"],
                        choices=["SKROCK", "MLA", "ULA"],
                        help="Sampling method")
    parser.add_argument("--poisson_level", type=float, default=DEFAULTS["poisson_level"],
                        help="Photon level alpha (lower = more noise)")
    parser.add_argument("--regularization", type=float, default=DEFAULTS["regularization"],
                        help="Regularization parameter rho")
    parser.add_argument("--step_size", type=float, default=DEFAULTS["step_size"],
                        help="MCMC step size delta")
    parser.add_argument("--iterations", type=int, default=DEFAULTS["iterations"],
                        help="Number of MCMC iterations")
    
    # Output arguments  
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to results directory (default: <repo>/results)")
    parser.add_argument("--save_samples", action="store_true",
                        help="Save posterior mean and std tensors")
    
    # Logging arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="pnp-langevin-ct",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (username or team)")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto-detect GPU)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable plot generation")
    parser.add_argument("--inset_plots", action="store_true",
                        help="Save inset crop plots for each image")
    parser.add_argument("--inset_loc", type=float, nargs=2, default=[0.52, 0.55],
                        metavar=("X", "Y"),
                        help="Extract location for inset crop (x, y) in [0,1]")
    
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def setup_device(device_arg=None):
    """Set up compute device."""
    if device_arg is not None:
        return device_arg
    if torch.cuda.is_available():
        return dinv.utils.get_freer_gpu()
    return "cpu"


def download_checkpoint(ckpt_path, url=CKPT_URL):
    """Download pretrained checkpoint if not present."""
    if not ckpt_path.exists():
        print(f"Downloading checkpoint to {ckpt_path}...")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url, str(ckpt_path), quiet=False)
    return ckpt_path


def load_test_image(data_dir, im_idx, device):
    """Load test image from LIDC dataset, downloading it from Hugging Face if needed."""
    dataset_path = data_dir / "Tomography" / "dinv_dataset0.h5"

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}. Downloading from Hugging Face...")
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=DATASET_HF_REPO,
            filename=DATASET_HF_FILE,
            local_dir=str(data_dir),
        )
    
    dataset = dinv.datasets.HDF5Dataset(path=str(dataset_path), train=False)
    
    # Get image at specified index
    for i, (x, _) in enumerate(dataset):
        if i == im_idx:
            return x.unsqueeze(0).to(device)
    
    raise IndexError(f"Image index {im_idx} out of range")


def create_ct_physics(img_size, poisson_level, device):
    """Create CT forward operator with Poisson noise."""
    noise_model = dinv.physics.PoissonNoise(torch.tensor(1.0 / poisson_level))
    physics = dinv.physics.Tomography(
        angles=360,
        img_width=img_size,
        device=device,
        noise_model=noise_model,
    )
    return physics


def load_denoiser(ckpt_path, device):
    """Load pretrained GS-DRUNet denoiser."""
    model = dinv.models.GSDRUNet(
        in_channels=1,
        out_channels=1,
        act_mode='s',
        device=device,
        pretrained=ckpt_path,
    )
    return model.to(device)


def create_sampling_callback(ground_truth, device):
    """Create callback for W&B logging during sampling."""
    def callback(X, statistics, iter, **kwargs):
        mean_estimate = statistics[0].mean().clamp(0, 1)
        psnr = dinv.metric.PSNR()(ground_truth, mean_estimate).item()

        print(f"Iteration {iter+1}: PSNR={psnr:.2f} dB")

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "PSNR": psnr,
                "Posterior mean": wandb.Image(mean_estimate.cpu().squeeze()),
                "Posterior std": wandb.Image(statistics[0].var().sqrt().cpu().squeeze()),
            }, step=iter + 1)
    
    return callback


def save_results(results_dir, mean, var, ground_truth, observation, fbp, physics, args):
    """Save reconstruction results and plots."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tensors
    if args.save_samples:
        torch.save(mean.cpu(), results_dir / "posterior_mean.pt")
        torch.save(torch.sqrt(var).cpu(), results_dir / "posterior_std.pt")
    
    # Generate plots
    if not args.no_plots:
        imgs = [observation, ground_truth, fbp, mean]
        titles = ["Observation", "Ground Truth", "FBP", "Reconstruction"]

        dinv.utils.plot(
            imgs + [torch.sqrt(var)],
            titles=titles + ["Posterior Std"],
            save_dir=results_dir,
            rescale_mode="min_max",
        )

        if args.inset_plots:
            inset_loc = tuple(args.inset_loc)
            dinv.utils.plot_inset(
                imgs, titles=["Observation", "Ground Truth", "FBP", "Reconstruction"],
                extract_loc=inset_loc, inset_loc=(0.0, 0.6),
                save_fn=results_dir / "inset.png",
            )
            for img, name in zip(
                [observation, ground_truth, fbp, mean],
                ["obs", "gt", "fbp", "recon"],
            ):
                dinv.utils.plot_inset(
                    [img], titles=[""],
                    extract_loc=inset_loc, inset_loc=(0.0, 0.6),
                    save_fn=results_dir / f"{name}_inset.png",
                )


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup paths
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    results_base = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    ckpt_path = CKPT_DIR / CKPT_FILENAME
    
    # Create experiment-specific results directory
    exp_name = (
        f"ct_{args.method}_im{args.im_idx}_"
        f"pl{args.poisson_level}_reg{args.regularization}_"
        f"step{args.step_size}_iter{args.iterations}"
    )
    results_dir = results_base / exp_name
    
    # Setup device
    device = setup_device(args.device)
    print(f"Using device: {device}")
    
    # Download checkpoint if needed
    download_checkpoint(ckpt_path)
    
    # Load test image
    print(f"Loading test image {args.im_idx}...")
    x = load_test_image(data_dir, args.im_idx, device)
    img_size = x.shape[-1]
    
    # Create physics operator
    physics = create_ct_physics(img_size, args.poisson_level, device)
    
    # Generate noisy measurement
    y = physics(x).clamp(1e-4, None)
    fbp = physics.A_dagger(y)
    
    print(f"Observation range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"Ground truth range: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"FBP range: [{fbp.min().item():.4f}, {fbp.max().item():.4f}]")
    
    # Setup data fidelity term
    image_mean = args.poisson_level * torch.mean(x)
    beta = image_mean * 0.02
    data_fidelity = dinv.optim.data_fidelity.PoissonLikelihood(
        gain=1.0 / args.poisson_level,
        bkg=beta.item(),
    )
    
    # Load denoiser and create prior
    denoiser = load_denoiser(ckpt_path, device)
    prior = dinv.optim.ScorePrior(denoiser=denoiser).to(device)
    
    # Sampling parameters
    params = {
        "step_size": args.step_size,
        "alpha": args.regularization,
        "sigma": DEFAULTS["sigma"],
        "eta": DEFAULTS["eta"],
        "inner_iter": DEFAULTS["inner_iter"],
        "method": args.method,
    }
    
    # Initialize W&B if requested
    if args.wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed, disabling logging")
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config={**params, **vars(args)},
            )
            wandb.log({
                "Observation": wandb.Image(y.cpu().squeeze()),
                "Ground truth": wandb.Image(x.cpu().squeeze()),
            })
    
    # Create sampler
    iterations = args.iterations # if torch.cuda.is_available() else 1000

    sampler = dinv.sampling.sampling_builder(
        iterator=args.method.upper(),
        prior=prior,
        data_fidelity=data_fidelity,
        max_iter=iterations,
        params_algo=params,
        thinning=DEFAULTS["thinning"],
        verbose=True,
        clip=[0, 1],
        callback=create_sampling_callback(x, device),
    )
    
    # Run sampling
    print(f"\nRunning {args.method} sampling for {iterations} iterations...")
    x_init = fbp.clamp(1e-4, 1)
    mean, var = sampler.sample(y, physics, x_init=x_init)
    
    # Compute final metrics
    psnr_init = dinv.metric.PSNR()(x, fbp.clamp(0, 1)).item()
    psnr_final = dinv.metric.PSNR()(x, mean.clamp(0, 1)).item()

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Initial PSNR (FBP): {psnr_init:.2f} dB")
    print(f"  Final PSNR:         {psnr_final:.2f} dB")
    print(f"{'='*50}")
    
    # Save results
    save_results(results_dir, mean, var, x, y, fbp, physics, args)
    print(f"\nResults saved to: {results_dir}")
    
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return mean, var


if __name__ == "__main__":
    main()
