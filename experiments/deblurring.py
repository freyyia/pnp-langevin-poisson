"""
Poisson deblurring using PnP Langevin sampling.

This experiment demonstrates Bayesian inference for Poisson image deblurring
using Plug-and-Play priors with MCMC sampling methods (RPnP-SKROCK, PnP-MLA, PPnP/RPnP-ULA).

Reference:
    Klatzer et al., "Efficient Bayesian Computation Using Plug-and-Play Priors
    for Poisson Inverse Problems", 2026.
"""

import argparse
import random
import sys
from pathlib import Path

import gdown
import h5py
import numpy as np
import torch
import torchvision.transforms as T
import deepinv as dinv
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

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CKPT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"

# Prox-DRUNet checkpoint (proxdrunet denoiser)
PROXDRUNET_HF_REPO = "deepinv/gradientstep"
PROXDRUNET_HF_FILE = "Prox-DRUNet.ckpt"

# BregmanDRUNet checkpoint (bregman_drunet denoiser, optional)
BREGMAN_CKPT_FILENAME = "inv_gamma.ckpt"
BREGMAN_CKPT_GDRIVE_ID = "1qCm6Fh3hkLW3orb3l1tghmxrU_j1m0KB"
BREGMAN_SUBMODULE_PATH = ROOT_DIR / "third_party" / "BregmanPnP" / "GS_denoising"

# Levin09 blur kernels (8 motion blur kernels, loaded via h5py)
LEVIN_KERNEL_FILENAME = "Levin09.mat"
LEVIN_KERNEL_URL = "https://drive.google.com/uc?id=15kFUtBF6hGicudW6hKYM-6p6rqYO4plD"
N_LEVIN_KERNELS = 8

# Test datasets (Google Drive folder IDs)
DATASET_GDRIVE_IDS = {
    "set3c":  "1VAYFqG5aJ7EMOlrgs00NV90sdrMJ6HpO",
    "cbsd10": "1s1KPsqPdo1a6rSffRFoZU6BhsYPHOJLm",
}

noise_lvl_denoiser = 20.0

DEFAULTS_SKROCK = {
    "dataset": "set3c",
    "im_idx": 2,
    "kernel": 4,
    "denoiser": "proxdrunet",
    "method": "SKROCK",
    "poisson_level": 20.0,
    "noise_lvl_denoiser": noise_lvl_denoiser,
    "regularization": 1.0 / (noise_lvl_denoiser / 255.0)**2,
    "delta_frac": 20.0,
    "sigma": noise_lvl_denoiser / 255.0,
    "eta": 0.05,
    "inner_iter": 10,
    "thinning": 10,
    "burnin_ratio": 0.02,
    "iterations": 1000,
    "patch_size": 256,
    "flip_n_rot": True,
    "seed": 0,
}

noise_lvl_denoiser = 25.0
DEFAULTS_MLA = {
    "dataset": "set3c",
    "im_idx": 2,
    "kernel": 4,
    "denoiser": "proxdrunet",
    "method": "MLA",
    "poisson_level": 20.0,
    "noise_lvl_denoiser": noise_lvl_denoiser,
    # (alpha/eps)*Dg prior scaling: alpha=1, eps=sigma^2 → regularization = 1/sigma^2
    "regularization": 1.0 / (noise_lvl_denoiser / 255.0)**2,
    # delta_frac=37.5 preserves the effective step size from the empirically tuned delta_frac=750
    # used with the old L_y=alpha^3 formula (now corrected to alpha^2, making delta_max 20x larger)
    "delta_frac": 37.5,
    "sigma": noise_lvl_denoiser / 255.0,
    # MLA has no inner_iter / eta (single-step method)
    "thinning": 10,
    "burnin_ratio": 0.02,
    # MLA mixes slower than SKROCK (no Chebyshev acceleration); use more iterations
    "iterations": 5000,
    "patch_size": 256,
    "flip_n_rot": True,
    "seed": 0,
}

_noise_lvl_ula = 20.0
DEFAULTS_PPNP_ULA = {
    "dataset": "set3c",
    "im_idx": 2,
    "kernel": 4,
    "denoiser": "proxdrunet",
    "method": "PPNP_ULA",
    "poisson_level": 20.0,
    "noise_lvl_denoiser": _noise_lvl_ula,
    "regularization": 1.0 / (_noise_lvl_ula / 255.0)**2,
    # delta_frac=1.0 uses the full delta_max = 1/(L/eps + L_y); PPnP convergence is guaranteed
    # for delta <= delta_max (reference: batch_Reflected_PnP_DRUNET.py, PPnP_method=True)
    "delta_frac": 1.0,
    "sigma": _noise_lvl_ula / 255.0,
    "thinning": 10,
    "burnin_ratio": 0.02,
    "iterations": 5000,
    "patch_size": 256,
    "flip_n_rot": True,
    "seed": 0,
}

DEFAULTS_RPNP_ULA = {
    "dataset": "set3c",
    "im_idx": 2,
    "kernel": 4,
    "denoiser": "proxdrunet",
    "method": "RPNP_ULA",
    "poisson_level": 20.0,
    "noise_lvl_denoiser": _noise_lvl_ula,
    "regularization": 1.0 / (_noise_lvl_ula / 255.0)**2,
    # same delta_max formula as PPnP; lambd is computed from max_lambd * lambd_frac
    "delta_frac": 1.0,
    "lambd_frac": 0.99,  # lambd = lambd_frac * max_lambd; max_lambd = 1/((2*L)/eps + 4*L_y)
    "sigma": _noise_lvl_ula / 255.0,
    "thinning": 10,
    "burnin_ratio": 0.02,
    "iterations": 5000,
    "patch_size": 256,
    "flip_n_rot": True,
    "seed": 0,
}

DEFAULTS = DEFAULTS_MLA


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Poisson deblurring with PnP Langevin sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument("--dataset", type=str, default=DEFAULTS["dataset"],
                        choices=["set3c", "cbsd10"],
                        help="Test dataset")
    parser.add_argument("--im_idx", type=int, default=DEFAULTS["im_idx"],
                        help="Image index within dataset")
    parser.add_argument("--kernel", type=int, default=DEFAULTS["kernel"],
                        choices=list(range(N_LEVIN_KERNELS)),
                        help="Levin09 blur kernel index (0-7)")
    parser.add_argument("--patch_size", type=int, default=DEFAULTS["patch_size"],
                        help="Center-crop size applied to input images")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to data directory (default: <repo>/data)")

    # Method arguments
    parser.add_argument("--denoiser", type=str, default=DEFAULTS["denoiser"],
                        choices=["proxdrunet", "lmmo", "gsdrunet", "bregman_drunet"],
                        help="Denoiser/prior: proxdrunet (GSPnP + Prox-DRUNet), "
                             "lmmo (ScorePrior + DnCNN lipschitz), "
                             "gsdrunet (GSPnP + GSDRUNet), "
                             "bregman_drunet (BregmanDRUNet; requires [bregman] extra + submodule)")
    parser.add_argument("--method", type=str, default=DEFAULTS["method"],
                        choices=["SKROCK", "MLA", "ULA", "PPNP_ULA", "RPNP_ULA"],
                        help="Sampling method: SKROCK (SK-ROCK accelerated Langevin), "
                             "MLA (Mirror Langevin), ULA/PPNP_ULA (projected ULA), "
                             "RPNP_ULA (reflected ULA with soft penalty)")
    parser.add_argument("--poisson_level", type=float, default=DEFAULTS["poisson_level"],
                        help="Photon level alpha (lower = more noise)")
    parser.add_argument("--regularization", type=float, default=DEFAULTS["regularization"],
                        help="Regularization parameter rho")
    parser.add_argument("--noise_lvl_denoiser", type=float, default=DEFAULTS["noise_lvl_denoiser"],
                        help="(Gaussian) noise level denoiser")
    parser.add_argument("--delta_frac", type=float, default=DEFAULTS["delta_frac"],
                        help="Step size as a fraction of delta_max (delta = delta_frac * delta_max)")
    parser.add_argument("--iterations", type=int, default=DEFAULTS["iterations"],
                        help="Number of MCMC iterations")
    parser.add_argument("--flip_n_rot", action="store_true", default=DEFAULTS["flip_n_rot"],
                        help="Apply random flip/rotation augmentation to denoiser gradient")
    parser.add_argument("--no_flip_n_rot", dest="flip_n_rot", action="store_false",
                        help="Disable flip/rotation augmentation")

    # Output arguments
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Path to results directory (default: <repo>/results)")
    parser.add_argument("--save_samples", action="store_true",
                        help="Save posterior mean and std tensors")

    # Logging arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="pnp-langevin-deblurring",
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


def load_levin_kernel(data_dir, kernel_idx, device):
    """Load a Levin09 blur kernel from data/kernels/Levin09.mat, downloading if needed.

    The file uses HDF5 format (MATLAB v7.3). Contains 8 motion blur kernels
    of varying sizes.
    """
    kernel_path = data_dir / "kernels" / LEVIN_KERNEL_FILENAME
    if not kernel_path.exists():
        print(f"Levin09.mat not found at {kernel_path}. Downloading...")
        kernel_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(LEVIN_KERNEL_URL, str(kernel_path), quiet=False)
    with h5py.File(str(kernel_path), "r") as f:
        ref = f["kernels"][kernel_idx, 0]
        k = np.array(f[ref], dtype=np.float32)
    k = k / k.sum()  # normalize
    k_tensor = torch.tensor(k).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, kH, kW)
    return k_tensor


def load_test_image(data_dir, dataset_name, im_idx, patch_size, device):
    """Load a test image from set3c or cbsd10, auto-downloading from Google Drive if needed."""
    dataset_dir = data_dir / dataset_name
    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        print(f"Dataset {dataset_name} not found at {dataset_dir}. Downloading...")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            id=DATASET_GDRIVE_IDS[dataset_name],
            output=str(dataset_dir),
            quiet=False,
        )

    transform = T.Compose([T.CenterCrop(patch_size), T.ToTensor()])
    dataset = dinv.datasets.ImageFolder(root=str(dataset_dir), transform=transform)
    if im_idx >= len(dataset):
        raise IndexError(
            f"Image index {im_idx} out of range for dataset {dataset_name} "
            f"(size {len(dataset)})"
        )
    x = dataset[im_idx].unsqueeze(0).to(device)  # (1, C, H, W)
    return x


def create_blur_physics(k_tensor, img_size, poisson_level, device):
    """Create blur forward operator with Poisson noise."""
    noise_model = dinv.physics.PoissonNoise(torch.tensor(1.0 / poisson_level))
    physics = dinv.physics.BlurFFT(
        img_size=img_size,
        filter=k_tensor,
        noise_model=noise_model,
        device=device,
    )
    return physics


# -----------------------------------------------------------------------------
# Augmentation helpers
# -----------------------------------------------------------------------------

def _augment(x, rot, flip):
    """Apply random rotation and flip."""
    if flip:
        x = torch.flip(x, dims=(-flip,))
    return torch.rot90(x, k=rot, dims=(-2, -1))


def _deaugment(x, rot, flip):
    """Reverse random rotation and flip."""
    x = torch.rot90(x, k=-rot, dims=(-2, -1))
    if flip:
        x = torch.flip(x, dims=(-flip,))
    return x


# -----------------------------------------------------------------------------
# Prior classes
# -----------------------------------------------------------------------------

class GSPnP(dinv.optim.prior.RED):
    """Gradient-Step Denoiser prior (GSPnP).

    Wraps a dinv.models.GSPnP denoiser as an explicit prior with potential g(x).
    See Hurault et al., "Gradient Step Denoiser for convergent Plug-and-Play", ICLR 2022.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def forward(self, x, *args, **kwargs):
        """Computes the prior potential g(x)."""
        return self.denoiser.potential(x, *args, **kwargs)


class AugmentedGSPnP(GSPnP):
    """GSPnP prior with optional random flip/rotation augmentation on the denoiser gradient."""

    def __init__(self, *args, flip_n_rot=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.flip_n_rot_enabled = flip_n_rot

    def grad(self, x, sigma_denoiser, **kwargs):
        if not self.flip_n_rot_enabled:
            return super().grad(x, sigma_denoiser, **kwargs)
        rot = random.choice([0, 1, 2, 3])
        flip = random.choice([0, 1, 2])
        x_aug = _augment(x, rot, flip)
        g = super().grad(x_aug, sigma_denoiser, **kwargs)
        return _deaugment(g, rot, flip)


class AugmentedScorePrior(dinv.optim.ScorePrior):
    """ScorePrior with optional random flip/rotation augmentation on the denoiser.

    Augmenting the denoiser call with random equivariant transforms reduces
    bias from directional artefacts in the denoiser.
    """

    def __init__(self, *args, flip_n_rot=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.flip_n_rot_enabled = flip_n_rot

    def grad(self, x, sigma_denoiser, **kwargs):
        if not self.flip_n_rot_enabled:
            return super().grad(x, sigma_denoiser, **kwargs)
        rot = random.choice([0, 1, 2, 3])
        flip = random.choice([0, 1, 2])
        x_aug = _augment(x, rot, flip)
        g = super().grad(x_aug, sigma_denoiser, **kwargs)
        return _deaugment(g, rot, flip)


class BregmanDRUNetPrior(dinv.optim.Prior):
    """Prior based on BregmanDRUNet (Hurault et al., NeurIPS 2023).

    Uses the Bregman gradient Dg(x) from a jacobian-constrained DRUNet trained
    with a Burg entropy Bregman divergence. Requires the BregmanPnP submodule
    and the [bregman] optional dependencies.

    Note: This prior was originally designed for Mirror Langevin sampling with
    Burg's entropy potential. When used with SKROCK/ULA, it acts as a standard
    prior gradient in the Euclidean metric.
    """

    def __init__(self, denoiser_model, flip_n_rot=True):
        super().__init__()
        self.denoiser_model = denoiser_model
        self.flip_n_rot_enabled = flip_n_rot
        self.explicit_prior = False

    def grad(self, x, sigma_denoiser, **kwargs):
        if self.flip_n_rot_enabled:
            rot = random.choice([0, 1, 2, 3])
            flip = random.choice([0, 1, 2])
            x_aug = _augment(x, rot, flip)
            Dg, _, _ = self.denoiser_model.calculate_grad(x_aug, sigma_denoiser)
            return _deaugment(Dg.detach(), rot, flip)
        else:
            Dg, _, _ = self.denoiser_model.calculate_grad(x, sigma_denoiser)
            return Dg.detach()


# -----------------------------------------------------------------------------
# Prior factory
# -----------------------------------------------------------------------------

def _download_proxdrunet(ckpt_dir):
    """Download Prox-DRUNet checkpoint from Hugging Face if not present."""
    ckpt_path = ckpt_dir / PROXDRUNET_HF_FILE
    if not ckpt_path.exists():
        print(f"Downloading Prox-DRUNet checkpoint to {ckpt_path}...")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=PROXDRUNET_HF_REPO,
            filename=PROXDRUNET_HF_FILE,
            local_dir=str(ckpt_path.parent),
        )
        downloaded = ckpt_path.parent / PROXDRUNET_HF_FILE
        if downloaded.exists() and not ckpt_path.exists():
            downloaded.rename(ckpt_path)
    return ckpt_path


def _load_bregman_drunet(ckpt_dir, device):
    """Download and load the BregmanDRUNet model from the BregmanPnP submodule."""
    # Lazy import from submodule
    gs_path = BREGMAN_SUBMODULE_PATH
    if not gs_path.exists():
        raise ImportError(
            f"BregmanPnP submodule not found at {gs_path}.\n"
            "Run: git submodule update --init --recursive"
        )
    try:
        sys.path.insert(0, str(gs_path))
        sys.path.insert(0, str(gs_path.parent / "PnP_restoration"))
        from lightning_denoiser import GradMatch  
    except ImportError as e:
        raise ImportError(
            "Failed to import BregmanPnP. Install optional deps:\n"
            "  pip install -e '.[bregman]'\n"
            "and ensure the submodule is initialised:\n"
            "  git submodule update --init --recursive"
        ) from e

    ckpt_path = ckpt_dir / BREGMAN_CKPT_FILENAME
    if not ckpt_path.exists():
        print(f"Downloading BregmanDRUNet checkpoint to {ckpt_path}...")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=BREGMAN_CKPT_GDRIVE_ID, output=str(ckpt_path), quiet=False)

    checkpoint = torch.load(str(ckpt_path), map_location=device)
    hparams = argparse.Namespace(**checkpoint["hyper_parameters"])
    model = GradMatch(hparams)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def _load_gsdrunet_from_ckpt(ckpt_path, act_mode="E", device="cpu"):
    """Load GSDRUNet weights from a local checkpoint.

    Bypasses deepinv's internal torch.load call, which fails on PyTorch >= 2.6
    when the checkpoint contains pytorch_lightning globals.
    Returns a dinv.models.GSPnP denoiser ready to be wrapped in a prior.
    """
    gsmodel = dinv.models.GSDRUNet(in_channels=3, out_channels=3, act_mode=act_mode, pretrained=None)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    gsmodel.load_state_dict(ckpt, strict=False)
    gsmodel.eval()
    return gsmodel.to(device)


def create_prior(denoiser_name, ckpt_dir, flip_n_rot, device):
    """Create the prior for the given denoiser name.

    :param denoiser_name: one of "proxdrunet", "lmmo", "gsdrunet", "bregman_drunet"
    :param ckpt_dir: directory for storing downloaded checkpoints
    :param flip_n_rot: whether to apply random flip/rotation augmentation
    :param device: torch device
    """
    if denoiser_name == "proxdrunet":
        ckpt_path = _download_proxdrunet(ckpt_dir)
        net = _load_gsdrunet_from_ckpt(ckpt_path, act_mode="s", device=device)
        return AugmentedGSPnP(denoiser=net, flip_n_rot=flip_n_rot).to(device)

    elif denoiser_name == "lmmo":
        net = dinv.models.DnCNN(pretrained="download_lipschitz")
        return AugmentedScorePrior(denoiser=net, flip_n_rot=flip_n_rot).to(device)

    elif denoiser_name == "gsdrunet":
        net = dinv.models.GSDRUNet(in_channels=3, out_channels=3, pretrained="download")
        return AugmentedGSPnP(denoiser=net, flip_n_rot=flip_n_rot).to(device)

    elif denoiser_name == "bregman_drunet":
        model = _load_bregman_drunet(ckpt_dir, device)
        return BregmanDRUNetPrior(model, flip_n_rot=flip_n_rot)

    else:
        raise ValueError(f"Unknown denoiser: {denoiser_name}")


# -----------------------------------------------------------------------------
# Callback and results
# -----------------------------------------------------------------------------

def create_sampling_callback(ground_truth, device, print_every=100):
    """Create sampling callback for diagnostics.

    :param print_every: print PSNR every this many iterations.
    """
    psnr_metric = dinv.metric.PSNR()
    ssim_metric = dinv.metric.SSIM()

    def callback(X, statistics, iter, **kwargs):
        mean_estimate = statistics[0].mean().clamp(0, 1)
        psnr = psnr_metric(ground_truth, mean_estimate).item()
        ssim = ssim_metric(ground_truth, mean_estimate).item()

        if (iter) % print_every == 0:
            cur_sample = X["x"]
            print(
                f"\nIteration {iter}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f} | "
                f"sample=[{cur_sample.min().item():.3f}, {cur_sample.max().item():.3f}] "
                f"mean=[{mean_estimate.min().item():.3f}, {mean_estimate.max().item():.3f}]"
            )

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "PSNR": psnr,
                "SSIM": ssim,
                "Posterior mean": wandb.Image(mean_estimate.cpu().squeeze().permute(1, 2, 0).numpy()),
                "Posterior std": wandb.Image(
                    statistics[0].var().sqrt().mean(dim=1).cpu().squeeze().numpy()
                ),
            }, step=iter)

    return callback


def save_results(results_dir, mean, var, ground_truth, observation, x_init, args):
    """Save reconstruction results and plots."""
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.save_samples:
        torch.save(mean.cpu(), results_dir / "posterior_mean.pt")
        torch.save(torch.sqrt(var).cpu(), results_dir / "posterior_std.pt")

    if not args.no_plots:
        std = torch.sqrt(var).mean(dim=1, keepdim=True).expand_as(mean)
        imgs = [observation, ground_truth, x_init, mean]
        titles = ["Observation", "Ground Truth", "Init", "Reconstruction"]

        dinv.utils.plot(
            imgs + [std],
            titles=titles + ["Posterior Std"],
            save_dir=results_dir,
            rescale_mode="min_max",
        )

        if args.inset_plots:
            inset_loc = tuple(args.inset_loc)
            dinv.utils.plot_inset(
                imgs, titles=titles,
                extract_loc=inset_loc, inset_loc=(0.0, 0.6),
                save_fn=results_dir / "inset.png",
            )
            for img, name in zip(imgs, ["obs", "gt", "init", "recon"]):
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

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup paths
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    results_base = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    exp_name = (
        f"deblur_{args.method}_{args.denoiser}_{args.dataset}_im{args.im_idx}_k{args.kernel}_"
        f"pl{args.poisson_level}_denoiser_lvl{args.noise_lvl_denoiser}_"
        f"dfrac{args.delta_frac}_iter{args.iterations}"
    )
    results_dir = results_base / exp_name

    device = setup_device(args.device)
    print(f"Using device: {device}")

    # Load test image
    print(f"Loading image {args.im_idx} from {args.dataset}...")
    x = load_test_image(data_dir, args.dataset, args.im_idx, args.patch_size, device)
    img_size = tuple(x.shape[1:])  # (C, H, W)
    print(f"Image size: {img_size}")

    # Load blur kernel
    k_tensor = load_levin_kernel(data_dir, args.kernel, device)
    print(f"Kernel {args.kernel} shape: {list(k_tensor.shape[-2:])}")

    # Create physics operator
    physics = create_blur_physics(k_tensor, img_size, args.poisson_level, device)

    # Generate noisy measurement
    y = physics(x).clamp(1e-4, None)
    x_init = (y / y.max()).clamp(1e-4, 1)

    print(f"Observation range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"Ground truth range: [{x.min().item():.4f}, {x.max().item():.4f}]")

    # Setup data fidelity
    image_mean = args.poisson_level * torch.mean(x)
    beta = image_mean * 0.02
    # MLA: no background term in grad_f (mirror map keeps iterates positive).
    # SKROCK/ULA: beta stabilises the likelihood near zero (reflbox/projbox boundary).
    # beta is still used for the Lipschitz bound (delta_max) in both cases.
    bkg = 0.0 if args.method.upper() == "MLA" else beta.item()
    data_fidelity = dinv.optim.data_fidelity.PoissonLikelihood(
        gain=1.0 / args.poisson_level,
        bkg=bkg,
    )

    # Lipschitz constant of the likelihood gradient:
    #   L_y = alpha^2 * max(y) / beta^2  (matches references for all methods)
    L_y = args.poisson_level ** 2 * (y.max() / beta ** 2)
    eps = DEFAULTS["sigma"]**2
    delta_max = 1.0 / (1.0 / eps + L_y.item())
    step_size = args.delta_frac * delta_max
    print(f"L_y: {L_y.item():.2e}, delta_max: {delta_max:.2e}, delta_frac: {args.delta_frac}, step_size: {step_size:.2e}")

    # Create prior
    print(f"Loading denoiser: {args.denoiser}...")
    prior = create_prior(args.denoiser, CKPT_DIR, args.flip_n_rot, device)

    # Sampling parameters
    params = {
        "step_size": step_size,
        "alpha": args.regularization,
        "sigma": DEFAULTS["sigma"],
        "method": args.method,
    }
    if args.method.upper() == "SKROCK":
        params["eta"] = DEFAULTS_SKROCK["eta"]
        params["inner_iter"] = DEFAULTS_SKROCK["inner_iter"]
    elif args.method.upper() == "RPNP_ULA":
        # RPnP-ULA penalty parameter: lambd = lambd_frac * max_lambd
        # max_lambd = 1/((2*L)/eps + 4*L_y) with L=1 (Lip. of prior gradient)
        max_lambd = 1.0 / (2.0 / eps + 4.0 * L_y.item())
        lambd_frac = DEFAULTS_RPNP_ULA.get("lambd_frac", 0.99)
        params["lambd"] = lambd_frac * max_lambd
        print(f"RPnP-ULA: max_lambd={max_lambd:.2e}, lambd={params['lambd']:.2e}")

    # Map method names to iterator names (PPNP_ULA and RPNP_ULA both use ULAIterator)
    _iterator_map = {"PPNP_ULA": "ULA", "RPNP_ULA": "ULA"}
    iterator_name = _iterator_map.get(args.method.upper(), args.method.upper())

    # Extra kwargs for the iterator constructor (e.g. boundary for RPnP-ULA)
    _iterator_kwargs = {}
    if args.method.upper() == "RPNP_ULA":
        _iterator_kwargs["boundary"] = "rpnp"

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
                "Observation": wandb.Image(
                    (y / y.max()).cpu().squeeze().permute(1, 2, 0).numpy()
                ),
                "Ground truth": wandb.Image(x.cpu().squeeze().permute(1, 2, 0).numpy()),
            })

    # Create sampler
    burnin_ratio = DEFAULTS["burnin_ratio"]
    print_every = max(1, args.iterations // 10)

    sampler = dinv.sampling.sampling_builder(
        iterator=iterator_name,
        prior=prior,
        data_fidelity=data_fidelity,
        max_iter=args.iterations,
        params_algo=params,
        thinning=DEFAULTS["thinning"],
        burnin_ratio=burnin_ratio,
        verbose=True,
        clip=[0, 1],
        callback=create_sampling_callback(x, device, print_every=print_every),
        **_iterator_kwargs,
    )

    # Run sampling
    print(f"\nRunning {args.method} sampling for {args.iterations} iterations...")
    mean, var = sampler.sample(y, physics, x_init=x_init)

    # Compute final metrics
    psnr_init = dinv.metric.PSNR()(x, x_init.clamp(0, 1)).item()
    psnr_final = dinv.metric.PSNR()(x, mean.clamp(0, 1)).item()
    ssim_final = dinv.metric.SSIM()(x, mean.clamp(0, 1)).item()

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Initial PSNR:  {psnr_init:.2f} dB")
    print(f"  Final PSNR:    {psnr_final:.2f} dB")
    print(f"  Final SSIM:    {ssim_final:.4f}")
    print(f"{'='*50}")

    # Save results
    save_results(results_dir, mean, var, x, y, x_init, args)
    print(f"\nResults saved to: {results_dir}")

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()

    return mean, var


if __name__ == "__main__":
    main()
