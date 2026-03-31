# PnP Langevin Sampling for Poisson Inverse Problems

Code accompanying the paper:

> **Efficient Bayesian Computation Using Plug-and-Play Priors for Poisson Inverse Problems**  
> Teresa Klatzer, Savvas Melidonis, Marcelo Pereyra, Konstantinos C. Zygalakis

## Installation

```bash
# Clone repository
git clone https://github.com/freyyia/pnp-langevin-poisson.git
cd pnp-langevin-poisson

# Create environment
conda create -n pnp-poisson python=3.10
conda activate pnp-poisson

# Install PyTorch with CUDA (check your version with nvidia-smi, adjust cu121 if needed)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install the package and its dependencies
pip install -e .

# Optional: install W&B logging support
pip install -e ".[wandb]"
```

## Repository Structure

```
pnp-langevin-poisson/
├── experiments/          # Experiment scripts
│   ├── low_dose_ct.py   # Low-dose CT reconstruction
│   └── deconvolution.py # Poisson deconvolution (to be added)
├── checkpoints/          # Pretrained model weights (auto-downloaded)
├── data/                 # Datasets (user-provided)
├── results/              # Output directory
└── configs/              # Example configuration files
```

## Quick Start

### Low-dose CT Experiment

```bash
# Run with default parameters (SKROCK method)
python experiments/low_dose_ct.py

# Run with Mirror Langevin Algorithm
python experiments/low_dose_ct.py --method MLA --iterations 5000

# Enable W&B logging
python experiments/low_dose_ct.py --wandb --wandb_project my-project
```

### Poisson Deblurring Experiment

```bash
# Run with default parameters (SKROCK, set3c, kernel 1)
python experiments/deblurring.py

# Run on cbsd10, kernel 3, with MLA
python experiments/deblurring.py --dataset cbsd10 --im_idx 2 --kernel 3 --method MLA

# Disable flip/rotation augmentation
python experiments/deblurring.py --no_flip_n_rot
```

The Levin09 blur kernels are **downloaded automatically** on first run to `data/kernels/Levin09.mat`.

### Command Line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--method` | SKROCK | Sampling method: SKROCK, MLA, or ULA |
| `--poisson_level` | 10.0 | Photon level α (lower = harder problem) |
| `--regularization` | 4.0 | Regularization parameter ρ |
| `--step_size` | 5e-5 | MCMC step size δ |
| `--iterations` | 2000 | Number of sampling iterations |
| `--im_idx` | 0 | Test image index |
| `--data_dir` | `<repo>/data` | Path to data directory |
| `--results_dir` | `<repo>/results` | Path to results directory |
| `--device` | auto | Device override (e.g. `cuda:0`, `cpu`) |
| `--seed` | 0 | Random seed |
| `--wandb` | False | Enable W&B logging |
| `--wandb_project` | pnp-langevin-ct | W&B project name |
| `--wandb_entity` | None | W&B entity (username or team) |
| `--save_samples` | False | Save posterior mean/std tensors |
| `--no_plots` | False | Disable plot generation |
| `--inset_plots` | False | Save inset crop plots for each image |
| `--inset_loc X Y` | 0.52 0.55 | Inset crop location in [0,1] |

## Data

### LIDC-IDRI Dataset (CT)

The preprocessed LIDC test set is **downloaded automatically** on first run from
[`jtachella/equivariant_bootstrap`](https://huggingface.co/datasets/jtachella/equivariant_bootstrap)
on Hugging Face and placed in `data/Tomography/dinv_dataset0.h5`.

### Pretrained Checkpoints

Checkpoints are automatically downloaded on first run. Manual download:
- [GS-DRUNet CT](https://drive.google.com/uc?id=1WuW6XUuf33P5odf91D_RtFXRrVpYVPhG) → `checkpoints/gsdrunet_ct.ckpt`

## Citation

```bibtex
@article{klatzer2025pnplangevin,
  title={Efficient Bayesian Computation Using Plug-and-Play Priors for Poisson Inverse Problems},
  author={Klatzer, Teresa and Melidonis, Savvas and Pereyra, Marcelo and Zygalakis, Konstantinos C.},
  journal={SIAM Journal on Imaging Sciences},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This work was supported by UKRI EPSRC (EP/V006134/1, EP/Z534481/1).
