# PnP Langevin Sampling for Poisson Inverse Problems

Code accompanying the paper:

> **Efficient Bayesian Computation Using Plug-and-Play Priors for Poisson Inverse Problems**  
> Teresa Klatzer, Savvas Melidonis, Marcelo Pereyra, Konstantinos C. Zygalakis

## Installation

```bash
git clone https://github.com/freyyia/pnp-langevin-poisson.git
cd pnp-langevin-poisson

conda create -n pnp-poisson python=3.10
conda activate pnp-poisson

# Install PyTorch with CUDA (adjust cu121 to match your driver, see nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -e .

# Optional extras
pip install -e ".[wandb]"   # Weights & Biases logging
pip install -e ".[eval]"    # evaluation metrics (LPIPS, pandas, scikit-image)
```

## Experiments

### Poisson deblurring

```bash
# Default: PnP-MLA, set3c image 2, Levin kernel 4, Poisson level 20
python experiments/deblurring.py

# RPnP-SKROCK
python experiments/deblurring.py --method SKROCK --im_idx 2 --kernel 4

# PPnP-ULA / RPnP-ULA
python experiments/deblurring.py --method PPNP_ULA
python experiments/deblurring.py --method RPNP_ULA

# Different dataset / image / kernel
python experiments/deblurring.py --dataset cbsd10 --im_idx 0 --kernel 9
```

Levin09 blur kernels are downloaded automatically to `data/kernels/Levin09.mat` on first run.

Key arguments:

| Argument | Description |
|----------|-------------|
| `--method` | `SKROCK`, `MLA`, `PPNP_ULA`, `RPNP_ULA` |
| `--dataset` | `set3c` (default) or `cbsd10` |
| `--im_idx` | Image index within the dataset |
| `--kernel` | Levin09 kernel index (0–7) |
| `--poisson_level` | Photon level α (default: 20) |
| `--delta_frac` | Step size as fraction of δ_max |
| `--iterations` | Number of MCMC iterations |
| `--denoiser` | `proxdrunet` (default), `gsdrunet`, `dncnn` |

### Low-dose CT

```bash
# Default: RPnP-SKROCK, Poisson level 10
python experiments/low_dose_ct.py

# With W&B logging
python experiments/low_dose_ct.py --wandb --wandb_project my-project
```

The LIDC test set is downloaded automatically from
[`jtachella/equivariant_bootstrap`](https://huggingface.co/datasets/jtachella/equivariant_bootstrap)
on Hugging Face (`data/Tomography/dinv_dataset0.h5`).

Key arguments:

| Argument | Description |
|----------|-------------|
| `--method` | `SKROCK` (default), `MLA`, `ULA` |
| `--poisson_level` | Photon level α (default: 10) |
| `--regularization` | Regularization ρ (default: 200.0) |
| `--step_size` | MCMC step size δ (default: 2.89e-9) |
| `--iterations` | Number of MCMC iterations (default: 2000) |

### Evaluation (state-of-the-art comparison)

```bash
# Download pre-computed results and evaluate all conditions
python experiments/eval_deblurring.py --download

# Single condition
python experiments/eval_deblurring.py --alpha 20 --kernel 4
```

Outputs (CSV tables and PSNR/LPIPS scatter plots)
are written to `results/eval/`.

## Pretrained Checkpoints

Downloaded automatically on first run:
- **Prox-DRUNet** (deblurring): from [`deepinv/gradientstep`](https://huggingface.co/deepinv/gradientstep) on Hugging Face
- **GS-DRUNet CT**: from [Google Drive](https://drive.google.com/uc?id=1WuW6XUuf33P5odf91D_RtFXRrVpYVPhG) → `checkpoints/gsdrunet_ct.ckpt`

## Citation

```bibtex
@article{klatzer2026pnplangevin,
  title={Efficient Bayesian Computation Using Plug-and-Play Priors for Poisson Inverse Problems},
  author={Klatzer, Teresa and Melidonis, Savvas and Pereyra, Marcelo and Zygalakis, Konstantinos C.},
  journal={SIAM Journal on Imaging Sciences},
  year={2026}
}
```

## Acknowledgements

This work was supported by UKRI EPSRC (EP/V006134/1, EP/Z534481/1, EP/Y028783/1).

## License

MIT, see [LICENSE](LICENSE).
