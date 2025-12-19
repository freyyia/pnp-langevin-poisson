# pnp-langevin-poisson

This repository contains the code for the paper [Efficient Bayesian Computation Using Plug-and-Play Priors for Poisson Inverse Problems](https://arxiv.org/abs/2503.16222) by Teresa Klatzer, Savvas Melidonis, Marcelo Pereyra, Konstantinos C. Zygalakis.

## Installing dependencies

```conda create -n env_name python=3.10```

Activate the environment

```conda activate env_name```

Some of the code is based on the Deepinv Libary, please checkout my fork at https://github.com/freyyia/deepinv. Clone the repository and from the deepinv directory install in editable mode with dependencies

```pip install -e .[dataset,denoisers]```

if you want to play around with it.

Further dependencies

``` pip install gdown ```

```conda install wandb```

Optional

``` conda install -c conda-forge ipywidgets jupyterlab ```

```jupyter nbextension enable --py widgetsnbextension```
