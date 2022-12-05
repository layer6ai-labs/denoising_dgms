<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# README

This is the codebase accompanying the paper ["Denoising Deep Generative Models"](https://arxiv.org/pdf/2212.01265.pdf) accepted for a spotlight presentation in the "I can't believe it's not better: Understanding Deep Learning Through Empirical Falsification" workshop at NeurIPS 2022. Our code is based on that of ["Diagnosing and Fixing Manifold Overfitting in Deep Generative Models"](https://github.com/layer6ai-labs/two_step_zoo), which we refer to as the "two step zoo codebase".
Here we discuss how to run the experiments in the paper, and we also outline the differences from the two step zoo codebase.

## Setup

The main prerequisite is to set up the python environment.
The command

    conda env create -f env-lock.yml

will create a `conda` environment called `two_step`.
Launch this environment using the command

    conda activate two_step

before running any experiments.

The file `env-lock.yml` contains strict versions of each of the packages.
If `conda` is not used, or some of the versions in `env-lock.yml` are no longer supported, we also provide a file `env-packages.yml` that simply lists the packages we used.
However, there are no guarantees that this will work out-of-the-box.

## Usage - `main.py`

Basic usage is as follows:

	python main.py --dataset <dataset> --model <model>

where:
- `<dataset>` is the dataset
  - The paper contains experiments with `mnist`, `fashion-mnist`, `svhn`, and `cifar10`
- `<model>` is the density estimator
  - Currently, we support any of the following: `arm`, `avb`, `ebm`, `flow`, and `vae`; although note that `arm` does not support any of the denoising methods reported in our paper. We also support other non-likelihood based models as leftover functionality from the two step zoo codebase.

The command above will by default train the vanilla version of a model, i.e. no noise added. Note that we do not report results on `avb` nor `ebm` models in the paper.

The config flag `denoising_sigma` (which defaults to `None`) adds Gaussian noise during training, and the config flag `use_tweedie_if_denoising` (which defaults to `True`) determines whether Tweedie denoising is used or not (if `denoising_sigma` is not `None`). For example, to train an ND-VAE model with $\sigma=0.01$ on MNIST we can run:

	python main.py --dataset mnist --model vae --is-gae --config sigma_denoising=0.01 --config use_tweedie_if_denoising=False

where the `--is-gae` flag is a leftover from the two step zoo codebase that should be used for VAEs and AVB. To run the analoguous TD-VAE model:

	python main.py --dataset mnist --model vae --is-gae --config sigma_denoising=0.01

The config flag `max_sigma` (which defaults to `None`) corresponds to $C$ in the paper. If not set to `None`, this will use conditional denoising. When using conditional denoising, conditioning networks might be needed. These are always MLPs in the current codebase if not `None`, and their architectures are specified through the config flags `conditioning_dims_1`, `conditioning_dims_2`, and `conditioning_dims_3`. `conditioning_dims_1` specifies the conditioning network's architecture for the encoder of VAEs and AVB, the energy function's for EBM, or the conditioning network for a flow. `conditioning_dims_2` specifies the decoder's conditioning network for VAEs and AVB, and does nothing for other models. `conditioning_dims_3` specifies the conditioning network for the discriminator of AVB, and does nothing for other models. For example, to run the CD-NF model from the paper on MNIST:

	python main.py --dataset mnist --model flow --config max_sigma=0.01 --config conditioning_dims_1="[256,64]"

In the case of CNNs, the conditioning network will automatically reshape its output, for example:

	python main.py --dataset svhn --model vae --is-gae --config max_sigma=0.01 --config conditioning_dims_1="[256,1024]" --config conditioning_dims_2="[256,8]"

## Differences with the two step zoo codebase

Note that the `denoising_sigma` flag behaves differently between codebases: in the two step zoo, it adds Gaussian noise before preprocessing data (i.e. scaling or whitening), whereas here the noise is added after preprocessing. This amounts to rescaling the variance of the noise that is added.

Note also that `main.py` in our codebase corresponds to `single_main.py` from the two step zoo codebase.

## BibTeX

```
@article{
  loaiza-ganem2022denoising,
  title = {Denoising Deep Generative Models},
  author = {Loaiza-Ganem, Gabriel and Ross, Brendan Leigh and Wu, Luhuan and Cunningham, John P. and Cresswell, Jesse C. and Caterini, Anthony L.},
  year = {2022},
  url = {https://arxiv.org/abs/2212.01265},
}

```
