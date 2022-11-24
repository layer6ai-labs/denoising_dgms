import torch

from nflows.distributions import Distribution, StandardNormal
from nflows.flows.base import Flow

from . import DensityEstimator
from ..utils import batch_or_dataloader, tweedie_denoising


class NormalizingFlow(DensityEstimator):

    model_type = "nf"

    def __init__(self, dim, transform, base_distribution: Distribution=None, **kwargs):
        super().__init__(**kwargs)
        self.transform = transform

        if base_distribution is None:
            self.base_distribution = StandardNormal([dim])
        else:
            self.base_distribution = base_distribution

        self._nflow = Flow(
            transform=self.transform,
            distribution=self.base_distribution
        )

    @tweedie_denoising
    def sample_transformed(self, n_samples):
        if self.max_sigma is not None:
            context = torch.zeros((1, 1)).to(self.device)
            if self.conditioning_network_1 is not None:
                context = self.conditioning_network_1.forward(context)
            samples = self._nflow.sample(n_samples, context=context)
        else:
            samples = self._nflow.sample(n_samples)
        return samples

    @batch_or_dataloader()
    def log_prob(self, x):
        x = self._data_transform(x)
        if self.max_sigma is not None:
            x, sigma_denoising = self._add_noise_for_denoising(x)
            context = sigma_denoising.unsqueeze(1)
            if self.conditioning_network_1 is not None:
                context = self.conditioning_network_1.forward(context)
            log_prob = self._nflow.log_prob(x, context=context)
        else:
            log_prob = self._nflow.log_prob(x)

        if len(log_prob.shape) == 1:
            log_prob = log_prob.unsqueeze(1)

        return log_prob

