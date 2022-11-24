import numpy as np
import torch

from .density_estimator import DensityEstimator
from .generalized_autoencoder import GeneralizedAutoEncoder
from .utils import batch_or_dataloader, tweedie_denoising
from .distributions import diagonal_gaussian_log_prob, diagonal_gaussian_entropy, diagonal_gaussian_sample


class GaussianVAE(GeneralizedAutoEncoder, DensityEstimator):
    model_type = "vae"

    @tweedie_denoising
    def sample_transformed(self, n_samples, true_sample=True):
        z = torch.randn((n_samples, self.latent_dim)).to(self.device)
        mu, log_sigma = self.decode_to_transformed(
            self._expand_input_for_denoising(z, torch.zeros(z.shape[0]).to(self.device), 'decoder')
        )
        sample = diagonal_gaussian_sample(mu, torch.exp(log_sigma)) if true_sample else mu
        return sample

    @batch_or_dataloader()
    def log_prob(self, x, k=1):
        # NOTE: With k=1, this gives the ELBO.
        batch_size = x.shape[0]

        # NOTE: Perform data transform _before_ repeat_interleave because we do not want
        #       to dequantize the same x point in several different ways.
        x = self._data_transform(x)

        sigma_denoising = 0.  # dummy value to define sigma_denoising in case max_sigma is None
        if self.max_sigma is not None:
            x, sigma_denoising = self._add_noise_for_denoising(x)

        x = x.repeat_interleave(k, dim=0)
        mu_z, log_sigma_z = self.encode_transformed(self._expand_input_for_denoising(x, sigma_denoising, 'encoder'))
        z = diagonal_gaussian_sample(mu_z, torch.exp(log_sigma_z))
        mu_x, log_sigma_x = self.decode_to_transformed(self._expand_input_for_denoising(z, sigma_denoising, 'decoder'))

        log_p_z = diagonal_gaussian_log_prob(z, torch.zeros_like(z), torch.zeros_like(z))
        log_p_x_given_z = diagonal_gaussian_log_prob(
            x.flatten(start_dim=1),
            mu_x.flatten(start_dim=1),
            log_sigma_x.flatten(start_dim=1)
        )
        if k == 1:
            h_z_given_x = diagonal_gaussian_entropy(log_sigma_z)
            return log_p_z + log_p_x_given_z + h_z_given_x
        else:
            log_q_z_given_x = diagonal_gaussian_log_prob(z, mu_z, log_sigma_z)
            elbo = log_p_z + log_p_x_given_z - log_q_z_given_x
            return torch.logsumexp(elbo.reshape(batch_size, k, 1), dim=1) - np.log(k)
