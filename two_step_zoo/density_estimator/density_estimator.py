from ..two_step import TwoStepComponent
import torch


class DensityEstimator(TwoStepComponent):

    def __init__(self, use_tweedie_if_denoising, max_sigma, conditioning_network_1=None, conditioning_network_2=None,
                 conditioning_network_3=None, **kwargs):
        super().__init__(**kwargs)
        self.use_tweedie_if_denoising = use_tweedie_if_denoising
        self.max_sigma = max_sigma
        if max_sigma is not None:
            self.conditioning_network_1 = conditioning_network_1
            self.conditioning_network_2 = conditioning_network_2
            self.conditioning_network_3 = conditioning_network_3

    @staticmethod
    def _reshape_sigma(sigma, len_x_shape):
        # TODO: the following if/else statements are hacky, change them
        if len_x_shape == 2:
            return sigma[:, None]
        elif len_x_shape == 4:
            return sigma[:, None, None, None]
        else:
            raise NotImplementedError("_reshape_sigma not implemented for this shape")

    def _add_noise_for_denoising(self, x):
        sigma = self.max_sigma * torch.rand(x.shape[0]).to(self.device)
        noise = torch.randn_like(x)
        reshaped_sigma = self._reshape_sigma(sigma, len(x.shape))
        return x + reshaped_sigma * noise, sigma

    def _expand_input_for_denoising(self, input_, sigma, type_):
        if self.max_sigma is None:
            return input_
        else:
            sigma = self._reshape_sigma(sigma, len(input_.shape))
            if self.conditioning_network_1 is None:  # append sigma as an extra input or as an extra channel
                return torch.cat((input_, sigma), 1)
            else:
                assert type_ == 'encoder' or type_ == 'decoder' or type_ == 'discriminator' or type_ == 'energy_func'
                if type_ == 'decoder':
                    cond_net = self.conditioning_network_2
                elif type_ == 'discriminator':
                    cond_net = self.conditioning_network_3
                else:
                    cond_net = self.conditioning_network_1
                to_concat = cond_net.forward(sigma)
                return torch.cat((input_, to_concat), 1)

    def sample(self, n_samples, **kwargs):
        samples = self.sample_transformed(n_samples, **kwargs)
        return self._inverse_data_transform(samples)

    def sample_transformed(self, n_samples, **kwargs):
        raise NotImplementedError("log_prob not implemented")

    def log_prob(self, x, **kwargs):
        raise NotImplementedError("log_prob not implemented")

    def loss(self, x, **kwargs):
        return -self.log_prob(x, **kwargs).mean()
