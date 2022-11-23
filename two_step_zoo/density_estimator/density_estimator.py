from ..two_step import TwoStepComponent


class DensityEstimator(TwoStepComponent):

    def __init__(self, use_tweedie_if_denoising, **kwargs):
        super().__init__(**kwargs)
        self.use_tweedie_if_denoising = use_tweedie_if_denoising

    def sample(self, n_samples, **kwargs):
        samples = self.sample_transformed(n_samples, **kwargs)
        return self._inverse_data_transform(samples)

    def sample_transformed(self, n_samples, **kwargs):
        raise NotImplementedError("log_prob not implemented")

    def log_prob(self, x, **kwargs):
        raise NotImplementedError("log_prob not implemented")

    def loss(self, x, **kwargs):
        return -self.log_prob(x, **kwargs).mean()
