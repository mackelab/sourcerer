"""Taken and adapted from https://github.com/MaximeVandegar/NEB/blob/main/ml/real_nvps.py"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import digamma, gamma
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

from sourcerer.utils import scale_tensor


class GaussianDistribution:
    """Multivariate standard normal"""

    @staticmethod
    def sample(size, dim=2):
        means = torch.zeros(size, dim)
        cov = torch.eye(dim)
        m = MultivariateNormal(means, cov)
        return m.sample()

    # infer dim from shape
    @staticmethod
    def log_prob(z):
        """
        Arguments:
        ----------
            - z: a batch of m data points (size: m x data_dim)
        """
        return -0.5 * (
            torch.log(torch.tensor([np.pi * 2], device=z.device)) + z**2
        ).sum(1)


class VariableTemperedUniform:
    def __init__(
        self,
        lower_bound=torch.Tensor([-1.0, -1.0]),
        upper_bound=torch.Tensor([1.0, 1.0]),
        outside_decay=100.0,
        device="cpu",
    ):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        assert len(lower_bound) == len(upper_bound)
        self.outside_decay = outside_decay
        self.n_dims = len(lower_bound)
        self.max_log_prob = -torch.sum(torch.log(upper_bound - lower_bound))
        self.device = device

    def to(self, device):
        self.device = device
        self.lower_bound = self.lower_bound.to(device)
        self.upper_bound = self.upper_bound.to(device)
        return self

    # log prob with tempered boundary
    def log_prob(self, x):
        max_val = torch.maximum(
            x - self.upper_bound, torch.zeros_like(x, device=self.device)
        )
        max_val += torch.maximum(
            self.lower_bound - x, torch.zeros_like(x, device=self.device)
        )
        return self.max_log_prob - self.outside_decay * torch.sum(max_val, dim=-1)

    def sample(self, num_samples):
        return (
            torch.rand(num_samples, self.n_dims, device=self.device)
            * (self.upper_bound - self.lower_bound)
            + self.lower_bound
        )


class TemperedUniform(VariableTemperedUniform):
    def __init__(self, low=-1.0, high=1.0, n_dims=2, outside_decay=100.0, device="cpu"):
        lower_bound = torch.ones(n_dims, device=device) * low
        upper_bound = torch.ones(n_dims, device=device) * high
        super().__init__(lower_bound, upper_bound, outside_decay, device)


class RealNVP(nn.Module):
    """
    RealNVP defined in "Density estimation using Real NVP,
                        Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio,
                        2017"
    """

    def __init__(self, data_dim, context_dim, hidden_layer_dim, alpha=1.9):
        super().__init__()

        self.d = data_dim // 2
        output_dim = data_dim - self.d
        self.alpha = alpha
        # alpha for the clamping operator defined in "Guided Image Generation
        #                                             with Conditional Invertible
        #                                             Neural Networks"

        self.s = nn.Sequential(
            nn.Linear(self.d + context_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, output_dim),
        )

        self.t = nn.Sequential(
            nn.Linear(self.d + context_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, output_dim),
        )

    def forward(self, y, context=None):
        return self.forward_and_compute_log_jacobian(y, context)[0]

    def forward_and_compute_log_jacobian(self, y, context=None):
        """
        Computes z = f(y) where f is bijective, and the log jacobian determinant of the transformation

        Arguments:
        ----------
            - y: a batch of input variables
            - context: conditioning variables (optional)

        Return:
        -------
            - z = f(y)
            - log_jac
        """

        z1 = y[:, : self.d]

        s = (
            self.s(torch.cat((z1, context), dim=1))
            if (context is not None)
            else self.s(z1)
        )
        s = self.clamp(s)

        t = (
            self.t(torch.cat((z1, context), dim=1))
            if (context is not None)
            else self.t(z1)
        )
        z2 = torch.mul(y[:, self.d :], torch.exp(s)) + t

        return torch.cat((z1, z2), dim=1), torch.sum(s, dim=1)

    def invert(self, z, context=None):
        """
        Computes y = f^{-1}(z)

        Arguments:
        ----------
            - z: a batch of input variables
            - context: conditioning variables (optional)
        """

        y1 = z[:, : self.d]

        s = (
            self.s(torch.cat((y1, context), dim=1))
            if (context is not None)
            else self.s(y1)
        )
        s = self.clamp(s)

        t = (
            self.t(torch.cat((y1, context), dim=1))
            if (context is not None)
            else self.t(y1)
        )
        y2 = torch.mul(z[:, self.d :] - t, torch.exp(-s))

        return torch.cat((y1, y2), dim=1)

    def clamp(self, s):
        """
        Clamping operator defined in "Guided Image Generation with Conditional Invertible Neural Networks
                                      Lynton Ardizzone, Carsten Lüth, Jakob Kruse, Carsten Rother, Ullrich Köthe
                                      2019"
        """
        return torch.tensor([2 * self.alpha / np.pi], device=s.device) * torch.atan(
            s / self.alpha
        )


class RealNVPs(nn.Module):
    """
    Stacking RealNVP layers. The same (almost) API than UMNN (https://github.com/AWehenkel/UMNN) is provided.
                             API: - forward(y, context=None)
                                  - invert(y, context=None)
                                  - compute_ll(y, context=None)
    """

    def __init__(
        self,
        flow_length,
        data_dim,
        context_dim,
        hidden_layer_dim,
        low=-1.0,
        high=1.0,
        alpha=1.9,
        device="cpu",
    ):
        super().__init__()

        self.data_dim = data_dim
        self.layers = nn.Sequential(
            *(
                RealNVP(data_dim, context_dim, hidden_layer_dim, alpha=alpha)
                for _ in range(flow_length)
            )
        )

        # NOTE: avoid buffers here for backwards compatibility
        if not isinstance(low, torch.Tensor):
            low = torch.ones(self.data_dim) * low
        self.low = low
        if not isinstance(high, torch.Tensor):
            high = torch.ones(self.data_dim) * high
        self.high = high

        assert self.high.shape[0] == self.low.shape[0] == self.data_dim

        self.device = device

    def forward_kld(self, y, context=None):
        return -torch.mean(self.log_prob(y, context))

    # Currently only supports (scaled) uniform distribution
    def reverse_kld(self, _target, num_samples, context=None):
        target = TemperedUniform(n_dims=self.data_dim, device=self.device)
        y = self.one_sample(num_samples, context)
        entro_term = -torch.mean(self.one_log_prob(y, context))
        # entro_term = kozachenko_leonenko_estimator(y)
        cross_entro_term = -torch.mean(target.log_prob(y))

        return -entro_term + cross_entro_term

    def one_log_prob(self, y, context=None):
        z, log_jacobian = self.forward_and_compute_log_jacobian(y, context)
        log_y_likelihood = GaussianDistribution.log_prob(z) + log_jacobian

        return log_y_likelihood

    def log_prob(self, y, context=None):
        scaled_y = scale_tensor(
            y,
            self.low,
            self.high,
            -torch.ones(self.data_dim, device=self.device),
            torch.ones(self.data_dim, device=self.device),
        )
        return self.one_log_prob(scaled_y, context)

    def forward_and_compute_log_jacobian(self, y, context=None):
        log_jacobians = 0
        z = y
        for layer in self.layers:
            # Every iteration, we swap the first and last variables so that no variable is left unchanged
            z = torch.cat(
                (z[:, self.data_dim // 2 :], z[:, : self.data_dim // 2]), dim=1
            )

            z, log_jacobian = layer.forward_and_compute_log_jacobian(z, context)
            log_jacobians += log_jacobian

        return z, log_jacobians

    def invert(self, z, context=None):
        y = z
        for i in range(len(self.layers) - 1, -1, -1):
            y = self.layers[i].invert(y, context)

            # Every iteration, we swap the first and last variables so that no variable is left unchanged
            y = torch.cat(
                (
                    y[:, (self.data_dim - self.data_dim // 2) :],
                    y[:, : (self.data_dim - self.data_dim // 2)],
                ),
                dim=1,
            )

        return y

    def one_sample(self, size=None, context=None):
        if size is None:
            size = context.shape[0]
        if context is not None:
            assert context.shape[0] == size
        y = GaussianDistribution.sample(size, self.data_dim)
        return self.invert(y.to(self.device), context)

    def sample(self, size=None, context=None):
        scaled_y = self.one_sample(size, context)
        return scale_tensor(
            scaled_y,
            -torch.ones(self.data_dim, device=self.device),
            torch.ones(self.data_dim, device=self.device),
            self.low,
            self.high,
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]
        self.low = self.low.to(self.device)
        self.high = self.high.to(self.device)
        return self


class Sampler(nn.Module):
    def __init__(
        self,
        xdim,
        input_noise_dim,
        hidden_layer_dim,
        num_hidden_layers=3,
        low=-1.0,
        high=1.0,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.input_noise_dim = input_noise_dim
        self.xdim = xdim

        layers = []
        layers.append(nn.Linear(input_noise_dim, hidden_layer_dim))
        layers.append(nn.BatchNorm1d(hidden_layer_dim))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_layer_dim, hidden_layer_dim))
            layers.append(nn.BatchNorm1d(hidden_layer_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_layer_dim, xdim))

        if not isinstance(low, torch.Tensor):
            low = torch.ones(self.xdim) * low
        self.register_buffer("low", low)
        if not isinstance(high, torch.Tensor):
            high = torch.ones(self.xdim) * high
        self.register_buffer("high", high)

        self.net = nn.Sequential(*layers)

    def one_forward(self, noise):
        # transform to [-1, 1]
        return -1.0 + 2.0 * F.sigmoid(self.net(noise))

    def forward(self, noise):
        return self.low + (self.high - self.low) * F.sigmoid(self.net(noise))

    def one_sample(self, size):
        return self.one_forward(
            torch.randn(size, self.input_noise_dim, device=self.device)
        )

    def sample(self, size):
        return self.forward(torch.randn(size, self.input_noise_dim, device=self.device))

    # Currently only supports (scaled) uniform distribution
    def reverse_kld(self, _target, num_samples):
        target = TemperedUniform(n_dims=self.xdim, device=self.device)
        y = self.one_sample(num_samples)
        entro_term = kozachenko_leonenko_estimator(y)
        cross_entro_term = -torch.mean(target.log_prob(y))

        return -entro_term + cross_entro_term

        # NOTE: with sampler boundaries, only minimizing negative entropy is also an option
        # return -entro_term

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]
        return self


def kozachenko_leonenko_estimator(samples, eps=1e-100, hack=1e10, on_torus=True):
    num_samples, dim = samples.shape
    assert num_samples <= 20000  # to not blow up memory
    if on_torus:
        pairdist = n_torus_pairwise(samples)
    else:
        pairdist = euclidean_pairwise(samples)
    nn_dist = torch.min(
        pairdist + torch.eye(len(samples), device=samples.device) * hack, dim=1
    ).values
    return (
        dim * torch.mean(torch.log(2.0 * nn_dist + eps))
        + np.log(((np.pi ** (dim / 2.0)) / gamma(1.0 + (dim / 2.0))) / (2.0**dim))
        - digamma(1.0)  # nearest neighbor
        + digamma(num_samples)
    )


# NOTE: This works only if the box "fits" the torus size.
# The default box is -1 to 1, so the default torus size is 2
def n_torus_pairwise(samples, torus_size=2.0, ord=2):
    diff = torch.cdist(
        samples.transpose(0, 1).unsqueeze(-1).double(),
        samples.transpose(0, 1).unsqueeze(-1).double(),
    )
    min_diff = torch.min(diff, torus_size - diff)
    return torch.linalg.vector_norm(min_diff, ord=ord, dim=0)


def euclidean_pairwise(samples, ord=2):
    return torch.cdist(
        samples.unsqueeze(0).double(),
        samples.unsqueeze(0).double(),
        p=ord,
    )[0]
