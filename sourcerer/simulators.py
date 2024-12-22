"""Taken from https://github.com/MaximeVandegar/NEB/blob/main/simulators"""

import numpy as np
import torch
from torch import nn
from torch.distributions import LogNormal, MultivariateNormal, Normal, Uniform
from torchdiffeq import odeint


class DeltaSimulator:
    def __init__(self, dim):
        self.ydim = dim
        self.xdim = dim
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def sample_prior(self, size):
        raise NotImplementedError

    def sample(self, context):
        return context

    def train(self):
        pass

    def eval(self):
        pass


class TwoMoonsSimulator:
    """
    Two Moons Simulator as in:

    - Vandegar et al. (2020): "Neural Empirical Bayes Neural Empirical Bayes: Source Distribution Estimation and its Applications to Simulation-Based Inference"
    - Lueckmann et al. (2021): "Benchmarking Simulation-Based Inference"
    """

    def __init__(self, mean_radius=0.1, std_radius=0.01):
        self.mean_radius = mean_radius
        self.std_radius = std_radius
        self.ydim = 2
        self.xdim = 2
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def get_ground_truth_parameters(self):
        return torch.tensor([0.0, 0.7])

    def sample_prior(self, size):
        m = Uniform(
            torch.zeros((size, 2), device=self.device) - 1.0,
            torch.zeros((size, 2), device=self.device) + 1.0,
        )
        theta = m.sample()
        return theta

    def sample(self, context):
        theta = context
        size_theta = theta.shape

        uniform = Uniform(
            torch.zeros(size_theta[0], device=self.device) + (-np.pi / 2),
            torch.zeros(size_theta[0], device=self.device) + (np.pi / 2),
        )
        normal = Normal(
            torch.zeros(size_theta[0], device=self.device) + self.mean_radius,
            torch.zeros(size_theta[0], device=self.device) + self.std_radius,
        )

        a = uniform.rsample()
        r = normal.rsample()

        p = torch.empty(size_theta, device=self.device)

        p[:, 0] = torch.mul(r, torch.cos(a)) + 0.25
        p[:, 1] = torch.mul(r, torch.sin(a))

        q = torch.empty(size_theta, device=self.device)
        q[:, 0] = -torch.abs(theta[:, 0] + theta[:, 1]) / np.sqrt(2)
        q[:, 1] = (-theta[:, 0] + theta[:, 1]) / np.sqrt(2)

        return p + q

    def train(self):
        pass

    def eval(self):
        pass


class SLCPSimulator:
    """
    Simple Likelihood Complex Posterior simulator, as in:

    - Vandegar et al. (2020): "Neural Empirical Bayes Neural Empirical Bayes: Source Distribution Estimation and its Applications to Simulation-Based Inference"
    - Lueckmann et al. (2021): "Benchmarking Simulation-Based Inference"
    """

    def __init__(self):
        self.ydim = 8
        self.xdim = 5
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def get_ground_truth_parameters(self):
        return torch.tensor([-0.7, -2.9, -1.0, -0.9, 0.6])

    def sample_prior(self, size):
        uniform = Uniform(
            (torch.zeros(size, 5) + torch.tensor([-3.0])).to(self.device),
            (torch.zeros(size, 5) + torch.tensor([3.0])).to(self.device),
        )
        return uniform.sample()

    def sample(self, context):
        means = context[:, :2]
        s1 = torch.pow(context[:, 2], 2)
        s2 = torch.pow(context[:, 3], 2)
        pho = torch.tanh(context[:, 4])

        # NOTE: add small values to the diagonal because of some numerical issues
        cov = torch.zeros(context.shape[0], 2, 2, device=self.device)
        cov[:, 0, 0] = torch.pow(s1, 2) + 1e-3
        cov[:, 0, 1] = pho * s1 * s2
        cov[:, 1, 0] = pho * s1 * s2
        cov[:, 1, 1] = torch.pow(s2, 2) + 1e-3

        # means.to(self.device) ensures that mean is on correct device independent of device of x
        normal = MultivariateNormal(means.to(self.device), cov)

        y = torch.zeros(context.shape[0], 8, device=self.device)
        y[:, :2] = normal.rsample()
        y[:, 2:4] = normal.rsample()
        y[:, 4:6] = normal.rsample()
        y[:, 6:] = normal.rsample()

        return y

    def train(self):
        pass

    def eval(self):
        pass


class InverseKinematicsSimulator:
    """
    Inverse Kinematic Simulator, as in:

    - Vandegar et al. (2020): "Neural Empirical Bayes Neural Empirical Bayes: Source Distribution Estimation and its Applications to Simulation-Based Inference"

    """

    def __init__(
        self, prior_var=None, l1=1 / 2, l2=1 / 2, l3=1, noise=0.01 / 180 * np.pi
    ):
        if prior_var is None:
            prior_var = [1 / 16, 1 / 4, 1 / 4, 1 / 4]
        self.prior_var = torch.tensor(prior_var)
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.noise = noise
        self.ydim = 2
        self.xdim = 4
        self.device = "cpu"

    def to(self, device):
        self.device = device
        self.prior_var = self.prior_var.to(device)
        return self

    def get_ground_truth_parameters(self):
        return torch.tensor([0.1, -0.4, 0.5, -0.1])

    def sample_prior(self, size):
        prior = MultivariateNormal(
            torch.zeros(size, 4, device=self.device),
            self.prior_var * torch.eye(4, device=self.device),
        )
        return prior.sample()

    def sample(self, context):
        y = torch.empty((context.shape[0], 2), device=self.device)

        noise_distribution = Normal(
            torch.zeros(context.shape[0], device=self.device), self.noise
        )

        context = context.to(self.device)
        y[:, 0] = (
            self.l1 * torch.sin(context[:, 1] + noise_distribution.rsample())
            + self.l2
            * torch.sin(context[:, 2] + context[:, 1] + noise_distribution.rsample())
            + self.l3
            * torch.sin(
                context[:, 3]
                + context[:, 1]
                + context[:, 2]
                + noise_distribution.rsample()
            )
            + context[:, 0]
        )
        y[:, 1] = (
            self.l1 * torch.cos(context[:, 1] + noise_distribution.rsample())
            + self.l2
            * torch.cos(context[:, 2] + context[:, 1] + noise_distribution.rsample())
            + self.l3
            * torch.cos(
                context[:, 3]
                + context[:, 1]
                + context[:, 2]
                + noise_distribution.rsample()
            )
        )

        return y

    def train(self):
        pass

    def eval(self):
        pass


class GaussianMixtureSimulator:
    """
    Gaussian Mixture Simulator, as in:

    - Lueckmann et al. (2021): "Benchmarking Simulation-Based Inference"
    """

    def __init__(self):
        self.xdim = 2
        self.ydim = 2
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def sample_prior(self, size):
        return 0.5 * torch.rand(size, self.xdim, device=self.device) + 0.5

    def sample(self, context):
        rand_ints = torch.randint(0, 2, (context.shape[0],))
        vector = torch.where(rand_ints == 0, 0.1, 1.0).unsqueeze(1).to(self.device)
        return context + vector * torch.randn(context.shape, device=self.device)

    def train(self):
        pass

    def eval(self):
        pass


class SIRSimulator:
    """
    Susecptible, Infectious, Recovered (SIR) simulator as in:

    - Lueckmann et al. (2021): "Benchmarking Simulation-Based Inference"

    """

    def __init__(self):
        super().__init__()
        self.ydim = 50
        self.xdim = 2
        self.device = "cpu"
        self.t = torch.linspace(0.0, 160.0, self.ydim)
        self.lN = 1000000.0

    def to(self, device):
        self.device = device
        return self

    def sample_prior(self, size):
        beta_log = LogNormal(
            torch.log(torch.tensor([0.4], device=self.device)),
            torch.tensor([0.5], device=self.device),
        )
        gamma_log = LogNormal(
            torch.log(torch.tensor([0.125], device=self.device)),
            torch.tensor([0.2], device=self.device),
        )
        return torch.cat([beta_log.sample((size,)), gamma_log.sample((size,))], dim=1)

    def sir(self, t, y):
        vals, par = y
        beta = par[:, 0]
        gamma = par[:, 1]
        N = par[:, 2]
        S = vals[:, 0]
        I = vals[:, 1]
        dS = -beta * ((S * I) / N)
        dI = beta * ((S * I) / N) - gamma * I
        dR = gamma * I
        return torch.stack([dS, dI, dR], dim=1), torch.zeros_like(
            par, device=self.device
        )

    def sample(self, context):
        batch_size, _num_par = context.shape

        initial = torch.ones(batch_size, 3, device=self.device)
        initial[:, 0] *= self.lN - 1.0
        initial[:, 1] *= 1.0
        initial[:, 2] *= 0.0

        params = torch.cat(
            [context, torch.ones(batch_size, 1, device=self.device) * self.lN], dim=1
        )

        sol, _param = odeint(
            self.sir, (initial, params), self.t.to(self.device), method="dopri5"
        )
        return sol[:, :, 1].T / self.lN

    def train(self):
        pass

    def eval(self):
        pass


class LotkaVolterraSimulator:
    """
    Lotka-Volterra simulator, as in:

    Gloeckler et al. (2023) - Adversarial Robustness of Amortized Bayesian Inference

    """

    def __init__(self):
        self.ydim = 100
        self.xdim = 4
        self.device = "cpu"
        self.t = torch.linspace(0.0, 20.0, 50)

    def to(self, device):
        self.device = device
        return self

    def sample_prior(self, size):
        return torch.exp(
            torch.sigmoid(0.5 * torch.randn((size, 4), device=self.device))
        )

    def lotka_volterra(self, t, y):
        vals, par = y
        alpha = par[:, 0]
        beta = par[:, 1]
        gamma = par[:, 2]
        delta = par[:, 3]
        x_val = vals[:, 0]
        y_val = vals[:, 1]
        dx_val = alpha * x_val - beta * x_val * y_val
        dy_val = delta * x_val * y_val - gamma * y_val
        return torch.stack([dx_val, dy_val], dim=1), torch.zeros_like(par)

    def sample(self, context):
        batch_size, _num_par = context.shape

        initial = torch.ones(batch_size, 2, device=self.device)

        try:
            sol, _param = odeint(
                self.lotka_volterra,
                (initial, context),
                self.t.to(self.device),
                method="dopri5",
            )
        except:
            print("Warning: Solving with fixed stepsize!")
            sol, _param = odeint(
                self.lotka_volterra,
                (initial, context),
                self.t.to(self.device),
                method="rk4",
                options={"step_size": 1e-2},
            )

        logged = sol.permute(1, 2, 0).reshape(batch_size, -1)

        mask = torch.isfinite(logged)
        logged[~mask] = 0.0

        return Normal(logged, 0.05).rsample()

    def train(self):
        pass

    def eval(self):
        pass
