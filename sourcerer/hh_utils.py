import torch
import torch.nn as nn


PRIOR_MIN = torch.tensor([0.1, 20, 0.1, 0, 0, 0, 0, 0, 0, -130, 50, -90, 0.1])
PRIOR_MAX = torch.tensor([15, 1000, 70, 250, 100, 30, 3, 250, 3, -50, 4000, -35, 3])
DEF_RESTRICTED = [8, 19, 20, 21, 22]


class HHSurro(nn.Module):
    """
    MLP Surrogate for likelihood of Hodgkin-Huxley model summary statistics.
    """
    def __init__(self, xdim=7, ydim=7, num_hidden_layers=4, hidden_layer_dim=128):
        super().__init__()
        self.xdim = xdim
        self.ydim = ydim

        self.mse = nn.MSELoss(reduction="mean")

        layers = []
        layers.append(nn.Linear(xdim, hidden_layer_dim))
        layers.append(nn.BatchNorm1d(hidden_layer_dim))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_layer_dim, hidden_layer_dim))
            layers.append(nn.BatchNorm1d(hidden_layer_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_layer_dim, ydim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    # Should conform to the interface of RealNVP but cannot really sample
    # Gives the mean of the Gaussian
    def sample(self, size=None, context=None):
        assert context is not None
        if size is not None:
            assert size == context.shape[0]
        return self.forward(context)

    def forward_kld(self, x, context):
        return self.mse(self.forward(context), x)


def sample_with_rejection(
    num_samples,
    base_distribution,
    classifier,
    batch_size=1000,
    strict_threshold=None,
    verbose=False,
):
    """
    Samples from a base distribution using rejection sampling based on a classifier.

    Parameters:
        num_samples: The number of samples to generate.
        base_distribution: The base distribution to sample from.
        classifier: The classifier used for rejection sampling.
        batch_size: The number of samples to generate in each batch. Defaults to 1000.
        strict_threshold: The threshold value for strict rejection. If None, uniform rejection is used. Defaults to None.
        verbose: Whether to print verbose output. Defaults to False.

    Returns:
        torch.Tensor: The generated samples.
    """

    samples = []
    num_accepted = 0
    while num_accepted < num_samples:
        with torch.no_grad():
            z = base_distribution.sample(batch_size)
            p = classifier(z)
            if strict_threshold is None:     # If no threshold is given, reject with uniform samples
                u = torch.rand_like(p, device=z.device)
                accepted = (p > u).squeeze()
            else:
                accepted = (p > strict_threshold).squeeze()
            if verbose:
                print(f"Accepted {accepted.sum()} of {batch_size} samples!")
            samples.append(z[accepted])
            num_accepted += accepted.sum()

    return torch.cat(samples)[:num_samples]
