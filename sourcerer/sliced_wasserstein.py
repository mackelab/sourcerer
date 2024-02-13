import torch


def rand_projections(n_projections, n_dims, device="cpu"):
    """
    Generate random projection vectors.
    """
    projections = torch.randn(n_projections, n_dims, device=device)
    projections /= torch.norm(projections, dim=1, keepdim=True)
    return projections


def sliced_wasserstein_distance(
    encoded_samples, distribution_samples, num_projections=50, p=2, device="cpu"
):
    """
    Sliced Wasserstein distance between encoded samples and distribution samples

    Args:
        encoded_samples (torch.Tensor): tensor of encoded training samples
        distribution_samples (torch.Tensor): tensor drawn from the prior distribution
        num_projection (int): number of projections to approximate sliced wasserstein distance
        p (int): power of distance metric
        device (torch.device): torch device 'cpu' or 'cuda' gpu

    Return:
        torch.Tensor: Tensor of wasserstein distances of size (num_projections, 1)
    """

    embedding_dim = distribution_samples.size(-1)

    projections = rand_projections(num_projections, embedding_dim, device=device)

    encoded_projections = encoded_samples.matmul(projections.transpose(-2, -1))

    distribution_projections = distribution_samples.matmul(
        projections.transpose(-2, -1)
    )

    wasserstein_distance = (
        torch.sort(encoded_projections.transpose(-2, -1), dim=-1)[0]
        - torch.sort(distribution_projections.transpose(-2, -1), dim=-1)[0]
    )

    wasserstein_distance = torch.pow(torch.abs(wasserstein_distance), p)

    # NOTE: currently computes the "squared" wasserstein distance
    # No p-th root is applied

    # return torch.pow(torch.mean(wasserstein_distance, dim=(-2, -1)), 1 / p)
    return torch.mean(wasserstein_distance, dim=(-2, -1))
