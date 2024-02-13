import torch

from sourcerer.sliced_wasserstein import sliced_wasserstein_distance


def train_source(
    source_model,
    optimizer,
    data,  # observations
    simulator,  # or surrogate
    entro_dist=None,  # TODO General target distributions are currently not supported
    kld_samples=512,
    num_chunks=1,
    entro_lambda=0.01 * torch.ones(1_000_000),
    scheduler=None,
    wasser_p=2,
    wasser_np=500,
    use_log_sw=False,
    epochs=1000,
    min_epochs_x_chus=500,
    early_stopping_patience=50,
    device="cpu",
):
    """
    Trains the source model by minimizing the wasserstein distance with an entropy maximization term. Currently only implemented for uniform reference distributions!

    Args:
        source_model: Source model to be trained.
        optimizer: Optimizer used for training.
        data: Observations to match.
        simulator: Differentiable simulator or surrogate model.
        entro_dist: Reference entropy distribution. Current implementation ignores this and takes a uniform distribution on a specified domain.
        kld_samples: Number of samples used for computing the reverse KLD. Defaults to 512.
        num_chunks: Number of chunks to divide the data into. Defaults to 1.
        entro_lambda: Values of the lambda regularization strength throughout optimization. Defaults to 0.01 for all optimization time.
        scheduler: Learning rate scheduler. Defaults to None.
        wasser_p: p value used in the Wasserstein distance computation. Defaults to 2.
        wasser_np: Number of projections used in the Sliced-Wasserstein distance computation. Defaults to 500.
        use_log_sw: Whether to use Sliced-Wasserstein distance or log of Sliced-Wasserstein distance in the optimization loss.
        epochs: Number of training epochs.
        min_epochs_x_chus: Minimum number of epochs multiplied by the number of chunks. Defaults to 500.
        early_stopping_patience: Number of chunks to wait for improvement in training loss before stopping training. Defaults to 50.
        device: The device to use for training. Defaults to "cpu".

    Returns:
        list: The training loss history per chunk.
    """
    training_loss = []
    best_training_loss = float("inf")
    num_observations = data.shape[0]

    source_model = source_model.to(device)
    simulator = simulator.to(device)
    data = data.to(device)  # Assumes all data fits on gpu
    entro_lambda = entro_lambda.to(device)

    log_or_id = torch.log if use_log_sw else lambda x: x

    source_model.train()
    simulator.eval()
    for epoch in range(epochs):
        chunks = torch.chunk(
            torch.randperm(num_observations, device=device), num_chunks
        )

        for n_chu, chu in enumerate(chunks):
            epoch_x_chu = epoch * num_chunks + n_chu
            source_samples = source_model.sample(len(chu))

            # reverse kld/entropy
            entropy_loss = source_model.reverse_kld(entro_dist, kld_samples)

            # compute constraint
            push_forward = simulator.sample(context=source_samples)
            y_loss = log_or_id(
                sliced_wasserstein_distance(
                    push_forward,
                    data[chu],
                    num_projections=wasser_np,
                    p=wasser_p,
                    device=device,
                )
            )

            loss = (1.0 - entro_lambda[epoch_x_chu]) * y_loss + entro_lambda[
                epoch_x_chu
            ] * entropy_loss

            if ~(torch.isnan(loss) | torch.isinf(loss)):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            else:
                print("Warning: NaN or Inf loss encountered, skipping step!")

            if epoch_x_chu % 10 == 0:
                print(
                    f"Epoch x Chunk: {epoch_x_chu}, Weighted Loss: {loss.item()}, "
                    f"Entropy loss: {entropy_loss.item()}, "
                    f"Wasser loss: {y_loss.item()}"
                )

            training_loss.append(loss.item())

            if epoch_x_chu < min_epochs_x_chus:
                continue

            if training_loss[-1] < best_training_loss:
                early_stopping_count = 0
                best_training_loss = training_loss[-1]
            else:
                early_stopping_count += 1
                if early_stopping_count > early_stopping_patience:
                    print(
                        "Training loss did not improve. "
                        f"Stopped training after {epoch_x_chu} epochs x chunks."
                    )
                    return training_loss

    return training_loss
