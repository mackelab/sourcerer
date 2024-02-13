"""Taken and adapted from https://github.com/MaximeVandegar/NEB/blob/main/inference/neb/mc_biased.py"""

import torch
import numpy as np


def compute_estimator(
    minibatch,
    nb_mc_integration_steps,
    source_model,
    likelihood_model,
):
    """
    Estimate the log marginal likelihood of the minibatch with Monte Carlo integration (biased)
    """

    # Shorter than original implementation
    minibatch = torch.repeat_interleave(minibatch, nb_mc_integration_steps, dim=0)

    # Samples from the prior
    x = source_model.sample(minibatch.shape[0])

    # Estimates the log marginal likelihood
    log_y_likelihoods = likelihood_model.log_prob(minibatch, context=x)
    log_y_likelihoods = log_y_likelihoods.view(-1, nb_mc_integration_steps)
    log_marginal_likelihood = torch.mean(torch.logsumexp(log_y_likelihoods, dim=1))
    return log_marginal_likelihood


def train_lml_source(
    observations,
    source_model,
    likelihood_model,
    optimizer,
    lam,
    entro_target=None,  # TODO General target distributions are currently not supported
    batch_size=128,
    nb_mc_integration_steps=1024,
    num_kole_samples=512,
    nb_epochs=300,
    validation_set_size=0.1,
    early_stopping_min_epochs=10,
    early_stopping_patience=10,
    print_every=10,
):
    validation_set_size = int(observations.shape[0] * validation_set_size)
    validation_set = observations[:validation_set_size]
    observations = observations[validation_set_size:]
    validation_loss = []
    best_validation_loss = float("inf")

    likelihood_model.eval()

    batchnum = 0
    training_loss = []
    for epoch in range(nb_epochs):
        # Training
        source_model.train()

        overall_batch_loss = []
        entropy_batch_loss = []
        lml_batch_loss = []
        dataloader = create_dataloader(observations, batch_size)

        for batch in dataloader:
            train_entropy_loss = source_model.reverse_kld(
                entro_target, num_kole_samples
            )

            lml = -compute_estimator(
                batch,
                nb_mc_integration_steps,
                source_model,
                likelihood_model,
            )

            optimizer.zero_grad()
            loss = (1.0 - lam[batchnum]) * lml + lam[batchnum] * train_entropy_loss
            loss.backward()
            optimizer.step()

            overall_batch_loss.append(loss.item())
            entropy_batch_loss.append(train_entropy_loss.item())
            lml_batch_loss.append(lml.item())

            batchnum += 1

        # print every 10
        if epoch % print_every == 0:
            print(
                f"Epoch: {epoch}, Train Loss: {np.mean(overall_batch_loss)}, "
                f"Entropy Loss: {np.mean(entropy_batch_loss)}, "
                f"MLL Loss: {np.mean(lml_batch_loss)}"
            )

        training_loss.append(np.mean(overall_batch_loss))

        if validation_set_size == 0:
            continue

        # Validation
        source_model.eval()

        lml_val_batch_loss = []
        dataloader = create_dataloader(validation_set, batch_size)
        for batch in dataloader:
            with torch.no_grad():
                lml = -compute_estimator(
                    batch,
                    nb_mc_integration_steps,
                    source_model,
                    likelihood_model,
                )
            lml_val_batch_loss.append(lml.item())

        validation_loss.append(np.mean(lml_val_batch_loss))

        if epoch % print_every == 0:
            print(f"Epoch: {epoch}, Val LML Loss: {np.mean(lml_val_batch_loss)}")

        if epoch < early_stopping_min_epochs:
            continue

        if validation_loss[-1] < best_validation_loss:
            early_stopping_count = 0
            best_validation_loss = validation_loss[-1]
        else:
            early_stopping_count += 1
            if early_stopping_count > early_stopping_patience:
                print(
                    "Validation loss did not improve. "
                    f"Stopped training after {epoch} epochs."
                )
                break

    return training_loss, validation_loss


def create_dataloader(dataset, batch_size):
    """
    Split the dataset into stochastic minibatches
    """
    r = torch.randperm(dataset.shape[0])
    dataset = dataset[r]
    dataloader = torch.split(dataset, batch_size)
    return dataloader
