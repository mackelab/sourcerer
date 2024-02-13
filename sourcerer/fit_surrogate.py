"""Taken and adapted from https://github.com/MaximeVandegar/NEB/blob/main/ml/ml_helper.py"""
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch


def create_train_val_dataloaders(
    y,
    x,
    validation_size=0.2,
    random_state=0,
    batch_size=256,
):
    """
    Create train and validation dataloaders for the given input data.

    Args:
        y (array-like): The target variable.
        x (array-like): The input features.
        validation_size (float, optional): The proportion of the data to include in the validation set.
            Defaults to 0.2.
        random_state (int, optional): The random seed for reproducibility. Defaults to 0.
        batch_size (int, optional): The batch size for the dataloaders. Defaults to 256.

    Returns:
        Training dataloader, validation dataloader
    """
    
    train_y, val_y, train_x, val_x = train_test_split(
        y, x, test_size=validation_size, random_state=random_state
    )
    train_loader = DataLoader(
        TensorDataset(train_y, train_x), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_y, val_x), batch_size=batch_size, shuffle=True
    )
    return train_loader, val_loader


def fit_conditional_normalizing_flow(
    network,
    optimizer,
    training_dataset,
    validation_dataset,
    early_stopping_patience=10,
    nb_epochs=300,
    print_every=10,
):
    """
    Fits a conditional normalizing flow model to the emulate a (probabilistic) simulator.
    
    Args:
        network: The conditional normalizing flow model.
        optimizer: The optimizer used for training.
        training_dataset: The dataset used for training.
        validation_dataset: The dataset used for validation.
        early_stopping_patience: The number of epochs to wait for validation loss improvement before stopping training. Defaults to 10.
        nb_epochs: The total number of training epochs. Defaults to 300.
        print_every: The frequency of printing training progress. Defaults to 10.
    
    Returns:
        tuple: A tuple containing the training loss and validation loss for each epoch.
    """
    
    training_loss = []
    validation_loss = []
    early_stopping_count = 0
    best_validation_loss = float("inf")

    for epoch in range(nb_epochs):
        # Training
        network.train()
        batch_losses = []
        for y_batch, x_batch in training_dataset:
            optimizer.zero_grad()
            loss = network.forward_kld(y_batch, context=x_batch)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_training_loss = np.mean(batch_losses)
        training_loss.append(epoch_training_loss)

        # Validation
        with torch.no_grad():
            batch_losses = []
            network.eval()
            for y_batch, x_batch in validation_dataset:
                loss = network.forward_kld(y_batch, context=x_batch)

                batch_losses.append(loss.item())

            epoch_validation_loss = np.mean(batch_losses)
            validation_loss.append(epoch_validation_loss)

        if epoch % print_every == 0:
            print(
                f"Epoch: {epoch}, Train Loss: {epoch_training_loss}, Val Loss: {epoch_validation_loss}"
            )

        if validation_loss[-1] < best_validation_loss:
            early_stopping_count = 0
            best_validation_loss = validation_loss[-1]
        else:
            early_stopping_count += 1
            if early_stopping_count > early_stopping_patience:
                print(
                    f"Validation loss did not improve for {early_stopping_patience} epochs, "
                    f"stopping training after {epoch} epochs."
                )
                break

    return training_loss, validation_loss
