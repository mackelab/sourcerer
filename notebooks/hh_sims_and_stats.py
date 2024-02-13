# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from corner import corner
from matplotlib import pyplot as plt

from sourcerer.fit_surrogate import (
    create_train_val_dataloaders,
    fit_conditional_normalizing_flow,
)
from sourcerer.hh_utils import HHSurro, PRIOR_MAX, PRIOR_MIN, DEF_RESTRICTED
from sourcerer.utils import scale_tensor


# %%
# Load data assuming the order here is sufficiently random
sim_data = np.load("TODO_path_to_simulations_array.npz")
theta = sim_data["theta"]
stats = sim_data["stats"]

# Remove undefined simulations (either only the 5 out of 15mil that completelly fail, or the ones without undefined stats)
keeping = (~np.isnan(np.mean(stats, axis=1))) & (~np.isinf(np.mean(stats, axis=1)))

moment_keeping = (~np.isnan(stats[:, 22])) & (~np.isinf(stats[:, 22]))  # 22 is a moment
print(theta[~moment_keeping, :])  # 5 sims out of 15mil completely fail

stats = stats[moment_keeping, :]  # delete Nan simulations that completely fail
theta = theta[moment_keeping, :]  # delete Nan simulations that completely fail

stats = stats[:, DEF_RESTRICTED]
# reverse engineer unnecessarily undefined counts
stats[:, :1][np.isnan(stats[:, :1])] = np.log(3)

number_of_sims = 1_000_000
stats = stats[:number_of_sims, :]
theta = theta[:number_of_sims, :]
keeping = keeping[:number_of_sims]

# %%
source_dim = 13
# standardize source to range from -1 to 1
source = scale_tensor(
    torch.from_numpy(np.float32(theta)),
    PRIOR_MIN,
    PRIOR_MAX,
    -torch.ones(source_dim),
    torch.ones(source_dim),
)
print(source.shape)

# %%
# move res to torch and standardize
stats_torch = torch.from_numpy(np.float32(stats))
stats_mean = torch.mean(stats_torch, dim=0)
stats_std = torch.std(stats_torch, dim=0)
print(stats_mean)
print(stats_std)
stats_torch = (stats_torch - stats_mean) / stats_std


# %%
for i in range(stats_torch.shape[1]):
    plt.hist(stats_torch.numpy()[:, i], bins=100)
    plt.axvline(
        np.percentile(stats_torch.numpy()[:, i], 10),
        color="red",
        linestyle="--",
        linewidth=1,
    )
    plt.axvline(
        np.percentile(stats_torch.numpy()[:, i], 90),
        color="red",
        linestyle="--",
        linewidth=1,
    )
    plt.show()


# %%
# Define surrogate model
ydim = 5
surrogate = HHSurro(hidden_layer_dim=256, xdim=13, ydim=ydim).to("cuda")
optimizer = optim.Adam(surrogate.parameters(), lr=5e-4, weight_decay=1e-5)

# %%
training_dataset, validation_dataset = create_train_val_dataloaders(
    y=stats_torch.to("cuda"),
    x=source.to("cuda"),
    batch_size=4096,
    validation_size=0.2,
    random_state=0,
)

# %%
# Train surrogate model
training_loss, validation_loss = fit_conditional_normalizing_flow(
    network=surrogate,
    optimizer=optimizer,
    training_dataset=training_dataset,
    validation_dataset=validation_dataset,
    nb_epochs=500,
    # early_stopping_patience=20,
    early_stopping_patience=10000,
    print_every=1,
)

# %%
plt.plot(training_loss)
plt.plot(validation_loss)
plt.show()

# %%
surrogate.eval()
with torch.no_grad():
    val_loss = 0.0
    for batch_Y, batch_X in validation_dataset:
        output = surrogate.sample(context=batch_X.to("cuda"))  # forward pass
        loss = surrogate.forward_kld(batch_Y.to("cuda"), batch_X.to("cuda"))
        val_loss += loss.item()

print(val_loss / len(validation_dataset))

# %%
rand_id = np.random.randint(0, batch_Y.shape[0])
print(batch_Y[rand_id, :])
print(output[rand_id, :])

print("aggregate")
print(torch.mean(torch.abs(batch_Y - output), dim=0))
print(torch.mean(batch_Y - output, dim=0))
print(torch.std(torch.abs(batch_Y - output), dim=0))


# %%
# Plot surrogate outputs vs. real data.
fig1 = corner(
    batch_Y.cpu().numpy(),
    hist_kwargs={"density": True},
    plot_density=False,
    # plot_contours=False,
)
corner(
    output.cpu().numpy(),
    hist_kwargs={"density": True},
    fig=fig1,
    color="red",
    plot_density=False,
    # plot_contours=False,
)

pass

# %%
torch.save(
    surrogate.state_dict(),
    f"TODO_path_save_surrogate.pt",
)
