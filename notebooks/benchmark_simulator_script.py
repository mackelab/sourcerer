# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from corner import corner
from omegaconf import OmegaConf as OC

from sourcerer.fit_surrogate import (
    create_train_val_dataloaders,
    fit_conditional_normalizing_flow,
)
from sourcerer.likelihood_estimator import train_lml_source
from sourcerer.real_nvp import (
    RealNVPs,
    Sampler,
    TemperedUniform,
    kozachenko_leonenko_estimator,
)
from sourcerer.sbi_classifier_two_sample_test import c2st_scores
from sourcerer.simulators import (
    GaussianMixtureSimulator,
    InverseKinematicsSimulator,
    LotkaVolterraSimulator,
    SIRSimulator,
    SLCPSimulator,
    TwoMoonsSimulator,
)
from sourcerer.sliced_wasserstein import sliced_wasserstein_distance
from sourcerer.utils import (
    save_cfg_as_yaml,
    save_fig,
    save_numpy_csv,
    save_state_dict,
    script_or_command_line_cfg,
    set_seed,
)
from sourcerer.wasserstein_estimator import train_source

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_simulator(cfg):
    if cfg.simulator.self == "two_moons":
        return TwoMoonsSimulator()
    elif cfg.simulator.self == "inverse_kinematics":
        return InverseKinematicsSimulator()
    elif cfg.simulator.self == "slcp":
        return SLCPSimulator()
    elif cfg.simulator.self == "gaussian_mixture":
        return GaussianMixtureSimulator()
    elif cfg.simulator.self == "sir":
        return SIRSimulator()
    elif cfg.simulator.self == "lotka_volterra":
        return LotkaVolterraSimulator()


# %%
# Define config
# NOTE: These overrides only take effect if this script is run interactively
local_overrides = [
    "base.tag=debug",
    "base.folder=misc",
    "simulator=inverse_kinematics",
]

cfg = script_or_command_line_cfg(
    config_name="config",
    config_path="../conf",
    local_overrides=local_overrides,
    name=__name__,
)

assert cfg.base.tag is not None
assert cfg.base.folder is not None

print(OC.to_yaml(cfg))

save_cfg_as_yaml(
    cfg,
    f"{cfg.base.tag}_cfg.yaml",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

if cfg.base.seed is None:
    random_random_seed = np.random.randint(2**16)
    set_seed(random_random_seed)
    save_numpy_csv(
        np.array([random_random_seed], dtype=int),
        file_name=f"{cfg.base.tag}_seed.csv",
        folder=cfg.base.folder,
        base_path=cfg.base.base_path,
    )
else:
    set_seed(cfg.base.seed)


# %%
# Define Simulator and reference domain. Train a surrogate/load an existing surrogate if necessary.
simulator = get_simulator(cfg)
simulator = simulator.to(device)

box_domain = TemperedUniform(
    cfg.simulator.box_domain_lower,
    cfg.simulator.box_domain_upper,
    simulator.xdim,
    device=device,
)

# %%
surrogate = RealNVPs(
    flow_length=cfg.surrogate.flow_length,
    data_dim=simulator.ydim,
    context_dim=simulator.xdim,
    hidden_layer_dim=cfg.surrogate.hidden_layer_dim,
)
surrogate = surrogate.to(device)

# %%
if cfg.surrogate.self == "load_surrogate":
    surrogate.load_state_dict(torch.load(cfg.surrogate.surrogate_path))
elif cfg.surrogate.self == "train_surrogate":
    surro_train_domain = box_domain.sample(cfg.surrogate.num_training_samples)
    surro_train_push_forward = simulator.sample(surro_train_domain)

    surro_optimizer = torch.optim.Adam(
        surrogate.parameters(),
        lr=cfg.surrogate.surrogate_lr,
        weight_decay=cfg.surrogate.surrogate_weight_decay,
    )

    train_dataset, val_dataset = create_train_val_dataloaders(
        surro_train_push_forward,
        surro_train_domain,
    )

    train_loss, val_loss = fit_conditional_normalizing_flow(
        network=surrogate,
        optimizer=surro_optimizer,
        training_dataset=train_dataset,
        validation_dataset=val_dataset,
        nb_epochs=cfg.surrogate.nb_epochs,
        early_stopping_patience=cfg.surrogate.early_stopping_patience,
    )

    torch.save(
        surrogate.state_dict(),
        os.path.join(
            cfg.base.base_path, cfg.base.folder, f"{cfg.base.tag}_surrogate.pt"
        ),
    )

    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="validation loss")
    plt.legend()
    plt.show()
else:
    assert cfg.surrogate.self == "no_surrogate"

# %%
# Sample from source distribution and push forward through surrogate simulator
# Here, we evaluate on the source distribution!
surrogate.eval()
gt_source = simulator.sample_prior(cfg.source.num_obs)
gt_source_two = simulator.sample_prior(cfg.source.num_eval_obs)
with torch.no_grad():
    gt_surrogate = surrogate.sample(cfg.source.num_obs, gt_source)
    gt_surrogate_two = surrogate.sample(cfg.source.num_eval_obs, gt_source_two)
    gt_simulator = simulator.sample(gt_source)
    gt_simulator_two = simulator.sample(gt_source_two)

# %%
if cfg.surrogate.self == "load_surrogate" or cfg.surrogate.self == "train_surrogate":
    print("Surrogate vs Simulator y-space C2ST AUC:")
    print(np.mean(c2st_scores(gt_simulator.cpu(), gt_surrogate.cpu())))

# %%
print("Sliced wassertein distance on groundtruth y:")
excepted_distance = sliced_wasserstein_distance(
    gt_simulator[: cfg.source.num_eval_obs],
    gt_simulator_two,
    num_projections=4096,
    device=device,
).item()
print(excepted_distance)
print(np.log(excepted_distance))

# %%
# Define source model
if cfg.source_model.self == "sampler":
    source = Sampler(
        xdim=simulator.xdim,
        input_noise_dim=simulator.xdim,
        hidden_layer_dim=cfg.source_model.hidden_layer_dim,
        num_hidden_layers=cfg.source_model.num_hidden_layers,
        low=cfg.simulator.box_domain_lower,
        high=cfg.simulator.box_domain_upper,
    )
elif cfg.source_model.self == "real_nvp":
    source = RealNVPs(
        data_dim=simulator.xdim,
        context_dim=0,
        hidden_layer_dim=cfg.source_model.hidden_layer_dim,
        flow_length=cfg.source_model.flow_length,
        low=cfg.simulator.box_domain_lower,
        high=cfg.simulator.box_domain_upper,
    )
else:
    raise ValueError

source = source.to(device)

# %%
# Train source model
optimizer = torch.optim.Adam(
    source.parameters(),
    lr=cfg.source.learning_rate,
    weight_decay=cfg.source.weight_decay,
)

# This is the scheduled values of lambda - first we linearly decay from lambda=1.0 (only entropy) until a minimum value of lambda. Then we stay at that value of lambda for more iterations.
schedule = torch.cat(
    [
        torch.ones(cfg.source.pretraining_steps),
        torch.linspace(
            1.0,
            cfg.source.fin_lambda,
            cfg.source.linear_decay_steps,
        ),
        cfg.source.fin_lambda * torch.ones(cfg.source.lambda_steps),
    ]
)

if cfg.source.self == "wasserstein":
    train_source(
        data=gt_simulator,
        source_model=source,
        simulator=simulator if cfg.surrogate.self == "no_surrogate" else surrogate,
        optimizer=optimizer,
        entro_dist=None,  # default uniform is used
        kld_samples=cfg.source.num_kole_samples,
        entro_lambda=schedule,
        wasser_p=cfg.source.wasserstein_order,
        wasser_np=cfg.source.wasserstein_slices,
        use_log_sw=cfg.source.use_log_sw,
        num_chunks=cfg.source.num_chunks,
        epochs=cfg.source.pretraining_steps
        + cfg.source.linear_decay_steps
        + cfg.source.lambda_steps,
        min_epochs_x_chus=cfg.source.pretraining_steps + cfg.source.linear_decay_steps,
        early_stopping_patience=cfg.source.early_stopping_patience,
        device=device,
    )
elif cfg.source.self == "mll":
    assert cfg.surrogate.self != "no_surrogate"

    train_lml_source(
        observations=gt_simulator,
        source_model=source,
        likelihood_model=surrogate,
        optimizer=optimizer,
        lam=schedule,
        entro_target=None,  # default uniform is used
        batch_size=cfg.source.batch_size,
        nb_mc_integration_steps=cfg.source.nb_mc_integration_steps,
        num_kole_samples=cfg.source.num_kole_samples,
        nb_epochs=cfg.source.nb_epochs,
        validation_set_size=cfg.source.validation_set_size,
        early_stopping_min_epochs=cfg.source.early_stopping_min_epochs,
        early_stopping_patience=cfg.source.early_stopping_patience,
        print_every=cfg.source.print_every,
    )
else:
    raise ValueError

# %%
save_state_dict(
    source,
    f"{cfg.base.tag}_learned_source.pt",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

# %%
# Evaluate trained source model
source.eval()
surrogate.eval()
with torch.no_grad():
    estimated_source = source.sample(cfg.source.num_eval_obs)
    surro_estimated_pf = surrogate.sample(cfg.source.num_eval_obs, estimated_source)
    simu_estimated_pf = simulator.sample(estimated_source)

# %%
# plot true and learned sources
fig_source = corner(
    gt_source_two.cpu().numpy(),
    color="black",
    bins=20,
    hist_kwargs={"density": True},
    plot_contours=False,
    plot_density=False,
    # plot_datapoints=False,
)
corner(
    estimated_source.cpu().numpy(),
    fig=fig_source,
    color="red",
    bins=20,
    hist_kwargs={"density": True},
    plot_contours=False,
    plot_density=False,
    # plot_datapoints=False,
)

save_fig(
    fig_source,
    file_name=f"{cfg.base.tag}_source_fig.pdf",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

pass

# %%
# Avoid plotting very large corner plots
plot_ss = slice(None)
if cfg.simulator.self == "sir":
    plot_ss = slice(1, 50, 4)
elif cfg.simulator.self == "lotka_volterra":
    plot_ss = slice(10, 20)

# %%
# Plot pairplots in observation space with surrogate
fig_surro = corner(
    gt_surrogate_two.cpu().numpy()[:, plot_ss],
    color="black",
    bins=20,
    hist_kwargs={"density": True},
    plot_contours=False,
    plot_density=False,
    # plot_datapoints=False,
)
corner(
    surro_estimated_pf.cpu().numpy()[:, plot_ss],
    fig=fig_surro,
    color="red",
    bins=20,
    hist_kwargs={"density": True},
    plot_contours=False,
    plot_density=False,
    # plot_datapoints=False,
)

save_fig(
    fig_surro,
    file_name=f"{cfg.base.tag}_surrogate_fig.pdf",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

pass

# %%
# Plot pairplots in observation space with true simulator
fig_simu = corner(
    gt_simulator_two.cpu().numpy()[:, plot_ss],
    color="black",
    bins=20,
    hist_kwargs={"density": True},
    plot_contours=False,
    plot_density=False,
    # plot_datapoints=False,
)
corner(
    simu_estimated_pf.cpu().numpy()[:, plot_ss],
    fig=fig_simu,
    color="red",
    bins=20,
    hist_kwargs={"density": True},
    plot_contours=False,
    plot_density=False,
    # plot_datapoints=False,
)

save_fig(
    fig_simu,
    file_name=f"{cfg.base.tag}_simulator_fig.pdf",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

pass


# %%
# Classifier two sample tests
c2st_ss = slice(None)
if cfg.simulator.self == "sir":
    c2st_ss = slice(1, 50)

print("y c2st AUC on simulator:")
simu_c2st = np.mean(
    c2st_scores(
        simu_estimated_pf.cpu()[:, c2st_ss],
        gt_simulator_two.cpu()[:, c2st_ss],
    )
)
print(simu_c2st)

print("y c2st AUC on surrogate:")
surro_c2st = np.mean(
    c2st_scores(
        surro_estimated_pf.cpu()[:, c2st_ss],
        gt_surrogate.cpu()[:, c2st_ss],
    )
)
print(surro_c2st)

print("Source c2st AUC:")
source_c2st = np.mean(
    c2st_scores(
        estimated_source.cpu(),
        gt_source_two.cpu(),
    )
)
print(source_c2st)

# %%
save_numpy_csv(
    np.array([simu_c2st]),
    file_name=f"{cfg.base.tag}_simu_c2st.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

save_numpy_csv(
    np.array([surro_c2st]),
    file_name=f"{cfg.base.tag}_surro_c2st.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

save_numpy_csv(
    np.array([source_c2st]),
    file_name=f"{cfg.base.tag}_source_c2st.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)


# %%
print("Estimate source entropies")
gt_source_kole = kozachenko_leonenko_estimator(gt_source_two, on_torus=False).item()
estimated_source_kole = kozachenko_leonenko_estimator(
    estimated_source, on_torus=False
).item()

print("Ground truth source entropy estimate:")
print(gt_source_kole)
print("Estimated source entropy estimate:")
print(estimated_source_kole)

save_numpy_csv(
    np.array([gt_source_kole]),
    file_name=f"{cfg.base.tag}_gt_source_kole.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)
save_numpy_csv(
    np.array([estimated_source_kole]),
    file_name=f"{cfg.base.tag}_estimated_source_kole.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)


# %%
# NOTE: For a better estimate, resample from ground truth here as well!
num_repeat = 10
surro_dists = np.zeros(num_repeat)
simu_dists = np.zeros(num_repeat)
for idx in range(10):
    with torch.no_grad():
        surro_est_pf_add = surrogate.sample(
            cfg.source.num_eval_obs, source.sample(cfg.source.num_eval_obs)
        )
        simu_est_pf_add = simulator.sample(source.sample(cfg.source.num_eval_obs))
    surro_dists[idx] = sliced_wasserstein_distance(
        gt_simulator_two,
        surro_est_pf_add,
        num_projections=4096,
        device=device,
    )
    simu_dists[idx] = sliced_wasserstein_distance(
        gt_simulator_two,
        simu_est_pf_add,
        num_projections=4096,
        device=device,
    )

print(np.mean(surro_dists))
print(np.std(surro_dists))

print(np.mean(simu_dists))
print(np.std(simu_dists))

save_numpy_csv(
    surro_dists,
    file_name=f"{cfg.base.tag}_surro_pf_swds.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)
save_numpy_csv(
    simu_dists,
    file_name=f"{cfg.base.tag}_simu_pf_swds.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)


# %%
# Only for Lotka Volterra
if cfg.simulator.self == "lotka_volterra":
    gt_simulator_two_np = gt_simulator_two.cpu().numpy()
    simu_estimated_pf_np = simu_estimated_pf.cpu().numpy()
    plt.fill_between(
        np.arange(50),
        np.quantile(simu_estimated_pf_np[:, 50:], 0.25, axis=0),
        np.quantile(simu_estimated_pf_np[:, 50:], 0.75, axis=0),
        color="r",
        alpha=0.3,
    )
    plt.fill_between(
        np.arange(50),
        np.quantile(simu_estimated_pf_np[:, :50], 0.25, axis=0),
        np.quantile(simu_estimated_pf_np[:, :50], 0.75, axis=0),
        color="blue",
        alpha=0.3,
    )
    plt.plot(np.mean(simu_estimated_pf_np[:, 50:], axis=0), c="r")
    plt.plot(np.mean(simu_estimated_pf_np[:, :50], axis=0), c="b")
    plt.show()
    plt.fill_between(
        np.arange(50),
        np.quantile(gt_simulator_two_np[:, 50:], 0.25, axis=0),
        np.quantile(gt_simulator_two_np[:, 50:], 0.75, axis=0),
        color="r",
        alpha=0.3,
    )
    plt.fill_between(
        np.arange(50),
        np.quantile(gt_simulator_two_np[:, :50], 0.25, axis=0),
        np.quantile(gt_simulator_two_np[:, :50], 0.75, axis=0),
        color="blue",
        alpha=0.3,
    )
    plt.plot(np.mean(gt_simulator_two_np[:, 50:], axis=0), c="r")
    plt.plot(np.mean(gt_simulator_two_np[:, :50], axis=0), c="b")
    plt.show()

# %%
if cfg.simulator.self == "sir":
    num_samps = 100
    rand_ids = np.random.choice(10000, size=num_samps, replace=False)
    gt_simulator_two_np = gt_simulator_two.cpu().numpy()
    simu_estimated_pf_np = simu_estimated_pf.cpu().numpy()
    # plot samples interleaved
    for i in rand_ids:
        plt.plot(gt_simulator_two_np[i], c="black", alpha=0.1)
        plt.plot(simu_estimated_pf_np[i], c="C3", alpha=0.1)
    for i in range(1, 10):
        plt.plot(
            np.quantile(gt_simulator_two_np, 0.1 * i, axis=0),
            c="black",
        )
        plt.plot(
            np.quantile(simu_estimated_pf_np, 0.1 * i, axis=0),
            c="C3",
        )

    plt.show()
