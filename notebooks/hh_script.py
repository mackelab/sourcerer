# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from corner import corner
from omegaconf import OmegaConf as OC

from sourcerer.hh_simulator import EphysModel
from sourcerer.hh_utils import DEF_RESTRICTED, PRIOR_MAX, PRIOR_MIN, HHSurro
from sourcerer.real_nvp import (
    Sampler,
    RealNVPs,
    VariableTemperedUniform,
    kozachenko_leonenko_estimator,
)
from sourcerer.sbi_classifier_two_sample_test import c2st_scores
from sourcerer.sliced_wasserstein import sliced_wasserstein_distance
from sourcerer.utils import (
    read_pickle,
    save_cfg_as_yaml,
    save_fig,
    save_numpy_csv,
    save_state_dict,
    scale_tensor,
    script_or_command_line_cfg,
    set_seed,
)
from sourcerer.wasserstein_estimator import train_source

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Define config
# NOTE: These overrides only take effect if this script is run interactively
local_overrides = [
    "base.tag=debug",
    "base.folder=misc",
    "source=wasserstein_hh",
    "source.fin_lambda=0.2",
    "surrogate=load_hh_surrogate",
    "surrogate.ydim=5",
    "surrogate.surrogate_path=../results_sourcerer/surrogates/hh_surrogate.pt",
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
# Load surrogate model
surrogate = HHSurro(
    hidden_layer_dim=cfg.surrogate.hidden_layer_dim,
    xdim=cfg.surrogate.xdim,
    ydim=cfg.surrogate.ydim,
)

state_dict = torch.load(cfg.surrogate.surrogate_path)
surrogate.load_state_dict(state_dict)
surrogate = surrogate.to(device)

# %%
# Load data and standardize to same scale as for training.
data = read_pickle(cfg.source.xo_path)
print(data["X_o"].head())

full_xo_stats_np = data["X_o"].to_numpy()
num_xo = full_xo_stats_np.shape[0]
print(num_xo)

xo_stats_np = full_xo_stats_np[:, DEF_RESTRICTED]

# NOTE: add standadization of summary stats used to train surrogate here
# This ensures, that the data is on the same scale as the data used to train the surrogate
xo_stats = torch.from_numpy(np.float32(xo_stats_np))
supervised_mean = torch.tensor(
    [  # correct restricted 1 mil
        2.3512,
        -93.2657,
        -52.7358,
        278.4319,
        0.4392,
    ],
)

supervised_std = torch.tensor(
    [  # correct restricted 1 mil
        1.1922,
        20.0920,
        19.6483,
        300.1352,
        4.4579,
    ],
)
xo_stats_norm = (xo_stats - supervised_mean) / supervised_std
xo_stats_norm = xo_stats_norm.to(device)

# %%
corner(
    xo_stats_np,
    plot_density=False,
)

pass

# %%
# Define source flows
if cfg.source_model.self == "sampler":
    source = Sampler(
        xdim=cfg.surrogate.xdim,
        input_noise_dim=cfg.surrogate.xdim,
        hidden_layer_dim=cfg.source_model.hidden_layer_dim,
        num_hidden_layers=cfg.source_model.num_hidden_layers,
    )
elif cfg.source_model.self == "real_nvp":
    source = RealNVPs(
        data_dim=cfg.surrogate.xdim,
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

train_loss = train_source(
    data=xo_stats_norm,
    source_model=source,
    simulator=surrogate,
    optimizer=optimizer,
    entro_dist=None,
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

plt.plot(train_loss)
plt.show()

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
    estimated_source = source.sample(10000)
    moved_estimated_source = scale_tensor(
        estimated_source.cpu(),
        -torch.ones(13),
        torch.ones(13),
        PRIOR_MIN,
        PRIOR_MAX,
    )
    surro_estimated_pf = surrogate.sample(10000, estimated_source)


# %%
# Plot estimated source
fig_source = corner(
    estimated_source.cpu().numpy(),
    color="red",
    bins=20,
    hist_kwargs={"density": True},
    # plot_contours=False,
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
# Estimate entropy of learned source
estimated_source_kole = kozachenko_leonenko_estimator(
    estimated_source, on_torus=False
).item()

print("Estimated source entropy estimate:")
print(estimated_source_kole)

save_numpy_csv(
    np.array([estimated_source_kole]),
    file_name=f"{cfg.base.tag}_estimated_source_kole.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

# %%
# Plot data with surrogate pushforward of learned source
fig_surro = corner(
    xo_stats_norm.cpu().numpy(),
    color="black",
    bins=20,
    hist_kwargs={"density": True},
    # plot_contours=False,
    plot_density=False,
    # plot_datapoints=False,
)
corner(
    surro_estimated_pf.cpu().numpy(),
    fig=fig_surro,
    color="red",
    bins=20,
    hist_kwargs={"density": True},
    # plot_contours=False,
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
# Evaluate Sliced-Wasserstein distance between surrogate pushforward and data
num_repeat = 10
surro_dists = np.zeros(num_repeat)
for idx in range(10):
    with torch.no_grad():
        surro_est_pf_add = surrogate.sample(num_xo, source.sample(num_xo))
    surro_dists[idx] = sliced_wasserstein_distance(
        xo_stats_norm,
        surro_est_pf_add,
        num_projections=4096,
        device=device,
    )

# %%
print(surro_dists)
print(np.mean(surro_dists))

save_numpy_csv(
    surro_dists,
    file_name=f"{cfg.base.tag}_surro_pf_swds.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

# %%
# Use the original simulator to pushforward the learned soruce
M1_model = EphysModel(
    name="M1",
    T=25.0,
    E_Na=69.0,
    E_K=-98.4,
    E_Ca=127.2,
    start=100,
    end=700,
    dt=0.04,
    n_processes=None,
    noise_factor=10,
)
sub_moved_estimated_source = moved_estimated_source[:num_xo]
simu_estimated_pf = M1_model.simulation_wrapper(sub_moved_estimated_source)

# %%
# NOTE: replace all Nans with log(3). This only works for count statistics!
simu_estimated_pf_np = simu_estimated_pf.numpy()
sub_moved_estimated_source_np = sub_moved_estimated_source.numpy()

# number of undefined for each stat
print(np.sum(np.isnan(simu_estimated_pf_np), axis=0))

strict_keeping = ~np.isnan(np.mean(simu_estimated_pf_np, axis=1))
print(np.sum(strict_keeping))  # number of undefined before subselect, all stats

def_simu_estimated_pf_np = simu_estimated_pf_np[:, DEF_RESTRICTED]

def_keeping = ~np.isnan(np.mean(def_simu_estimated_pf_np, axis=1))
print(np.sum(def_keeping))  # number of undefined after subselect
def_simu_estimated_pf_np[np.isnan(def_simu_estimated_pf_np)] = np.log(3)
def_keeping = ~np.isnan(np.mean(def_simu_estimated_pf_np, axis=1))
print(np.sum(def_keeping))  # fixing the subselect should result in no more undefined


# %%
# Plot data with simulator pushforward
fig_simu = corner(
    xo_stats_np,
    color="black",
    bins=20,
    hist_kwargs={"density": True},
    # plot_contours=False,
    plot_density=False,
    # plot_datapoints=False,
)
corner(
    def_simu_estimated_pf_np,
    fig=fig_simu,
    color="red",
    bins=20,
    hist_kwargs={"density": True},
    # plot_contours=False,
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
simu_dist = sliced_wasserstein_distance(
    xo_stats_norm.cpu(),
    (torch.from_numpy(np.float32(def_simu_estimated_pf_np)) - supervised_mean)
    / supervised_std,
    num_projections=4096,
)

print(simu_dist)

# %%
save_numpy_csv(
    np.array([simu_dist]),
    file_name=f"{cfg.base.tag}_simu_pf_swd.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)


# %%
# Evaluate C2ST between data and simulator pushforward
# NOTE: C2ST standardizes the data internally, so we don't need to do it here
simu_c2st = np.mean(
    c2st_scores(
        torch.from_numpy(xo_stats_np),
        torch.from_numpy(def_simu_estimated_pf_np),
        seed=10,
    )
)
print(simu_c2st)

# %%
save_numpy_csv(
    np.array([simu_c2st]),
    file_name=f"{cfg.base.tag}_simu_c2st.csv",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

# %%
# Plot some traces from simulator pushforward of learned source
fig_samp_source, axs = plt.subplots(8, 5, figsize=(12, 15))
for i in range(8):
    for j in range(5):
        rand = np.random.randint(sub_moved_estimated_source.shape[0])
        trial = sub_moved_estimated_source[rand][np.newaxis, :]
        x = M1_model.run_HH_model(trial)
        axs[i, j].plot(x["time"], x["data"][0, 0, :], c="C3")
        axs[i, j].set_title(rand)
        axs[i, j].set_axis_off()

save_fig(
    fig_samp_source,
    file_name=f"{cfg.base.tag}_samp_source_fig.pdf",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)


# %%
# Plot some traces from simulated pushforward of reference box prior
prior = VariableTemperedUniform(lower_bound=PRIOR_MIN, upper_bound=PRIOR_MAX)
prior_samples = prior.sample(num_xo)

simu_box_pf = M1_model.simulation_wrapper(prior_samples)

# %%
# same procedure as above (this is a bit messy)
simu_box_pf_np = simu_box_pf.numpy()

strict_keeping_box = ~np.isnan(np.mean(simu_box_pf_np, axis=1))
print(np.sum(strict_keeping_box))

def_stats_box_np = simu_box_pf_np[:, DEF_RESTRICTED]

def_keeping_box = ~np.isnan(np.mean(def_stats_box_np, axis=1))
print(np.sum(def_keeping_box))
def_stats_box_np[np.isnan(def_stats_box_np)] = np.log(3)
def_keeping_box = ~np.isnan(np.mean(def_stats_box_np, axis=1))
print(np.sum(def_keeping_box))


# %%
fig_simu_extend = corner(
    xo_stats_np,
    color="black",
    bins=20,
    hist_kwargs={"density": True},
    # plot_contours=False,
    plot_density=False,
    # plot_datapoints=False,
)
corner(
    def_stats_box_np,
    fig=fig_simu_extend,
    color="blue",
    bins=20,
    hist_kwargs={"density": True},
    # plot_contours=False,
    plot_density=False,
    # plot_datapoints=False,
)
corner(
    def_simu_estimated_pf_np,
    fig=fig_simu_extend,
    color="red",
    bins=20,
    hist_kwargs={"density": True},
    # plot_contours=False,
    plot_density=False,
    # plot_datapoints=False,
)

save_fig(
    fig_simu_extend,
    file_name=f"{cfg.base.tag}_simulator_extended_fig.pdf",
    folder=cfg.base.folder,
    base_path=cfg.base.base_path,
)

pass
