# %%
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sourcerer.real_nvp import Sampler, VariableTemperedUniform
from sourcerer.utils import read_pickle, scale_tensor, set_seed
from sourcerer.hh_utils import PRIOR_MAX, PRIOR_MIN, DEF_RESTRICTED

set_seed(42)

# NOTE: fin_lambda is 0.25

samp = Sampler(xdim=13, input_noise_dim=13, hidden_layer_dim=100)
samp.load_state_dict(
    torch.load("../../results_sourcerer/estimated_hh_source_state_dict.pt")
)

prior = VariableTemperedUniform(lower_bound=PRIOR_MIN, upper_bound=PRIOR_MAX)


num_samples = 1033
with torch.no_grad():
    source = samp.sample(num_samples)
    moved_estimated_source = scale_tensor(
        source,
        -torch.ones(13),
        torch.ones(13),
        PRIOR_MIN,
        PRIOR_MAX,
    )
    # moved_estimated_source = prior.sample(num_samples)


# convert to numpy
moved_estimated_source_np = moved_estimated_source.numpy()
moved_estimated_source_np = moved_estimated_source_np[:, [1, 7, 11]]

# %%
df2 = pd.DataFrame(moved_estimated_source_np)
df2["group"] = "source"


g = sns.PairGrid(
    df2,
    hue="group",
    diag_sharey=False,
    palette={"source": "C0"},
    height=3.25 / 4,
)

g.map_diag(sns.kdeplot, fill=False, hue_order=["source"], bw_adjust=0.5)
g.map_upper(sns.scatterplot, alpha=0.5, s=1)
g.set(xticklabels=[], yticklabels=[])
g.set(xlabel=None, ylabel=None)

for ax in g.axes.flatten():
    if ax is None:
        continue
    ax.set_xticks(ax.get_xlim())
    ax.set_yticks(ax.get_ylim())

g.axes[1, 0].set_axis_off()
g.axes[2, 0].set_axis_off()
g.axes[2, 1].set_axis_off()

g.axes[0, 0].tick_params(left=False)
g.axes[0, 0].spines["left"].set_visible(False)
g.axes[1, 1].tick_params(left=False)
g.axes[1, 1].spines["left"].set_visible(False)
g.axes[2, 2].tick_params(left=False)
g.axes[2, 2].spines["left"].set_visible(False)

g.savefig(
    "../../results_sourcerer/figures/hh_source.pdf",
    transparent=True,
)


# %%
data = read_pickle("TODO_path_to_processed_data.pickle")
print(data["X_o"].head())

full_xo_stats_np = data["X_o"].to_numpy()
num_xo = full_xo_stats_np.shape[0]
print(num_xo)
xo_stats_np = full_xo_stats_np[:, DEF_RESTRICTED]

simu_estimated_pf_np = np.load("TODO_path_simulator_pusforward_of_estimated_source.npy")
simu_box_pf_np = np.load("TODO_path_simulator_pushforward_of_box_prior.npy")

xo_stats_np = xo_stats_np[:, [0, 1, 4]]
simu_estimated_pf_np = simu_estimated_pf_np[:, [0, 1, 4]]
simu_box_pf_np = simu_box_pf_np[:, [0, 1, 4]]

df1 = pd.DataFrame(simu_box_pf_np)
df1["group"] = "box_prior"
df2 = pd.DataFrame(xo_stats_np)
df2["group"] = "x_o"
df3 = pd.DataFrame(simu_estimated_pf_np)
df3["group"] = "estimated_pf"
df = pd.concat([df1, df2, df3])


# %%
g = sns.PairGrid(
    df,
    hue="group",
    diag_sharey=False,
    corner=True,
    palette={"box_prior": "C1", "x_o": "black", "estimated_pf": "C3"},
    height=3.25 / 4,
)

g.map_diag(
    sns.kdeplot,
    fill=False,
    hue_order=["estimated_pf", "x_o", "box_prior"],
    bw_adjust=0.3,
)
g.map_lower(
    sns.scatterplot,
    hue_order=["estimated_pf", "x_o", "box_prior"],
    alpha=0.5,
    s=1,
)
g.set(xticklabels=[], yticklabels=[])
g.set(xlabel=None, ylabel=None)

for ax in g.axes.flatten():
    if ax is None:
        continue
    ax.set_xticks(ax.get_xlim())
    ax.set_yticks(ax.get_ylim())

g.savefig(
    "../../results_sourcerer/figures/hh_pf.pdf",
    transparent=True,
)
