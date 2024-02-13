# %%
import pandas as pd
import seaborn as sns
import torch

from sourcerer.real_nvp import Sampler
from sourcerer.simulators import InverseKinematicsSimulator
from sourcerer.utils import set_seed

set_seed(42)

# NOTE: Lambda 11 is 0.35

samp = Sampler(xdim=4, input_noise_dim=4, hidden_layer_dim=100)
samp.load_state_dict(torch.load("TODO_path_estimated_ik_source.pt"))

simu = InverseKinematicsSimulator()

num_samples = 1000
with torch.no_grad():
    prior = simu.sample_prior(num_samples)
    simu_pf = simu.sample(prior)

    source = samp.sample(num_samples)
    samp_pf = simu.sample(source)


# convert to numpy
prior = prior.numpy()
simu_pf = simu_pf.numpy()
source = source.numpy()
samp_pf = samp_pf.numpy()

# %%
df1 = pd.DataFrame(prior)
df1["group"] = "prior"
df2 = pd.DataFrame(source)
df2["group"] = "source"
df = pd.concat([df1, df2])

# %%
# Plot the pairplot
g = sns.PairGrid(
    df,
    hue="group",
    diag_sharey=False,
    corner=True,
    palette={"prior": "black", "source": "C0"},
    height=3.25 / 4,
)

g.map_diag(sns.kdeplot, fill=False, hue_order=["source", "prior"])
g.map_lower(sns.scatterplot, alpha=0.5, s=1)
g.set(xticks=[], yticks=[])
g.set(xticklabels=[], yticklabels=[])
g.set(xlabel=None, ylabel=None)

g.savefig(
    "../../results_sourcerer/ik_source.png",
    dpi=300,
    transparent=True,
)

# %%
df11 = pd.DataFrame(simu_pf)
df11["group"] = "prior_pf"
df22 = pd.DataFrame(samp_pf)
df22["group"] = "source_pf"
dff = pd.concat([df11, df22])

# %%
# Plot the pairplot
g = sns.PairGrid(
    dff,
    hue="group",
    diag_sharey=False,
    palette={"prior_pf": "black", "source_pf": "C3"},
    height=3.25 / 4,
)

g.map_diag(sns.kdeplot, fill=False, hue_order=["source_pf", "prior_pf"])
g.map_upper(sns.scatterplot, alpha=0.5, s=1)
g.set(xticklabels=[], yticklabels=[])
g.set(xticks=[], yticks=[])
g.set(xlabel=None, ylabel=None)
g.axes[1, 0].set_axis_off()

g.axes[0, 0].tick_params(left=False)
g.axes[0, 0].spines["left"].set_visible(False)
g.axes[1, 1].tick_params(left=False)
g.axes[1, 1].spines["left"].set_visible(False)


g.savefig(
    "../../results_sourcerer/figures/ik_pf.png",
    dpi=300,
    transparent=True,
)
