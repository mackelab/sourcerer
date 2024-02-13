# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

from sourcerer.real_nvp import Sampler
from sourcerer.simulators import LotkaVolterraSimulator
from sourcerer.utils import FigureLayout, set_seed

set_seed(22)

matplotlibrc_path = "../../matplotlibrc"

# NOTE: fin_lambda is 0.5
samp = Sampler(xdim=4, input_noise_dim=4, hidden_layer_dim=100)
samp.load_state_dict(torch.load("TODO_path_to_estimated_sir_source_state_dict.pt"))

simu = LotkaVolterraSimulator()

num_samples = 10000
with torch.no_grad():
    prior = simu.sample_prior(num_samples)
    simu_pf = simu.sample(prior)

    source = samp.sample(num_samples)
    samp_pf = simu.sample(source)

prior = prior.numpy()
simu_pf = simu_pf.numpy()
source = source.numpy()
samp_pf = samp_pf.numpy()

FL_LV = FigureLayout(234, 100, 8)
rand_ids = np.random.choice(np.arange(10000), size=10, replace=False)

# %%
gt_simulator_two_np = simu_pf
simu_estimated_pf_np = samp_pf


line_style_obs = "-"
line_style_sim = "-"

with plt.rc_context(rc=FL_LV.get_rc(80, 60), fname=matplotlibrc_path):
    fig, axs = plt.subplots(2)
    plt.subplots_adjust(hspace=15)

    axs[0].plot(
        gt_simulator_two_np[rand_ids, 50:].T, c="black", alpha=0.5, linewidth=0.5
    )
    axs[0].plot(simu_estimated_pf_np[rand_ids, 50:].T, c="C3", alpha=0.5, linewidth=0.5)

    axs[0].plot(
        np.quantile(gt_simulator_two_np[:, 50:], 0.10, axis=0),
        c="black",
        ls=line_style_obs,
    )
    axs[0].plot(
        np.quantile(gt_simulator_two_np[:, 50:], 0.90, axis=0),
        c="black",
        ls=line_style_obs,
    )

    axs[0].plot(
        np.quantile(simu_estimated_pf_np[:, 50:], 0.1, axis=0),
        c="C3",
        ls=line_style_sim,
    )
    axs[0].plot(
        np.quantile(simu_estimated_pf_np[:, 50:], 0.9, axis=0),
        c="C3",
        ls=line_style_sim,
    )

    axs[0].plot(np.median(gt_simulator_two_np[:, 50:], axis=0), c="black")
    axs[0].plot(
        np.median(simu_estimated_pf_np[:, 50:], axis=0), c="C3", ls=line_style_sim
    )

    axs[1].plot(
        gt_simulator_two_np[rand_ids, 50:].T, c="black", alpha=0.5, linewidth=0.5
    )
    axs[1].plot(simu_estimated_pf_np[rand_ids, 50:].T, c="C3", alpha=0.5, linewidth=0.5)

    axs[1].plot(np.quantile(gt_simulator_two_np[:, :50], 0.1, axis=0), c="black")
    axs[1].plot(np.quantile(gt_simulator_two_np[:, :50], 0.9, axis=0), c="black")

    axs[1].plot(
        np.quantile(simu_estimated_pf_np[:, :50], 0.1, axis=0),
        c="C3",
        ls=line_style_sim,
    )
    axs[1].plot(
        np.quantile(simu_estimated_pf_np[:, :50], 0.9, axis=0),
        c="C3",
        ls=line_style_sim,
    )

    axs[1].plot(np.median(gt_simulator_two_np[:, :50], axis=0), c="black")
    axs[1].plot(
        np.median(simu_estimated_pf_np[:, :50], axis=0), c="C3", ls=line_style_sim
    )

    axs[0].set_xticklabels([])

    axs[0].set_yticks(axs[0].get_ylim())
    axs[0].set_yticklabels([])
    axs[0].set_xlim(0, 50)

    axs[1].set_yticks(axs[1].get_ylim())
    axs[1].set_yticklabels([])
    axs[1].set_xlim(0, 50)

    fig.tight_layout()

    fig.savefig(
        "../../results_sourcerer/figures/lv_simulations.pdf",
        transparent=True,
    )
    plt.show()
