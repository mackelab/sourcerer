# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

from sourcerer.real_nvp import Sampler
from sourcerer.simulators import SIRSimulator
from sourcerer.utils import FigureLayout, set_seed

set_seed(42)


matplotlibrc_path = "../../matplotlibrc"
FL_SIR = FigureLayout(234, 100, 8)
with plt.rc_context(rc=FL_SIR.get_rc(77, 45), fname=matplotlibrc_path):
    fig, ax = plt.subplots()

# NOTE: fin_lambda is 0.25
samp = Sampler(xdim=2, input_noise_dim=2, hidden_layer_dim=100)
samp.load_state_dict(torch.load("TODO_path_to_estimated_SIR_source_state_dict.pt"))

simu = SIRSimulator()

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

FL_SIR = FigureLayout(234, 100, 8)

num_samps = 50
rand_ids = np.random.choice(10000, size=num_samps, replace=False)

# %%

line_style_obs = "-"
line_style_sim = "-"

with plt.rc_context(rc=FL_SIR.get_rc(77, 45), fname=matplotlibrc_path):
    fig, ax = plt.subplots()

    gt_simulator_two_np = simu_pf
    simu_estimated_pf_np = samp_pf

    for i in rand_ids[:30]:
        ax.plot(gt_simulator_two_np[i], c="black", alpha=0.2, linewidth=0.5)
        ax.plot(simu_estimated_pf_np[i], c="C3", alpha=0.2, linewidth=0.5)
    for i in [5, 7, 9]:
        ax.plot(
            np.quantile(gt_simulator_two_np, 0.1 * i, axis=0),
            c="black",
        )
        ax.plot(
            np.quantile(simu_estimated_pf_np, 0.1 * i, axis=0),
            c="C3",
            ls=line_style_sim,
        )

    ax.set_ylim(0.0, 0.7)
    ax.set_yticks([0.0, 0.7])
    ax.set_yticklabels([])
    ax.set_xlim(0, 50)

    fig.tight_layout()
    fig.savefig(
        "../../results_sourcerer/figures/sir_simulations.pdf",
        transparent=True,
    )
    plt.show()
