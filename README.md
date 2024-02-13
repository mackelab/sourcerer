# Sourcerer

This is the repository for the preprint [**Sourcerer: Sample-based Maximum Entropy Source Distribution Estimation**](https://arxiv.org/abs/2402.07808).

## Initial setup

Install all dependencies with `pip install -e .`.

## Running Two Moons, IK, SLCP or SIR, Lotka-Volterra:

`hydra` is used for configuration management and command line interface.

You can run all benchmark either via the commandline or interactively (with VSCode cells or jupyter notebooks) using `benchmark_simulator_script.py`.

To run the interactively, overwrite the `local_overrides` list with your desired configuration. 

Alternatively, launch the script in the command line `python3 benchmark_simulator_script.py simulator=two_moons` with your desired configuration (as an example, the `two_moons` task is selected here).

By default, the Inverse Kinematics task with a differentiable simulator will be performed and results will be saved in `results_sourcerer`.


## Running the Hodgkin-Huxley experiment

To run the Hodgkin-Huxley experiment, the public dataset from Scala et al. (2020) is required. The dataset can be downloaded at `https://dandiarchive.org/dandiset/000008/draft`.

Alternatively, a preprocessed dataset, together with a set of simulations to train the surrogate is publically available at `https://github.com/berenslab/hh_sbi`.

To train the surrogate, use the `hh_sims_and_stats.py` script.
To perform source estimation after training the surrogate, use the `hh_script.py` script.
Both scripts can be run interactively using VSCode cells.


## Figures
Code for the figures can be found in the `figures` folder.
To reproduce the figures, it is required to first run the experiments and update the paths to point to the result files.

## References
The code is mainly based on the `pytorch` library. Configuration management is performed with `hydra`.

Parts of the code are based on code from the following publically available repositories:
- `https://github.com/berenslab/hh_sbi`
- `https://github.com/MaximeVandegar/NEB`



