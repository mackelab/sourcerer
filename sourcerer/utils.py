import os
import pickle
import random
import sys
from dataclasses import dataclass
from time import time_ns

import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf as OC

# Normalization utils


def standardize_array(arr, ax, set_mean=None, set_std=None, return_mean_std=False):
    if set_mean is None:
        arr_mean = np.mean(arr, axis=ax, keepdims=True)
    else:
        arr_mean = set_mean
    if set_std is None:
        arr_std = np.std(arr, axis=ax, keepdims=True)
    else:
        arr_std = set_std

    assert np.min(arr_std) > 0.0
    if return_mean_std:
        return (arr - arr_mean) / arr_std, arr_mean, arr_std
    else:
        return (arr - arr_mean) / arr_std


def scale_tensor(tensor, current_min, current_max, target_min, target_max):
    assert torch.all(current_min < current_max)
    assert torch.all(target_min < target_max)

    tensor = (tensor - current_min) / (current_max - current_min)
    tensor = tensor * (target_max - target_min) + target_min

    return tensor


# Scripting utils


def is_running_in_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

# Manages configs to allows the same file to be run as a notebook or called as a script.
def script_or_command_line_cfg(config_name, config_path, local_overrides, name):
    ipython = is_running_in_ipython()
    if ipython:
        cfg = get_cfg(config_name, config_path, local_overrides)
    elif name == "__main__" and not ipython:
        cfg = get_cfg(config_name, config_path, sys.argv[1:])
    else:
        raise ValueError("Not running in ipython or as __main__ from command line!")
    return cfg


def get_cfg(config_name, config_path, overrides):
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Saving utils


def save_pickle(obj, file_name, folder="", base_path=""):
    """
    Save object as pickle file.

    Args:
        obj: Object to be saved.
        file_path: Path to save file to.
    """
    assert file_name.endswith(".pkl")
    file_path = os.path.join(base_path, folder, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        print(f"WARNING: File {file_path} already exists. Save with timestamp.")
        file_path = file_path.replace(".pkl", "") + "_" + time_ns_string() + ".pkl"

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(file_name, folder="", base_path=""):
    file_path = os.path.join(base_path, folder, file_name)
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_numpy_csv(arr, file_name, folder="", base_path=""):
    assert file_name.endswith(".csv")
    assert arr.ndim <= 2
    file_path = os.path.join(base_path, folder, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        print(f"WARNING: File {file_path} already exists. Save with timestamp.")
        file_path = file_path.replace(".csv", "") + "_" + time_ns_string() + ".csv"

    np.savetxt(file_path, arr, delimiter=",")


def save_cfg_as_yaml(cfg, file_name, folder="", base_path=""):
    assert file_name.endswith(".yaml")
    file_path = os.path.join(base_path, folder, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        print(f"WARNING: File {file_path} already exists. Save with timestamp.")
        file_path = file_path.replace(".yaml", "") + "_" + time_ns_string() + ".yaml"

    with open(file_path, "w") as f:
        f.write(OC.to_yaml(cfg))


def save_fig(fig, file_name, folder="", base_path="", tight_layout=True):
    assert file_name.endswith(".pdf")
    file_path = os.path.join(base_path, folder, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        print(f"WARNING: File {file_path} already exists. Save with timestamp.")
        file_path = file_path.replace(".pdf", "") + "_" + time_ns_string() + ".pdf"

    if tight_layout:
        fig.tight_layout()

    fig.savefig(file_path, transparent=True)


def save_state_dict(model, file_name, folder="", base_path=""):
    assert file_name.endswith(".pt")
    file_path = os.path.join(base_path, folder, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        print(f"WARNING: File {file_path} already exists. Save with timestamp.")
        file_path = file_path.replace(".pt", "") + "_" + time_ns_string() + ".pt"

    torch.save(model.state_dict(), file_path)


def time_ns_string():
    return str(time_ns())

# Plotting rcparam management
@dataclass
class FigureLayout:
    width_in_pt: float
    width_grid: int
    base_font_size: int = 10
    scale_factor: float = 1.0

    # matplotlib uses inch
    def _get_grid_in_inch(self, w_grid, h_grid):
        pt_to_inch = 1 / 72
        assert w_grid <= self.width_grid
        return (
            (w_grid / self.width_grid) * self.width_in_pt * pt_to_inch,
            (h_grid / self.width_grid) * self.width_in_pt * pt_to_inch,
        )

    def get_rc(self, w_grid, h_grid):
        return {
            "figure.figsize": self._get_grid_in_inch(w_grid, h_grid),
            "font.size": self.base_font_size * self.scale_factor,
        }
