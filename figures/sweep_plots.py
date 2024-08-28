# %%
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from sourcerer.utils import FigureLayout

matplotlibrc_path = "../matplotlibrc"

# %%
FL = FigureLayout(486, 100, 8)

with plt.rc_context(rc=FL.get_rc(50, 50), fname=matplotlibrc_path):
    fig, axs = plt.subplots(3, 2)

    base_path = "TODO_path_sweep_folder_benchmark"
    meta_folders = [
        "tm/tm_ws_diff_",
        "tm/tm_ws_surro_",
        "ik/ik_ws_diff_",
        "ik/ik_ws_surro_",
        "slcp/slcp_ws_diff_",
        "slcp/slcp_ws_surro_",
    ]

    for pp in range(len(meta_folders)):
        folders = [meta_folders[pp] + str(i) for i in range(5)]
        print(meta_folders[pp])

        tag = "run"

        num_runs = 15
        kole = np.zeros((len(folders), num_runs))
        c2st = np.zeros((len(folders), num_runs))
        swd = np.zeros((len(folders), num_runs))
        for ff, folder in enumerate(folders):
            for i in range(num_runs):
                kole[ff, i] = np.loadtxt(
                    os.path.join(
                        base_path, folder, f"{tag}_{i}_estimated_source_kole.csv"
                    ),
                )
                c2st[ff, i] = np.loadtxt(
                    os.path.join(base_path, folder, f"{tag}_{i}_simu_c2st.csv"),
                )
                swd[ff, i] = np.mean(
                    np.loadtxt(
                        os.path.join(
                            base_path, folder, f"{tag}_{i}_simu_pf_swds.csv"
                        ),
                    )
                )

        kole_mean = np.mean(kole, axis=0)
        c2st_mean = np.mean(c2st, axis=0)
        swd_mean = np.mean(swd, axis=0)

        # lambdas = np.arange(num_runs)
        lambdas = 1 / (np.sqrt(2) ** np.arange(num_runs))
        lambdas = lambdas[::-1]
        lambdas[0] *= 0.5
        print(lambdas[0])
        print(lambdas[11])

        print(pp // 2, pp % 2)
        ax1 = axs[pp // 2, pp % 2]

        # ax1.set_title(meta_folders[pp])
        ax1.spines["right"].set_visible(True)
        ax1.spines["top"].set_visible(True)
        ax1.errorbar(
            lambdas[1:],
            kole_mean[1:],
            yerr=np.std(kole[:, 1:], axis=0),
            fmt="-o",
            color="C0",
        )
        ax1.errorbar(
            lambdas[0],
            kole_mean[0],
            yerr=np.std(kole[:, 0], axis=0),
            fmt="o",
            color="C0",
        )
        if meta_folders[pp][:2] == "tm":
            ax1.set_ylim(0.8, 1.8)
            ax1.set_yticks([0.8, 1.8])
        elif meta_folders[pp][:2] == "ik":
            ax1.set_ylim(1.0, 5.0)
            ax1.set_yticks([1.0, 5.0])
        elif meta_folders[pp][:2] == "sl":
            ax1.set_ylim(6.0, 12.0)
            ax1.set_yticks([6.0, 12.0])

        ax2 = ax1.twinx()
        ax2.errorbar(
            lambdas[1:],
            c2st_mean[1:],
            yerr=np.std(c2st[:, 1:], axis=0),
            fmt="-o",
            color="C3",
        )
        ax2.errorbar(
            lambdas[0],
            c2st_mean[0],
            yerr=np.std(c2st[:, 0], axis=0),
            fmt="o",
            color="C3",
        )
        ax2.vlines(lambdas[11], -10, 10, color="darkgrey", zorder=10)
        ax2.set_ylim(0.48, 1)
        ax2.set_yticks([0.5, 1.0])
        ax1.set_xticks(lambdas)
        ax1.set_xscale("log")
        ax1.set_xticks(
            [1.0 - i * 0.125 for i in range(1, 7)]
            + [0.125 - i * 0.015625 for i in range(1, 7)],
            minor=True,
        )
        ax1.set_xticklabels([], minor=True)
        ax1.set_xticks([1, 0.125, 0.015625, lambdas[0]])
        ax1.set_xticklabels(["1.0", "0.125", "0.015", "NA"])
        if pp % 2 == 0:
            ax2.set_yticklabels([])
        if pp % 2 == 1:
            ax1.set_yticklabels([])

    # turn of xticks at 0, 0
    axs[0, 0].set_xticklabels([])
    axs[0, 1].set_xticklabels([])
    axs[1, 0].set_xticklabels([])
    axs[1, 1].set_xticklabels([])

    fig.tight_layout()
    fig.savefig(
        "../results_sourcerer/benchmark_sweeps.pdf",
        transparent=True,
    )
    plt.show()

# %%
FL_SIR = FigureLayout(234, 100, 8)

with plt.rc_context(rc=FL_SIR.get_rc(94, 40), fname=matplotlibrc_path):
    fig, ax = plt.subplots()

    ground_truth_swd = np.array(
        [
            0.00224643,
            0.00187718,
            0.00263804,
            0.00231465,
            0.00219216
        ]
    )

    base_path = "TODO_path_sweep_folder_SIR"
    folders = [f"sir_ws_diff_{i}" for i in range(5)]
    tag = "run"

    num_runs = 15
    kole = np.zeros((len(folders), num_runs))
    c2st = np.zeros((len(folders), num_runs))
    swd = np.zeros((len(folders), num_runs))

    for ff, folder in enumerate(folders):
        for i in range(num_runs):
            kole[ff, i] = np.loadtxt(
                os.path.join(base_path, folder, f"{tag}_{i}_estimated_source_kole.csv"),
            )
            c2st[ff, i] = np.loadtxt(
                os.path.join(base_path, folder, f"{tag}_{i}_simu_c2st.csv"),
            )
            swd[ff, i] = np.mean(
                np.loadtxt(
                    os.path.join(base_path, folder, f"{tag}_{i}_simu_pf_swds.csv"),
                )
            )

    kole_mean = np.mean(kole, axis=0)
    c2st_mean = np.mean(c2st, axis=0)
    swd_mean = np.median(swd, axis=0)

    lambdas = 1 / (np.sqrt(2) ** np.arange(num_runs))
    lambdas = lambdas[::-1]
    lambdas[0] *= 0.5
    print(lambdas[0])
    print(lambdas[10])

    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.errorbar(
        lambdas[1:],
        kole_mean[1:],
        yerr=np.std(kole[:, 1:], axis=0),
        fmt="-o",
        color="C0",
    )
    ax.errorbar(
        lambdas[0],
        kole_mean[0],
        yerr=np.std(kole[:, 0], axis=0),
        fmt="o",
        color="C0",
    )
    ax.set_ylim([-3.5, 3])
    ax.set_yticks([-3.5, 3])

    ax2 = ax.twinx()
    ax2.axhline(np.mean(ground_truth_swd), color="black", linestyle="dashdot")
    ax2.errorbar(
        lambdas[1:],
        swd_mean[1:],
        yerr=np.std(swd[:, 1:], axis=0),
        fmt="-o",
        color="C3",
    )
    ax2.errorbar(
        lambdas[0],
        swd_mean[0],
        yerr=np.std(swd[:, 0], axis=0),
        fmt="o",
        color="C3",
    )
    ax2.axvline(lambdas[10], 0, 100, color="darkgrey", zorder=10)
    ax2.set_yscale("log")

    ax.set_xscale("log")
    ax.set_xticks(
        [1.0 - i * 0.125 for i in range(1, 7)]
        + [0.125 - i * 0.015625 for i in range(1, 7)],
        minor=True,
    )
    ax.set_xticklabels([], minor=True)
    ax.set_xticks([1, 0.125, 0.015625, lambdas[0]])
    ax.set_xticklabels(["1.0", "0.125", "0.015", "NA"])

    line = mlines.Line2D(
        [], [], color="black", linestyle="dashdot", label="ground truth SWD"
    )
    ax.legend(handles=[line])
    fig.tight_layout()
    fig.savefig(
        "../results_sourcerer/sir_curves.pdf",
        transparent=True,
    )

    plt.show()

# %%
FL_LV = FigureLayout(234, 100, 8)


with plt.rc_context(rc=FL_LV.get_rc(94, 40), fname=matplotlibrc_path):
    fig, ax = plt.subplots()

    ground_truth_swd = np.array(
        [
            0.0081546,
            0.00827313,
            0.00856958,
            0.00799153,
            0.00838649
        ]
    )

    base_path = "TODO_path_sweep_folder_LV"
    folders = [f"lv_ws_diff_{i}" for i in [0, 1, 3, 4]]
    tag = "run"

    num_runs = 15
    kole = np.zeros((len(folders), num_runs))
    c2st = np.zeros((len(folders), num_runs))
    swd = np.zeros((len(folders), num_runs))

    for ff, folder in enumerate(folders):
        for i in range(num_runs):
            kole[ff, i] = np.loadtxt(
                os.path.join(base_path, folder, f"{tag}_{i}_estimated_source_kole.csv"),
            )
            c2st[ff, i] = np.loadtxt(
                os.path.join(base_path, folder, f"{tag}_{i}_simu_c2st.csv"),
            )
            swd[ff, i] = np.mean(
                np.loadtxt(
                    os.path.join(base_path, folder, f"{tag}_{i}_simu_pf_swds.csv"),
                )
            )

    kole_mean = np.mean(kole, axis=0)
    c2st_mean = np.mean(c2st, axis=0)
    swd_mean = np.median(swd, axis=0)

    lambdas = 1 / (np.sqrt(2) ** np.arange(num_runs))
    lambdas = lambdas[::-1]
    lambdas[0] *= 0.5
    print(lambdas[0])
    print(lambdas[12])

    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.errorbar(
        lambdas[1:],
        kole_mean[1:],
        yerr=np.std(kole[:, 1:], axis=0),
        fmt="-o",
        color="C0",
    )
    ax.errorbar(
        lambdas[0],
        kole_mean[0],
        yerr=np.std(kole[:, 0], axis=0),
        fmt="o",
        color="C0",
    )
    ax.set_ylim([-2, 2])
    ax.set_yticks([-2, 2])

    ax2 = ax.twinx()
    ax2.axhline(np.mean(ground_truth_swd), color="black", linestyle="dashdot")
    ax2.errorbar(
        lambdas[1:],
        swd_mean[1:],
        yerr=np.std(swd[:, 1:], axis=0),
        fmt="-o",
        color="C3",
    )
    ax2.errorbar(
        lambdas[0],
        swd_mean[0],
        yerr=np.std(swd[:, 0], axis=0),
        fmt="o",
        color="C3",
    )
    ax2.axvline(lambdas[12], 0, 100, color="darkgrey", zorder=10)
    ax2.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(
        [1.0 - i * 0.125 for i in range(1, 7)]
        + [0.125 - i * 0.015625 for i in range(1, 7)],
        minor=True,
    )
    ax.set_xticklabels([], minor=True)
    ax.set_xticks([1, 0.125, 0.015625, lambdas[0]])
    ax.set_xticklabels(["1.0", "0.125", "0.015", "NA"])

    line = mlines.Line2D(
        [], [], color="black", linestyle="dashdot", label="ground truth SWD"
    )
    ax.legend(handles=[line])

    fig.tight_layout()
    fig.savefig("../results_sourcerer/lv_curves.pdf", transparent=True)

    plt.show()


# %%
# NOTE: fin_lambda is 0.25
FL_HH = FigureLayout(486, 100, 8)

with plt.rc_context(rc=FL_HH.get_rc(45, 15), fname=matplotlibrc_path):
    fig, ax = plt.subplots()

    base_path = "TODO_path_sweep_folder_HH"
    folders = [f"hh_ws_surro_{i}" for i in range(5)]
    tag = "run"

    num_runs = 15
    kole = np.zeros((len(folders), num_runs))
    c2st = np.zeros((len(folders), num_runs))
    swd = np.zeros((len(folders), num_runs))

    for ff, folder in enumerate(folders):
        for i in range(num_runs):
            kole[ff, i] = np.loadtxt(
                os.path.join(base_path, folder, f"{tag}_{i}_estimated_source_kole.csv"),
            )
            c2st[ff, i] = np.loadtxt(
                os.path.join(base_path, folder, f"{tag}_{i}_simu_c2st.csv"),
            )
            swd[ff, i] = np.mean(
                np.loadtxt(
                    os.path.join(base_path, folder, f"{tag}_{i}_simu_pf_swd.csv"),
                )
            )

    kole_mean = np.mean(kole, axis=0)
    c2st_mean = np.mean(c2st, axis=0)
    swd_mean = np.median(swd, axis=0)

    lambdas = 1 / (np.sqrt(2) ** np.arange(num_runs))
    lambdas = lambdas[::-1]
    lambdas[0] *= 0.5
    print(lambdas[0])
    print(lambdas[10])

    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.errorbar(
        lambdas[1:],
        kole_mean[1:],
        yerr=np.std(kole[:, 1:], axis=0),
        fmt="-o",
        color="C0",
    )
    ax.errorbar(
        lambdas[0],
        kole_mean[0],
        yerr=np.std(kole[:, 0], axis=0),
        fmt="o",
        color="C0",
    )
    ax.set_ylim([-20, 15])
    ax.set_yticks([-20, 15])

    ax2 = ax.twinx()
    ax2.errorbar(
        lambdas[1:],
        swd_mean[1:],
        yerr=np.std(swd[:, 1:], axis=0),
        fmt="-o",
        color="C3",
    )
    ax2.errorbar(
        lambdas[0],
        swd_mean[0],
        yerr=np.std(swd[:, 0], axis=0),
        fmt="o",
        color="C3",
    )
    ax2.axvline(lambdas[10], 0, 100, color="darkgrey", zorder=10)
    ax2.set_yscale("log")

    ax.set_xscale("log")
    ax.set_xticks(
        [1.0 - i * 0.125 for i in range(1, 7)]
        + [0.125 - i * 0.015625 for i in range(1, 7)],
        minor=True,
    )
    ax.set_xticklabels([], minor=True)
    ax.set_xticks([1, 0.125, 0.015625, lambdas[0]])
    ax.set_xticklabels(["1.0", "0.125", "0.015", "NA"])

    fig.tight_layout()
    fig.savefig(
        "../results_sourcerer/hh_curves_swd.pdf",
        transparent=True,
    )

    plt.show()


with plt.rc_context(rc=FL_HH.get_rc(44, 15), fname=matplotlibrc_path):
    fig, ax = plt.subplots()

    base_path = "TODO_path_sweep_folder_HH"
    folders = [f"hh_ws_surro_{i}" for i in range(5)]
    tag = "run"

    num_runs = 15
    kole = np.zeros((len(folders), num_runs))
    c2st = np.zeros((len(folders), num_runs))
    swd = np.zeros((len(folders), num_runs))

    for ff, folder in enumerate(folders):
        for i in range(num_runs):
            kole[ff, i] = np.loadtxt(
                os.path.join(base_path, folder, f"{tag}_{i}_estimated_source_kole.csv"),
            )
            c2st[ff, i] = np.loadtxt(
                os.path.join(base_path, folder, f"{tag}_{i}_simu_c2st.csv"),
            )
            swd[ff, i] = np.mean(
                np.loadtxt(
                    os.path.join(base_path, folder, f"{tag}_{i}_simu_pf_swd.csv"),
                )
            )

    kole_mean = np.mean(kole, axis=0)
    c2st_mean = np.mean(c2st, axis=0)
    swd_mean = np.median(swd, axis=0)

    lambdas = 1 / (np.sqrt(2) ** np.arange(num_runs))
    lambdas = lambdas[::-1]
    lambdas[0] *= 0.5

    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.errorbar(
        lambdas[1:],
        kole_mean[1:],
        yerr=np.std(kole[:, 1:], axis=0),
        fmt="-o",
        color="C0",
    )
    ax.errorbar(
        lambdas[0],
        kole_mean[0],
        yerr=np.std(kole[:, 0], axis=0),
        fmt="o",
        color="C0",
    )
    ax.set_ylim([-20, 15])
    ax.set_yticks([-20, 15])

    ax2 = ax.twinx()
    ax2.errorbar(
        lambdas[1:],
        c2st_mean[1:],
        yerr=np.std(c2st[:, 1:], axis=0),
        fmt="-o",
        color="C3",
    )
    ax2.errorbar(
        lambdas[0],
        c2st_mean[0],
        yerr=np.std(c2st[:, 0], axis=0),
        fmt="o",
        color="C3",
    )
    ax2.axvline(lambdas[10], -1, 1, color="darkgrey", zorder=10)
    ax2.set_ylim([0.5, 1])
    ax2.set_yticks([0.5, 1.0])

    ax.set_xscale("log")
    ax.set_xticks(
        [1.0 - i * 0.125 for i in range(1, 7)]
        + [0.125 - i * 0.015625 for i in range(1, 7)],
        minor=True,
    )
    ax.set_xticklabels([], minor=True)
    ax.set_xticks([1, 0.125, 0.015625, lambdas[0]])
    ax.set_xticklabels(["1.0", "0.125", "0.015", "NA"])

    fig.tight_layout()
    fig.savefig(
        "../results_sourcerer/path_hh_curves_c2st.pdf",
        transparent=True,
    )

    plt.show()
