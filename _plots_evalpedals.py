import csv
import json
import random
from random import randint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from natsort import natsorted

random.seed(10)


def set_size(width, fraction=1, scale_height=0.5):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in * scale_height)
    return fig_dim


fig, axes = plt.subplots(
    2, 2, figsize=set_size(426.79137, fraction=0.9, scale_height=0.8)
)

with open("./pedal_logs/1_cond_ds1_pedal/summary/losses.json") as f:
    worse_ds = json.load(f)

with open("./pedal_logs/3_cond_ds1_pedal/summary/losses.json") as f:
    better_ds = json.load(f)

with open("./pedal_logs/1_cond_big_rat_pedal/summary/losses.json") as f:
    worse_rat = json.load(f)

with open("./pedal_logs/3_cond_big_rat_pedal/summary/losses.json") as f:
    better_rat = json.load(f)

worse_esr1 = worse_ds["per_cond_ESR"]
worse_stft1 = worse_ds["per_cond_STFT"]

better_esr1 = better_ds["per_cond_ESR"]
better_stft1 = better_ds["per_cond_STFT"]

worse_esr2 = worse_rat["per_cond_ESR"]
worse_stft2 = worse_rat["per_cond_STFT"]

better_esr2 = better_rat["per_cond_ESR"]
better_stft2 = better_rat["per_cond_STFT"]

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# plt.xticks([0, 2, 4, 6, 8, 10])
# plt.yticks([0.01, 0.025, 0.04])

# plt.scatter([0, 2, 4, 6, 8, 10], worse_stft[0: 6])
# plt.scatter([0, 2, 4, 6, 8, 10], worse_stft[6: 12])
# plt.scatter([0, 2, 4, 6, 8, 10], worse_stft[12:])

colors = ["#2e2d94", "#69b9dc"]

axes[0, 0].plot(
    [0, 0.2, 0.4, 0.6, 0.8, 1],
    worse_esr1,
    marker="o",
    markersize=5,
    linewidth=0.8,
    alpha=1,
    color=colors[0],
)

axes[0, 0].plot(
    [0, 0.2, 0.4, 0.6, 0.8, 1],
    better_esr1,
    marker="o",
    markersize=5,
    linewidth=0.8,
    alpha=1,
    color=colors[1],
)

axes[0, 1].plot(
    [0, 0.2, 0.4, 0.6, 0.8, 1],
    worse_esr2,
    marker="o",
    markersize=5,
    linewidth=0.8,
    alpha=1,
    color=colors[0],
)

axes[0, 1].plot(
    [0, 0.2, 0.4, 0.6, 0.8, 1],
    better_esr2,
    marker="o",
    markersize=5,
    linewidth=0.8,
    alpha=1,
    label="3 targets",
    color=colors[1],
)

axes[1, 0].plot(
    [0, 0.2, 0.4, 0.6, 0.8, 1],
    worse_stft1,
    marker="o",
    markersize=5,
    linewidth=0.8,
    alpha=1,
    label="1 target",
    color=colors[0],
)

axes[1, 0].plot(
    [0, 0.2, 0.4, 0.6, 0.8, 1],
    better_stft1,
    marker="o",
    markersize=5,
    linewidth=0.8,
    alpha=1,
    label="3 targets",
    color=colors[1],
)

axes[1, 1].plot(
    [0, 0.2, 0.4, 0.6, 0.8, 1],
    worse_stft2,
    marker="o",
    markersize=5,
    linewidth=0.8,
    alpha=1,
    color=colors[0],
)

axes[1, 1].plot(
    [0, 0.2, 0.4, 0.6, 0.8, 1],
    better_stft2,
    marker="o",
    markersize=5,
    linewidth=0.8,
    alpha=1,
    color=colors[1],
)

axes[0, 0].tick_params(axis="x", labelsize=9)
axes[0, 0].tick_params(axis="y", labelsize=9)
axes[0, 0].set_xlim([-0.05, 1.05])
axes[0, 0].set_ylim([-0.005, 0.035])
axes[0, 0].set_yticks([0.0, 0.01, 0.02, 0.03])
axes[0, 0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axes[0, 0].set_ylabel(r"$\mathcal{E}_{\mathrm{ESR}}$", fontsize=11)
axes[0, 0].set_xticklabels([])
axes[0, 0].grid("both", alpha=0.5)

axes[0, 1].tick_params(axis="x", labelsize=9)
axes[0, 1].tick_params(axis="y", labelsize=9)
axes[0, 1].set_xlim([-0.05, 1.05])
axes[0, 1].set_ylim([-0.005, 0.035])
axes[0, 1].set_yticks([0.0, 0.01, 0.02, 0.03])
axes[0, 1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axes[0, 1].set_yticklabels([])
axes[0, 1].set_xticklabels([])
axes[0, 1].grid("both", alpha=0.5)
axes[1, 0].legend(fontsize=9, loc="lower right")


# axes[0, 2].tick_params(axis="x", labelsize=9)
# axes[0, 2].tick_params(axis="y", labelsize=9)
# axes[0, 2].set_xlim([-0.05, 1.05])
# axes[0, 2].set_ylim([-0.005, 0.055])
# axes[0, 2].set_yticks([0.0, 0.025, 0.05])
# axes[0, 2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# axes[0, 2].set_yticklabels([])
# axes[0, 2].set_xticklabels([])
# axes[0, 2].grid("both", alpha=0.5)

#  STFT
axes[1, 0].tick_params(axis="x", labelsize=9)
axes[1, 0].tick_params(axis="y", labelsize=9)
axes[1, 0].set_xlim([-0.05, 1.05])
axes[1, 0].set_ylim([0.0, 2])
# axes[1, 0].set_yticks([0.0, 0.025, 0.05])
axes[1, 0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axes[1, 0].set_xlabel(r"$c_{\mathrm{t}}$ (DS1)", fontsize=10)
axes[1, 0].set_ylabel(r"$\mathcal{E}_{\mathrm{STFT}}$", fontsize=10)
axes[1, 0].grid("both", alpha=0.5)

axes[1, 1].tick_params(axis="x", labelsize=9)
axes[1, 1].tick_params(axis="y", labelsize=9)
axes[1, 1].set_xlim([-0.05, 1.05])
axes[1, 1].set_ylim([0.0, 2])
# axes[1, 1].set_yticks([0.0, 0.025, 0.05])
axes[1, 1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axes[1, 1].set_xlabel(r"$c_{\mathrm{t}}$ (RAT)", fontsize=10)
axes[1, 1].grid("both", alpha=0.5)
# axes[1, 1].legend(fontsize=5, loc='upper center')


# axes[1, 2].tick_params(axis="x", labelsize=9)
# axes[1, 2].tick_params(axis="y", labelsize=9)
# axes[1, 2].set_xlim([-0.05, 1.05])
# axes[1, 2].set_ylim([0.5, 0.9])
# # axes[1, 2].set_yticks([0.0, 0.025, 0.05])
# axes[1, 2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
# axes[1, 2].set_yticklabels([])
# axes[1, 2].set_xlabel(r"$c_{\mathrm{t}}$", fontsize=11)
# axes[1, 2].grid("both", alpha=0.5)

plt.tight_layout(pad=0.2)
plt.savefig("./dplots/pedals_eval.pdf")

plt.show()
