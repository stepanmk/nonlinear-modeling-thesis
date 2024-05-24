import json

import numpy as np
from matplotlib import pyplot as plt

with open("./runs1/jvm_autoclip_lstm_16/summary/losses.json") as f:
    hg = json.load(f)
# with open("./_unused_logs/jvm_lowgain_lstm_16/summary/losses.json") as f:
#     lg = json.load(f)


tw = 426.79137


def set_size(width, fraction=1, scale_height=0.6):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in * scale_height)
    return fig_dim


fig, axes = plt.subplots(
    num=1,
    nrows=1,
    ncols=3,
    sharex=True,
    sharey=True,
    figsize=set_size(tw, scale_height=0.6),
)

axes[0].plot(hg["grad_norms"][0:50000], "k", linewidth=0.8)
axes[0].set_xticks([0, 25000, 50000], ["0", "25k", "50k"])
axes[0].set_yticks([0, 5, 10], ["0", "5", "10"])
axes[0].set_xlim([0, 50000])
axes[0].set_ylim([0, 10])
axes[0].grid("both", alpha=0.5)
axes[0].set_title("No clipping", fontsize=10)

axes[1].plot(np.array(hg["grad_norms"][0:50000]).clip(max=1), "k", linewidth=0.8)
axes[1].set_xticks([0, 25000, 50000], ["0", "25k", "50k"])
axes[1].set_xlim([0, 50000])
axes[1].set_ylim([0, 10])
axes[1].grid("both", alpha=0.5)
axes[1].set_title("$v_{\\mathrm{clip}}$ = 1", fontsize=10)

axes[2].set_xticks([0, 25000, 50000], ["0", "25k", "50k"])
axes[2].plot(hg["clipped_grad_norms"][0:50000], "k", linewidth=0.8)
axes[2].set_xlim([0, 50000])
axes[2].set_ylim([0, 10])
axes[2].grid("both", alpha=0.5)
axes[2].set_title("$q$ = 0.1", fontsize=10)

fig.supxlabel("Iterations", fontsize=10, x=0.5, y=0.1)
fig.supylabel(
    "$||\\mathrm{g}||_2$",
    fontsize=10,
    x=0.05,
    y=0.5,
)
plt.tight_layout(pad=0.9)
plt.savefig("./dplots/clip_compare.pdf")
plt.show()
