import json

from matplotlib import pyplot as plt

with open("./runs1/jvm_no_clip_lstm_16/summary/losses.json") as f:
    hg = json.load(f)
with open("./runs1/jvm_autoclip_lstm_16/summary/losses.json") as f:
    lg = json.load(f)


tw = 426.79137


def set_size(width, fraction=1, scale_height=0.8):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in * scale_height)
    return fig_dim


fig, axes = plt.subplots(
    num=1,
    nrows=2,
    ncols=2,
    sharex="row",
    sharey="row",
    figsize=set_size(tw, scale_height=0.85),
)
import numpy as np

axes[0, 0].plot(hg["grad_norms"][0:40000], "k", linewidth=0.8)
axes[0, 0].set_xlim([0, 40000])
axes[0, 0].set_ylim([0, 100])
axes[0, 0].set_xticks([0, 20000, 40000], ["0", "20k", "40k"])
axes[0, 0].set_ylabel("$||\\mathrm{g}||_2$")
axes[0, 0].grid("both", alpha=0.5)
axes[0, 0].set_xlabel("Iterations")

axes[0, 1].plot(lg["grad_norms"][0:40000], "k", linewidth=0.8)
axes[0, 1].set_xlim([0, 40000])
axes[0, 1].set_ylim([0, 100])
axes[0, 1].grid("both", alpha=0.5)
axes[0, 1].set_xlabel("Iterations")

axes[1, 0].plot(hg["train_ep_losses"], "k", linewidth=0.8)
axes[1, 0].plot(np.array(hg["ESR"]["val"]).repeat(2), "--k", linewidth=0.8)
axes[1, 0].set_xlim([0, 300])
axes[1, 0].set_ylim([0, 0.6])
axes[1, 0].set_ylabel("$\\mathcal{L}_{\\mathrm{ESR}}, \\mathcal{E}_{\\mathrm{ESR}}$")
axes[1, 0].grid("both", alpha=0.5)
axes[1, 0].set_xlabel("Epochs")

axes[1, 1].plot(lg["train_ep_losses"], "k", linewidth=0.8)
axes[1, 1].plot(np.array(lg["ESR"]["val"]).repeat(2), "--k", linewidth=0.8)
axes[1, 1].set_xlim([0, 300])
axes[1, 1].set_ylim([0, 0.6])
axes[1, 1].set_xlabel("Epochs")

axes[1, 1].grid("both", alpha=0.5)
axes[1, 1].legend(
    [
        "$\\mathcal{L}_{\\mathrm{ESR}}$ (train.)",
        "$\\mathcal{E}_{\\mathrm{ESR}}$ (val.)",
    ],
    loc="upper right",
)


# fig.supxlabel("Iterations", fontsize=10, x=0.5, y=0.1)
# fig.supylabel("Gradient $l^2$-norm", fontsize=10, x=0.05, y=0.5)
plt.tight_layout(pad=0.2)
plt.savefig("./dplots/exploded.pdf")
plt.show()
