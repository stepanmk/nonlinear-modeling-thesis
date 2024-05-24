import json

from matplotlib import pyplot as plt

with open("./runs1/jvm_no_clip_lstm_16/summary/losses.json") as f:
    hg = json.load(f)
with open("./_unused_logs/jvm_lowgain_lstm_16/summary/losses.json") as f:
    lg = json.load(f)


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
    ncols=2,
    sharex=True,
    sharey=True,
    figsize=set_size(tw, scale_height=0.6),
)

axes[0].plot(lg["grad_norms"][0:2000], "k", linewidth=0.8)
axes[0].set_xlim([0, 2000])
axes[0].set_ylim([0, 100])
axes[0].grid("both", alpha=0.5)
axes[0].set_title("Gain = 2", fontsize=10)

axes[1].plot(hg["grad_norms"][0:2000], "k", linewidth=0.8)
axes[1].set_xlim([0, 2000])
axes[1].set_ylim([0, 100])
axes[1].grid("both", alpha=0.5)
axes[1].set_title("Gain = 8", fontsize=10)

fig.supxlabel("Iterations", fontsize=10, x=0.5, y=0.1)
fig.supylabel(
    "$||\\mathrm{g}||_2$",
    fontsize=10,
    x=0.05,
    y=0.5,
)
plt.tight_layout()
plt.savefig("./dplots/grad_compare.pdf")
plt.show()
