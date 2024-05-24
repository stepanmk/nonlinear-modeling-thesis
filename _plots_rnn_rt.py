from matplotlib import pyplot as plt

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
    ncols=1,
    sharex=True,
    sharey=True,
    figsize=set_size(tw / 1.2, scale_height=0.9),
)


hs = [1, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192]

lstm = [
    0.00534156,
    0.0141728,
    0.0271456,
    0.0458755,
    0.0756835,
    0.0904434,
    0.139653,
    0.163071,
    0.234514,
    0.274696,
    0.333946,
    0.396803,
    0.455504,
]


gru = [
    0.00632091,
    0.0141149,
    0.0236248,
    0.0387378,
    0.0536908,
    0.0736579,
    0.0925126,
    0.12305,
    0.160071,
    0.207718,
    0.248903,
    0.307482,
    0.357832,
]


axes.plot(hs, lstm, "k", linewidth=0.8, marker="+", markersize=6)
axes.plot(hs, gru, "--k", linewidth=0.8, marker="x", markersize=5)
axes.set_xlim([1, 192])
axes.set_ylim([0, 0.5])
axes.set_xticks(
    [1, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192],
    ["1", "", "32", "", "64", "", "96", "", "128", "", "160", "", "192"],
)
axes.grid("both", alpha=0.5)

plt.legend(["LSTM", "GRU"], loc="upper left")
plt.xlabel("Hidden state size", fontsize=10)
plt.ylabel("Compute time [s]", fontsize=10)
plt.xticks(hs)
plt.tight_layout()
plt.savefig("./dplots/rnn_rt.pdf")
plt.show()
