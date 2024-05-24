import os

import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from natsort import natsorted


def set_size(width, fraction=1, scale_height=0.5):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in * scale_height)
    return fig_dim


tw = 426.79137


wavenet_path = "wavenet_times/wavenetva/"
path_static = "wavenet_times/proposed_static/"
path_dynamic = "wavenet_times/proposed_dynamic/"

wavenet_sizes = natsorted(os.listdir(wavenet_path))
wavenet_sizes = wavenet_sizes[-7:] + wavenet_sizes[0:14]
static_sizes = natsorted(os.listdir(path_static))
dynamic_sizes = natsorted(os.listdir(path_dynamic))

wave_1 = []
wave_2 = []
wave_3 = []

dynamic_1 = []
dynamic_2 = []
dynamic_3 = []

static_1 = []
static_2 = []
static_3 = []

for i in range(len(wavenet_sizes)):
    f_wavenet = wavenet_path + wavenet_sizes[i]
    f_static = path_static + static_sizes[i]
    f_dynamic = path_dynamic + dynamic_sizes[i]
    with open(f_wavenet, "r") as f:
        vals_str = f.read().split("\n")[:-1]
        vals = list(map(float, vals_str))
        result = sum(vals) / len(vals)
        if "WaveNet_" in f_wavenet:
            wave_1.append(result)
        elif "WaveNet2" in f_wavenet:
            wave_2.append(result)
        elif "WaveNet3" in f_wavenet:
            wave_3.append(result)
    with open(f_static, "r") as f:
        vals_str = f.read().split("\n")[:-1]
        vals = list(map(float, vals_str))
        result = sum(vals) / len(vals)
        if "WaveNet1" in f_static:
            static_1.append(result)
        elif "WaveNet2" in f_static:
            static_2.append(result)
        elif "WaveNet3" in f_static:
            static_3.append(result)
    with open(f_dynamic, "r") as f:
        vals_str = f.read().split("\n")[:-1]
        vals = list(map(float, vals_str))
        result = sum(vals) / len(vals)
        if "WaveNet1" in f_dynamic:
            dynamic_1.append(result)
        elif "WaveNet2" in f_dynamic:
            dynamic_2.append(result)
        elif "WaveNet3" in f_dynamic:
            dynamic_3.append(result)

d_rates = [1, 2, 3, 4, 5, 6, 7]

plt.style.use("tableau-colorblind10")

fig, ax = plt.subplots(figsize=set_size(tw, scale_height=1))
ax.plot(d_rates, 1 / np.array(wave_1), "--^", markersize=5, linewidth=0.8)
ax.plot(d_rates, 1 / np.array(wave_2), "--x", markersize=5, linewidth=0.8)
ax.plot(d_rates, 1 / np.array(wave_3), "--o", markersize=5, linewidth=0.8)

ax.plot(d_rates, 1 / np.array(dynamic_1), "-^", markersize=5, linewidth=0.8)
ax.plot(d_rates, 1 / np.array(dynamic_2), "-x", markersize=5, linewidth=0.8)
ax.plot(d_rates, 1 / np.array(dynamic_3), "-o", markersize=5, linewidth=0.8)

ax.set_ylim([0.04, 20])
ax.set_xticks(d_rates)
ax.set_yscale("log")

ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

x = [1, 2, 4, 8, 16, 32, 64]
ax.set_xticklabels(x)

ax.set_ylabel("Processing speed (× RT)")
ax.set_xlabel("Convolution channels")

ax.legend(
    [
        "Reference (10 layers)",
        "Reference (18 layers)",
        "Reference (24 layers)",
        "Proposed dynamic (10 layers)",
        "Proposed dynamic (18 layers)",
        "Proposed dynamic (24 layers)",
    ],
    fontsize=9.5,
    loc=3,
)

fig.tight_layout()

plt.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)

plt.grid("both")
plt.savefig("wavenet_dynamic.pdf")
plt.show()


rnn_path = "./rnn_times"
