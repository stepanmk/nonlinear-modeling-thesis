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


txt_path = "conv_compare/"

kernel_sizes = natsorted(os.listdir(txt_path + "kernel"))

wave_d = []
rt_d = []
fast_d = []
fast_d2 = []

for d in kernel_sizes:
    f_name = txt_path + "kernel/" + d
    with open(f_name, "r") as f:
        vals_str = f.read().split("\n")[:-1]
        vals = list(map(float, vals_str))
        result = sum(vals) / len(vals)
        if "RTNeural" in f_name:
            rt_d.append(result)
        elif "WaveNetVA" in f_name:
            wave_d.append(result)
        elif "FastConvD" in f_name:
            fast_d.append(result)
        elif "FastConvV2" in f_name:
            fast_d2.append(result)

d_rates = [1, 2, 3, 4, 5, 6, 7]

plt.style.use("tableau-colorblind10")

fig, ax = plt.subplots(figsize=set_size(tw, scale_height=0.82))
ax.plot(d_rates, 1 / np.array(wave_d), "-^", markersize=5, linewidth=0.8)
ax.plot(d_rates, 1 / np.array(rt_d), "-s", markersize=5, linewidth=0.8)
ax.plot(d_rates, 1 / np.array(fast_d), "-x", markersize=5, linewidth=0.8)
ax.plot(d_rates, 1 / np.array(fast_d2), "-o", markersize=5, linewidth=0.8)
ax.set_ylim([3, 10000])
ax.set_xticks(d_rates)
ax.set_yscale("log")

ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

x = [1, 2, 4, 8, 16, 32, 64]
ax.set_xticklabels(x)

ax.set_ylabel("Processing speed (× RT)")
ax.set_xlabel("Kernel size")

ax.legend(
    ["WaveNetVA", "RTNeural", "Proposed (dynamic)", "Proposed (static)"], fontsize=8.3
)

fig.tight_layout()
plt.grid("both")
plt.savefig("kernel_size.pdf")
plt.show()
