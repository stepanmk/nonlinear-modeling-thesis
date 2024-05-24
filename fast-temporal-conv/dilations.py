import os

import matplotlib.ticker
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted

# plt.rcParams.update({"font.family": "Arial"})


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

dilations = natsorted(os.listdir(txt_path + "dilation"))

wave_d = []
rt_d = []
fast_d = []
fast_d2 = []

for d in dilations:
    f_name = txt_path + "dilation/" + d
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

d_rates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

plt.style.use("tableau-colorblind10")

fig, ax = plt.subplots(figsize=set_size(tw, scale_height=0.82))
ax.plot(d_rates, 1 / np.array(wave_d)[:-2], "-^", markersize=5, linewidth=0.8)
ax.plot(d_rates, 1 / np.array(rt_d)[:-2], "-s", markersize=5, linewidth=0.8)
ax.plot(d_rates, 1 / np.array(fast_d)[:-2], "-x", markersize=5, linewidth=0.8)
ax.plot(d_rates, 1 / np.array(fast_d2)[:-2], "-o", markersize=5, linewidth=0.8)
ax.set_ylim([3, 10000])
ax.set_xticks(d_rates)

ax.set_yscale("log")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.set_ylabel("Processing speed (× RT)")
ax.set_xlabel("Dilation rate")

x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
ax.set_xticklabels(x)

ax.legend(
    ["WaveNetVA", "RTNeural", "Proposed (dynamic)", "Proposed (static)"], fontsize=9
)

fig.tight_layout()
plt.grid("both")
plt.savefig("dilations.pdf")
plt.show()
