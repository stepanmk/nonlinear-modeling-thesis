import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import freqz

from nlmodel.dk_method.ds1_tone import DS1Tone
from nlmodel.dk_method.rat_tone import RATTone

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
in_t = torch.zeros((4096, 1))
in_t[0, 0] = 1.0

colors = [
    "black",
    "#227093",
    "#706fd3",
    "#34ace0",
    "#33d9b2",
    "#ff5252",
    "black",
]

for i, t in enumerate([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]):
    alpha_t = 0.536 * np.tanh(5.113 * t - 3.787) + 0.535
    rat = RATTone(alpha_t=alpha_t, sr=192000)
    out = rat.process(in_t)
    out = out.numpy()
    w, h = freqz(b=out, worN=4096, fs=192000)
    axes[1].semilogx(w, 20 * np.log10(np.abs(h)), color=colors[i], linewidth=0.8)

axes[1].set_xticks([100, 1000, 10000])
axes[1].set_xticklabels(["100", "1k", "10k"])
axes[1].set_xlim([20, 20000])
axes[1].set_ylim([-40, 0])
axes[1].grid(which="both", alpha=0.5)
axes[1].set_title("RAT tone section", fontsize=10)
axes[0].set_ylabel("Magnitude [dB]")

for i, t in enumerate([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]):
    rat = DS1Tone(alpha_t=t, sr=192000)
    out = rat.process(in_t)
    out = out.numpy()
    w, h = freqz(b=out, worN=4096, fs=192000)
    axes[0].semilogx(w, 20 * np.log10(np.abs(h)), color=colors[i], linewidth=0.8)

axes[0].set_xticks([100, 1000, 10000])
axes[0].set_xticklabels(["100", "1k", "10k"])
axes[0].set_xlim([20, 20000])
axes[0].set_ylim([-40, 0])
axes[0].grid(which="both", alpha=0.5)
axes[0].set_title("DS1 tone section", fontsize=10)

fig.supxlabel("Frequency [Hz]", fontsize=10, x=0.5, y=0.1)
plt.tight_layout()
plt.savefig("./dplots/pedal_tones.pdf")
plt.show()
