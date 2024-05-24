import numpy as np
from matplotlib import pyplot as plt
from soundfile import read


def set_size(width, fraction=1, scale_height=0.5):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in * scale_height)
    return fig_dim


def harm(amplitude=0.5, freq=100, seconds=1, fs=48000):
    n_samples = fs * seconds
    n = np.arange(n_samples)
    x = amplitude * np.cos(2 * np.pi * n * freq / fs)
    return x, (n / fs)


tw = 426.79137
x, n = harm(amplitude=0.9)

audio_in, fs = read(file="./dataset/final/B5_M5_T5_G1/B5_M5_T5_G1-input.wav")
audio_out, _ = read(file="./dataset/final/B5_M5_T5_G1/B5_M5_T5_G1-speakerout.wav")
print(fs)

n_samples = int(fs * 0.2)
t = np.arange(n_samples) / fs
start1 = 222
end1 = start1 + 1

start2 = 226
end2 = start2 + 1

start3 = 338
end3 = 339

fig, axes = plt.subplots(
    num=1,
    nrows=2,
    ncols=3,
    sharex=True,
    sharey=True,
    figsize=set_size(tw, scale_height=0.8),
)

axes[0, 0].plot(t, audio_in[n_samples * start1 : n_samples * end1], "k", linewidth=0.8)
axes[0, 0].set_xlim([0, 0.2])
axes[0, 0].set_ylim([-1, 1])
axes[0, 0].grid(axis="x", alpha=0.5, color="blue")
axes[0, 0].set_xticks(
    [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2],
    ["0", "", "", "", "0.1", "", "", "", "0.2"],
)


axes[1, 0].plot(t, audio_out[n_samples * start1 : n_samples * end1], "k", linewidth=0.8)
axes[1, 0].set_xlim([0, 0.2])
axes[1, 0].set_ylim([-1, 1])
axes[1, 0].grid(axis="x", alpha=0.5, color="blue")

axes[0, 1].plot(t, audio_in[n_samples * start2 : n_samples * end2], "k", linewidth=0.8)
axes[0, 1].set_xlim([0, 0.2])
axes[0, 1].set_ylim([-1, 1])
axes[0, 1].grid(axis="x", alpha=0.5, color="blue")

axes[1, 1].plot(t, audio_out[n_samples * start2 : n_samples * end2], "k", linewidth=0.8)
axes[1, 1].set_xlim([0, 0.2])
axes[1, 1].set_ylim([-1, 1])
axes[1, 1].grid(axis="x", alpha=0.5, color="blue")

axes[0, 2].plot(t, audio_in[n_samples * start3 : n_samples * end3], "k", linewidth=0.8)
axes[0, 2].set_xlim([0, 0.2])
axes[0, 2].set_ylim([-1, 1])
axes[0, 2].grid(axis="x", alpha=0.5, color="blue")

axes[1, 2].plot(t, audio_out[n_samples * start3 : n_samples * end3], "k", linewidth=0.8)
axes[1, 2].set_xlim([0, 0.2])
axes[1, 2].set_ylim([-1, 1])
axes[1, 2].grid(axis="x", alpha=0.5, color="blue")

fig.supxlabel("Time [s]", fontsize=9, x=0.5, y=0.05)
fig.supylabel("Amplitude [–]", fontsize=9, x=0.05, y=0.5)

for i in range(2):
    for j in range(3):
        axes[i, j].xaxis.set_tick_params(labelsize=9)
        axes[i, j].yaxis.set_tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("./dplots/paired_data.pdf")
