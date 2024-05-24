import numpy as np
from matplotlib import pyplot as plt


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

fig, axes = plt.subplots(num=1, nrows=1, ncols=3, figsize=set_size(tw))

axes[0].plot(
    np.linspace(-6, 6, 1000),
    np.maximum(0, np.linspace(-6, 6, 1000)),
    "k",
    linewidth=0.8,
)
axes[0].set_title("ReLU", fontsize=10)
axes[0].set_xlim([-6, 6])
axes[0].set_ylim([-0.8, 4.8])
axes[0].set_xlabel("Input", fontsize=9)
axes[0].set_ylabel("Output", fontsize=9)
axes[0].grid("both", alpha=0.5)

axes[2].plot(
    np.linspace(-6, 6, 1000), np.tanh(np.linspace(-5, 5, 1000)), "k", linewidth=0.8
)
axes[2].set_title("tanh", fontsize=10)
axes[2].set_xlim([-6, 6])
axes[2].set_ylim([-1.4, 1.4])
axes[2].grid("both", alpha=0.5)
axes[2].set_xlabel("Input", fontsize=9)
axes[2].set_ylabel("Output", fontsize=9)

axes[1].plot(
    np.linspace(-6, 6, 1000),
    1 / (1 + np.exp(-np.linspace(-6, 6, 1000))),
    "k",
    linewidth=0.8,
)
axes[1].set_title("$\sigma$", fontsize=10)
axes[1].set_xlim([-6, 6])
axes[1].set_ylim([-0.2, 1.2])
axes[1].grid("both", alpha=0.5)
axes[1].set_xlabel("Input", fontsize=9)
axes[1].set_ylabel("Output", fontsize=9)

for ax in axes:
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

plt.tight_layout()
plt.savefig("./dplots/act_func.pdf")

# fft = 20 * np.log10(np.abs(np.fft.rfft(x)) / len(x))
# fft_nonlin = 20 * np.log10(np.abs(np.fft.rfft(0.5 * np.tanh(5 * x))) / len(x))
# fft_nonlin2 = 20 * np.log10(
#     np.abs(np.fft.rfft(0.5 * np.tanh(1.5 * x))) / len(x))

# freq = np.linspace(0, 48000 / 2, len(fft))

# fig, axes = plt.subplots(num=2, nrows=1, ncols=2, figsize=set_size(tw))

# axes[0].semilogx(freq, fft, 'k', linewidth=0.8)
# axes[1].semilogx(freq, fft_nonlin, 'k', linewidth=0.8)

# for i, ax in enumerate(axes):
#     ax.set_xlim([20, 20000])
#     ax.set_ylim([-60, 0])
#     ax.set_xlabel('Frequency [Hz]', fontsize=9)
#     if i == 0:
#         ax.set_ylabel('Magnitude [dB]', fontsize=9)
#     ax.set_xticks([100, 1000, 10000], labels=['100', '1k', '10k'])
#     ax.grid('both', alpha=0.5)
#     ax.xaxis.set_tick_params(labelsize=8)
#     ax.yaxis.set_tick_params(labelsize=8)

# plt.tight_layout()
# plt.savefig('./dplots/act_func.pdf')
