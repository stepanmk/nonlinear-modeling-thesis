import auraloss
import numpy as np
import torch
from matplotlib import pyplot as plt
from soundfile import read

from nlmodel.losses.time import ESR


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


def test_signal(amplitude=1.0, phase=np.pi, freq=20, fs=48000):
    seconds = 1 / 20
    n_samples = int(seconds * fs)
    n = np.linspace(0, 2 * np.pi, n_samples)
    x = amplitude * (
        0.5
        * np.sin(0.5 * n + phase)
        * (0.5 * np.cos((2 + 3 * n) * n) + 0.5 * np.cos(freq * n))
    )
    return x, np.arange(n_samples) / fs


tw = 426.79137

l_mae = torch.nn.L1Loss()
l_mse = torch.nn.MSELoss()
l_esr = ESR()
# l_mrstft = auraloss.time.DCLoss()

fig, axes = plt.subplots(
    num=1, nrows=2, ncols=3, sharey="row", figsize=set_size(tw, scale_height=0.9)
)

x_def, n_def = test_signal()

amplitudes = np.linspace(-1, 2, 50)
mae_amp = []
mse_amp = []
esr_amp = []
for i, a in enumerate(amplitudes):
    x, n = test_signal(amplitude=a)
    if i % 5 == 0:
        axes[0, 0].plot(n, x, "gray", linewidth=0.5, alpha=0.8)
    axes[0, 0].set_xlim([0, 0.05])
    axes[0, 0].set_ylim([-1, 1])
    axes[0, 0].grid("both", alpha=0.5)
    x_t_def = torch.tensor(x_def).unsqueeze(0).unsqueeze(-1)
    x_t = torch.tensor(x).unsqueeze(0).unsqueeze(-1)
    mae_amp.append(l_mae(x_t, x_t_def).item())
    mse_amp.append(l_mse(x_t, x_t_def).item())
    esr_amp.append(l_esr(x_t, x_t_def).item())

axes[0, 0].plot(n_def, x_def, "k", linewidth=0.8)
axes[0, 0].set_title("Amplitude variation", fontsize=9)
axes[0, 1].set_title("Phase variation", fontsize=9)
axes[0, 2].set_title("Frequency variation", fontsize=9)
axes[0, 0].set_xticks([0, 0.025, 0.05], ["0", "", "0.05"])
axes[0, 0].set_xlabel("Time [s]", fontsize=9)
axes[0, 2].set_xlabel("Time [s]", fontsize=9)
axes[0, 1].set_xlabel("Time [s]", fontsize=9)
axes[0, 0].set_ylabel("Amplitude [–]", fontsize=9)

axes[1, 0].set_ylabel("Error [–]", fontsize=9)
axes[1, 0].set_xlabel("Amplitude [–]", fontsize=9)
axes[1, 1].set_xlabel("Phase [rad]", fontsize=9)
axes[1, 2].set_xlabel("Frequency [Hz]", fontsize=9)

axes[1, 0].plot(amplitudes, mae_amp, "k", linewidth=0.8, linestyle="dotted")
axes[1, 0].plot(amplitudes, mse_amp, "k", linewidth=0.8, linestyle="dashed")
axes[1, 0].plot(amplitudes, esr_amp, "k", linewidth=0.8)
axes[1, 0].set_xlim([-1, 2])
axes[1, 0].set_ylim([0, 1.5])
axes[1, 0].grid("both", alpha=0.5)
axes[1, 0].legend(["MAE", "MSE", "ESR"], fontsize=8, loc=(0.435, 0.4))

phases = np.linspace(0, 2 * np.pi, 50)
mae_ph = []
mse_ph = []
esr_ph = []
mrstft_ph = []
for i, p in enumerate(phases):
    x, n = test_signal(phase=p)
    if i % 5 == 0:
        axes[0, 1].plot(n, x, "gray", linewidth=0.5, alpha=0.8)
    axes[0, 1].set_xlim([0, 0.05])
    axes[0, 1].set_ylim([-1, 1])
    axes[0, 1].grid("both", alpha=0.5)
    x_t_def = torch.tensor(x_def).unsqueeze(0).unsqueeze(-1)
    x_t = torch.tensor(x).unsqueeze(0).unsqueeze(-1)
    mae_ph.append(l_mae(x_t, x_t_def).item())
    mse_ph.append(l_mse(x_t, x_t_def).item())
    esr_ph.append(l_esr(x_t, x_t_def).item())

axes[0, 1].plot(n_def, x_def, "k", linewidth=0.8)
axes[0, 1].set_xticks([0, 0.025, 0.05], ["0", "", "0.05"])
axes[1, 1].plot(phases, mae_ph, "k", linewidth=0.8, linestyle="dotted")
axes[1, 1].plot(phases, mse_ph, "k", linewidth=0.8, linestyle="dashed")
axes[1, 1].plot(phases, esr_ph, "k", linewidth=0.8)
axes[1, 1].set_xlim([0, 2 * np.pi])
axes[1, 1].set_ylim([0, 1.5])
axes[1, 1].set_xticks(
    [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
    ["$0$", "$\pi/2$", "$\pi$", "$3\pi/2$", "$2\pi$"],
)
axes[1, 1].grid("both", alpha=0.5)

freqs = np.linspace(15, 25, 75)
mae_f = []
mse_f = []
esr_f = []
for i, f in enumerate(freqs):
    x, n = test_signal(freq=f)
    if i % 10 == 0:
        print(i)
        axes[0, 2].plot(n, x, "gray", linewidth=0.5, alpha=0.8)
    axes[0, 2].set_xlim([0, 0.05])
    axes[0, 2].set_ylim([-1, 1])
    axes[0, 2].grid("both", alpha=0.5)
    x_t_def = torch.tensor(x_def).unsqueeze(0).unsqueeze(-1)
    x_t = torch.tensor(x).unsqueeze(0).unsqueeze(-1)
    mae_f.append(l_mae(x_t, x_t_def).item())
    mse_f.append(l_mse(x_t, x_t_def).item())
    esr_f.append(l_esr(x_t, x_t_def).item())

axes[0, 2].plot(n_def, x_def, "k", linewidth=0.8)
axes[0, 2].set_xticks([0, 0.025, 0.05], ["0", "", "0.05"])
axes[1, 2].plot(freqs, mae_f, "k", linewidth=0.8, linestyle="dotted")
axes[1, 2].plot(freqs, mse_f, "k", linewidth=0.8, linestyle="dashed")
axes[1, 2].plot(freqs, esr_f, "k", linewidth=0.8)
axes[1, 2].set_xlim([15, 25])
axes[1, 2].set_ylim([0, 1.5])
axes[1, 2].set_xticks([15, 20, 25])
axes[1, 2].grid("both", alpha=0.5)

for i in range(2):
    for j in range(3):
        axes[i, j].xaxis.set_tick_params(labelsize=9)
        axes[i, j].yaxis.set_tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("./dplots/losses.pdf")
