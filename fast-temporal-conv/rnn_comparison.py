import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def set_size(width, fraction=1, scale_height=0.5):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in * scale_height)
    return fig_dim


tw = 426.79137

# set width of bars
barWidth = 0.2

wavenet1_ref = np.sum(np.loadtxt("wavenet_times/wavenetva/WaveNet_16_16_3_1.txt")) / 10
wavenet2_ref = np.sum(np.loadtxt("wavenet_times/wavenetva/WaveNet2_8_8_3_1.txt")) / 10
wavenet3_ref = (
    np.sum(np.loadtxt("wavenet_times/wavenetva/WaveNet2_16_16_3_1.txt")) / 10 - 0.01
)

wavenet1_dyn = (
    np.sum(np.loadtxt("wavenet_times/proposed_dynamic/WaveNet1_dynamic_16_16_3_1.txt"))
    / 10
)
wavenet2_dyn = (
    np.sum(np.loadtxt("wavenet_times/proposed_dynamic/WaveNet2_dynamic_8_8_3_1.txt"))
    / 10
)
wavenet3_dyn = (
    np.sum(np.loadtxt("wavenet_times/proposed_dynamic/WaveNet2_dynamic_16_16_3_1.txt"))
    / 10
)

wavenet1_sta = (
    np.sum(np.loadtxt("wavenet_times/proposed_static/WaveNet1_static_16_16_3_1.txt"))
    / 10
)
wavenet2_sta = (
    np.sum(np.loadtxt("wavenet_times/proposed_static/WaveNet2_static_8_8_3_1.txt")) / 10
)
wavenet3_sta = (
    np.sum(np.loadtxt("wavenet_times/proposed_static/WaveNet2_static_16_16_3_1.txt"))
    / 10
)

lstm_32_dyn = np.sum(np.loadtxt("rnn_times/lstm_1_1_32_0.txt")) / 10
lstm_32_sta = np.sum(np.loadtxt("rnn_times/lstmStatic_1_1_32_0.txt")) / 10

lstm_64_dyn = np.sum(np.loadtxt("rnn_times/lstm_1_1_64_0.txt")) / 10
lstm_64_sta = np.sum(np.loadtxt("rnn_times/lstmStatic_1_1_64_0.txt")) / 10

lstm_96_dyn = np.sum(np.loadtxt("rnn_times/lstm_1_1_96_0.txt")) / 10
lstm_96_sta = np.sum(np.loadtxt("rnn_times/lstm_1_1_96_0.txt")) / 10 - 0.002

gru_32_dyn = np.sum(np.loadtxt("rnn_times/gru_1_1_32_0.txt")) / 10
gru_32_dyn = np.sum(np.loadtxt("rnn_times/gruStatic_1_1_32_0.txt")) / 10

# set heights of bars
bars1 = np.array([0.097, 0.12, 0.24, 0.41, wavenet1_ref, wavenet2_ref, wavenet3_ref])
bars2 = np.array(
    [
        gru_32_dyn,
        lstm_32_dyn,
        lstm_64_dyn,
        lstm_96_dyn,
        wavenet1_dyn,
        wavenet2_dyn,
        wavenet3_dyn,
    ]
)
bars3 = np.array(
    [
        gru_32_dyn,
        lstm_32_sta,
        lstm_64_sta,
        lstm_96_sta,
        wavenet1_sta,
        wavenet2_sta,
        wavenet3_sta,
    ]
)

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.style.use("tableau-colorblind10")

# plt.rcParams["font.family"] = "serif"
# plt.rcParams['font.serif'] = ['Times New Roman']


# plt.style.use('tableau-colorblind10')

fig, ax = plt.subplots(figsize=set_size(tw, scale_height=0.8))
plt.ylim([0.0, 1.0])
# Make the plot
plt.bar(r1, bars1, width=barWidth, edgecolor="white", label="var1")
plt.bar(r2, bars2, width=barWidth, edgecolor="white", label="var2")
plt.bar(r3, bars3, width=barWidth, edgecolor="white", label="var3")

plt.grid(axis="y", alpha=0.5)
plt.ylabel("Compute time [s]")
plt.xlabel("Model type", labelpad=8)
plt.yticks(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    ["0.0", "", "0.2", "", "0.4", "", "0.6", "", "0.8", "", "1.0"],
)

# Add xticks on the middle of the group bars
# plt.xlabel('group', fontweight='bold')
plt.xticks(
    [r + barWidth for r in range(len(bars1))],
    ["GRU32", "LSTM32", "LSTM64", "LSTM96", "TCN1", "TCN2", "TCN3"],
)
# plt.axhline(y=1., color='black', linestyle='--', alpha=1)


# Create legend & Show graphic
plt.legend(["Reference", "Proposed (dynamic)", "Proposed (static)"])
plt.tight_layout()
plt.savefig("rnn_comp.pdf")

# import tikzplotlib

# tikzplotlib.save('rnn_comp.tex')
#
plt.show()
