import torch

from nlmodel.loaders import SingleFileLoader
from nlmodel.losses.time import ESR
from nlmodel.nets.rnn import RNNBase

data_dir = "./dataset/"
inp_file = "ht1-input.wav"
tgt_file = "ht1-target.wav"
test_start_sec = 375.0
test_end_sec = 435.0

test1 = SingleFileLoader(
    data_dir=data_dir,
    inp_file=inp_file,
    tgt_file=tgt_file,
    start_sec=test_start_sec,
    end_sec=test_end_sec,
    seg_len=5,
    batch_size=12,
    shuffle=False,
)

data_dir = f"./dataset/final/B5_M5_T5_G8/"
inp_file = f"B5_M5_T5_G8-input.wav"
tgt_file = f"B5_M5_T5_G8-speakerout.wav"
test_start_sec = 300.0
test_end_sec = 360.0

test2 = SingleFileLoader(
    data_dir=data_dir,
    inp_file=inp_file,
    tgt_file=tgt_file,
    start_sec=test_start_sec,
    end_sec=test_end_sec,
    seg_len=5,
    batch_size=12,
    shuffle=False,
)


def test_model(net, loader, val_steps=22050):
    train_loss = ESR()
    for batch in loader:
        seg_samples = loader.dataset.seg_samples
        ep_loss = 0
        for batch in loader:
            x_inp = batch[0].to("cuda")
            y_tgt = batch[1].to("cuda")
            net.reset_states()
            batch_loss = 0
            y_hat_list = []
            for n in range(0, seg_samples, val_steps):
                with torch.no_grad():
                    y_hat = net(x_inp[:, n : n + val_steps, :])
                y_hat_list.append(y_hat)
                seg_loss = train_loss(y_hat, y_tgt[:, n : n + val_steps, :])
                batch_loss += seg_loss
            ep_loss += batch_loss / (seg_samples / val_steps)
            y_hat_cat = torch.cat(y_hat_list, dim=1)
            diff = torch.abs(y_tgt - y_hat_cat)
    return ep_loss, y_tgt, y_hat_cat, diff


import numpy as np
from matplotlib import pyplot as plt

comp1 = [
    "./runs3/ht1_no_clip_3_gru_32/checkpoints/best_model.ckpt",
    "./runs1/ht1_no_clip_gru_48/checkpoints/best_model.ckpt",
    "./runs2/jvm_no_clip_2_gru_16/checkpoints/best_model.ckpt",
    "./runs2/jvm_no_clip_2_lstm_16/checkpoints/best_model.ckpt",
]

comp2 = [
    "./runs3/ht1_clip_3_gru_32/checkpoints/best_model.ckpt",
    "./runs1/ht1_autoclip_gru_48/checkpoints/best_model.ckpt",
    "./runs3/jvm_clip_3_gru_16/checkpoints/best_model.ckpt",
    "./runs2/jvm_clip_2_lstm_16/checkpoints/best_model.ckpt",
]


def compare(models1, models2):
    loader1 = test1.get_loader()
    loader2 = test2.get_loader()
    ht1 = []
    jvm = []
    for i in range(len(models1)):
        ct1 = models1[i].split("/")[2].split("_")[-2].upper()
        ct2 = models2[i].split("/")[2].split("_")[-2].upper()
        hs1 = int(models1[i].split("/")[2].split("_")[-1])
        hs2 = int(models2[i].split("/")[2].split("_")[-1])
        net1 = RNNBase(cell_type=ct1, hidden_size=hs1, skip=True, batch_size=12)
        net2 = RNNBase(cell_type=ct2, hidden_size=hs2, skip=True, batch_size=12)
        net1.eval()
        net2.eval()
        best_checkpoint_1 = torch.load(models1[i])
        best_checkpoint_2 = torch.load(models2[i])
        net1.load_state_dict(best_checkpoint_1["model_state_dict"])
        net2.load_state_dict(best_checkpoint_2["model_state_dict"])
        if i < 2:
            loss1, y_tgt, y_hat_cat, diff = test_model(net1, loader1)
            ht1.append(
                {"loss": loss1, "y_tgt": y_tgt, "y_hat": y_hat_cat, "diff": diff}
            )
            loss2, y_tgt, y_hat_cat, diff = test_model(net2, loader1)
            ht1.append(
                {"loss": loss2, "y_tgt": y_tgt, "y_hat": y_hat_cat, "diff": diff}
            )
        else:
            loss1, y_tgt, y_hat_cat, diff = test_model(net1, loader2)
            jvm.append(
                {"loss": loss1, "y_tgt": y_tgt, "y_hat": y_hat_cat, "diff": diff}
            )
            loss2, y_tgt, y_hat_cat, diff = test_model(net2, loader2)
            jvm.append(
                {"loss": loss2, "y_tgt": y_tgt, "y_hat": y_hat_cat, "diff": diff}
            )
    return ht1, jvm


tw = 426.79137


def set_size(width, fraction=1, scale_height=0.8):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in * scale_height)
    return fig_dim


ht1, jvm = compare(comp1, comp2)

fig, axes = plt.subplots(
    num=1,
    nrows=4,
    ncols=4,
    sharey=True,
    sharex=True,
    figsize=set_size(tw, scale_height=1.5),
)

ht1start = 45000
ht1end = ht1start + 1103

jvmstart = 199000
jvmend = jvmstart + (1103)
t = np.arange(1103) / 44100
q = 3

compcolor = "tab:red"
h0 = ht1[0]
axes[0, 0].plot(
    t,
    h0["y_tgt"][q, ht1start:ht1end, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="tab:blue",
)
axes[0, 0].plot(
    t,
    h0["y_hat"][q, ht1start:ht1end, 0].detach().cpu().numpy(),
    linewidth=0.6,
    linestyle="--",
    color=compcolor,
)
axes[0, 0].grid("both", alpha=0.5)
axes[0, 0].set_ylim([-1, 1])
axes[0, 0].set_xticks([0, 0.025], ["0", "0.025"])
axes[0, 0].set_yticks([-1, 0.0, 1])
axes[0, 0].set_xlim([0, 0.025])
axes[0, 0].text(0.001, 0.75, "HT1 GRU32", fontsize=9)
axes[0, 0].set_title("Signal", fontsize=9)

axes[0, 1].plot(
    t,
    h0["diff"][q, ht1start:ht1end, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="black",
    alpha=0.9,
)
axes[0, 1].grid("both", alpha=0.5)

h1 = ht1[1]
axes[0, 2].plot(
    t,
    h1["y_tgt"][q, ht1start:ht1end, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="tab:blue",
)
axes[0, 2].plot(
    t,
    h1["y_hat"][q, ht1start:ht1end, 0].detach().cpu().numpy(),
    linewidth=0.6,
    linestyle="--",
    color=compcolor,
)
axes[0, 2].grid("both", alpha=0.5)
axes[0, 2].set_ylim([-1, 1])

from matplotlib.lines import Line2D

custom_lines = [
    Line2D([0], [0], color="tab:blue", lw=0.6),
    Line2D([0], [0], color="red", lw=0.6, linestyle="--"),
    Line2D([0], [0], color="black", lw=0.6),
]

axes[0, 3].legend(
    custom_lines,
    ["Target", "Prediction", "Abs. error"],
    fontsize=6.5,
    loc="lower center",
)

axes[0, 3].plot(
    t,
    h1["diff"][q, ht1start:ht1end, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="black",
    alpha=0.9,
)
axes[0, 1].set_title("Abs. error", fontsize=9)
axes[0, 2].set_title("Signal", fontsize=9)
axes[0, 3].set_title("Abs. error", fontsize=9)
axes[0, 3].grid("both", alpha=0.5)

h2 = ht1[2]
axes[1, 0].plot(
    t,
    h2["y_tgt"][q, ht1start:ht1end, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="tab:blue",
)
axes[1, 0].plot(
    t,
    h2["y_hat"][q, ht1start:ht1end, 0].detach().cpu().numpy(),
    linewidth=0.6,
    linestyle="--",
    color=compcolor,
)
axes[1, 0].text(0.001, 0.75, "HT1 GRU48", fontsize=9)
axes[1, 0].grid("both", alpha=0.5)
axes[1, 0].set_ylim([-1, 1])
axes[1, 1].plot(
    t,
    h2["diff"][q, ht1start:ht1end, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="black",
    alpha=0.9,
)
axes[1, 1].grid("both", alpha=0.5)

h3 = ht1[3]
axes[1, 2].plot(
    t,
    h3["y_tgt"][q, ht1start:ht1end, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="tab:blue",
)
axes[1, 2].plot(
    t,
    h3["y_hat"][q, ht1start:ht1end, 0].detach().cpu().numpy(),
    linewidth=0.6,
    linestyle="--",
    color=compcolor,
)
axes[1, 2].grid("both", alpha=0.5)
axes[1, 2].set_ylim([-1, 1])
axes[1, 3].plot(
    t,
    h3["diff"][q, ht1start:ht1end, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="black",
    alpha=0.9,
)
axes[1, 3].grid("both", alpha=0.5)


j0 = jvm[0]
axes[2, 0].plot(
    t,
    j0["y_tgt"][0, jvmstart:jvmend, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="tab:blue",
)
axes[2, 0].plot(
    t,
    j0["y_hat"][0, jvmstart:jvmend, 0].detach().cpu().numpy(),
    linewidth=0.6,
    linestyle="--",
    color=compcolor,
)
axes[2, 0].grid("both", alpha=0.5)
axes[2, 0].text(0.001, 0.75, "JVM GRU16", fontsize=9)
axes[2, 0].set_ylim([-1, 1])
axes[2, 1].plot(
    t,
    j0["diff"][0, jvmstart:jvmend, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="black",
    alpha=0.9,
)
axes[2, 1].grid("both", alpha=0.5)

j1 = jvm[1]
axes[2, 2].plot(
    t,
    j1["y_tgt"][0, jvmstart:jvmend, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="tab:blue",
)
axes[2, 2].plot(
    t,
    j1["y_hat"][0, jvmstart:jvmend, 0].detach().cpu().numpy(),
    linewidth=0.6,
    linestyle="--",
    color=compcolor,
)
axes[2, 2].grid("both", alpha=0.5)
axes[2, 2].set_ylim([-1, 1])
axes[2, 3].plot(
    t,
    j1["diff"][0, jvmstart:jvmend, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="black",
    alpha=0.9,
)
axes[2, 3].grid("both", alpha=0.5)

j2 = jvm[2]
axes[3, 0].plot(
    t,
    j2["y_tgt"][0, jvmstart:jvmend, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="tab:blue",
)
axes[3, 0].plot(
    t,
    j2["y_hat"][0, jvmstart:jvmend, 0].detach().cpu().numpy(),
    linewidth=0.6,
    linestyle="--",
    color=compcolor,
)
axes[3, 0].text(0.001, 0.75, "JVM LSTM16", fontsize=9)
axes[3, 0].grid("both", alpha=0.5)
axes[3, 0].set_ylim([-1, 1])
axes[3, 1].plot(
    t,
    j2["diff"][0, jvmstart:jvmend, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="black",
    alpha=0.9,
)
axes[3, 1].grid("both", alpha=0.5)

j3 = jvm[3]
axes[3, 2].plot(
    t,
    j3["y_tgt"][0, jvmstart:jvmend, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="tab:blue",
)
axes[3, 2].plot(
    t,
    j3["y_hat"][0, jvmstart:jvmend, 0].detach().cpu().numpy(),
    linewidth=0.6,
    linestyle="--",
    color=compcolor,
)
axes[3, 2].grid("both", alpha=0.5)
axes[3, 2].set_ylim([-1, 1])
axes[3, 3].plot(
    t,
    j3["diff"][0, jvmstart:jvmend, 0].detach().cpu().numpy(),
    linewidth=0.6,
    color="black",
    alpha=0.9,
)
axes[3, 3].grid("both", alpha=0.5)
fig.supxlabel("Time [s]", fontsize=9, x=0.5, y=0)
fig.supylabel(
    "Amplitude [–]",
    fontsize=9,
    x=0.015,
    y=0.5,
)

plt.tight_layout(pad=0.4)
plt.savefig("./dplots/timedomain.pdf")
plt.show()
