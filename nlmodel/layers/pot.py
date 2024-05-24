import pytorch_lightning as pl
import torch
import torch.nn as nn


class Potentiometer(nn.Module):
    def __init__(self, init="lin", device="cuda"):
        super().__init__()
        self.init = init
        self.device = device
        if init == "lin":
            self.t2 = nn.Parameter(
                torch.tensor([1.79], device=self.device), requires_grad=True
            )
            self.t3 = nn.Parameter(
                torch.tensor([-0.919], device=self.device), requires_grad=True
            )
        if init == "log":
            self.t2 = nn.Parameter(
                torch.tensor([4.4], device=self.device), requires_grad=True
            )
            self.t3 = nn.Parameter(
                torch.tensor([-3.38], device=self.device), requires_grad=True
            )

    def forward(self, x):
        t1 = 1 / (torch.tanh(self.t2 + self.t3) - torch.tanh(self.t3))
        t4 = -t1 * torch.tanh(self.t3)
        return t1 * torch.tanh(self.t2 * x + self.t3) + t4


class CondBlock(nn.Module):
    def __init__(
        self, n_targets, cond_input="onehot", cond_process="pot", device="cuda"
    ):
        super().__init__()
        self.device = device
        self.n_targets = n_targets
        self.cond_input = cond_input
        self.cond_process = cond_process
        if self.cond_input == "onehot":
            self.bass_from_onehot = nn.Sequential(
                nn.Linear(in_features=n_targets, out_features=1), nn.Sigmoid()
            )
            self.mid_from_onehot = nn.Sequential(
                nn.Linear(in_features=n_targets, out_features=1), nn.Sigmoid()
            )
            self.treble_from_onehot = nn.Sequential(
                nn.Linear(in_features=n_targets, out_features=1), nn.Sigmoid()
            )
        if self.cond_process == "pot":
            self.bass_nn = Potentiometer(init="log", device=self.device)
            self.mid_nn = Potentiometer(init="lin", device=self.device)
            self.treble_nn = Potentiometer(init="lin", device=self.device)
        else:
            self.bass_nn = nn.Sequential(
                nn.Linear(in_features=1, out_features=1),
                nn.Tanh(),
                nn.Linear(in_features=1, out_features=1),
                nn.Sigmoid(),
            )
            self.mid_nn = nn.Sequential(
                nn.Linear(in_features=1, out_features=1),
                nn.Tanh(),
                nn.Linear(in_features=1, out_features=1),
                nn.Sigmoid(),
            )
            self.treble_nn = nn.Sequential(
                nn.Linear(in_features=1, out_features=1),
                nn.Tanh(),
                nn.Linear(in_features=1, out_features=1),
                nn.Sigmoid(),
            )

    def forward(self, cond_batch):
        if self.cond_input == "onehot":
            bass_cond = self.bass_from_onehot(cond_batch)
            mid_cond = self.mid_from_onehot(cond_batch)
            treble_cond = self.treble_from_onehot(cond_batch)
        else:
            bass_cond = cond_batch[:, 0:1]
            mid_cond = cond_batch[:, 1:2]
            treble_cond = cond_batch[:, 2:3]
        bass_cond = self.bass_nn(bass_cond)
        mid_cond = self.mid_nn(mid_cond)
        treble_cond = self.treble_nn(treble_cond)
        return [bass_cond.squeeze(), mid_cond.squeeze(), treble_cond.squeeze()]


class CondBlockTone(nn.Module):
    def __init__(
        self,
        n_targets,
        cond_input="onehot",
        cond_process="pot",
        device="cuda",
        pot_char="lin",
    ):
        super().__init__()
        self.device = device
        self.n_targets = n_targets
        self.cond_input = cond_input
        self.cond_process = cond_process
        self.pot_char = pot_char
        if self.cond_process == "pot":
            self.bass_nn = Potentiometer(init=self.pot_char, device=self.device)

    def forward(self, cond_batch):
        if self.cond_input == "onehot":
            bass_cond = self.bass_from_onehot(cond_batch)
        else:
            bass_cond = cond_batch[:, 0:1]
        bass_cond = self.bass_nn(bass_cond)
        return [bass_cond.squeeze()]
