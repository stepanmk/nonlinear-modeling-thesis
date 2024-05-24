import torch
import torch.nn as nn
from auraloss.freq import STFTLoss


class STFT(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = STFTLoss()

    def forward(self, output, target):
        return self.loss(
            torch.permute(output, (0, 2, 1)), torch.permute(target, (0, 2, 1))
        )
