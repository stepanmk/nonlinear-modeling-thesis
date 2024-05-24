import torch
import torch.nn as nn


class ESR(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps = 1e-05

    def forward(self, output, target):
        nom = torch.mean(torch.square(target - output))
        den = torch.mean(torch.square(target)) + self.eps
        return nom / den
