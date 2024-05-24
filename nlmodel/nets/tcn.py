import torch
import torch.nn as nn

from ..layers.conv import ConvBlock, ConvBlockAd


class TCN(nn.Module):

    def __init__(
        self,
        num_channels: int = 16,
        kernel_size: int = 3,
        layers: int = 10,
        repeats: int = 1,
        groups: int = 1,
        df: int = 1,
        act_type: str = "gated",
        block_type: str = "standard",
        device: str = "cuda",
        conv_in: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.layers = layers
        self.repeats = repeats
        self.groups = groups
        self.act_type = act_type
        self.block_type = block_type
        self.df = df
        self.dilations = self.repeats * [2**i * df for i in range(self.layers)]
        self.device = device
        self.conv_in = None
        if conv_in:
            self.conv_in = nn.Conv1d(
                in_channels=1,
                out_channels=self.num_channels,
                kernel_size=1,
                device=self.device,
            )
        if self.block_type == "standard":
            self.block_class = ConvBlock
        else:
            self.block_class = ConvBlockAd
        self.conv_blocks = nn.ModuleList()
        for i, dilation in enumerate(self.dilations):
            in_channels = 1 if i == 0 and not self.conv_in else self.num_channels
            self.conv_blocks.append(
                self.block_class(
                    in_channels=in_channels,
                    out_channels=self.num_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    groups=self.groups,
                    act_type=self.act_type,
                    device=self.device,
                    **kwargs,
                )
            )
        self.conv_out = nn.Conv1d(
            in_channels=self.num_channels * len(self.dilations),
            out_channels=1,
            kernel_size=1,
            device=self.device,
        )
        self.num_params = None
        self.model_name = (
            f"tcn_ch{num_channels}_l{layers}_r{repeats}_act_{act_type}_{block_type}"
        )
        self.count_params()

    def count_params(self) -> None:
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        # make dims consistent with recurrent nets
        z_out = x.permute((0, 2, 1))
        if self.conv_in:
            z_out = self.conv_in(z_out)
        for block in self.conv_blocks:
            z_skip, z_out = block(z_out)
            skips.append(z_skip)
        return torch.permute(self.conv_out(torch.cat(skips, dim=1)), (0, 2, 1))
