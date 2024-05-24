from typing import Tuple

import torch
import torch.nn as nn


# dim order for conv layers is (B, C, L)
class Conv1dCausal(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 1,
        dilation: int = 1,
        groups: int = 1,
        device: str = "cuda",
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.device = device
        self.conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros((x.shape[0], x.shape[1], self.padding), device=self.device)
        return self.conv_layer(torch.cat([pad, x], dim=-1))


class Conv1dStateful(Conv1dCausal):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 1,
        dilation: int = 1,
        groups: int = 1,
        batch_size: int = 1,
        device: str = "cuda",
    ):
        Conv1dCausal.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            device=device,
        )
        self.in_channels = in_channels
        self.device = device
        self.batch_size = batch_size
        self.conv_state = None
        self.reset_state()

    def change_batch_size(self, new_batch_size: int) -> None:
        self.batch_size = new_batch_size
        self.reset_state()

    def reset_state(self) -> None:
        self.conv_state = torch.zeros(
            (self.batch_size, self.in_channels, self.padding), device=self.device
        )

    def detach_state(self) -> None:
        self.conv_state = self.conv_state.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_layer(torch.cat([self.conv_state, x], dim=-1))
        self.conv_state = x[:, :, -self.padding :]
        return out


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 1,
        dilation: int = 1,
        groups: int = 1,
        act_type: str = "gated",
        device: str = "cuda",
    ):
        super().__init__()
        self.act_type = act_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.device = device
        self.act = None
        self.conv_out = nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            device=self.device,
        )
        self.init_activation(act_type=self.act_type)
        self.conv_in = Conv1dCausal(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=self.groups,
            device=self.device,
        )

    def init_activation(self, act_type: str) -> None:
        if act_type == "gated":
            self.out_channels *= 2
            self.tanh = nn.Tanh()
            self.sigm = nn.Sigmoid()
        elif act_type == "softsign":
            self.act = nn.Softsign()
        elif act_type == "hardtanh":
            self.act = nn.Hardtanh()
        elif act_type == "tanh":
            self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        res = x
        if self.act_type == "gated":
            z_in = self.conv_in(x)
            z_tanh = self.tanh(z_in[:, 0 : int(self.out_channels / 2), :])
            z_sigm = self.sigm(z_in[:, int(self.out_channels / 2) :, :])
            z_gated = z_tanh * z_sigm
            z_out = self.conv_out(z_gated) + res
            return z_gated, z_out
        else:
            z_tanh = self.act(self.conv_in(x))
            z_out = self.conv_out(z_tanh) + res
            return z_tanh, z_out


class ConvBlockAd(ConvBlock):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 1,
        dilation: int = 1,
        groups: int = 1,
        act_type: str = "gated",
        ad_type: str = "simple",
        device: str = "cuda",
    ):
        ConvBlock.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            act_type=act_type,
            device=device,
        )
        self.ad_type = ad_type
        if self.ad_type == "simple":
            self.param_shape = (1,)
        else:
            self.param_shape = (1, in_channels, 1)
        if self.act_type == "gated":
            self.alpha_tanh_in = nn.Parameter(
                torch.randn(self.param_shape, device=self.device)
            )
            self.alpha_tanh_out = nn.Parameter(
                torch.randn(self.param_shape, device=self.device)
            )
            self.alpha_sigm_in = nn.Parameter(
                torch.randn(self.param_shape, device=self.device)
            )
            self.alpha_sigm_out = nn.Parameter(
                torch.randn(self.param_shape, device=self.device)
            )
        else:
            self.alpha_in = nn.Parameter(
                torch.randn(self.param_shape, device=self.device)
            )
            self.alpha_out = nn.Parameter(
                torch.randn(self.param_shape, device=self.device)
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        res = x
        if self.act_type == "gated":
            z_in = self.conv_in(x)
            z_tanh = self.alpha_tanh_out * self.tanh(
                self.alpha_tanh_in * z_in[:, 0 : int(self.out_channels / 2), :]
            )
            z_sigm = self.alpha_sigm_out * self.sigm(
                self.alpha_sigm_in * z_in[:, 0 : int(self.out_channels / 2), :]
            )
            z_gated = z_tanh * z_sigm
            z_out = self.conv_out(z_gated) + res
            return z_gated, z_out
        else:
            z_tanh = self.alpha_out * self.act(self.alpha_in * self.conv_in(x))
            z_out = self.conv_out(z_tanh) + res
            return z_tanh, z_out
