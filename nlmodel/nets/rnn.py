import torch
import torch.nn as nn

from ..layers.conv import Conv1dStateful


class RNNBase(nn.Module):

    def __init__(
        self,
        cell_type: str = "RNN",
        input_size: int = 1,
        hidden_size: int = 32,
        batch_size: int = 40,
        skip: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.cell_type = cell_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.skip = skip
        self.device = device
        self.rnn = None
        self.hidden_state = None
        self.init_rnn()
        self.linear = nn.Linear(
            in_features=self.hidden_size, out_features=1, device=self.device
        )
        self.model_name = f"{cell_type.lower()}_{hidden_size}"

    def init_rnn(self) -> None:
        if self.cell_type == "RNN":
            self.rnn = nn.RNN(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                batch_first=True,
                device=self.device,
            )
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                batch_first=True,
                device=self.device,
            )
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                batch_first=True,
                device=self.device,
            )

    def change_batch_size(self, new_batch_size: int) -> None:
        self.batch_size = new_batch_size
        self.reset_states()

    def reset_states(self) -> None:
        if self.cell_type == "LSTM":
            self.hidden_state = (
                torch.zeros((1, self.batch_size, self.hidden_size), device=self.device),
                torch.zeros((1, self.batch_size, self.hidden_size), device=self.device),
            )
        else:
            self.hidden_state = torch.zeros(
                (1, self.batch_size, self.hidden_size), device=self.device
            )

    def detach_states(self) -> None:
        if self.cell_type == "LSTM":
            hidden_state = list(self.hidden_state)
            hidden_state[0] = self.hidden_state[0].detach()
            hidden_state[1] = self.hidden_state[1].detach()
            self.hidden_state = tuple(hidden_state)
        else:
            self.hidden_state = self.hidden_state.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x[:, :, 0:1]
        x, self.hidden_state = self.rnn(x, self.hidden_state)
        x = self.linear(x)
        if self.skip:
            x = x + res
        return x


class LinRNN(nn.Module):

    def __init__(
        self,
        cell_type: str = "RNN",
        hidden_size: int = 32,
        batch_size: int = 40,
        rnn_skip: bool = False,
        global_skip: bool = False,
        device: str = "cuda",
        lin_size: int = 4,
    ):
        super().__init__()
        self.rnn = RNNBase(
            cell_type=cell_type,
            input_size=lin_size,
            hidden_size=hidden_size,
            batch_size=batch_size,
            skip=rnn_skip,
            device=device,
        )
        self.lin_in = nn.Linear(in_features=1, out_features=lin_size, device=device)
        self.global_skip = global_skip
        self.model_name = f"lin_{lin_size}_{cell_type.lower()}_{hidden_size}"

    def change_batch_size(self, new_batch_size) -> None:
        self.rnn.change_batch_size(new_batch_size=new_batch_size)

    def reset_states(self) -> None:
        self.rnn.reset_states()

    def detach_states(self) -> None:
        self.rnn.detach_states()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin_out = self.lin_in(x)
        out = self.rnn(lin_out)
        if self.global_skip:
            out = out + x
        return out


class TCNRNN(nn.Module):

    def __init__(
        self,
        cell_type: str = "RNN",
        hidden_size: int = 32,
        batch_size: int = 40,
        rnn_skip: bool = False,
        global_skip: bool = True,
        device: str = "cuda",
        out_channels: int = 1,
        kernel_size: int = 3,
        num_layers: int = 4,
    ):

        super().__init__()
        self.rnn = RNNBase(
            cell_type=cell_type,
            input_size=1,
            hidden_size=hidden_size,
            batch_size=batch_size,
            skip=rnn_skip,
            device=device,
        )
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.conv_blocks = nn.ModuleList()
        self.dilations = 1 * [2**i * 1 for i in range(self.num_layers)]
        for i, dilation in enumerate(self.dilations):
            in_channels = 1 if i == 0 else self.out_channels
            self.conv_blocks.append(
                Conv1dStateful(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    batch_size=batch_size,
                )
            )
        self.conv_out = nn.Conv1d(
            in_channels=self.out_channels * len(self.dilations),
            out_channels=1,
            kernel_size=1,
            device=device,
        )
        self.global_skip = global_skip
        self.model_name = f"tcn_{cell_type.lower()}_{hidden_size}_ch{out_channels}"
        self.model_name += f"_ks{kernel_size}_d{dilation}"

    def change_batch_size(self, new_batch_size: int) -> None:
        for conv in self.conv_blocks:
            conv.change_batch_size(new_batch_size)
        self.rnn.change_batch_size(new_batch_size=new_batch_size)

    def reset_states(self) -> None:
        for conv in self.conv_blocks:
            conv.reset_state()
        self.rnn.reset_states()

    def detach_states(self) -> None:
        for conv in self.conv_blocks:
            conv.detach_state()
        self.rnn.detach_states()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        # make dims consistent with recurrent nets
        z_out = x.permute((0, 2, 1))
        for block in self.conv_blocks:
            z_out = block(z_out)
            skips.append(z_out)
        out = self.rnn(torch.permute(self.conv_out(torch.cat(skips, dim=1)), (0, 2, 1)))
        if self.global_skip:
            out = out + x
        return out


class ConvRNN(nn.Module):

    def __init__(
        self,
        cell_type: str = "RNN",
        hidden_size: int = 32,
        batch_size: int = 40,
        rnn_skip: bool = False,
        global_skip: bool = True,
        device: str = "cuda",
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
    ):

        super().__init__()
        self.rnn = RNNBase(
            cell_type=cell_type,
            input_size=out_channels,
            hidden_size=hidden_size,
            batch_size=batch_size,
            skip=rnn_skip,
            device=device,
        )
        self.conv_in = Conv1dStateful(
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            batch_size=batch_size,
        )
        self.global_skip = global_skip
        self.model_name = f"conv_{cell_type.lower()}_{hidden_size}_ch{out_channels}"
        self.model_name += f"_ks{kernel_size}_d{dilation}"

    def change_batch_size(self, new_batch_size: int) -> None:
        self.conv_in.change_batch_size(new_batch_size=new_batch_size)
        self.rnn.change_batch_size(new_batch_size=new_batch_size)

    def reset_states(self) -> None:
        self.conv_in.reset_state()
        self.rnn.reset_states()

    def detach_states(self) -> None:
        self.conv_in.detach_state()
        self.rnn.detach_states()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv_in(x.permute((0, 2, 1)))
        out = self.rnn(conv_out.permute((0, 2, 1)))
        if self.global_skip:
            out = out + x
        return out
