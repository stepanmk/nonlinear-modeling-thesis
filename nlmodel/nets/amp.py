import torch
import torch.nn as nn

from ..layers.preamp_gain import PreampGainLayer
from ..layers.tonestack import ToneStack, ToneStackSimple
from .rnn import RNNBase


class AmpModel(nn.Module):

    def __init__(
        self,
        freeze_cond_block: bool = False,
        batch_size: int = 80,
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = f"amp"
        self.freeze_cond_block = freeze_cond_block
        self.batch_size = batch_size
        self.device = device

        self.preamp = RNNBase(
            cell_type="LSTM",
            input_size=1,
            hidden_size=40,
            batch_size=self.batch_size,
            device=self.device,
            skip=True,
        )

        self.tonestack = ToneStackSimple(
            batch_size=self.batch_size,
            device=self.device,
            freeze_cond_block=self.freeze_cond_block,
        )

        self.poweramp = RNNBase(
            cell_type="GRU",
            input_size=1,
            hidden_size=8,
            batch_size=self.batch_size,
            device=self.device,
            skip=True,
        )

    def change_batch_size(self, new_batch_size: int) -> None:
        self.batch_size = new_batch_size
        self.preamp.change_batch_size(new_batch_size=new_batch_size)
        self.preamp.reset_states()
        self.tonestack.change_batch_size(new_batch_size=new_batch_size)
        self.tonestack.reset_states()
        self.poweramp.change_batch_size(new_batch_size=new_batch_size)
        self.poweramp.reset_states()

    def reset_states(self) -> None:
        self.preamp.reset_states()
        self.tonestack.reset_states()
        self.poweramp.reset_states()

    def detach_states(self) -> None:
        self.preamp.detach_states()
        self.tonestack.detach_states()
        self.poweramp.detach_states()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        preamp_out = self.preamp(x)
        tonestack_out = self.tonestack(preamp_out, c)
        poweramp_out = self.poweramp(tonestack_out)
        return poweramp_out


class AmpModelGain(nn.Module):

    def __init__(
        self,
        freeze_cond_block: bool = False,
        batch_size: int = 80,
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = f"amp"
        self.freeze_cond_block = freeze_cond_block
        self.batch_size = batch_size
        self.device = device

        self.preamp = RNNBase(
            cell_type="LSTM",
            input_size=2,
            hidden_size=40,
            batch_size=self.batch_size,
            device=self.device,
            skip=True,
        )

        self.tonestack = ToneStackSimple(
            batch_size=self.batch_size,
            device=self.device,
            freeze_cond_block=self.freeze_cond_block,
        )

        self.poweramp = RNNBase(
            cell_type="GRU",
            input_size=1,
            hidden_size=8,
            batch_size=self.batch_size,
            device=self.device,
            skip=True,
        )

    def change_batch_size(self, new_batch_size: int) -> None:
        self.batch_size = new_batch_size
        self.preamp.change_batch_size(new_batch_size=new_batch_size)
        self.preamp.reset_states()
        self.tonestack.change_batch_size(new_batch_size=new_batch_size)
        self.tonestack.reset_states()
        self.poweramp.change_batch_size(new_batch_size=new_batch_size)
        self.poweramp.reset_states()

    def reset_states(self) -> None:
        self.preamp.reset_states()
        self.tonestack.reset_states()
        self.poweramp.reset_states()

    def detach_states(self) -> None:
        self.preamp.detach_states()
        self.tonestack.detach_states()
        self.poweramp.detach_states()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, c[:, 3:4].repeat((1, x.shape[1])).unsqueeze(-1)], dim=-1)
        preamp_out = self.preamp(x)
        tonestack_out = self.tonestack(preamp_out, c[:, 0:3])
        poweramp_out = self.poweramp(tonestack_out)
        return poweramp_out
