import torch
import torch.nn as nn

from ..layers.ds1_tone import DS1
from ..layers.rat_tone import RAT
from .rnn import RNNBase


class PedalModel(nn.Module):

    def __init__(
        self,
        freeze_cond_block: bool = False,
        batch_size: int = 80,
        device: str = "cuda",
        pedal: str = "ds1",
    ):
        super().__init__()
        self.model_name = f"{pedal}_pedal"
        self.freeze_cond_block = freeze_cond_block
        self.batch_size = batch_size
        self.device = device
        self.pedal = pedal

        self.preamp = RNNBase(
            cell_type="GRU",
            input_size=1,
            hidden_size=32,
            batch_size=self.batch_size,
            device=self.device,
            skip=True,
        )
        if self.pedal == "ds1":
            self.tonestack = DS1(
                batch_size=self.batch_size,
                device=self.device,
                freeze_cond_block=self.freeze_cond_block,
            )
        else:
            self.tonestack = RAT(
                batch_size=self.batch_size,
                device=self.device,
                freeze_cond_block=self.freeze_cond_block,
            )

    def change_batch_size(self, new_batch_size: int) -> None:
        self.batch_size = new_batch_size
        self.preamp.change_batch_size(new_batch_size=new_batch_size)
        self.preamp.reset_states()
        self.tonestack.change_batch_size(new_batch_size=new_batch_size)
        self.tonestack.reset_states()

    def reset_states(self) -> None:
        self.preamp.reset_states()
        self.tonestack.reset_states()

    def detach_states(self) -> None:
        self.preamp.detach_states()
        self.tonestack.detach_states()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        preamp_out = self.preamp(x)
        tonestack_out = self.tonestack(preamp_out, c)
        return tonestack_out
