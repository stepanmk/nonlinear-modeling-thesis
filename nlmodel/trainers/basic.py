import json
import os
from datetime import timedelta
from time import time
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from autoclip.torch import QuantileClip
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses.freq import STFT
from ..losses.time import ESR


class BasicTrainer:

    def __init__(
        self,
        net: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader = None,
        train_loss_name: str = "ESR",
        losses: dict = {
            "ESR": ESR(),
            "MSE": nn.MSELoss(),
            "MAE": nn.L1Loss(),
            "STFT": STFT(),
        },
        device: str = "cuda",
        logs_path: str = "./custom_logs/",
        run_name: str = "double_trunc",
        max_epochs: int = 350,
        val_freq: int = 2,
        lr: float = 0.004,
        lr_reduce_factor: float = 0.5,
        lr_reduce_patience: int = 10,
        stop_patience: int = 50,
        batch_resize: bool = False,
        verbose: bool = True,
    ):
        self.net = net
        self.device = device
        self.logs_path = logs_path
        self.run_name = run_name
        self.model_name = f"{run_name}_{self.net.model_name}"
        self.max_epochs = max_epochs
        self.val_freq = val_freq
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.bs_train = train_loader.batch_size
        self.bs_val = val_loader.batch_size
        if self.test_loader:
            self.bs_test = test_loader.batch_size
        self.batch_resize = batch_resize
        self.verbose = verbose

        # optimizer
        self.lr = lr
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_reduce_patience = lr_reduce_patience
        self.stop_patience = stop_patience
        self.stop_count = 0
        self.optimizer = None
        self.scheduler = None
        self.init_optim()

        # train and test losses
        self.losses = losses
        self.train_loss_name = train_loss_name
        self.train_loss = self.losses[train_loss_name]
        self.losses.pop(train_loss_name)

        # loss tracking
        self.train_ep_losses = []
        self.val_ep_losses = []
        self.best_val_loss = float("inf")
        self.test_loss = None
        self.losses_dict = {}
        self.create_losses_dict()
        self.time_started = None
        # grad tracking
        self.grad_norms = []
        self.clipped_grad_norms = []

        # log paths
        self.ckpt_path = None
        self.summary_path = None
        self.create_logs_path()

    def init_optim(self) -> None:
        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=self.lr_reduce_factor,
            patience=self.lr_reduce_patience,
            verbose=True,
        )

    def early_stopping(self, current_val_loss: float, ep: int) -> bool:
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.save_checkpoint(ep)
            self.stop_count = 0
        else:
            self.stop_count += 1
            print(
                f"Val loss did not get better. Patience: {self.stop_count}/{self.stop_patience}."
            )
        return False if self.stop_count > self.stop_patience else True

    def save_checkpoint(self, ep: int) -> None:
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": self.net.state_dict(),
                "loss": self.best_val_loss,
            },
            os.path.join(f"{self.ckpt_path}/", "best_model.ckpt"),
        )

    def save_loss_plot(self, ep: int) -> None:
        ep_axis = np.linspace(1, ep + 1, ep + 1, dtype=int)
        val_ep_axis = ep_axis[1 :: self.val_freq].astype(int)
        if len(ep_axis) == len(self.train_ep_losses) and len(val_ep_axis) == len(
            self.val_ep_losses
        ):
            plt.figure(1, figsize=(6, 3))
            plt.plot(ep_axis, self.train_ep_losses)
            plt.plot(val_ep_axis, self.val_ep_losses)
            plt.xlabel("Epoch")
            plt.ylabel(self.train_loss_name)
            plt.xlim([1, self.max_epochs])
            plt.ylim([0, 0.5])
            plt.grid(which="both")
            plt.tight_layout()
            plt.savefig(os.path.join(f"{self.summary_path}/", f"{ep + 1}_losses.png"))
            plt.close(1)

    def save_grad_norm_plot(self) -> None:
        plt.figure(1, figsize=(6, 3))
        plt.plot(self.grad_norms)
        plt.plot(self.clipped_grad_norms)
        plt.xlabel("Iteration")
        plt.ylabel("Grad l2 norm")
        plt.grid(which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(f"{self.summary_path}/", f"grad_norms.png"))
        plt.close(1)

    def create_losses_dict(self) -> None:
        for loss_name in self.losses:
            self.losses_dict[loss_name] = {"val": [], "test": []}

    def create_logs_path(self) -> None:
        self.ckpt_path = os.path.join(self.logs_path, f"{self.model_name}/checkpoints")
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.summary_path = os.path.join(self.logs_path, f"{self.model_name}/summary")
        os.makedirs(self.summary_path, exist_ok=True)

    def get_empty_loss_dict(self) -> dict:
        l_dict = {}
        for loss_name in self.losses:
            l_dict[loss_name] = 0
        return l_dict

    def calc_losses(self, l_dict: dict, num_batches: int, stage: str = "val") -> None:
        for loss_name in l_dict:
            val_loss = l_dict[loss_name] / num_batches
            self.losses_dict[loss_name][stage].append(round(val_loss.item(), 5))

    def train(self) -> None:
        self.time_started = time()
        for ep in range(self.max_epochs):
            self.train_epoch(ep)
            if (ep + 1) % self.val_freq == 0:
                continue_flag = self.val_epoch(ep)
                delta_t = time() - self.time_started
                print(f"Time elapsed: {timedelta(seconds=round(delta_t))}\n")
                if not continue_flag:
                    break
        self.losses_dict[self.train_loss_name] = {"val": self.val_ep_losses, "test": []}
        # load best checkpoint
        best_checkpoint = torch.load(
            os.path.join(f"{self.ckpt_path}/", "best_model.ckpt")
        )
        self.net.load_state_dict(best_checkpoint["model_state_dict"])
        # final testing
        if self.test_loader:
            self.test_epoch(ep=best_checkpoint["epoch"])
            self.losses_dict[self.train_loss_name]["test"].append(self.test_loss)
        self.losses_dict["grad_norms"] = self.grad_norms
        self.losses_dict["clipped_grad_norms"] = self.clipped_grad_norms
        self.losses_dict["train_ep_losses"] = self.train_ep_losses
        self.losses_dict["avg_grad_norm"] = sum(self.grad_norms) / len(self.grad_norms)
        self.save_grad_norm_plot()
        with open(os.path.join(f"{self.summary_path}/", "losses.json"), "w") as f:
            json.dump(self.losses_dict, f, indent=4)
        print("Training finished.")

    def _train_batch(self, batch: tuple) -> torch.Tensor:
        x_inp = batch[0].to(self.device)
        y_tgt = batch[1].to(self.device)
        y_hat = self.net(x_inp)
        batch_loss = self.train_loss(y_hat, y_tgt)
        batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return batch_loss

    def _val_test(
        self, num_batches: int, stage: str = "val"
    ) -> Union[torch.Tensor, dict]:
        l_dict = self.get_empty_loss_dict()
        loader = self.val_loader if stage == "val" else self.test_loader
        ep_loss = 0
        for batch in loader:
            x_inp = batch[0].to(self.device)
            y_tgt = batch[1].to(self.device)
            with torch.no_grad():
                y_hat = self.net(x_inp)
            batch_loss = self.train_loss(y_hat, y_tgt)
            ep_loss += batch_loss
            # compute additional losses
            for loss_name in l_dict:
                l_dict[loss_name] += self.losses[loss_name](y_hat, y_tgt)
        return ep_loss / num_batches, l_dict

    def train_epoch(self, ep: int) -> None:
        self.net.train()
        num_batches = len(self.train_loader)
        if self.batch_resize:
            self.net.change_batch_size(self.train_loader.batch_size)
        ep_loss = 0
        for batch in tqdm(
            self.train_loader,
            f"Ep. {ep + 1}",
            total=num_batches,
            ncols=80,
            unit=" batches",
            leave=False,
        ):
            ep_loss += self._train_batch(batch)
        avg_ep_loss = ep_loss / num_batches
        self.train_ep_losses.append(round(avg_ep_loss.item(), 5))
        self.save_loss_plot(ep=ep)
        if self.verbose:
            print(
                f"Ep. {ep + 1} train {self.train_loss_name}:",
                round(avg_ep_loss.item(), 3),
            )

    def compute_norm(self, clipped: bool):
        grads = [
            param.grad.detach().flatten()
            for param in self.net.parameters()
            if param.grad is not None
        ]
        if clipped:
            self.clipped_grad_norms.append(torch.cat(grads).norm().item())
        else:
            self.grad_norms.append(torch.cat(grads).norm().item())

    def val_epoch(self, ep: int) -> bool:
        self.net.eval()
        num_batches = len(self.val_loader)
        if self.batch_resize:
            self.net.change_batch_size(self.val_loader.batch_size)
        avg_ep_loss, l_dict = self._val_test(num_batches, "val")
        # lr scheduler
        self.scheduler.step(avg_ep_loss)
        self.val_ep_losses.append(round(avg_ep_loss.item(), 5))
        # save additional val losses
        self.calc_losses(l_dict, num_batches, "val")
        if self.verbose:
            print(
                f"Ep. {ep + 1} val {self.train_loss_name}:",
                round(avg_ep_loss.item(), 3),
            )
        return self.early_stopping(current_val_loss=avg_ep_loss.item(), ep=ep)

    def test_epoch(self, ep: int) -> None:
        self.net.eval()
        num_batches = len(self.test_loader)
        if self.batch_resize:
            self.net.change_batch_size(self.test_loader.batch_size)
        avg_ep_loss, l_dict = self._val_test(num_batches, "test")
        self.test_loss = round(avg_ep_loss.item(), 5)
        # save final test losses
        self.calc_losses(l_dict, num_batches, "test")
        if self.verbose:
            print(
                f"Ep. {ep + 1} test {self.train_loss_name}:",
                round(avg_ep_loss.item(), 3),
            )


class BasicTruncTrainer(BasicTrainer):

    def __init__(
        self,
        clip_grad: bool = True,
        cond: bool = True,
        warmup: bool = False,
        warmup_steps: int = 1000,
        trunc_steps: int = 2048,
        val_steps: int = 22050,
        max_norm: float = 1.0,
        clip_method: str = "norm",
        **kwargs,
    ):
        super().__init__(**kwargs, batch_resize=True)
        self.clip_grad = clip_grad
        self.cond = cond
        self.trunc_steps = trunc_steps
        self.val_steps = val_steps
        self.test_steps = val_steps
        self.warmup = warmup
        self.warmup_steps = warmup_steps
        self.max_norm = max_norm
        self.clip_method = clip_method
        if self.clip_method == "auto":
            self.clipper = QuantileClip(
                self.net.parameters(), quantile=0.1, history_length=500
            )

    def _train_batch(self, batch: tuple) -> torch.Tensor:
        x_inp = batch[0].to(self.device)
        y_tgt = batch[1].to(self.device)
        if self.cond:
            c = batch[2].to(self.device)
        seg_samples = self.train_loader.dataset.seg_samples
        start_sample = 0
        self.net.reset_states()
        # warmup hidden states
        if self.warmup:
            start_sample = self.warmup_steps
            with torch.no_grad():
                if self.cond:
                    y_hat = self.net(x_inp[:, 0 : self.warmup_steps, :], c)
                else:
                    y_hat = self.net(x_inp[:, 0 : self.warmup_steps, :])
        batch_loss = 0
        for n in range(start_sample, seg_samples, self.trunc_steps):
            if self.cond:
                y_hat = self.net(x_inp[:, n : n + self.trunc_steps, :], c)
            else:
                y_hat = self.net(x_inp[:, n : n + self.trunc_steps, :])
            tbptt_loss = self.train_loss(y_hat, y_tgt[:, n : n + self.trunc_steps, :])
            tbptt_loss.backward()
            batch_loss += tbptt_loss
            self.compute_norm(clipped=False)
            if self.clip_grad:
                if self.clip_method == "auto":
                    self.clipper.step()
                else:
                    clip_grad_norm_(self.net.parameters(), max_norm=self.max_norm)
                self.compute_norm(clipped=True)
            self.optimizer.step()
            self.net.detach_states()
            self.optimizer.zero_grad()
        return batch_loss / (seg_samples / self.trunc_steps)

    def _val_test(
        self, num_batches: int, stage: str = "val"
    ) -> Union[torch.Tensor, dict]:
        l_dict = self.get_empty_loss_dict()
        loader = self.val_loader if stage == "val" else self.test_loader
        seg_samples = loader.dataset.seg_samples
        ep_loss = 0
        per_cond_losses = []
        per_cond_stft = []
        for batch in loader:
            x_inp = batch[0].to(self.device)
            y_tgt = batch[1].to(self.device)
            if self.cond:
                c = batch[2].to(self.device)
            self.net.reset_states()
            batch_loss = 0
            y_hat_list = []
            for n in range(0, seg_samples, self.val_steps):
                with torch.no_grad():
                    if self.cond:
                        y_hat = self.net(x_inp[:, n : n + self.val_steps, :], c)
                    else:
                        y_hat = self.net(x_inp[:, n : n + self.val_steps, :])
                y_hat_list.append(y_hat)
                seg_loss = self.train_loss(y_hat, y_tgt[:, n : n + self.val_steps, :])
                batch_loss += seg_loss
            ep_loss += batch_loss / (seg_samples / self.val_steps)
            if stage == "test":
                per_cond_losses.append(
                    (batch_loss / (seg_samples / self.val_steps)).item()
                )
            # compute additional losses
            for loss_name in l_dict:
                l_dict[loss_name] += self.losses[loss_name](
                    torch.cat(y_hat_list, dim=1), y_tgt
                )
                if stage == "test" and loss_name == "STFT":
                    per_cond_stft.append(
                        (
                            self.losses[loss_name](torch.cat(y_hat_list, dim=1), y_tgt)
                        ).item()
                    )
        if stage == "test" and self.cond:
            self.losses_dict["cond"] = self.test_loader.dataset.settings_names
            self.losses_dict["per_cond_ESR"] = per_cond_losses
            self.losses_dict["per_cond_STFT"] = per_cond_stft
        return ep_loss / num_batches, l_dict
