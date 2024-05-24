import os
from typing import Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


class SingleTargetDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        inp_file: str,
        tgt_file: str,
        start_sec: float,
        end_sec: float,
        seg_len: float,
        fs: int,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.inp_file = inp_file
        self.tgt_file = tgt_file
        self.start_sec = int(start_sec)
        self.end_sec = int(end_sec)
        self.seg_len = seg_len
        self.fs = fs
        self.seg_samples = int(self.seg_len * self.fs)

        inp_path = os.path.join(self.data_dir, self.inp_file)
        tgt_path = os.path.join(self.data_dir, self.tgt_file)
        inp_data, _ = torchaudio.load(inp_path, channels_first=False)
        tgt_data, _ = torchaudio.load(tgt_path, channels_first=False)
        inp_data = inp_data[self.start_sec * self.fs : self.end_sec * self.fs, :]
        tgt_data = tgt_data[self.start_sec * self.fs : self.end_sec * self.fs, :]

        self.input = inp_data
        self.target = tgt_data
        self.num_segments = self.input.shape[0] // self.seg_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = index * self.seg_samples
        stop = (index + 1) * self.seg_samples
        return self.input[start:stop, :], self.target[start:stop, :]

    def __len__(self) -> int:
        return self.num_segments


class SingleFileLoader:

    def __init__(
        self,
        data_dir: str,
        inp_file: str,
        tgt_file: str,
        start_sec: float,
        end_sec: float,
        seg_len: float = 0.5,
        fs: int = 44100,
        batch_size: int = 40,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = SingleTargetDataset(
            data_dir=data_dir,
            inp_file=inp_file,
            tgt_file=tgt_file,
            start_sec=start_sec,
            end_sec=end_sec,
            seg_len=seg_len,
            fs=fs,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last

    def get_loader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )


class AmpAudioDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        file_names: Tuple,
        start_sec: float,
        end_sec: float,
        segment_length_seconds: float = 0.5,
        use_gain_value: bool = False,
        cond_type: str = "labels",
        fs: int = 44100,
    ):
        self.inputs = []
        self.targets = []
        self.settings = []
        self.settings_names = []
        self.fs = fs
        self.seg_samples = int(segment_length_seconds * self.fs)
        self.cond_type = cond_type
        self.use_gain_value = use_gain_value

        for i, target_file in enumerate(file_names):
            input_file = target_file[:-14] + "input.wav"
            inp_data, self.fs = torchaudio.load(
                os.path.join(data_dir, input_file), channels_first=False
            )
            tgt_data, self.fs_t = torchaudio.load(
                os.path.join(data_dir, target_file), channels_first=False
            )
            settings = None
            if end_sec is None:
                inp_data = inp_data[int(start_sec * self.fs) :, :]
                tgt_data = tgt_data[int(start_sec * self.fs) :, :]
            else:
                inp_data = inp_data[
                    int(start_sec * self.fs) : int(end_sec * self.fs), :
                ]
                tgt_data = tgt_data[
                    int(start_sec * self.fs) : int(end_sec * self.fs), :
                ]
            assert self.fs == self.fs_t
            self.inputs.append(inp_data)
            self.targets.append(tgt_data)
            settings_string = target_file.split("/")[0].split("_")
            self.settings_names.append("_".join(settings_string))
            if self.cond_type == "labels":
                if len(settings_string) == 1:
                    settings = torch.zeros(1)
                    settings[0] = float(settings_string[0][1:]) / 10
                else:
                    if self.use_gain_value:
                        settings = torch.zeros(4)
                        settings[0] = float(settings_string[0][1:]) / 10
                        settings[1] = float(settings_string[1][1:]) / 10
                        settings[2] = float(settings_string[2][1:]) / 10
                        settings[3] = float(settings_string[3][1:]) / 10
                    else:
                        settings = torch.zeros(3)
                        settings[0] = float(settings_string[0][1:]) / 10
                        settings[1] = float(settings_string[1][1:]) / 10
                        settings[2] = float(settings_string[2][1:]) / 10
            if self.cond_type == "onehot":
                settings = torch.zeros(len(file_names))
                settings[i] = 1.0
            self.settings.append(settings)
        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)
        self.num_segments_in_cond = self.inputs.shape[1] // self.seg_samples
        self.num_segments = len(self.settings) * self.num_segments_in_cond
        self.num_conds = len(self.settings)
        self.segs_per_cond = self.num_segments // self.num_conds

    def __getitem__(self, index):
        cond_val = index // self.segs_per_cond
        index = index % self.segs_per_cond
        start = index * self.seg_samples
        stop = (index + 1) * self.seg_samples
        return (
            self.inputs[cond_val, start:stop, :],
            self.targets[cond_val, start:stop, :],
            self.settings[cond_val],
        )

    def __len__(self):
        return self.num_segments


class AmpLoader:

    def __init__(
        self,
        data_dir: str,
        file_names: tuple,
        start_sec: float,
        end_sec: float,
        seg_len: float = 0.5,
        fs: int = 44100,
        batch_size: int = 80,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        use_gain_value: bool = False,
    ):
        self.dataset = AmpAudioDataset(
            data_dir=data_dir,
            file_names=file_names,
            start_sec=start_sec,
            end_sec=end_sec,
            segment_length_seconds=seg_len,
            use_gain_value=use_gain_value,
            fs=fs,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last

    def get_loader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )


class PedalAudioDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        file_names: Tuple,
        start_sec: float,
        end_sec: float,
        segment_length_seconds: float = 0.5,
        use_gain_value: bool = False,
        cond_type: str = "labels",
        fs: int = 44100,
    ):
        self.inputs = []
        self.targets = []
        self.settings = []
        self.settings_names = []
        self.fs = fs
        self.seg_samples = int(segment_length_seconds * self.fs)
        self.cond_type = cond_type
        self.use_gain_value = use_gain_value

        for i, target_file in enumerate(file_names):
            input_file = "input.wav"
            inp_data, self.fs = torchaudio.load(
                os.path.join(data_dir, input_file), channels_first=False
            )
            tgt_data, self.fs_t = torchaudio.load(
                os.path.join(data_dir, target_file), channels_first=False
            )
            settings = None
            if end_sec is None:
                inp_data = inp_data[int(start_sec * self.fs) :, :]
                tgt_data = tgt_data[int(start_sec * self.fs) :, :]
            else:
                inp_data = inp_data[
                    int(start_sec * self.fs) : int(end_sec * self.fs), :
                ]
                tgt_data = tgt_data[
                    int(start_sec * self.fs) : int(end_sec * self.fs), :
                ]
            assert self.fs == self.fs_t
            self.inputs.append(inp_data)
            self.targets.append(tgt_data)
            settings_string = target_file.split("-")[0].replace("T", "")
            self.settings_names.append(f"T_{settings_string}")
            if self.cond_type == "labels":
                settings = torch.zeros(1)
                settings[0] = float(settings_string) / 10

            if self.cond_type == "onehot":
                settings = torch.zeros(len(file_names))
                settings[i] = 1.0
            self.settings.append(settings)
        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)
        self.num_segments_in_cond = self.inputs.shape[1] // self.seg_samples
        self.num_segments = len(self.settings) * self.num_segments_in_cond
        self.num_conds = len(self.settings)
        self.segs_per_cond = self.num_segments // self.num_conds

    def __getitem__(self, index):
        cond_val = index // self.segs_per_cond
        index = index % self.segs_per_cond
        start = index * self.seg_samples
        stop = (index + 1) * self.seg_samples
        return (
            self.inputs[cond_val, start:stop, :],
            self.targets[cond_val, start:stop, :],
            self.settings[cond_val],
        )

    def __len__(self):
        return self.num_segments


class PedalLoader:

    def __init__(
        self,
        data_dir: str,
        file_names: tuple,
        start_sec: float,
        end_sec: float,
        seg_len: float = 0.5,
        fs: int = 44100,
        batch_size: int = 80,
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        use_gain_value: bool = False,
    ):
        self.dataset = PedalAudioDataset(
            data_dir=data_dir,
            file_names=file_names,
            start_sec=start_sec,
            end_sec=end_sec,
            segment_length_seconds=seg_len,
            use_gain_value=use_gain_value,
            fs=fs,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last

    def get_loader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )
