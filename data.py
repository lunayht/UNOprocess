import os
from typing import Tuple, Union
from functools import lru_cache

import cv2
import librosa
import numpy as np
import torch
import torchaudio.transforms as AT
import torchvision.transforms as VT
from torch.utils.data import Dataset

from configs_.arguments import FrontendArguments

cv2.setNumThreads(0)


class AudioDataset(Dataset):
    r"""Dataset class that loads audio file and returns a tuple of sample (fbank, label) when called."""

    def __init__(
        self,
        root: str,
        signals: np.ndarray,
        labels: np.ndarray,
        frontend_configs: FrontendArguments = None,
        mode: str = "train",
        get_weights: bool = False,
    ) -> None:
        super(AudioDataset, self).__init__()

        assert len(signals) and len(labels) > 0, "signals and/or labels list is empty."
        assert mode in [
            "train",
            "test",
        ], "mode must be either 'train' or 'test'."

        self.root = root
        self.signals = signals
        self.labels = labels
        self.frontend_configs = frontend_configs
        self.mode = mode

        self.normed_weights = self._get_weights() if get_weights else None
        self.transform = self._get_transform()

    def _get_weights(self) -> torch.Tensor:
        r"""Get class weights for imbalanced dataset."""
        counts = np.bincount(self.labels)
        normed_weights = 1.0 - np.divide(counts, counts.sum())
        return torch.as_tensor(normed_weights, dtype=torch.float32)

    def _get_transform(self) -> VT.Compose:
        r"""Returns transform for audio data."""
        transform = [VT.ToTensor()]
        if self.mode == "train":
            if self.frontend_configs.spec_aug_kwargs["tm"]:
                transform.append(
                    AT.TimeMasking(self.frontend_configs.spec_aug_kwargs["tm"])
                )
            if self.frontend_configs.spec_aug_kwargs["fm"]:
                transform.append(
                    AT.FrequencyMasking(self.frontend_configs.spec_aug_kwargs["fm"])
                )
            transform.append(
                VT.RandomCrop(
                    size=(
                        self.frontend_configs.n_mels,
                        self.frontend_configs.spec_width,
                    ),
                    pad_if_needed=True,
                    padding_mode="constant",
                )
            )
            if self.frontend_configs.wav_aug_kwargs:
                raise NotImplementedError  # add audio augmentation manually if needed.
        else:
            transform.append(
                VT.CenterCrop(
                    size=(
                        self.frontend_configs.n_mels,
                        self.frontend_configs.spec_width,
                    )
                )
            )
        return VT.Compose(transform)

    def __len__(self) -> int:
        return len(self.signals)

    def _sig2fbank(self, sig: np.ndarray) -> torch.Tensor:
        r"""Converts audio signal to grayscale fbank using librosa and performs transformation to `torch.Tensor`.
        Arg:
            sig: Audio signal in `np.ndarray`.
        Returns:
            fbank: Filterbank in `torch.Tensor` of shape `(n_mels, spec_width)`.
        """
        M = librosa.feature.melspectrogram(
            y=sig,
            sr=self.frontend_configs.target_sample_rate,
            n_mels=self.frontend_configs.n_mels,
            n_fft=self.frontend_configs.nfft,
            hop_length=self.frontend_configs.hop_length,
        )
        M_db = librosa.power_to_db(M, ref=np.max)
        fbank = (M_db - M_db.min()) / (M_db.max() - M_db.min())
        if self.mode == "test":
            fbank = self._right_pad_if_necessary(fbank)
        fbank = self.transform(fbank)
        return fbank

    @lru_cache(maxsize=None)
    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Union[int, Tuple[int, int, float]]]:
        wav_file = os.path.join(self.root, self.signals[idx])
        label = self.labels[idx]

        if self.mode == "train" and self.frontend_configs.mixup_alpha:
            mix_sample_idx = np.random.randint(0, len(self.signals) - 1)
            mix_wav_file = os.path.join(self.root, self.signals[mix_sample_idx])
            label_2 = self.labels[mix_sample_idx]
            mixup_sig, mixup_lambda = self._get_mixup_sig(
                wav_file_1=wav_file, wav_file_2=mix_wav_file
            )
            mixup_fbank = self._sig2fbank(mixup_sig)
            return mixup_fbank, label, label_2, mixup_lambda
        else:
            sig, _ = librosa.load(wav_file, sr=self.frontend_configs.target_sample_rate)
            fbank = self._sig2fbank(sig)
            return fbank, label

    def _align_time_length(self, sig_1: np.ndarray, sig_2: np.ndarray) -> np.ndarray:
        r"""Pads sig_2 at time axis with sig_1 or truncates it to the target length."""
        sig_1_length = sig_1.shape[0]
        sig_2_length = sig_2.shape[0]
        if sig_2_length < sig_1_length:
            pad_length = (sig_1_length // sig_2_length) + 1
            sig_2 = sig_2.repeat(pad_length)
            sig_2 = sig_2[:sig_1_length]
        elif sig_2_length > sig_1_length:
            start_time = np.random.randint(0, sig_2_length - sig_1_length)
            sig_2 = sig_2[start_time : start_time + sig_1_length]
        return sig_2

    def _get_mixup_sig(
        self, wav_file_1: str, wav_file_2: str
    ) -> Tuple[torch.Tensor, float]:
        r"""Performs mixup on two wav files.
        Args:
            wav_file_1: First wav file.
            wav_file_2: Second wav file.
        Returns:
            Tuple of `(mixup_sig, mixup_lambda)`
        """
        sig_1, _ = librosa.load(wav_file_1, sr=self.frontend_configs.target_sample_rate)
        if np.random.random() < 0.5:
            sig_2, _ = librosa.load(
                wav_file_2, sr=self.frontend_configs.target_sample_rate
            )

            sig_2 = self._align_time_length(sig_1, sig_2)

            mixup_lambda = np.random.beta(
                self.frontend_configs.mixup_alpha, self.frontend_configs.mixup_alpha
            )

            mixup_sig = mixup_lambda * sig_1 + (1 - mixup_lambda) * sig_2
        else:
            mixup_sig = sig_1
            mixup_lambda = 1.0
        return mixup_sig, mixup_lambda

    def _right_pad_if_necessary(self, fbank: np.ndarray) -> np.ndarray:
        r"""Pads fbank at time axis with zeros"""
        _, w = fbank.shape
        if fbank.shape[1] < self.frontend_configs.spec_width:
            w_to_pad = self.frontend_configs.spec_width - w
            padded_fbank = cv2.copyMakeBorder(
                fbank,
                0,
                0,
                0,
                w_to_pad,
                cv2.BORDER_CONSTANT,
                (0, 0),
            )
            return padded_fbank
        else:
            return fbank
