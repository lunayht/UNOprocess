import os
from dataclasses import dataclass, field
from typing import Dict, List, Union

import yaml


@dataclass
class DataArguments:
    dataset: str = field(default="icbhi_t11")

    wavs_dir: str = field(default="icbhi_chunks_16k/")
    label_file: str = field(default="icbhi_t11.txt")

    official_split: str = field(default=None)

    batch_size: int = field(default=32)
    val_batch_size: int = field(default=32)
    num_workers: int = field(default=8)

    num_classes: int = field(default=4)

    kwargs_: field(default_factory=Dict) = None

    def __post_init__(self) -> None:
        if self.kwargs_ is not None:
            for key, value in self.kwargs_.items():
                setattr(self, key, value)


@dataclass
class FrontendArguments:
    frontend_name: str = field(default="mel")

    target_sample_rate: int = field(default=16000)
    n_mels: int = field(default=32)
    nfft: int = field(default=401)
    hop_length: int = field(default=None)
    win_length: int = field(default=None)
    spec_width: int = field(default=512)
    mixup_alpha: float = field(default=None)

    wav_aug_kwargs: field(default_factory=Dict) = None
    spec_aug_kwargs: field(default_factory=Dict) = None
    kwargs_: field(default_factory=Dict) = None

    def __post_init__(self) -> None:
        if self.wav_aug_kwargs is not None:
            for key, value in self.wav_aug_kwargs.items():
                setattr(self, key, value)
        if self.spec_aug_kwargs is not None:
            for key, value in self.spec_aug_kwargs.items():
                setattr(self, key, value)
        if self.kwargs_ is not None:
            for key, value in self.kwargs_.items():
                setattr(self, key, value)


@dataclass
class ModelArguments:
    model_name: str = field(default="ast")

    from_checkpoint: str = field(default=None)
    imgnet_pretrain: bool = field(default=False)

    kwargs_: field(default_factory=Dict) = None

    def __post_init__(self) -> None:
        if self.kwargs_ is not None:
            for key, value in self.kwargs_.items():
                setattr(self, key, value)


@dataclass
class TrainArguments:
    project_name: str = field(default="respiratory-sound-analysis")
    gpus: Union[int, str, List[str], List[int]] = field(default="cpu")
    seed: int = field(default=1234)
    wandb: bool = field(default=False)
    val_check_interval: int = field(default=1)
    save_stats_dir: str = field(default="./checkpoints")
    onnx: bool = field(default=False)

    max_epochs: int = field(default=10)
    lr: float = field(default=1e-4)

    loss: str = field(default="CrossEntropyLoss")
    optimizer: str = field(default="Adam")
    lr_scheduler: str = field(default="CosineAnnealingWarmRestarts")

    kwargs_: field(default_factory=Dict) = None

    def __post_init__(self) -> None:
        if self.kwargs_ is not None:
            for key, value in self.kwargs_.items():
                setattr(self, key, value)


def yaml_to_args(
    arg_type: str = "data",
) -> Union[DataArguments, FrontendArguments, ModelArguments, TrainArguments]:
    r"""Take a yaml file and return the corresponding arguments.

    Args:
        `arg_type`: The type of arguments to return. One of `data`, `frontend`, `model`, `train`. (default: `data`)

    Returns:
        `Union[DataArguments, FrontendArguments, ModelArguments, TrainArguments]`: The corresponding arguments.
    """
    yml_file = os.path.join("./configs_", arg_type + ".yml")

    with open(yml_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    if arg_type == "data":
        return DataArguments(**config)
    elif arg_type == "frontend":
        return FrontendArguments(**config)
    elif arg_type == "model":
        return ModelArguments(**config)
    elif arg_type == "train":
        return TrainArguments(**config)
    else:
        raise ValueError("Invalid arg_type.")
