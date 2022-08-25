import os
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import randomname
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from configs_.arguments import *
from pipeline import UNOModule


class Colors:
    r"""ANSI escape sequence color code constants."""

    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    ENDC = "\033[0m"


def labels_file_to_dict(labels_file: str, sep: str = "\t") -> Dict[str, int]:
    labels_pd = pd.read_csv(labels_file, sep=sep, header=None)
    labels_dict = labels_pd.set_index(0).T.to_dict("records")[0]
    return labels_dict


def set_logger(
    save_dir: str,
    train_configs: TrainArguments,
    model_configs: ModelArguments,
    data_configs: DataArguments,
    frontend_configs: FrontendArguments,
    group_name: str = None,
    name: str = None,
) -> Union[TensorBoardLogger, WandbLogger]:
    r"""Configure logger for the experiment: TensorBoard or Wandb."""
    configs = {
        **train_configs.__dict__,
        **data_configs.__dict__,
        **model_configs.__dict__,
        **frontend_configs.__dict__,
    }
    if train_configs.wandb:
        tags = list(train_configs.kwargs_["wandb_tags"])
        logger = WandbLogger(
            project=train_configs.project_name,
            name=name,
            group=group_name,
            save_dir=save_dir,
            tags=tags,
            config=configs,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=save_dir,
        )
        logger.log_hyperparams(configs)
    return logger


def set_trainer(
    save_dir: str,
    train_configs: TrainArguments,
    model_configs: ModelArguments,
    data_configs: DataArguments,
    frontend_configs: FrontendArguments,
    group_name: str = None,
    name: str = None,
) -> pl.Trainer:
    r"""Configure Pytorch Lightning trainer."""
    logger = set_logger(
        save_dir=save_dir,
        train_configs=train_configs,
        model_configs=model_configs,
        data_configs=data_configs,
        frontend_configs=frontend_configs,
        group_name=group_name,
        name=name,
    )

    score_callback = ModelCheckpoint(
        monitor="valid/sc",
        dirpath=save_dir,
        filename="epoch={epoch:02d}_val_sc={valid/sc:.2f}",
        every_n_train_steps=0,
        every_n_epochs=1,
        save_top_k=1,
        mode="max",
        save_last=False,
        auto_insert_metric_name=False,
    )

    trainer_args = {
        "gpus": train_configs.gpus,
        "callbacks": [score_callback],
        "logger": logger,
        "max_epochs": train_configs.max_epochs,
        "log_every_n_steps": 1,
        "check_val_every_n_epoch": train_configs.val_check_interval,
        "deterministic": True,
        "precision": 16,
    }

    return pl.Trainer(**trainer_args)


def main():
    train_configs = yaml_to_args(arg_type="train")
    data_configs = yaml_to_args(arg_type="data")
    model_configs = yaml_to_args(arg_type="model")
    frontend_configs = yaml_to_args(arg_type="frontend")

    SUFFIX = randomname.get_name()

    pl.seed_everything(train_configs.seed, workers=True)
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    all_configs = {
        "train_configs": train_configs,
        "data_configs": data_configs,
        "model_configs": model_configs,
        "frontend_configs": frontend_configs,
    }

    all_wavs = np.array(os.listdir(data_configs.wavs_dir))

    name = f"{model_configs.model_name}_{train_configs.optimizer}{train_configs.lr}_BS{data_configs.batch_size}_{SUFFIX}".upper()

    df = pd.read_csv(data_configs.official_split, header=None)
    file_name = lambda full_name: "_".join(full_name.split("_")[:5])
    labels_dict = labels_file_to_dict(
        labels_file=data_configs.label_file, sep=data_configs.kwargs_["labels_sep"]
    )
    if "t1" in data_configs.dataset:
        get_labels = lambda filename: labels_dict[filename.split(".wav")[0]]
    else:
        get_labels = lambda filename: labels_dict[
            "_".join(filename.split(".wav")[0].split("_")[:5])
        ]
    test_mask = np.array(
        [df.loc[df[0] == file_name(wav)][1].to_list()[0] == "test" for wav in all_wavs]
    )
    train_wavs = all_wavs[~test_mask]
    test_wavs = all_wavs[test_mask]

    pl_module = UNOModule(
        train_wavs=train_wavs,
        test_wavs=test_wavs,
        train_labels=np.array([get_labels(wav) for wav in train_wavs]),
        test_labels=np.array([get_labels(wav) for wav in test_wavs]),
        **all_configs,
    )

    save_dir = os.path.join(
        train_configs.save_stats_dir, train_configs.project_name, name
    )
    os.makedirs(save_dir, exist_ok=True)

    print(
        f"{Colors.BLUE}******************{Colors.ENDC} {Colors.BOLD}{Colors.GREEN}JOB NAME: {name}{Colors.ENDC} {Colors.BLUE}******************{Colors.ENDC}"
    )

    trainer = set_trainer(save_dir=save_dir, group_name=None, name=name, **all_configs)
    trainer.fit(pl_module)

    for filename in os.listdir(save_dir):
        if "val_sc" in filename:
            ckpt_path = os.path.join(save_dir, filename)
            break
    trainer.test(pl_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
