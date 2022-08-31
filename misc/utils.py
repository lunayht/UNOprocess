import os
import sys

sys.path.append(os.path.abspath(os.path.join("", "..")))


from typing import Dict, Union

import pandas as pd
import pytorch_lightning as pl
from configs_ import DataArguments, FrontendArguments, ModelArguments, TrainArguments
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


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
