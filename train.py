import argparse
import os
import random

import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from configs_ import yaml_to_args
from misc.utils import *
from pipeline import NCSSLPretrainModule, TaskEvaluationModule


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type",
        type=str,
        default="task_evaluation",
        required=True,
        help="Task Type: task_evaluation or ssl (default: task_evaluation)",
    )
    args = parser.parse_args()

    return args


def ssl_pretrain():
    ssl_configs = yaml_to_args(arg_type="ssl")

    SUFFIX = train_init(seed=ssl_configs.seed)

    all_wavs = np.array(os.listdir(ssl_configs.wavs_dir))

    name = f"{ssl_configs.model}_{ssl_configs.optimizer}{ssl_configs.lr}_BS{ssl_configs.batch_size}_{SUFFIX}".upper()

    df = pd.read_csv(ssl_configs.official_split, header=None)
    file_name = lambda full_name: "_".join(full_name.split(".wav")[0].split("_")[:5])
    test_mask = np.array(
        [df.loc[df[0] == file_name(wav)][1].to_list()[0] == "test" for wav in all_wavs]
    )
    train_wavs = all_wavs[~test_mask]
    random.shuffle(train_wavs)

    save_dir = os.path.join(ssl_configs.save_stats_dir, ssl_configs.project_name, name)
    os.makedirs(save_dir, exist_ok=True)

    if ssl_configs.wandb:
        logger = WandbLogger(
            project=ssl_configs.project_name,
            name=name,
            config=ssl_configs.__dict__,
            save_dir=save_dir,
        )
    else:
        logger = TensorBoardLogger(
            save_dir=save_dir,
            name=name,
        )

    checkpoints = ModelCheckpoint(
        filename="{epoch}-{step}-{loss:.4f}",
        dirpath=os.path.join(save_dir, "monitor_loss"),
        monitor="loss",
        mode="min",
        save_last=True,
    )

    trainer_args = {
        "gpus": ssl_configs.gpus,
        "max_epochs": ssl_configs.max_epochs,
        "callbacks": checkpoints,
        "logger": logger,
        "log_every_n_steps": 1,
        "deterministic": True,
        "accumulate_grad_batches": 2,
        "check_val_every_n_epoch": 1,
    }

    trainer = pl.Trainer(**trainer_args)

    print(
        f"{Colors.BOLD}{Colors.RED}****************** SSL Pretrain: {name} ******************{Colors.ENDC}"
    )

    pl_module = NCSSLPretrainModule(
        configs=ssl_configs,
        train_wavs=train_wavs,
    )

    trainer.fit(pl_module)


def task_evaluation():
    train_configs = yaml_to_args(arg_type="train")
    data_configs = yaml_to_args(arg_type="data")
    model_configs = yaml_to_args(arg_type="model")
    frontend_configs = yaml_to_args(arg_type="frontend")

    SUFFIX = train_init(seed=train_configs.seed)

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

    pl_module = TaskEvaluationModule(
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
    task_type = args_parser().task_type

    if task_type == "task_evaluation":
        task_evaluation()
    elif task_type == "ssl":
        ssl_pretrain()
    else:
        raise ValueError(
            f"Task type {task_type} is not supported. Please choose from [task_evaluation, ssl]"
        )
