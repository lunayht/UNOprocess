import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader

import models
from configs_.arguments import (
    DataArguments,
    FrontendArguments,
    ModelArguments,
    TrainArguments,
)
from data import AudioDataset
from eval import Evaluation_Metrics


class UNOModule(pl.LightningModule):
    def __init__(
        self,
        model_configs: ModelArguments,
        train_configs: TrainArguments,
        data_configs: DataArguments,
        frontend_configs: FrontendArguments,
        train_wavs: np.ndarray,
        test_wavs: np.ndarray,
        train_labels: np.ndarray,
        test_labels: np.ndarray,
    ) -> None:
        super().__init__()

        self.model_configs = model_configs
        self.train_configs = train_configs
        self.data_configs = data_configs
        self.frontend_configs = frontend_configs

        self.train_wavs = train_wavs
        self.test_wavs = test_wavs
        self.train_labels = train_labels
        self.test_labels = test_labels

        self.model = getattr(models, self.model_configs.model_name)(
            num_classes=self.data_configs.num_classes,
            seq_length=self.frontend_configs.spec_width,
            n_mels=self.frontend_configs.n_mels,
            imgnet_pretrain=self.model_configs.imgnet_pretrain,
            **self.model_configs.kwargs_,
        )

        if self.model_configs.from_checkpoint:
            state_dict = torch.load(
                self.model_configs.from_checkpoint, map_location=self.device
            )
            self._load_pretrain_state_dict(state_dict)

        self.example_input_array = torch.randn(
            self.data_configs.batch_size,
            1,
            self.frontend_configs.n_mels,
            self.frontend_configs.spec_width,
            device=self.device,
        )

        self.train_step_count = 0
        self.valid_step_count = 0

    def prepare_data(self) -> None:
        self.train_dataset = AudioDataset(
            root=self.data_configs.wavs_dir,
            signals=self.train_wavs,
            labels=self.train_labels,
            frontend_configs=self.frontend_configs,
            mode="train",
            get_weights=self.train_configs.kwargs_["weighted_criterion"],
        )
        self.val_dataset = AudioDataset(
            root=self.data_configs.wavs_dir,
            signals=self.test_wavs,
            labels=self.test_labels,
            frontend_configs=self.frontend_configs,
            mode="test",
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_criterion = getattr(torch.nn, self.train_configs.loss)(
                weight=self.train_dataset.normed_weights,
                reduction="none" if self.frontend_configs.mixup_alpha else "mean",
            )
            self.val_criterion = getattr(torch.nn, self.train_configs.loss)()

            self.train_metrics = Evaluation_Metrics(
                num_classes=self.data_configs.num_classes,
                normal_class_label=self.data_configs.kwargs_["normal_class_label"],
                mixup=True if self.frontend_configs.mixup_alpha else False,
            )
            self.val_metrics = Evaluation_Metrics(
                num_classes=self.data_configs.num_classes,
                normal_class_label=self.data_configs.kwargs_["normal_class_label"],
            )

        if stage in (None, "test"):
            self.test_metrics = Evaluation_Metrics(
                self.data_configs.num_classes,
                normal_class_label=self.data_configs.kwargs_["normal_class_label"],
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_configs.batch_size,
            shuffle=True,
            num_workers=self.data_configs.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_configs.val_batch_size,
            shuffle=False,
            num_workers=self.data_configs.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_configs.val_batch_size,
            shuffle=False,
            num_workers=self.data_configs.num_workers,
            pin_memory=True,
        )

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        optimizer = getattr(torch.optim, self.train_configs.optimizer)(
            self.model.parameters(),
            lr=self.train_configs.lr,
            weight_decay=self.train_configs.kwargs_["weight_decay"],
        )
        lr_scheduler = getattr(
            torch.optim.lr_scheduler, self.train_configs.lr_scheduler
        )(optimizer, **self.train_configs.kwargs_["lr_scheduler_args"])
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self) -> None:
        if self.train_configs.onnx:
            model_onnx = os.path.join(self.logger.save_dir, "model.onnx")
            torch.onnx.export(
                self, self.example_input_array.to(self.device), model_onnx
            )
            if self.train_configs.wandb:
                wandb.save(model_onnx, base_path=self.logger.save_dir)

    def training_step(self, batch: Tuple[torch.Tensor], batch_id: int) -> torch.Tensor:
        if self.frontend_configs.mixup_alpha:
            mixed_x, y_a, y_b, lam = batch
            logits = self(mixed_x)
            loss = self.mixup_criterion(
                criterion=self.train_criterion,
                pred=logits,
                y_a=y_a,
                y_b=y_b,
                lam=lam,
            )
            self.train_metrics.update_mixup_stats(
                logits=logits, y_true_a=y_a, y_true_b=y_b, lam=lam
            )
        else:
            x, y = batch
            logits = self(x)
            loss = self.train_criterion(logits, y)
            self.train_metrics.update_lists(logits=logits, y_true=y)

        self.train_step_count += y_a.size(0)
        self.log_dict(
            {
                "train/loss": loss,
                "trainer/train_steps": float(self.train_step_count),
            },
            on_step=True,
            on_epoch=False,
        )
        return {"loss": loss}

    def training_epoch_end(self, outputs: None) -> None:
        if self.frontend_configs.mixup_alpha:
            acc = self.train_metrics.get_mixup_stats()
            self.log_dict(
                {"train/acc": acc},
                on_step=False,
                on_epoch=True,
            )
        else:
            acc, se, sp, sc = self.train_metrics.get_stats()
            self.log_dict(
                {
                    "train/acc": acc,
                    "train/se": se,
                    "train/sp": sp,
                    "train/sc": sc,
                },
                on_step=False,
                on_epoch=True,
            )
        self.train_metrics.reset_metrics()

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_id: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.val_criterion(logits, y)
        self.val_metrics.update_lists(logits=logits, y_true=y)
        self.valid_step_count += y.size(0)
        self.log_dict(
            {
                "valid/loss": loss,
                "trainer/valid_steps": float(self.valid_step_count),
            },
            on_step=True,
            on_epoch=False,
        )
        return {"loss": loss}

    def validation_epoch_end(self, outputs: None) -> None:
        acc, se, sp, sc = self.val_metrics.get_stats()
        self.log_dict(
            {
                "valid/acc": acc,
                "valid/se": se,
                "valid/sp": sp,
                "valid/sc": sc,
            },
            on_step=False,
            on_epoch=True,
        )
        if self.train_configs.kwargs_["pre_rec_fbeta"]:
            precision, recall, f1, _ = self.val_metrics.get_precision_recall_fbeta(
                fbeta=self.train_configs.kwargs_["pre_rec_fbeta"]
            )
            self.log_dict(
                {
                    "valid/precision": precision,
                    "valid/recall": recall,
                    "valid/f1": f1,
                },
                on_step=False,
                on_epoch=True,
            )

        self.val_metrics.reset_metrics()

    def test_step(self, batch: Tuple[torch.Tensor], batch_id: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        self.test_metrics.update_lists(logits=logits, y_true=y)

    def test_epoch_end(self, outputs: Union[Any, List[Any]]) -> None:
        acc, se, sp, sc = self.test_metrics.get_stats()
        self.log_dict(
            {
                "test/acc": acc,
                "test/se": se,
                "test/sp": sp,
                "test/sc": sc,
            }
        )
        if self.train_configs.wandb:
            wandb.log(
                {
                    "test/cm": wandb.plot.confusion_matrix(
                        y_true=self.test_metrics.y_true.numpy(),
                        preds=self.test_metrics.y_pred.numpy(),
                    ),
                    "test/roc": wandb.plot.roc_curve(
                        y_true=self.test_metrics.y_true.numpy(),
                        y_probas=self.test_metrics.y_pred_prob.numpy(),
                    ),
                }
            )
        if self.train_configs.kwargs_["pre_rec_fbeta"]:
            precision, recall, f1, _ = self.test_metrics.get_precision_recall_fbeta(
                fbeta=self.train_configs.kwargs_["pre_rec_fbeta"]
            )
            self.log_dict(
                {
                    "test/precision": precision,
                    "test/recall": recall,
                    "test/f1": f1,
                }
            )

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: torch.optim, optimizer_idx: int
    ) -> None:
        optimizer.zero_grad(set_to_none=True)

    def _load_pretrain_state_dict(self, state_dict: OrderedDict) -> None:
        r"""Loads the state dict from self-supervised pretraining."""
        try:
            params = state_dict["state_dict"]
            new_state_dict = OrderedDict()
            for key, value in params.items():
                if key.startswith("learner.online_encoder.net."):
                    if "mlp" in key:
                        continue
                    else:
                        new_key = key.replace("learner.online_encoder.net.", "")
                        new_state_dict[new_key] = value
                else:
                    continue
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"=======> Loaded state dict from pretrain.\n")
        except Exception as e:
            print(f"=======> Failed to load state dict.\n{e}")
            raise e

    @staticmethod
    def mixup_criterion(
        criterion: torch.nn,
        pred: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        mixup_loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        return mixup_loss.mean()
