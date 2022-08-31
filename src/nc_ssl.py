# Modified from: https://github.com/lucidrains/byol-pytorch

import copy
from functools import wraps
from typing import Any, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as AT

from uno import UNO


class ExponentialMovingAverage:
    def __init__(self, beta: float = 0.99) -> None:
        super().__init__()
        self.beta = beta

    def update(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class ProjectorMLP(nn.Module):
    def __init__(self, dim: int, hidden_size: int, projection_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NetWrapper(nn.Module):
    r"""A wrapper class for the base network."""

    def __init__(
        self,
        net: nn.Module,
        projection_size: int,
        projection_hidden_size: int,
        layer: int = -1,
    ) -> None:
        super().__init__()
        self.net = net
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.layer = layer

        self.hidden = dict()
        self.hook_registered = False
        self.projector = None

    def singleton(cache_key: str) -> Any:
        def inner_fn(fn):
            @wraps(fn)
            def wrapper(self, *args, **kwargs):
                instance = getattr(self, cache_key)
                if instance is not None:
                    return instance
                instance = fn(self, *args, **kwargs)
                setattr(self, cache_key, instance)
                return instance

            return wrapper

        return inner_fn

    @staticmethod
    def flatten(t: torch.Tensor) -> torch.Tensor:
        return t.view(t.size(0), -1)

    @singleton("projector")
    def _get_projector(self, hidden: torch.Tensor) -> nn.Module:
        _, dim = hidden.size()
        projector = ProjectorMLP(
            dim=dim,
            projection_size=self.projection_size,
            hidden_size=self.projection_hidden_size,
        )
        return projector.to(hidden)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer == -1:
            return self.net(x)
        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        assert hidden is not None, f"Hidden layer {self.layer} never emitted an output."
        return hidden

    def forward(
        self, x: torch.Tensor, get_projection: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        representation = self.get_representation(x)

        if get_projection:
            projector = self._get_projector(representation)
            projection = projector(representation)
            return projection, representation
        return representation

    def _find_hook_layer(self) -> nn.Module:
        hook_layer: nn.Module = None
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            hook_layer = modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            hook_layer = children[self.layer]
        return hook_layer

    def _hook(self, _, x: torch.Tensor, y: torch.Tensor) -> None:
        device = x[0].device
        self.hidden[device] = self.flatten(y)

    def _register_hook(self) -> None:
        layer = self._find_hook_layer()
        assert layer is not None, f"Cannot find layer {self.layer}."
        _ = layer.register_forward_hook(self._hook)
        self.hook_registered = True


class NonContrastiveSSL(nn.Module):
    DEFAULT_AUGMENT = nn.Sequential(
        AT.TimeMasking(16),
        AT.FrequencyMasking(16),
    )

    def __init__(
        self,
        net: nn.Module,
        spec_size: Tuple[int, int] = (64, 128),  # (freq_bins, time_axis)
        hidden_layer: int = -1,
        projection_size: int = 256,
        projection_hidden_size: int = 4096,
        augment_fn: nn.Sequential = None,
        augment_fn2: nn.Sequential = None,
        ema_beta: float = 0.99,
        use_momentum: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = net
        self.dropout = dropout
        self.use_momentum = use_momentum

        self.aug_1 = self.augmentation_wrapper(aug=augment_fn, def_aug=nn.Sequential())
        self.aug_2 = self.augmentation_wrapper(
            aug=augment_fn2, def_aug=self.DEFAULT_AUGMENT
        )

        self.online_encoder = NetWrapper(
            net=net,
            projection_size=projection_size,
            projection_hidden_size=projection_hidden_size,
            layer=hidden_layer,
        )

        if dropout > 0:
            self.online_predictor = UNO(
                dim=projection_size,
                hidden_size=projection_hidden_size,
                projection_size=projection_size,
                dropout=dropout,
            )
        else:
            self.online_predictor = ProjectorMLP(
                dim=projection_size,
                hidden_size=projection_hidden_size,
                projection_size=projection_size,
            )

        self.target_encoder = None
        self.target_ema = ExponentialMovingAverage(beta=ema_beta)

        device = next(net.parameters()).device
        self.to(device)

        self.forward(torch.randn(2, 1, spec_size[0], spec_size[1], device=device))

    @NetWrapper.singleton("target_encoder")
    def _get_target_encoder(self) -> nn.Module:
        target_encoder = copy.deepcopy(self.online_encoder)
        self.set_requires_grad(target_encoder, False)
        print("Freezing target encoder.")
        return target_encoder

    @staticmethod
    def augmentation_wrapper(
        aug: nn.Sequential, def_aug: nn.Sequential
    ) -> nn.Sequential:
        return def_aug if aug is None else aug

    @staticmethod
    def l2_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    @staticmethod
    def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for param in module.parameters():
            param.requires_grad = requires_grad

    def reset_ema(self) -> None:
        del self.target_encoder
        self.target_encoder = None

    def update_ema(self) -> None:
        assert (
            self.use_momentum
        ), "Do not need to update EMA if you have turned off momentum for the target encoder."
        assert (
            self.target_encoder is not None
        ), "Cannot update EMA if you have not yet created the target encoder."
        for online_parms, target_parms in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            old_weights, new_weights = target_parms.data, online_parms.data
            target_parms.data = self.target_ema.update(old_weights, new_weights)

    def forward(self, x: torch.Tensor, return_projection: bool = False) -> torch.Tensor:
        if return_projection:
            return self.online_encoder(x, get_projection=return_projection)

        x_1, x_2 = self.aug_1(x), self.aug_2(x)

        online_projection_1, _ = self.online_encoder(x_1)
        online_projection_2, _ = self.online_encoder(x_2)
        online_prediction_1 = self.online_predictor(online_projection_1)
        online_prediction_2 = self.online_predictor(online_projection_2)

        with torch.no_grad():
            target_encoder = (
                self._get_target_encoder() if self.use_momentum else self.online_encoder
            )
            target_projection_1, _ = target_encoder(x_1)
            target_projection_2, _ = target_encoder(x_2)
            target_projection_1.detach_()
            target_projection_2.detach_()

        loss_1 = self.l2_loss(online_prediction_1, target_projection_2.detach())
        loss_2 = self.l2_loss(online_prediction_2, target_projection_1.detach())
        loss = (loss_1 + loss_2).mean()
        return loss
