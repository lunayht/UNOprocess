# Modified from: https://github.com/lucidrains/byol-pytorch

from functools import wraps
from typing import Any, Tuple, Union

import torch
import torch.functional as F
import torch.nn as nn


class Exponential_Moving_Average:
    def __init__(self, beta: float = 0.99) -> None:
        super().__init__()
        self.beta = beta

    def update(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


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

    @singleton("projector")
    def _get_projector(self, hidden: torch.Tensor) -> nn.Module:
        _, dim = hidden.size()
        projector = Projector_MLP(
            dim=dim,
            hidden_size=self.projection_size,
            output_size=self.projection_hidden_size,
        )
        return projector.to(hidden)

    @staticmethod
    def flatten(t: torch.Tensor) -> torch.Tensor:
        return t.view(t.size(0), -1)


class Projector_MLP(nn.Module):
    def __init__(self, dim: int, hidden_size: int, projection_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, projection_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return x
