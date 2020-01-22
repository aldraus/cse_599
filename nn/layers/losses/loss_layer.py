from abc import ABC

from ..layer import Layer


class LossLayer(Layer, ABC):
    def forward(self, *args, **kwargs) -> float:
        raise NotImplementedError
