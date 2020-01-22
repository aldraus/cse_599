import pdb
from abc import ABC

from nn.layers import LayerUsingLayer
from nn.layers.losses import LossLayer


class Network(LayerUsingLayer, ABC):
    def __init__(self, loss_layer: LossLayer):
        super(Network, self).__init__()
        self.loss_layer: LossLayer = loss_layer

    @property
    def final_layer(self):
        return self.loss_layer

    def loss(self, *args, **kwargs) -> float:
        raise NotImplementedError
