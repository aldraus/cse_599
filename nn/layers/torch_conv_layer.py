from typing import Optional, Callable

import numpy as np
import torch.nn.functional as F
import torch.nn.grad as grad

from nn import Parameter
from tests import utils
from .layer import Layer


class TorchConvLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(TorchConvLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros((output_channels), dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.data = None
        self.initialize()
        self.weight_tensor = None
        self.bias_tensor = None
        self.output = None

    def forward(self, data):
        self.weight_tensor = utils.from_numpy(self.weight.data.swapaxes(0, 1))
        self.bias_tensor = utils.from_numpy(self.bias.data)
        self.data = utils.from_numpy(data)
        self.output = F.conv2d(self.data, self.weight_tensor, self.bias_tensor, self.stride, self.padding)
        return utils.to_numpy(self.output)

    def backward(self, previous_partial_gradient):
        gradients = utils.from_numpy(previous_partial_gradient)
        input_grad = grad.conv2d_input(self.data.shape, self.weight_tensor, gradients, self.stride, self.padding)
        weight_grad = grad.conv2d_weight(self.data, self.weight_tensor.shape, gradients, self.stride, self.padding)
        bias_grad = gradients.sum((0, 2, 3))

        self.weight.grad = utils.to_numpy(weight_grad.transpose(1, 0))
        self.bias.grad = utils.to_numpy(bias_grad)
        data_gradient = utils.to_numpy(input_grad)
        return data_gradient

    def selfstr(self):

        return "Kernel: (%s, %s) In Channels %s Out Channels %s Stride %s" % (
            self.weight.data.shape[2],
            self.weight.data.shape[3],
            self.weight.data.shape[0],
            self.weight.data.shape[1],
            self.stride,
        )

    def initialize(self, initializer: Optional[Callable[[Parameter], None]] = None):
        if initializer is None:
            self.weight.data = np.random.normal(0, 0.1, self.weight.data.shape)
            self.bias.data = 0
        else:
            for param in self.own_parameters():
                initializer(param)
        super(TorchConvLayer, self).initialize()



