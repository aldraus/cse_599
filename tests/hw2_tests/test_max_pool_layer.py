import numpy as np
import torch
from torch import nn

from nn.layers.max_pool_layer import MaxPoolLayer, MaxPoolNumbaLayer
from tests import utils

TOLERANCE = 1e-4


def _test_max_pool_forward(input_shape, kernel_size, stride):
    np.random.seed(0)
    torch.manual_seed(0)
    padding = (kernel_size - 1) // 2
    input = np.random.random(input_shape).astype(np.float32) * 20
    original_input = input.copy()
    layer = MaxPoolLayer(kernel_size, stride)
    #layer = MaxPoolNumbaLayer(kernel_size, stride)


    torch_layer = nn.MaxPool2d(kernel_size, stride, padding)
    output = layer.forward(input)

    torch_data = utils.from_numpy(input)
    torch_out = utils.to_numpy(torch_layer(torch_data))
    output[np.abs(output) < 1e-4] = 0
    torch_out[np.abs(torch_out) < 1e-4] = 0

    assert np.all(input == original_input)
    assert output.shape == torch_out.shape
    utils.assert_close(output, torch_out, atol=TOLERANCE)

def test_max_pool_forward_batch_input_output():
    width = 10
    height = 10
    kernel_size = 3
    stride = 1
    for batch_size in range(1, 5):
        for input_channels in range(1, 5):
            input_shape = (batch_size, input_channels, width, height)
            _test_max_pool_forward(input_shape, kernel_size, stride)

def test_max_pool_forward_width_height_stride_kernel_size():
    batch_size = 2
    input_channels = 2
    for width in range(10, 21):
        for height in range(11, 21):
            for stride in range(1, 3):
                for kernel_size in range(stride, 6):
                    input_shape = (batch_size, input_channels, width, height)
                    _test_max_pool_forward(input_shape, kernel_size, stride)



def _test_max_pool_backward(input_shape, kernel_size, stride):


    np.random.seed(0)
    torch.manual_seed(0)
    padding = (kernel_size - 1) // 2
    input = np.random.random(input_shape).astype(np.float32) * 20
    layer = MaxPoolLayer(kernel_size, stride)
    # layer = MaxPoolNumbaLayer(kernel_size, stride)


    torch_layer = nn.MaxPool2d(kernel_size, stride, padding)

    output = layer.forward(input)
    out_grad = layer.backward(2 * np.ones_like(output) / output.size)

    torch_input = utils.from_numpy(input).requires_grad_(True)
    torch_out = torch_layer(torch_input)
    (2 * torch_out.mean()).backward()

    torch_out_grad = utils.to_numpy(torch_input.grad)
    utils.assert_close(out_grad, torch_out_grad, atol=TOLERANCE)


def test_max_pool_backward_batch_input_output():
    width = 10
    height = 10
    kernel_size = 3
    stride = 1
    for batch_size in range(1, 5):
        for input_channels in range(1, 5):
            input_shape = (batch_size, input_channels, width, height)
            _test_max_pool_backward(input_shape, kernel_size, stride)




def test_max_pool_backward_width_height_stride_kernel_size():
    batch_size = 2
    input_channels = 2
    for width in range(10, 21):
        for height in range(10, 21):
            for stride in range(1, 3):
                for kernel_size in range(stride, 6):
                    input_shape = (batch_size, input_channels, width, height)
                    _test_max_pool_backward(input_shape, kernel_size, stride)


