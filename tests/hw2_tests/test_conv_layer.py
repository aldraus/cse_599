import pdb
import numpy as np
import torch
from torch import nn

from nn.layers.conv_layer import ConvLayer, ConvNumbaLayer
from tests import utils

TOLERANCE = 1e-4


def _test_conv_forward(input_shape, out_channels, kernel_size, stride):
    np.random.seed(0)
    torch.manual_seed(0)
    in_channels = input_shape[1]
    padding = (kernel_size - 1) // 2
    input = np.random.random(input_shape).astype(np.float32) * 20
    original_input = input.copy()
    layer = ConvNumbaLayer(in_channels, out_channels, kernel_size, stride)

    torch_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    utils.assign_conv_layer_weights(layer, torch_layer)

    output = layer.forward(input)

    torch_data = utils.from_numpy(input)
    torch_out = torch_layer(torch_data)

    assert np.all(input == original_input)
    assert output.shape == torch_out.shape
    utils.assert_close(output, torch_out, atol=TOLERANCE)
#
def test_conv_forward_batch_input_output():
    width = 10
    height = 10
    kernel_size = 3
    stride = 1
    for batch_size in range(1, 5):
        for input_channels in range(1, 5):
            for output_channels in range(2, 5):
                input_shape = (batch_size, input_channels, width, height)
                _test_conv_forward(input_shape, output_channels, kernel_size, stride)


def test_conv_forward_width_height_stride_kernel_size():
    batch_size = 2
    input_channels = 2
    output_channels = 3
    for width in range(10, 21):
        for height in range(10, 21):
            for stride in range(1, 3):
                for kernel_size in range(stride, 6):
                    input_shape = (batch_size, input_channels, width, height)
                    _test_conv_forward(input_shape, output_channels, kernel_size, stride)




def _test_conv_backward(input_shape, out_channels, kernel_size, stride):
    np.random.seed(0)
    torch.manual_seed(0)
    in_channels = input_shape[1]
    padding = (kernel_size - 1) // 2
    input = np.random.random(input_shape).astype(np.float32) * 20
    layer = ConvNumbaLayer(in_channels, out_channels, kernel_size, stride)

    torch_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)
    utils.assign_conv_layer_weights(layer, torch_layer)

    output = layer.forward(input)
    out_grad = layer.backward(2 * np.ones_like(output) / output.size)

    torch_input = utils.from_numpy(input).requires_grad_(True)
    torch_out = torch_layer(torch_input)
    (2 * torch_out.mean()).backward()

    utils.assert_close(out_grad, torch_input.grad, atol=TOLERANCE)
    utils.check_conv_grad_match(layer, torch_layer) # only fails this, I am passing the wrong values to the grads

def test_conv_backward_batch_input_output():
    width = 10
    height = 10
    kernel_size = 3
    stride = 1
    for batch_size in range(1, 5):
        for input_channels in range(2, 5):
            for output_channels in range(2, 5):
                input_shape = (batch_size, input_channels, width, height)
                _test_conv_backward(input_shape, output_channels, kernel_size, stride)


def test_conv_backward_width_height_stride_kernel_size():
    batch_size = 2
    input_channels = 2
    output_channels = 3
    for width in range(10, 21):
        for height in range(10, 21):
            for stride in range(1, 3):
                for kernel_size in range(stride, 6):
                    input_shape = (batch_size, input_channels, width, height)
                    _test_conv_backward(input_shape, output_channels, kernel_size, stride)

