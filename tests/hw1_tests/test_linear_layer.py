import numpy as np
import torch
from torch import nn

from nn.layers.linear_layer import LinearLayer
from tests import utils

TOLERANCE = 1e-4


def _test_linear_forward(input_shape, out_channels):
    in_channels = input_shape[1]
    input = np.random.random(input_shape).astype(np.float32) * 20
    original_input = input.copy()
    layer = LinearLayer(in_channels, out_channels)

    torch_layer = nn.Linear(in_channels, out_channels, bias=True)
    utils.assign_linear_layer_weights(layer, torch_layer)

    output = layer.forward(input)

    torch_data = utils.from_numpy(input)
    torch_out = torch_layer(torch_data)

    assert np.all(input == original_input)
    assert output.shape == torch_out.shape
    utils.assert_close(output, torch_out, atol=TOLERANCE)


def test_linear_forward():
    for batch_size in range(1, 4):
        for input_channels in range(1, 5):
            for output_channels in range(1, 5):
                input_shape = (batch_size, input_channels)
                _test_linear_forward(input_shape, output_channels)


def _test_linear_backward(input_shape, out_channels):
    in_channels = input_shape[1]
    input = np.random.random(input_shape).astype(np.float32) * 20
    layer = LinearLayer(in_channels, out_channels)

    torch_layer = nn.Linear(in_channels, out_channels, bias=True)
    utils.assign_linear_layer_weights(layer, torch_layer)

    output = layer.forward(input)
    out_grad = layer.backward(np.ones_like(output) * 2)

    torch_input = utils.from_numpy(input).requires_grad_(True)
    torch_out = torch_layer(torch_input)
    (2 * torch_out).sum().backward()

    utils.assert_close(out_grad, torch_input.grad, atol=TOLERANCE)
    utils.check_linear_grad_match(layer, torch_layer, tolerance=TOLERANCE)


def test_linear_backward():
    for batch_size in range(1, 4):
        for input_channels in range(1, 5):
            for output_channels in range(1, 5):
                input_shape = (batch_size, input_channels)
                _test_linear_backward(input_shape, output_channels)
