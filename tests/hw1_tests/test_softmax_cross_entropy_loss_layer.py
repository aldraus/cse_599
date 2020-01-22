<<<<<<< HEAD:test/hw1_tests/test_softmax_cross_entropy_loss_layer.py
import numpy as np
import torch.nn.functional as F

from nn.layers.losses import SoftmaxCrossEntropyLossLayer
from test import utils


def _test_forward(input_shape, reduction, axis):
    layer = SoftmaxCrossEntropyLossLayer(reduction=reduction)
    data = np.random.random(input_shape) * 2 - 1
    labels_shape = list(data.shape)
    labels_shape.pop(axis)
    labels = np.random.randint(0, data.shape[axis], labels_shape).astype(np.int64)
    labels = np.random.randint(0, data.shape[axis], labels_shape)
    print(np.shape(labels))
    loss = layer(data, labels, axis=axis)
    if axis == 1:
        pytorch_loss = F.cross_entropy(utils.from_numpy(data), utils.from_numpy(labels), reduction=reduction)
    else:
        pytorch_loss = F.cross_entropy(
            utils.from_numpy(data.swapaxes(1, axis)), utils.from_numpy(labels), reduction=reduction
        )
    pytorch_loss = utils.to_numpy(pytorch_loss)

<<<<<<< HEAD
    assert np.allclose(loss, pytorch_loss, atol=0.01)
=======
    utils.assert_close(loss, pytorch_loss, atol=0.001)
>>>>>>> 3eccfe31aacb9d0a4a15dc27c51a2bf86c11ab0b


def test_forward_easy():
    input_shape = (20, 10)
    _test_forward(input_shape, "mean", 1)
    _test_forward(input_shape, "sum", 1)


def _test_forward_overflow(input_shape, reduction, axis):
    layer = SoftmaxCrossEntropyLossLayer(reduction=reduction)
    data = np.random.random(input_shape) * 10000 - 1
    labels_shape = list(data.shape)
    labels_shape.pop(axis)
    labels = np.random.randint(0, data.shape[axis], labels_shape).astype(np.int64)
    labels = np.random.randint(0, data.shape[axis], labels_shape)
    print(np.shape(labels))
    loss = layer(data, labels, axis=axis)
    if axis == 1:
        pytorch_loss = F.cross_entropy(utils.from_numpy(data), utils.from_numpy(labels), reduction=reduction)
    else:
        pytorch_loss = F.cross_entropy(
            utils.from_numpy(data.swapaxes(1, axis)), utils.from_numpy(labels), reduction=reduction
        )
    pytorch_loss = utils.to_numpy(pytorch_loss)

    utils.assert_close(loss, pytorch_loss, atol=0.001)


def test_forward_overflow():
    input_shape = (20, 10)
    _test_forward_overflow(input_shape, "mean", 1)
    _test_forward_overflow(input_shape, "sum", 1)


def test_forward_hard():
    input_shape = (20, 10, 5)
    _test_forward(input_shape, "mean", 1)
    _test_forward(input_shape, "sum", 1)
    _test_forward(input_shape, "mean", 2)
    _test_forward(input_shape, "sum", 2)


def _test_backward(input_shape, reduction, axis):
    layer = SoftmaxCrossEntropyLossLayer(reduction=reduction)
    data = np.random.random(input_shape) * 2 - 1
    labels_shape = list(data.shape)
    labels_shape.pop(axis)
    labels = np.random.randint(0, data.shape[axis], labels_shape).astype(np.int64)
    labels = np.random.randint(0, data.shape[axis], labels_shape)
    loss = layer(data, labels, axis=axis)
    if axis == 1:
        torch_input = utils.from_numpy(data).requires_grad_(True)
    else:
        torch_input = utils.from_numpy(np.moveaxis(data, axis, 1)).requires_grad_(True)
    pytorch_loss = F.cross_entropy(torch_input, utils.from_numpy(labels), reduction=reduction)
    if len(pytorch_loss.shape) > 0:
        pytorch_loss.sum().backward()
    else:
        pytorch_loss.backward()

    utils.assert_close(loss, utils.to_numpy(pytorch_loss))

    grad = layer.backward()
    torch_grad = utils.to_numpy(torch_input.grad)
    if axis != 1:
        torch_grad = np.moveaxis(torch_grad, 1, axis)

    utils.assert_close(grad, torch_grad, atol=0.001)


def test_backward_easy():
    input_shape = (20, 10)
    _test_backward(input_shape, "mean", 1)
    _test_backward(input_shape, "sum", 1)


def test_backward_hard():
    input_shape = (20, 10, 5)
    _test_backward(input_shape, "mean", 1)
    _test_backward(input_shape, "sum", 1)
    _test_backward(input_shape, "mean", 2)
    _test_backward(input_shape, "sum", 2)


def test_backward_harder():
    input_shape = (20, 10, 5, 23)
    _test_backward(input_shape, "mean", 0)
    _test_backward(input_shape, "sum", 0)
    _test_backward(input_shape, "mean", 1)
    _test_backward(input_shape, "sum", 1)
    _test_backward(input_shape, "mean", 2)
    _test_backward(input_shape, "sum", 2)
    _test_backward(input_shape, "mean", 3)
    _test_backward(input_shape, "sum", 3)
=======
import numpy as np
import torch.nn.functional as F

from nn.layers.losses import SoftmaxCrossEntropyLossLayer
from tests import utils


def _test_forward(input_shape, reduction, axis):
    layer = SoftmaxCrossEntropyLossLayer(reduction=reduction)
    data = np.random.random(input_shape) * 2 - 1
    labels_shape = list(data.shape)
    labels_shape.pop(axis)
    labels = np.random.randint(0, data.shape[axis], labels_shape).astype(np.int64)
    loss = layer(data, labels, axis=axis)
    if axis == 1:
        pytorch_loss = F.cross_entropy(utils.from_numpy(data), utils.from_numpy(labels), reduction=reduction)
    else:
        pytorch_loss = F.cross_entropy(
            utils.from_numpy(data.swapaxes(1, axis)), utils.from_numpy(labels), reduction=reduction
        )
    pytorch_loss = utils.to_numpy(pytorch_loss)

    utils.assert_close(loss, pytorch_loss, atol=0.001)


def test_forward_easy():
    input_shape = (20, 10)
    _test_forward(input_shape, "mean", 1)
    _test_forward(input_shape, "sum", 1)


def _test_forward_overflow(input_shape, reduction, axis):
    layer = SoftmaxCrossEntropyLossLayer(reduction=reduction)
    data = np.random.random(input_shape) * 10000 - 1
    labels_shape = list(data.shape)
    labels_shape.pop(axis)
    labels = np.random.randint(0, data.shape[axis], labels_shape).astype(np.int64)
    loss = layer(data, labels, axis=axis)
    if axis == 1:
        pytorch_loss = F.cross_entropy(utils.from_numpy(data), utils.from_numpy(labels), reduction=reduction)
    else:
        pytorch_loss = F.cross_entropy(
            utils.from_numpy(data.swapaxes(1, axis)), utils.from_numpy(labels), reduction=reduction
        )
    pytorch_loss = utils.to_numpy(pytorch_loss)

    utils.assert_close(loss, pytorch_loss, atol=0.001)


def test_forward_overflow():
    input_shape = (20, 10)
    _test_forward_overflow(input_shape, "mean", 1)
    _test_forward_overflow(input_shape, "sum", 1)


def test_forward_hard():
    input_shape = (20, 10, 5)
    _test_forward(input_shape, "mean", 1)
    _test_forward(input_shape, "sum", 1)
    _test_forward(input_shape, "mean", 2)
    _test_forward(input_shape, "sum", 2)


def _test_backward(input_shape, reduction, axis):
    layer = SoftmaxCrossEntropyLossLayer(reduction=reduction)
    data = np.random.random(input_shape) * 2 - 1
    labels_shape = list(data.shape)
    labels_shape.pop(axis)
    labels = np.random.randint(0, data.shape[axis], labels_shape).astype(np.int64)
    loss = layer(data, labels, axis=axis)
    if axis == 1:
        torch_input = utils.from_numpy(data).requires_grad_(True)
    else:
        torch_input = utils.from_numpy(np.moveaxis(data, axis, 1)).requires_grad_(True)
    pytorch_loss = F.cross_entropy(torch_input, utils.from_numpy(labels), reduction=reduction)
    if len(pytorch_loss.shape) > 0:
        pytorch_loss.sum().backward()
    else:
        pytorch_loss.backward()

    utils.assert_close(loss, utils.to_numpy(pytorch_loss))

    grad = layer.backward()
    torch_grad = utils.to_numpy(torch_input.grad)
    if axis != 1:
        torch_grad = np.moveaxis(torch_grad, 1, axis)

    utils.assert_close(grad, torch_grad, atol=0.001)


def test_backward_easy():
    input_shape = (20, 10)
    _test_backward(input_shape, "mean", 1)
    _test_backward(input_shape, "sum", 1)


def test_backward_hard():
    input_shape = (20, 10, 5)
    _test_backward(input_shape, "mean", 1)
    _test_backward(input_shape, "sum", 1)
    _test_backward(input_shape, "mean", 2)
    _test_backward(input_shape, "sum", 2)


def test_backward_harder():
    input_shape = (20, 10, 5, 23)
    _test_backward(input_shape, "mean", 0)
    _test_backward(input_shape, "sum", 0)
    _test_backward(input_shape, "mean", 1)
    _test_backward(input_shape, "sum", 1)
    _test_backward(input_shape, "mean", 2)
    _test_backward(input_shape, "sum", 2)
    _test_backward(input_shape, "mean", 3)
    _test_backward(input_shape, "sum", 3)
>>>>>>> 4cbb235ae962e8c897fc9fcb67b88739b1829319:tests/hw1_tests/test_softmax_cross_entropy_loss_layer.py
