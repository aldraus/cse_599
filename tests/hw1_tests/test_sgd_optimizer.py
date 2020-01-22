<<<<<<< HEAD:test/hw1_tests/test_sgd_optimizer.py
import numpy as np
import torch

import nn
import nn.layers
from nn.optimizers import sgd_optimizer
from test import utils


class TorchNet(torch.nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.layer = torch.nn.Linear(100, 10)

    def forward(self, x):
        return self.layer(x)


def test_sgd_update():
    np.random.seed(0)
    torch.manual_seed(0)
    net = nn.layers.LinearLayer(100, 10)
    learning_rate = 1
    optimizer = sgd_optimizer.SGDOptimizer(net.parameters(), learning_rate)

    data = np.random.random((20, 100)).astype(np.float32) * 2 - 1

    initial_weight = net.weight.data.copy()
    initial_bias = net.bias.data.copy()

    torch_net = TorchNet()
    utils.assign_linear_layer_weights(net, torch_net.layer)
    torch_optimizer = torch.optim.SGD(torch_net.parameters(), learning_rate)

    optimizer.zero_grad()
    out = net(data)
    loss = out.sum()
    net.backward(np.ones_like(out))

    torch_optimizer.zero_grad()
    torch_out = torch_net(utils.from_numpy(data))
    utils.assert_close(out, utils.to_numpy(torch_out.clone().detach()), atol=0.01)

    torch_loss = torch_out.sum()
<<<<<<< HEAD
    assert np.allclose(loss, torch_loss.item(), atol=0.01)
    assert np.allclose(out, utils.to_numpy(torch_out.clone().detach()), atol=0.001)

    torch_loss = torch_out.sum()
    assert np.allclose(loss, torch_loss.item(), atol=0.001)
=======
    utils.assert_close(loss, torch_loss.item(), atol=0.01)
>>>>>>> 3eccfe31aacb9d0a4a15dc27c51a2bf86c11ab0b
    torch_loss.backward()

    utils.check_linear_grad_match(net, torch_net.layer)

    optimizer.step()
    torch_optimizer.step()

    utils.assert_close(net.weight.data.T, utils.to_numpy(torch_net.layer.weight))
    utils.assert_close(net.bias.data, utils.to_numpy(torch_net.layer.bias))

    assert not np.allclose(net.weight.data, initial_weight)
    assert not np.allclose(net.bias.data, initial_bias)
=======
import numpy as np
import torch

import nn
import nn.layers
from nn.optimizers import sgd_optimizer
from tests import utils


class TorchNet(torch.nn.Module):
    def __init__(self):
        super(TorchNet, self).__init__()
        self.layer = torch.nn.Linear(100, 10)

    def forward(self, x):
        return self.layer(x)


def test_sgd_update():
    np.random.seed(0)
    torch.manual_seed(0)
    net = nn.layers.LinearLayer(100, 10)
    learning_rate = 1
    optimizer = sgd_optimizer.SGDOptimizer(net.parameters(), learning_rate)

    data = np.random.random((20, 100)).astype(np.float32) * 2 - 1

    initial_weight = net.weight.data.copy()
    initial_bias = net.bias.data.copy()

    torch_net = TorchNet()
    utils.assign_linear_layer_weights(net, torch_net.layer)
    torch_optimizer = torch.optim.SGD(torch_net.parameters(), learning_rate)

    optimizer.zero_grad()
    out = net(data)
    loss = out.sum()
    net.backward(np.ones_like(out))

    torch_optimizer.zero_grad()
    torch_out = torch_net(utils.from_numpy(data))
    utils.assert_close(out, utils.to_numpy(torch_out.clone().detach()), atol=0.01)

    torch_loss = torch_out.sum()
    utils.assert_close(loss, torch_loss.item(), atol=0.01)
    torch_loss.backward()

    utils.check_linear_grad_match(net, torch_net.layer)

    optimizer.step()
    torch_optimizer.step()

    utils.assert_close(net.weight.data.T, utils.to_numpy(torch_net.layer.weight))
    utils.assert_close(net.bias.data, utils.to_numpy(torch_net.layer.bias))

    assert not np.allclose(net.weight.data, initial_weight)
    assert not np.allclose(net.bias.data, initial_bias)
>>>>>>> 4cbb235ae962e8c897fc9fcb67b88739b1829319:tests/hw1_tests/test_sgd_optimizer.py
