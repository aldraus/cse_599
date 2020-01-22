from typing import Optional, Callable
import numpy as np

from numba import njit, prange

from nn import Parameter
from .layer import Layer


class ConvLayer(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(ConvLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.initialize()
        self.conv_layer_input = np.array([])
        self.conv_layer_reshaped = np.array([])
        self.newwidth = 0
        self.newheight = 0
        self.initial = 0


    def forward(self, data):
        self.conv_layer_input = data.copy()
        A = data.copy()
        self.initial = A
        sampls, chnls, wdth, hght = np.shape(A)
        Areshaped = im2col_indices(A,self.kernel_size,self.kernel_size,padding=self.padding,stride= self.stride)
        self.conv_layer_reshaped = Areshaped
        weights = self.weight.data
        wdims = np.shape(weights)
        weights_reshaped = np.reshape(weights, (wdims[0], wdims[1], np.square(self.kernel_size)))
        weights_reshaped = np.moveaxis(weights_reshaped,0,1)
        weights_reshaped = np.reshape(weights_reshaped,(weights_reshaped.shape[0],-1))
        bias_reshaped = self.bias.data[:,np.newaxis]
        output = np.dot(weights_reshaped, Areshaped) + bias_reshaped
        ## Output should be (W−F+2P)/S+1
        self.newwidth = int((wdth - self.kernel_size + 2 * self.padding) / self.stride + 1)
        self.newheight = int((hght - self.kernel_size + 2 * self.padding) / self.stride + 1)
        out = output.reshape(wdims[1], self.newwidth,self.newheight,sampls)
        out = np.moveaxis(out, -1, 0)

        return out


    def backward(self, previous_partial_gradient):
        # TODO
        input_channel, output_channel, width, height = self.weight.data.shape
        dout_flat = previous_partial_gradient.transpose(1, 2, 3, 0).reshape(output_channel, -1)

        dW = np.dot(dout_flat, self.conv_layer_reshaped.T)
        dW = np.moveaxis(dW,0,-1)
        dW = dW.reshape(input_channel,width,height,-1).transpose(3,0,1,2)
        dW = dW.swapaxes(0, 1)
        self.weight.grad = dW
        db = np.sum(previous_partial_gradient, axis=(0, 2, 3))
        self.bias.grad = db
        W_flat = np.swapaxes(self.weight.data, 0, 1).reshape(output_channel, -1)
        dX_col = np.dot(W_flat.T, dout_flat)
        dX = col2im_indices(dX_col, self.initial.shape, self.kernel_size, self.kernel_size, self.padding, self.stride)

        return dX

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
        super(ConvLayer, self).initialize()


class ConvNumbaLayer(Layer):

    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, parent=None):
        super(ConvNumbaLayer, self).__init__(parent)
        self.weight = Parameter(np.zeros((input_channels, output_channels, kernel_size, kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros(output_channels, dtype=np.float32))
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.initialize()
        self.conv_layer_input = np.array([])
        self.conv_layer_reshaped = np.array([])
        self.newwidth = 0
        self.newheight = 0
        self.initial = 0

    def forward(self, data):
        self.conv_layer_input = data.copy()
        stride = self.stride
        kernel_size = self.kernel_size
        p = self.padding
        x_padded = np.pad(data, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        self.input = x_padded
        out = self.forward_numba(x_padded, self.weight.data, self.bias.data, stride, kernel_size)
        return out

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(self, data, weights, bias, stride, kernel_size):
        f = kernel_size
        s = stride
        ichn, ochn, k1, k2 = weights.shape
        data_size, c_n, w_prev, h_prev, = data.shape
        h_n = int((h_prev - f) / s) + 1
        w_n = int((w_prev - f) / s) + 1

        A = np.zeros((data_size, ochn, w_n, h_n))

        for i in prange(data_size):
            for c in prange(ochn):
                for w in prange(w_n):
                    for h in prange(h_n):
                        vert_start = h * stride
                        vert_end = h * stride + f
                        horiz_start = w * stride
                        horiz_end = w * stride + f
                        a_prev_slice = data[i, :, horiz_start:horiz_end, vert_start:vert_end]
                        s = 0
                        for col in range(kernel_size):
                            for ro in range(kernel_size):
                                val = int(a_prev_slice[:, col, ro])[0] * int(weights[:, c, col, ro])[0]
                                s = s + val
                        Z = Z + bias[c]
                        A[i, c, w, h] = Z
        return A

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data, kernel, kernel_grad):
        # data is N x C x H x W
        # kernel is COld x CNew x K x K
        # After implementing it in max pooling layer, I decided numba version is extremely slow compared to im2col version. For this reason, I will only use im2col version for all my implementations
        return None

    def selfstr(self):
        return "Kernel: (%s, %s) In Chann" \
               "els %s Out Channels %s Stride %s" % (
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
        super(ConvNumbaLayer, self).initialize()

# im2col and col2im indices are taken from https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py and slightly modified. This basically reforms the inpu image in a way that the activation map can be calculated with a single dot product as shown in the class.


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = np.int((H + 2 * padding - field_height) / stride + 1)
    out_width = np.int((W + 2 * padding - field_width) / stride + 1)
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return (k, i, j)

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]

def im2col_indices(x, field_height, field_width, padding=0, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)

    return cols