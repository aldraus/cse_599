import numbers

import numpy as np
from numba import njit, prange

from .layer import Layer


class MaxPoolNumbaLayer(Layer):

    def __init__(self, kernel_size: int = 2, stride: int = 2, parent=None):
        super(MaxPoolNumbaLayer, self).__init__(parent)
        self.stride = stride
        self.input = []
        self.coords = []
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, data):
        stride = self.stride
        kernel_size = self.kernel_size
        p = self.padding
        x_padded = np.pad(data, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        self.input = x_padded
        downsampled, coords  = self.forward_numba(x_padded,stride,kernel_size)
        self.coords = coords
        return downsampled

    @staticmethod
    @njit(parallel=True, cache=True)
    def forward_numba(data,stride,kernel_size):
        # data is N x C x H x W
        f = kernel_size
        s = stride
        data_size, c_n, w_prev,h_prev,  = data.shape
        h_n = int((h_prev - f) / s) + 1
        w_n = int((w_prev - f) / s) + 1
        A = np.zeros((data_size,c_n, w_n,h_n))
        coords = np.zeros((data_size,c_n, w_n,h_n))
        for i in prange(data_size):
            for c in prange(c_n):
                for w in prange(w_n):
                    for h in prange(h_n):
                        vert_start = h * stride
                        vert_end = h * stride + f
                        horiz_start = w * stride
                        horiz_end = w * stride + f
                        a_prev_slice = data[i, c, horiz_start:horiz_end,vert_start:vert_end,]
                        A[i,c, w,h] = np.max(a_prev_slice)
                        coords[i,c,w,h] = np.argmax(a_prev_slice)

        return A, coords


    def backward(self, previous_partial_gradient):
        stride = self.stride
        kernel_size = self.kernel_size
        input_data = self.input
        padval = self.padding
        coords = self.coords
        data_size, c_n, w_prev, h_prev = coords.shape
        realcoords = np.zeros((data_size, c_n, w_prev, h_prev,2))
        for i in prange(data_size):
            for c in prange(c_n):
                for w in prange(w_prev):
                    for h in prange(h_prev):
                        realcoords[i,c,w,h,:]=np.unravel_index(int(coords[i,c,w,h]),(kernel_size,kernel_size))
        upsampled  = self.backward_numba(previous_partial_gradient,input_data,stride,kernel_size,padval,realcoords)
        m,c,w,h = upsampled.shape
        upsampled = upsampled[:,:,padval:w-padval,padval:h-padval]


        return upsampled

    @staticmethod
    @njit(cache=True, parallel=True)
    def backward_numba(previous_grad, data, stride_val,kernel_size,padval,realcoords):
        dA = previous_grad
        A_prev = data
        p = padval
        stride = stride_val
        f = kernel_size
        m, n_C_prev, n_W_prev, n_H_prev = A_prev.shape
        m, n_C,n_W, n_H,  = dA.shape
        dA_prev = np.zeros((m, n_C_prev, n_W_prev, n_H_prev))
        for i in range(m):
            a_prev = A_prev[i]
            for w in range(0,n_W):
                for h in range(0,n_H):
                    for c in range(n_C):
                        vert_start = h*stride
                        vert_end = h*stride + f
                        horiz_start = w*stride
                        horiz_end = w*stride + f
                        a_prev_slice = a_prev[c, horiz_start:horiz_end,vert_start:vert_end]
                        mask = np.zeros_like((a_prev_slice))
                        indx = realcoords[i,c,w,h,:]
                        mask[int(indx[0]),int(indx[1])]= 1
                        dA_prev[i, c, horiz_start:horiz_end,vert_start:vert_end] += np.multiply(mask,dA[i,c,w-p,h-p])

        return dA_prev


class MaxPoolLayer(Layer):
    def __init__(self, kernel_size: int = 2, stride: int = 2, parent=None):
        super(MaxPoolLayer, self).__init__(parent)
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.stride = stride
        self.reshapedinput = []
        self.input = []
        self.coords = []
        self.samples = 0
        self.chnls = 0
        self.width = 0
        self.height = 0
        self.newheight = 0
        self.newwidth = 0

    def forward(self, data):

        A = data.copy()
        self.input = A
        coords = []
        self.samples, self.chnls, self.width, self.height = np.shape(A)
        sampls, chnls, wdth, hght = np.shape(A)
        Areshaped = im2col_indices(A, self.kernel_size, self.kernel_size, padding = self.padding,stride=self.stride)
        self.reshapedinput = Areshaped
        channelsdata = np.zeros((chnls, Areshaped.shape[1]))
        for i in range(0, chnls):
            val = Areshaped[int(i * Areshaped.shape[0] / chnls):int((i + 1) * Areshaped.shape[0] / chnls), :]
            max_idx = np.argmax(val, axis=0)
            coords.append(max_idx)
            channelsdata[i,:] = val[max_idx, range(max_idx.size)]
        self.coords = np.asarray(coords)
        out = channelsdata.reshape(chnls,sampls,-1,order = 'F')
        self.newheight = int((hght - self.kernel_size + 2 * self.padding) / self.stride + 1)
        self.newwidth = int((wdth - self.kernel_size + 2 * self.padding) / self.stride + 1)
        out = out.swapaxes(0, 1).reshape(sampls,chnls, self.newwidth,self.newheight)

        return out

    def backward(self, previous_partial_gradient):

        dX_col = np.zeros_like(self.reshapedinput)
        dout_flat = previous_partial_gradient.transpose(2, 3, 0, 1).ravel()
        coords = self.coords.reshape(dout_flat.shape, order='F')
        dX_col = np.reshape(dX_col, (np.square(self.kernel_size), self.newheight*self.newwidth*self.chnls*self.samples))
        dX_col[coords, range(coords.size)] = dout_flat
        shape = (self.samples * self.chnls, 1, self.width, self.height)
        dX = col2im_indices(dX_col, shape, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        dX = dX.reshape(self.input.shape)

        return dX

    def selfstr(self):
        return str("kernel: " + str(self.kernel_size) + " stride: " + str(self.stride))

# im2col and col2im indices are taken from https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py and slightly modified. This basically reforms the input image in a way that the activation map can be calculated with a single dot product as shown in the class.


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