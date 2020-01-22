import numpy as np

from .loss_layer import LossLayer


class SoftmaxCrossEntropyLossLayer(LossLayer):

    def __init__(self, reduction="mean", parent=None):
        """
        #TODO: consider reduction case!!!!!!!!!!!!!!!!!!!!!!!
        :param reduction: mean reduction indicates the results should be summed and scaled by the size of the input (excluding the axis dimension).
            sum reduction means the results should be summed.
        """
        self.reduction = reduction
        super(SoftmaxCrossEntropyLossLayer, self).__init__(parent)
        self.softmaxval = np.array([])
        self.truelabels = np.array([])
        self.batchsize = 0

    def forward(self, logits, targets, axis=-1) -> float:

        logits_shifted = np.moveaxis(logits, axis, -1)
        logitsreshaped = np.reshape(logits_shifted,(-1, logits_shifted.shape[-1]))
        logitsreshapednew_maxsubtracted = logitsreshaped - np.moveaxis(np.max(logitsreshaped, -1)[np.newaxis], 0, -1)
        targets = targets.reshape(-1)
        target_one_hot_mapping = np.eye(logits.shape[axis])
        targets = target_one_hot_mapping[targets]
        targetsreturned = np.reshape(targets,(logits_shifted.shape))
        self.truelabels = np.moveaxis(targetsreturned, -1, axis)
        logq = logitsreshapednew_maxsubtracted - (np.log(np.sum(np.exp(logitsreshapednew_maxsubtracted), axis=1)))[:,np.newaxis]
        softmax = np.exp(logitsreshapednew_maxsubtracted) / np.sum(np.exp(logitsreshapednew_maxsubtracted),axis=1)[:,np.newaxis]
        softmax = softmax.reshape(logits_shifted.shape)
        self.softmaxval = np.moveaxis(softmax, -1, axis)
        self.batchsize = logq.shape[0]
        if self.reduction == 'sum':
            loss_val = -(np.sum(targets * logq))
        else:
            loss_val = -(np.sum(targets * logq)) / self.batchsize  # if don't scale test fails

        return loss_val





    def backward(self) -> np.ndarray:
        """
        Takes no inputs (should reuse computation from the forward pass)
        :return: gradients wrt the logits the same shape as the input logits
        """
        if self.reduction == 'mean':
            grad_softmax = (self.softmaxval - self.truelabels) /self.batchsize # division with batch size might be unneccessary ,check the assingment discussions page
        else:
            grad_softmax = (self.softmaxval - self.truelabels) # division with batch size might be unneccessary ,check the assingment discussions page

        return grad_softmax

