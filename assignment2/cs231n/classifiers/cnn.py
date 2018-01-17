from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, dropout=0, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.usedropout = dropout > 0
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.pool_height = 2
        self.pool_width = 2
        self.pool_stride = 2

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # NUmber of channels
        C, H, W = input_dim
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)
        H_pool = (H - self.pool_height) / 2 + 1
        W_pool = (W - self.pool_width) / 2 + 1
        self.params['W2'] = np.random.randn(np.prod((num_filters, H_pool, W_pool)), hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)

        # Set dropout parameters if necessary
        self.dropout_param={}
        if self.usedropout:
            self.dropout_param ={'mode':'train', 'p':dropout}

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': self.pool_height, 'pool_width': self.pool_width, 
                      'stride': self.pool_stride}

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        first_layer_scores, first_layer_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        second_layer_scores, second_layer_cache = affine_relu_forward(first_layer_scores, W2, b2, dropout=self.usedropout,
                                                                      dropout_param=self.dropout_param)
        scores, output_cache = affine_forward(second_layer_scores, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        grads = {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # Compute loss
        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
        
        # Compute the gradient
        grads['W1'] = 2 * self.reg * np.sum(W1 * W1)
        grads['W2'] = 2 * self.reg * np.sum(W2 * W2)
        grads['W3'] = 2 * self.reg * np.sum(W3 * W3)

        dx, dw, db = affine_backward(dscores, output_cache)
        grads['W3'] += dw
        grads['b3'] = db
        dx, dw, db = affine_relu_backward(dx, second_layer_cache, dropout=self.usedropout)
        grads['W2'] += dw
        grads['b2'] = db
        _, dw, db = conv_relu_pool_backward(dx, first_layer_cache)
        grads['W1'] += dw
        grads['b1'] = db

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
