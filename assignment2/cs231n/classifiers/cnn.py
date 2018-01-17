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
                 use_batch_norm=False, dtype=np.float32):
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
        self.use_dropout = dropout > 0
        self.use_batch_norm = use_batch_norm
        self.params = {}
        self.reg = reg
        self.num_layers = 3
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

        # Initialize the parameters for batch normalization if necessary
        if self.use_batch_norm:
            self.params['gamma1'] = np.ones(num_filters) 
            self.params['beta1'] = np.zeros(num_filters)
            self.params['gamma2'] = np.ones(hidden_dim)
            self.params['beta2'] = np.zeros(hidden_dim)

        # Set dropout parameters if necessary
        self.dropout_param={}
        if self.use_dropout:
            self.dropout_param ={'mode':'train', 'p':dropout}

        self.bn_params = []
        if self.use_batch_norm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

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
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        if self.use_batch_norm:
            gamma1 = self.params['gamma1']
            beta1 = self.params['beta1']
            gamma2 = self.params['gamma2']
            beta2 = self.params['beta2']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': self.pool_height, 'pool_width': self.pool_width, 
                      'stride': self.pool_stride}

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batch_norm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        # Convolutional layer going forward
        if self.use_batch_norm:
            first_layer_scores, first_layer_cache = conv_bn_relu_pool_forward(X, W1, b1,
                                                                              gamma1, beta1,
                                                                              conv_param,
                                                                              self.bn_params[0],
                                                                              pool_param)
        else:
            first_layer_scores, first_layer_cache = conv_relu_pool_forward(X, W1, b1, 
                                                                           conv_param,
                                                                           pool_param)

        # Fully connected layers going forward
        if self.use_batch_norm:    
            second_layer_scores, second_layer_cache = affine_bn_relu_forward(first_layer_scores,
                                                                             W2, b2, gamma2, beta2, 
                                                                             self.bn_params[1], 
                                                                             dropout=self.use_dropout, 
                                                                             dropout_param=self.dropout_param)
        else:
            second_layer_scores, second_layer_cache = affine_relu_forward(first_layer_scores, 
                                                                          W2, b2, 
                                                                          dropout=self.use_dropout,
                                                                          dropout_param=self.dropout_param)

        # Output layer going forward
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

        # Output layer going backward
        dx, dw, db = affine_backward(dscores, output_cache)
        grads['W3'] += dw
        grads['b3'] = db

        # Fully connected layers going backward
        if self.use_batch_norm:
            dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dx, second_layer_cache, dropout=self.use_dropout)
            grads['gamma2'] = dgamma
            grads['beta2'] = dbeta

        else:
            dx, dw, db = affine_relu_backward(dx, second_layer_cache, dropout=self.use_dropout)
        grads['W2'] += dw
        grads['b2'] = db

        # Convolutional layers going backward.
        if self.use_batch_norm:
            _, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(dx, first_layer_cache)
            grads['gamma1'] = dgamma
            grads['beta1'] = dbeta

        else:
            _, dw, db = conv_relu_pool_backward(dx, first_layer_cache)
        grads['W1'] += dw
        grads['b1'] = db

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
