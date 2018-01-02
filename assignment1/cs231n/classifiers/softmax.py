import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = dW.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  max_scores = np.max(scores, axis=1)[:, np.newaxis]
  scores -= max_scores
  correct_class_scores = scores[np.arange(len(scores)), y][:, np.newaxis]
  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores, axis=1)[:, np.newaxis]
  loss = -correct_class_scores + np.log(sum_exp_scores)

  # Compute the gradient using loops
  for i in range(num_train):
    for j in range(num_classes):
      if j == y[i]:
        dW[:, y[i]] += -X[i]
      else:
        dW[:, j] += (1.0 / sum_exp_scores[i]) * exp_scores[i,j] * X[i]

  # Average the loss and gradient over the traiing data
  loss = np.sum(loss) / num_train
  dW /= num_train

  # Add regularization
  loss += reg * np.sum(W * W) 
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  max_scores = np.max(scores, axis=1)[:, np.newaxis]
  scores -= max_scores
  correct_class_scores = scores[np.arange(len(scores)), y][:, np.newaxis]
  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores, axis=1)
  loss = -correct_class_scores + np.log(sum_exp_scores[:, np.newaxis])

  # Computethe gradient in vectorized format
  exp_scores[np.arange(len(exp_scores)), y] = -sum_exp_scores
  dW = X.T.dot(exp_scores * 1/ sum_exp_scores[:, np.newaxis])


  # Average the loss and gradient over the traiing data
  loss = np.sum(loss) / num_train
  dW /= num_train

  # Add regularization
  loss += reg * np.sum(W * W) 
  dW += 2 * reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

