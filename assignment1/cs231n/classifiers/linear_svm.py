import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    numPosMargin = 0
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        numPosMargin += 1
        loss += margin
        dW[ :,j] += X[i,:]
    dW[ :,y[i]] -= (numPosMargin * X[i,:])

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = len(X)

  # Compute loss
  scores = np.dot(X, W)
  correct_scores = scores[np.arange(len(scores)), y]
  loss = np.maximum(scores - correct_scores[:, np.newaxis] + 1, 0)
  loss[np.arange(len(loss)), y] = 0

  # Also compute gradient
  isLossPositive = (loss > 0) + 0
  numPosLoss = np.sum(isLossPositive, axis=1)
  isLossPositive[np.arange(len(isLossPositive)), y] = -numPosLoss
  dW = X.T.dot(isLossPositive)

  loss = np.sum(loss, axis=1)
  loss = sum(loss) / num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  return loss, dW
