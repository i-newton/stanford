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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  #############################################################################
  for i in range(num_train):
    scores = X[i].dot(W)
    C = np.amax(scores)
    scores -= C
    loss += -np.log(np.exp(scores[y[i]])/np.sum(np.exp(scores)))
    for j in np.arange(0, scores.shape[0]):
      dw_j = X[i]*np.exp(scores[j])/np.sum(np.exp(scores))
      if j == y[i]:
        dw_j -= X[i]
      dW[:,j] += dw_j
  loss /= float(num_train)
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
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
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  max_scores = np.max(scores, axis=1)
  scores -= max_scores[:, None]
  correct_scores = scores[np.arange(num_train), y]
  exp_correct_scores = np.exp(correct_scores)
  exp_scores = np.exp(scores)
  sum_scores = np.sum(exp_scores, axis=1)
  losses = -np.log(exp_correct_scores / sum_scores)
  loss = np.sum(losses)
  grad_correct = np.zeros_like(scores)
  grad_correct[np.arange(num_train), y] = 1
  grad_correct = X.transpose().dot(grad_correct)
  sk = exp_scores / sum_scores[:, None]
  grad_all = X.transpose().dot(sk)
  dW = grad_all - grad_correct
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= float(num_train)
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  return loss, dW

