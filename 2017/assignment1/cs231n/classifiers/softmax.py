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
  num_classes = W.shape[1]  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  for i in range(num_train):
        score = scores[i,:]
        normalised_score = score - np.max(score)
        loss += -normalised_score[y[i]] + np.log(np.sum(np.exp(normalised_score)))
  
        for j in range(num_classes):
            softmax_score = np.exp(normalised_score[j])/np.sum(np.exp(normalised_score))
            dW[:,j] += (softmax_score - (j==y[i]))*X[i] 
        
            
  
  loss /= num_train
  loss += reg*np.sum(W*W)

  dW /= num_train
  dW += 2*reg*W 
    
    

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot(X,W)
  normalised_scores = scores - np.max(scores,axis=1)[...,np.newaxis]
  softmax_scores = np.exp(normalised_scores)/ np.sum(np.exp(normalised_scores), axis=1)[..., np.newaxis]
  
  dScore = softmax_scores
  dScore[range(num_train),y] = dScore[range(num_train),y] - 1
  dW = np.dot(X.T, dScore)  
  correct_class_scores = np.choose(y, normalised_scores.T)  # Size N vector
  loss = -correct_class_scores + np.log(np.sum(np.exp(normalised_scores), axis=1))
  loss = np.sum(loss)
  loss /= num_train
  loss += reg * np.sum(W*W)

  dW /= num_train
  dW += 2*reg*W
    

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

