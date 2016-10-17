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
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
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
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    
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
    C, H, W = input_dim
    pad = (filter_size - 1) / 2
    stride = 1
    conv_out_height = 1 + (H + 2 * pad - filter_size) / stride
    conv_out_width = 1 + (W + 2 * pad - filter_size) / stride
    pool_out_height = 1 + (conv_out_height - 2) / 2
    pool_out_width = 1 + (conv_out_width - 2) / 2
    self.params["W1"] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
    self.params["b1"] = np.zeros(num_filters)
    self.params["W2"] = np.random.randn(num_filters * pool_out_height * pool_out_width, hidden_dim) * weight_scale
    self.params["b2"] = np.zeros(hidden_dim)
    self.params["W3"] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params["b3"] = np.zeros(num_classes)


    if self.use_batchnorm:
      self.bn_params = {}
      self.bn_params['bn_param1'] = {'mode': 'train',
                     'running_mean': np.zeros(num_filters),
                     'running_var': np.zeros(num_filters)}
      self.params['gamma1'] = np.ones(num_filters)
      self.params['beta1'] = np.zeros(num_filters)

      self.bn_params['bn_param2'] = {'mode': 'train',
                     'running_mean': np.zeros(hidden_dim),
                     'running_var': np.zeros(hidden_dim)}
      self.params['gamma2'] = np.ones(hidden_dim)
      self.params['beta2'] = np.zeros(hidden_dim)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    if self.use_batchnorm:
      for key, bn_param in self.bn_params.iteritems():
        bn_param['mode'] = mode


      bn_param1, gamma1, beta1 = self.bn_params['bn_param1'], self.params['gamma1'], self.params['beta1']
      bn_param2, gamma2, beta2 = self.bn_params['bn_param2'], self.params['gamma2'], self.params['beta2']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    if self.use_batchnorm:
      pool_out, pool_cache = conv_norm_relu_pool_forward(X, W1, b1, conv_param, pool_param, gamma1, beta1, bn_param1)
    else:
      pool_out, pool_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

    N, F, H, W = pool_out.shape

    if self.use_batchnorm:
      relu_out, relu_cache = affine_batchnorm_relu_forward(pool_out.reshape((N, F * H * W)), W2, b2, gamma2, beta2, bn_param2, False, {})
    else:
      relu_out, relu_cache = affine_relu_forward(pool_out.reshape((N, F * H * W)), W2, b2, False, {})

    scores, affine_cache = affine_forward(relu_out, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dx1 = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))


    dx2, dW3, db3 = affine_backward(dx1, affine_cache)
    dW3 += self.reg * W3

    if self.use_batchnorm:
      dx3, dW2, db2, dgamma2, dbeta2 = affine_batchnorm_relu_backward(dx2, relu_cache, False)
    else:
      dx3, dW2, db2 = affine_relu_backward(dx2, relu_cache, False)

    dW2 += self.reg * W2

    if self.use_batchnorm:
      dx4, dW1, db1, dgamma1, dbeta1 = conv_norm_relu_pool_backward(dx3.reshape(N, F, H, W), pool_cache)
    else:
      dx4, dW1, db1 = conv_relu_pool_backward(dx3.reshape(N, F, H, W), pool_cache)

    dW1 += self.reg * W1

    grads["W1"], grads["b1"] = dW1, db1
    grads["W2"], grads["b2"] = dW2, db2
    grads["W3"], grads["b3"] = dW3, db3
    if self.use_batchnorm:
        grads["gamma1"], grads["beta1"] = dgamma1, dbeta1
        grads["gamma2"], grads["beta2"] = dgamma2, dbeta2

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
