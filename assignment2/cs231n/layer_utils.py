from cs231n.layers import *
from cs231n.fast_layers import *

def conv_norm_relu_forward(x, w, b, conv_param, gamma, beta, bn_param):
    """Convenience layer that performs a convolution, spatial
    batchnorm, a ReLU, and a pool.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    conv, conv_cache = conv_forward_fast(x, w, b, conv_param)
    norm, norm_cache = spatial_batchnorm_forward(conv, gamma, beta, bn_param)
    out, relu_cache = relu_forward(norm)

    cache = (conv_cache, norm_cache, relu_cache)

    return out, cache


def conv_norm_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, norm_cache, relu_cache = cache

    drelu = relu_backward(dout, relu_cache)
    dnorm, dgamma, dbeta = spatial_batchnorm_backward(drelu, norm_cache)
    dx, dw, db = conv_backward_fast(dnorm, conv_cache)

    return dx, dw, db, dgamma, dbeta


def conv_norm_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
    """Convenience layer that performs a convolution, spatial
    batchnorm, a ReLU, and a pool.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    conv, conv_cache = conv_forward_fast(x, w, b, conv_param)
    norm, norm_cache = spatial_batchnorm_forward(conv, gamma, beta, bn_param)
    relu, relu_cache = relu_forward(norm)
    out, pool_cache = max_pool_forward_fast(relu, pool_param)

    cache = (conv_cache, norm_cache, relu_cache, pool_cache)

    return out, cache


def conv_norm_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, norm_cache, relu_cache, pool_cache = cache

    dpool = max_pool_backward_fast(dout, pool_cache)
    drelu = relu_backward(dpool, relu_cache)
    dnorm, dgamma, dbeta = spatial_batchnorm_backward(drelu, norm_cache)
    dx, dw, db = conv_backward_fast(dnorm, conv_cache)

    return dx, dw, db, dgamma, dbeta

def affine_relu_forward(x, w, b, use_dropout, dropout_param):
  """
  Convenience layer that performs an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  if use_dropout:
    out, dropout_cache = dropout_forward(out, dropout_param)
    cache = (fc_cache, relu_cache, dropout_cache)
  return out, cache

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param, use_dropout, dropout_param):
  """
  Convenience layer that performs an affine transform 
  followed by a Batch Normalization 
  followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma, beta: scale and shift parameters for batchnorm layer
  - bn_param: 

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  b, batch_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(b)
  cache = (fc_cache, batch_cache, relu_cache)
  if use_dropout:
    out, dropout_cache = dropout_forward(out, dropout_param)
    cache = (fc_cache, batch_cache, relu_cache, dropout_cache)
  return out, cache


def affine_relu_backward(dout, cache, use_dropout):
  """
  Backward pass for the affine-relu convenience layer
  """
  if use_dropout:
    fc_cache, relu_cache, dropout_cache = cache
    dout = dropout_backward(dout, dropout_cache)
  else:
    fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def affine_batchnorm_relu_backward(dout, cache, use_dropout):
  """
  Backward pass for the affine-relu convenience layer
  """
  if use_dropout:
    fc_cache, batch_cache, relu_cache, dropout_cache = cache
    dout = dropout_backward(dout, dropout_cache)
  else:
    fc_cache, batch_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dbatch, dgamma, dbeta = batchnorm_backward(da, batch_cache)
  dx, dw, db = affine_backward(dbatch, fc_cache)
  return dx, dw, db, dgamma, dbeta


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

