pass
from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

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
    return out, cache


def affine_relu_dropout_forward(x, w, b, dropout_param):
    out, cache = affine_relu_forward(x, w, b)
    dout, dout_cache = dropout_forward(out, dropout_param)
    all_caches = cache + (dout_cache,)
    return dout, all_caches


def affine_relu_dropout_backward(dout, cache):
    dropout_cache = cache[-1]
    dr_out = dropout_backward(dout, dropout_cache)
    cache = cache[:2]
    return affine_relu_backward(dr_out, cache)


def affine_batchnorm_relu_forward(x, w, b, bn_param):
    aout, acache = affine_forward(x, w, b)
    all_caches = [acache]
    N,D = aout.shape
    beta = bn_param.get("beta", None)
    if beta is None:
        #beta = np.sum(aout, axis=0)/N
        beta = np.random.randn(D)
        bn_param['beta'] = beta
    gamma = bn_param.get("gamma", None)
    if gamma is None:
#        gamma = np.sum((aout - beta)**2, axis=0)
#        gamma = np.sqrt(gamma)
        gamma = np.random.randn(D)
        bn_param['gamma'] = gamma
    bout, bcache = batchnorm_forward(aout, gamma, beta, bn_param)
    all_caches.append(bcache)
    rout, rcache = relu_forward(bout)
    all_caches.append(rcache)
    return rout, all_caches


def affine_batchnorm_relu_backward(dout, cache):
    relu_cache = cache[-1]
    relu_out = relu_backward(dout, relu_cache)
    b_cache = cache[-2]
    bout, dg, db = batchnorm_backward(relu_out, b_cache)
    af_cache = cache[0]
    return affine_backward(bout, af_cache) + (dg, db)


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


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


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


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
