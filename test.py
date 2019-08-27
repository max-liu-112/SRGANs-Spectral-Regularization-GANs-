import chainer.functions as F
from chainer import cuda
import numpy as np
from chainer import Variable

def _l2normalize(v, eps=1e-12):
    norm = cuda.reduce('T x', 'T out',
                       'x * x', 'a + b', 'out = sqrt(a)', 0,
                       'norm_sn')
    div = cuda.elementwise('T x, T norm, T eps',
                           'T out',
                           'out = x / (norm + eps)',
                           'div_sn')
    return div(v, norm(v), eps)


def max_singular_value(W, u=None, Ip=1):
    """
    Apply power iteration for the weight parameter
    """
    if not Ip >= 1:
        raise ValueError("The number of power iterations should be positive integer")

    xp = cuda.get_array_module(W.data)
    if u is None:
        u = xp.random.normal(size=(1, W.shape[0])).astype(xp.float32)
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(xp.dot(_u, W.data), eps=1e-12)
        _u = _l2normalize(xp.dot(_v, W.data.transpose()), eps=1e-12)
    sigma = F.sum(F.linear(_u, F.transpose(W)) * _v)
    return sigma, _u, _v




def max_singular_value_fully_differentiable(W,  Ip=1):
    """
    Apply power iteration for the weight parameter (fully differentiable version)
    """

    xp = cuda.get_array_module(W.data)
    u = xp.random.normal(size=(1, W.shape[0])).astype(dtype="f")

    for _ in range(Ip):
        _v = F.normalize(F.matmul(u, W), eps=1e-12)
        _u = F.normalize(F.matmul(_v, F.transpose(W)), eps=1e-12)
    sigma = F.sum(F.linear(_u, F.transpose(W)) * _v)
    return sigma



W =10* np.random.normal(size=(2, 3)).astype(dtype="f")
W = Variable(W)
U, s, V = np.linalg.svd(W.data, full_matrices=True)
print(U.shape)
print(s.shape)
print(V.shape)
S = np.zeros(W.data.shape)
s_num = min(W.data.shape)
S[:s_num,:s_num] = np.identity(s.shape[0])
W_hat = np.dot(U, np.dot(S, V))
error = np.sum(np.square(W.data/max(s)-W_hat))
print(np.allclose(W.data/max(s), W_hat))
print(error)
print(W.data/max(s))
W.data = W_hat
print(W.data)
U, s, V = np.linalg.svd(W.data, full_matrices=True)
print(s)

