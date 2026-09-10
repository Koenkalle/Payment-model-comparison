import json
import math
import sys
from pathlib import Path
import numpy as np

sys.setrecursionlimit(30000)
H, K = 8, 5
ROOT = Path(__file__).resolve().parent


def unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, size in enumerate(shape):
        if size == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


class T:
    """Small reverse-mode autodiff engine for the exact operations used here."""
    def __init__(self, data, parents=(), back=None):
        self.d = np.asarray(data, dtype=np.float64)
        self.g = np.zeros_like(self.d)
        self.parents, self.back = parents, back

    def __add__(self, other):
        o = other if isinstance(other, T) else T(other)
        z = T(self.d + o.d, (self, o))
        def back():
            self.g += unbroadcast(z.g, self.d.shape)
            o.g += unbroadcast(z.g, o.d.shape)
        z.back = back
        return z
    __radd__ = __add__

    def __mul__(self, other):
        o = other if isinstance(other, T) else T(other)
        z = T(self.d * o.d, (self, o))
        def back():
            self.g += unbroadcast(z.g * o.d, self.d.shape)
            o.g += unbroadcast(z.g * self.d, o.d.shape)
        z.back = back
        return z
    __rmul__ = __mul__

    def __matmul__(self, other):
        z = T(self.d @ other.d, (self, other))
        def back():
            self.g += z.g @ other.d.T
            other.g += self.d.reshape(-1, self.d.shape[-1]).T @ z.g.reshape(-1, z.g.shape[-1])
        z.back = back
        return z

    def tanh(self):
        z = T(np.tanh(self.d), (self,))
        z.back = lambda: self.acc(z.g * (1 - z.d * z.d))
        return z

    def sigmoid(self):
        z = T(1 / (1 + np.exp(-np.clip(self.d, -50, 50))), (self,))
        z.back = lambda: self.acc(z.g * z.d * (1 - z.d))
        return z

    def acc(self, g):
        self.g += g

    def sum(self, axis, keepdims=False):
        z = T(self.d.sum(axis, keepdims=keepdims), (self,))
        z.back = lambda: self.acc(np.broadcast_to(z.g if keepdims else np.expand_dims(z.g, axis), self.d.shape))
        return z

    def expand(self, axis):
        z = T(np.expand_dims(self.d, axis), (self,))
        z.back = lambda: self.acc(np.squeeze(z.g, axis))
        return z

    def softmax(self, axis=-1):
        d = np.exp(self.d - self.d.max(axis, keepdims=True))
        z = T(d / d.sum(axis, keepdims=True), (self,))
        z.back = lambda: self.acc(z.d * (z.g - (z.g * z.d).sum(axis, keepdims=True)))
        return z

    def backward(self):
        order, seen = [], set()
        def visit(x):
            if id(x) not in seen:
                seen.add(id(x))
                for p in x.parents:
                    visit(p)
                order.append(x)
        visit(self)
        self.g = np.ones_like(self.d)
        for x in reversed(order):
            if x.back:
                x.back()


def cat(*xs):
    out = T(np.concatenate([x.d for x in xs], axis=-1), xs)
    def back():
        start = 0
        for x in xs:
            w = x.d.shape[-1]
            x.g += out.g[..., start:start+w]
            start += w
    out.back = back
    return out


def gather(x, ids):
    batches = np.arange(x.d.shape[0]).reshape((-1,) + (1,) * (ids.ndim - 1))
    out = T(x.d[batches, ids], (x,))
    out.back = lambda: np.add.at(x.g, (batches, ids), out.g)
    return out


