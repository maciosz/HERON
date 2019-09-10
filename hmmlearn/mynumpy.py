# source of idea: https://stackoverflow.com/questions/5350342/can-i-set-float128-as-the-standard-float-array-in-numpy
from numpy import *

_empty = empty
_ndarray = ndarray
_zeros = zeros
_ones = ones
_full = full

def empty(*args, **kwargs):
    if 'dtype' not in kwargs.keys():
        kwargs.update(dtype=float128)
    return _empty(*args, **kwargs)

def ndarray(*args, **kwargs):
    kwargs.update(dtype=float128)
    return _ndarray(*args, **kwargs)

def ones(*args, **kwargs):
    kwargs.update(dtype=float128)
    return _ones(*args, **kwargs)

def zeros(*args, **kwargs):
    if 'dtype' not in kwargs.keys():
        kwargs.update(dtype=float128)
    return _zeros(*args, **kwargs)

def full(*args, **kwargs):
    if 'dtype' not in kwargs.keys():
        kwargs.update(dtype=float128)
    return _full(*args, **kwargs)
