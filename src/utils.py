import functools

import numpy as np

from scipy.special import roots_legendre


def nquad_vec(func, bounds, n_roots=21, args=()):
    
    # bounds = [bnd if callable(bnd) else _RangeFunc(bnd) for bnd in bounds]
    
    x, w = roots_legendre(n_roots)
    x = np.real(x)
    
    a, b = np.expand_dims(np.transpose(bounds), -1)
    
    if np.any(np.isinf(a)) or np.any(np.isinf(b)):
            raise ValueError("Gaussian quadrature is only available for "
                             "finite limits.")
    
    y = (b - a) * (x + 1) / 2 + a
    weights = (b - a) / 2 * w
    weights = functools.reduce(np.multiply.outer, weights)
    integ = weights * func(*y, *args)
    # integ = np.array(integ).astype(float)
    
    # sums over all dimensionality axes
    # ignores first axis if func returns a vector
    res = np.nansum(integ, axis=tuple(-(1 + i) for i in range(len(bounds))))
    return res


def nquad_vec_ab(func, a, b, n_roots=21, args=()):
    
    # bounds = [bnd if callable(bnd) else _RangeFunc(bnd) for bnd in bounds]
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    ndim = len(a)
    
    x, w = roots_legendre(n_roots)
    x = np.real(x)
    
    a, b = np.expand_dims([a, b], -1)
    
    if np.any(np.isinf(a)) or np.any(np.isinf(b)):
            raise ValueError("Gaussian quadrature is only available for "
                             "finite limits.")
    
    y = (b - a) * (x + 1) / 2 + a
    weights = (b - a) / 2 * w
    weights = functools.reduce(np.multiply.outer, weights)
    integ = weights * func(*y, *args)
    
    # sums over all dimensionality axes
    # ignores first axis if func returns a vector
    res = np.sum(integ, axis=tuple(-(1 + i) for i in range(len(a))))
    return res
