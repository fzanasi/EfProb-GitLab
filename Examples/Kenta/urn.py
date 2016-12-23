from functools import reduce
import operator
import numpy as np
from efprob_dc import *

# Example from
# https://people.eecs.berkeley.edu/~russell/papers/ijcai05-blog.pdf
# See also ../Bart/blog-urn.py

col = [['B', 'G']]
blue = predicate([1, 0], col)
green = predicate([0, 1], col)

s1 = uniform_state(col)

def draw1(n):
    """Returns a channel to draw a ball from an urn at random"""
    r = 1.0 / n
    def f(*xs):
        return convex_sum((r, uniform_state([xs[i]], col))
                          for i in range(n))
    return chan_fromklmap(f, col * n, col)

def proj(i, n):
    return joint(idn(col) if j == i else discard(col)
                 for j in range(n))

def draw2(n):
    r = 1.0 / n
    return convex_sum((r, proj(i, n)) for i in range(n))

# np.allclose(draw1(5).array, draw2(5).array)

draw = draw2

def blues(n):
    """Returns a predicate that a drawn ball is blue"""
    return draw(n) << blue

s5 = s1 ** 5
b5 = blues(5)
print(s5 >= b5)
print(s5 / b5)
# Assume we know the number of balls is 5. If we draw a blue ball, the
# urn is likely to contain more blue balls than green ones.
print(s5 >= b5 & b5)
print(s5 / (b5 & b5))
# If we draw a blue ball twice, then the urn is likely to contain
# even more blue balls.

def noisy_draw1(n):
    r = 1.0 / n
    def f(*xs):
        return convex_sum(
            (r, state([0.8, 0.2] if xs[i] == 'B' else [0.2, 0.8], col))
            for i in range(n))
    return chan_fromklmap(f, col * n, col)

noise = channel([0.8, 0.2,
                 0.2, 0.8], col, col)

def noisy_draw2(n):
    return noise * draw(n)

# np.allclose(noisy_draw1(5).array, noisy_draw2(5).array)

noisy_draw = noisy_draw2

def noisy_blues(n):
    return noisy_draw(n) << blue

def iter_noisy_blues(n, m):
    return andthen([noisy_blues(n)] * m)

M = 8
# We want to define a predicate as the composition of a map
#
#     M --> col ** 1 + ... + col ** 8
#
# and a map / predicate
#
#     col ** 1 + ... + col ** 8 --> 2 .
#
# But we can't (at least easily), since we do not have coproduct
# types. Instead, we explicitly define the predicate (via meta-level
# computation) as follows:

tenbluesM = predicate([s1 ** i >= iter_noisy_blues(i, 10)
                       for i in range(1, M+1)],
                      range(1, M+1))

#
# uniform prior
#
uprior = uniform_state(range(1, M+1))

uposterior = uprior / tenbluesM
uposterior.plot()

#
# Poisson prior
# Note that Poisson state starts with 0
#
N = 20
pprior = poisson(6, N)

tenbluesN = predicate([0] + [s1 ** i >= iter_noisy_blues(i, 10)
                             for i in range(1, N)],
                      range(N))

pposterior = pprior / tenbluesN
pprior.plot()
pposterior.plot()
