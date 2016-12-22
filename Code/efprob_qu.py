#
# Quantum probability library, prototype version
#
# Started by Bart Jacobs, nov. 2016
#
#
from functools import reduce
import functools
import itertools
import operator
import math
import cmath
import numpy as np
import numpy.linalg
import numpy.random
import random
import scipy.linalg
#import discprob as d

# http://qutip.org/docs/2.2.0/guide/guide-basics.html

# About arbitrary quantum channels:
# https://arxiv.org/abs/1611.03463
# https://arxiv.org/abs/1609.08103
#
# Security protocol with quantum channel:
# https://arxiv.org/pdf/1407.3886
#
# Explanations at:
# https://quantiki.org/wiki/basic-concepts-quantum-computation
#

float_format_spec = ".3g"

tolerance = 1e-10


########################################################################
# 
# Preliminary definitions
#
########################################################################


def approx_eq_num(r, s):
    return r.real - s.real <= tolerance and s.real - r.real <= tolerance and \
        r.imag - s.imag <= tolerance and s.imag - r.imag <= tolerance

def round(x):
    sign = 1 if x >= 0 else -1
    y = math.floor(sign * x / tolerance) 
    return sign * y * tolerance

def round_matrix(m):
    return np.vectorize(round)(m)

def approx_eq_mat(M, N):
    out = (M.shape == N.shape)
    n = M.shape[0]
    for i in range(n):
        for j in range(n):
            out = out and approx_eq_num(M[i][j], N[i][j])
    return out

def prod(iterable):
    """Returns the product of elements from iterable."""
    return reduce(operator.mul, iterable, 1)

# print("problem! The square root need for & below fails.")
# v = vector_state(0.5 * math.sqrt(3), complex(0, 0.5))
# p = v.as_pred()
# print(p, is_positive(p.array))
# print(p & p)
def matrix_square_root(mat):
    E = np.linalg.eigh(mat)
    sq = np.dot(np.dot(E[1], np.sqrt(np.diag(E[0]))),  np.linalg.inv(E[1]))
    return sq

#
# Produce the list of eigenvectors vi, with (roots of )eigenvalues
# incorporated so that mat = sum |vi><vi|, given in python as:
#
# sum([ np.outer(v, v) for v in spectral_decomposition(mat) ]) 
#
# For some reason there is no complex-conjugate involved.
#
def spectral_decomposition(mat):
    E = np.linalg.eigh( mat )
    EVs = [cmath.sqrt(x) * y for x,y in zip(list(E[0]), list(E[1].T))]
    return EVs

#
# We concentrate on square matrices/arrays
#

def is_square(mat):
    return len(mat.shape) == 2 and mat.shape[0] == mat.shape[1]

def is_zero(mat):
    return is_square(mat) \
        and np.all(mat <= tolerance) \
        and np.all(mat >= -tolerance)

def conjugate_transpose(mat):
    return mat.conj().T

def is_symmetric(mat):
    return is_square(mat) and (mat == mat.T).all()

def is_hermitian(mat):
    #print("hermitian\n", mat, "\n", conjugate_transpose(mat))
    return is_square(mat) and np.allclose(mat, conjugate_transpose(mat))

def is_unitary(mat):
    out = is_square(mat) 
    n = mat.shape[0]
    out = out and np.allclose(np.dot(mat, conjugate_transpose(mat)), 
                              np.eye(n))
    out = out and np.allclose(np.dot(conjugate_transpose(mat), mat), 
                              np.eye(n))
    return out


def is_positive(mat):
    if not is_hermitian(mat):
        return False
    E = np.linalg.eigvals(mat)
    #print("eigenvalues", E)
    out = all([e.real > -tolerance for e in E])
    #print("is_positive", out)
    return out
    # if is_zero(mat):
    #     return True
    # try:
    #     # approach from http://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    #     ch = np.linalg.cholesky(mat)
    #     return True
    # except:
    #     return False

def lowner_le(m1, m2):
    return is_positive(m2 - m1)

def is_effect(mat):
    if not is_positive(mat):
        print("not positive!")
        return False
    n = mat.shape[0]
    out = lowner_le(mat, np.eye(n))
    if not out:
        print("Not below the identity; difference is:\n",
              np.eye(n) - mat)
    return out

def is_state(mat):
    tr = np.trace(mat)
    return is_positive(mat) \
        and approx_eq_num(tr.real, 1.0) \
        and approx_eq_num(tr.imag, 0.0)


def entanglement_test(s):
    return np.allclose(s.array,
                       np.kron((s%[1,0]).array, (s%[0,1]).array))

########################################################################
# 
# Classes
#
########################################################################


class Dom:
    """(Co)domains of states, predicates and channels"""
    def __init__(self, dims):
        self.dims = dims
        self.size = prod(dims)

    def __repr__(self):
        return str(self.dims)

    def __eq__(self, other):
        return len(self.dims) == len(other.dims) and \
            all([self.dims[i] == other.dims[i] for i in range(len(self.dims))])

    def __ne__(self, other):
        return not (self == other)

    def __add__(self, other):
        """ concatenation of lists of dimensions """
        return Dom(self.dims + other.dims)

    def __mul__(self, n):
        if n == 0:
            raise Exception('Non-zero number required in Domain multiplication')
        if n == 1:
            return self
        return self + (self * (n-1))


class State:
    def __init__(self, ar, dom):
        self.array = ar
        self.dom = dom if isinstance(dom, Dom) else Dom(dom)
        # print(is_positive(ar), np.trace(ar))
        # print(ar)
        # if not is_state(self.array):
        #     raise Exception('State creation requires a state matrix')
        if ar.shape[0] != self.dom.size:
            raise Exception('Non-matching matrix in state creation')

    #def __str__(self):
    #    return str(self.array)

    # unfinished
    def __repr__(self):
        return str(self.array)
    # np.array2string(self.array, 
    #                            separator=',  ',
    #                            formatter={'complexfloat':lambda x: '%3g + %3gi' 
#                                          % (x.real, x.imag)})

    def __eq__(self, other):
        return self.dom == other.dom \
            and np.all(np.isclose(self.array, other.array))

    def __ne__(self, other):
        return not self == other

    # experimental
    def is_pure(self):
        # purity criterion: trace of square is 1, see Nielen-Chang Exc 2.71
        return approx_eq_num(np.trace(np.dot(self.array, self.array)), 1)

    # validity
    def __ge__(self, p):
        if not is_effect(p.array):
            raise Exception('Non-predicate used in validity')
        if self.dom != p.dom:
            raise Exception('State and predicate with different domains in validity')
        return np.trace(np.dot(self.array, p.array)).real

    # conditioning
    def __truediv__(self, p):
        if not is_effect(p.array):
            raise Exception('Non-predicate used in conditioning')
        #print("conditioning:",  self.dims, p.dims)
        if self.dom != p.dom:
            raise Exception('State and predicate with different domains in conditioning')
        v = self >= p
        if v == 0:
            raise Exception('Zero-validity excludes conditioning')
        sq = matrix_square_root(p.array)
        """scipy does not work well, for instance not on matrices:
        sq = scipy.linalg.sqrtm(p.array)
        np.array([[1, 0, 0, 0], 
                  [0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
        np.array([[1, 0, 0, 0], 
                  [0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]])
        It returns a matrix with only 'nan'
        """
        #print("square root", p.array, sq, np.dot(sq, np.dot(self.array, sq)))
        return State( np.dot(sq, np.dot(self.array, sq)) / v, self.dom)
        
    # parallel product
    def __matmul__(self, s):
        return State(np.kron(self.array, s.array), self.dom + s.dom)

    # iterated product
    def __pow__(self, n):
        if n == 0:
            raise Exception('Power of a state must be at least 1')
        return reduce(lambda s1, s2: s1 @ s2, [self] * n)
    # 
    # The selection is a dim-length list of 0's and 1's, where 0
    # corresponds to marginalisation of the corresponding component.
    #
    def __mod__(self, selection):
        """ How this works in a state with 3 components
        third marginal
        mat.trace(axis1 = 0, axis2 = 3).trace(axis1 = 0, axis2 = 2))
        second marginal
        mat.trace(axis1 = 0, axis2 = 3).trace(axis1 = 1, axis2 = 3))
        first marginal
        mat.trace(axis1 = 1, axis2 = 4).trace(axis1 = 1, axis2 = 3))
        """
        n = len(selection)
        if n != len(self.dom.dims):
            raise Exception('Wrong length of marginal selection')
        dims = [n for (n,i) in zip(self.dom.dims, selection) if i == 1]
        mat = self.array.reshape(tuple(self.dom.dims + self.dom.dims))
        #print(mat.shape)
        marg_pos = 0
        marg_dist = n
        for i in range(n):
            if selection[i] == 1:
                marg_pos = marg_pos+1
                continue
            #print(i, marg_pos, marg_dist)
            mat = mat.trace(axis1 = marg_pos, axis2 = marg_pos + marg_dist)
            marg_dist = marg_dist - 1
        p = prod(dims)
        mat.resize(p,p)
        return State(mat, Dom(dims))

    # convex sum of two states
    def __add__(self, stat):
        return lambda r: convex_state_sum(*[(r, self), (1-r, stat)])

    # Turn a state into a predicate
    def as_pred(self):
        return Predicate(self.array, self.dom)

    #
    # Turn a state on n into a channel 0 -> n. This is useful for
    # bringing in extra "ancillary" bits into the system.
    #
    # Not that the empty list [] is the appropriate domain type, where
    # the product over this list is 1, as used in the dimension of the
    # matrix.
    #
    def as_chan(self):
        return Channel(self.array.reshape(self.dom.size, self.dom.size, 1, 1), 
                       Dom([]),
                       self.dom)

    # experimental conjugate transpose
    def conjugate(self):
        return State(conjugate_transpose(self.array), self.dom)


class Predicate:
    def __init__(self, ar, dom):
        self.array = ar
        self.dom = dom if isinstance(dom, Dom) else Dom(dom)
        if not is_effect(self.array):
            raise Exception('Predicate creation requires an effect matrix')
        if ar.shape[0] != self.dom.size:
            raise Exception('Non-matching matrix in predicate creation')

    #def __str__(self):
    #    return str(self.array)

    def __repr__(self):
        return str(self.array)

    def __eq__(self, other):
        return self.dom == other.dom \
            and np.all(np.isclose(self.array, other.array))

    def __ne__(self, other):
        return not self == other

    # 
    # The selection is a dim-length list of 0's and 1's, where 0
    # corresponds to marginalisation of the corresponding component.
    # This copied from the same operations for states.  operation is
    # not guaranteed to produce a predicate. Hence it is a "partial"
    # partial trace!
    #
    def __mod__(self, selection):
        n = len(selection)
        if n != len(self.dims):
            raise Exception('Wrong length of marginal selection')
        dims = [n for (n,i) in zip(self.dims, selection) if i == 1]
        mat = self.array.reshape(tuple(self.dims + self.dims)) + 0.j
        #print(mat.shape)
        marg_pos = 0
        marg_dist = n
        for i in range(n):
            if selection[i] == 1:
                marg_pos = marg_pos+1
                continue
            #print(i, marg_pos, marg_dist)
            mat = mat.trace(axis1 = marg_pos, axis2 = marg_pos + marg_dist)
            marg_dist = marg_dist - 1
        p = prod(dims)
        mat.resize(p,p)
        return Predicate(mat, dims)

    def __matmul__(self, p):
        return Predicate(np.kron(self.array, p.array), self.dom + p.dom)

    def __invert__(self):
        return Predicate(np.eye(self.dom.size) - self.array, self.dom)

    def __mul__(self, r):
        if r < 0.0 or r > 1.0:
            raise Exception('Scalar multiplication only allow with scalar from the unit interval')
        return Predicate(r * self.array, self.dom)

    def __rmul__(self, r):
        return self * r

    def __add__(self, p):
        if self.dom != p.dom:
            raise Exception('Mismatch of dimensions in sum of predicates')
        mat = self.array + p.array
        if not lowner_le(mat, np.eye(self.dom.size)):
            raise Exception('Sum of predicates undefined since above 1')
        return Predicate(mat, self.dom)

    def __and__(self, p):
        sq = matrix_square_root(self.array)
        conj = np.dot(sq, np.dot(p.array, sq))
        # print(is_positive(sq), is_positive(conj))
        return Predicate(np.dot(sq, np.dot(p.array, sq)), self.dom)

    #
    # Turn a predicate on n into a channel n -> 0. The validity s >= p
    # is the same as p.as_chan() >> s, except that the latter is a 1x1
    # matrix, from which the validity can be extracted via indices
    # [0][0].
    #
    def as_chan(self):
        return Channel(self.array.reshape(1,1, self.dom.size, self.dom.size), 
                       self.dom, Dom([]))
        

# A channel A -> B
#
# Schroedinger picture: CP-map C : L(H_A) -> L(H_B) preserving traces 
#
# Heisenberg picture: unital CP-map D : L(H_B) -> L(H_A)
#
# This D = C*, satisfying <A, C(rho)> = <D(A), rho> which is the
# transformation validity rule, using that < , > is the
# Hilbert-Schmidt inner product, given by tr(-.-).
#
# Here we follow the Heisenberg picture, as used in von Neumann algebras.
#
class Channel:
    def __init__(self, ar, dom, cod):
        self.array = ar
        self.dom = dom if isinstance(dom, Dom) else Dom(dom)
        self.cod = cod if isinstance(cod, Dom) else Dom(cod)
        if ar.shape[0] != self.cod.size or ar.shape[1] != self.cod.size \
           or ar.shape[2] != self.dom.size or ar.shape[3] != self.dom.size:
            raise Exception('Non-matching matrix in channel creation')
        # mat is a (cod.size x cod.size) matrix of (dom.size x
        # dom.size) matrices Hence its shape is (cod.size, cod.size,
        # dom.size, dom.size) This a map dom -> cod in vNA^op

    def __repr__(self):
        return str(self.array)
    # "channel from" + str(self.dom_dims) + "to" + str(self.cod_dims)

    def __eq__(self, other):
        return self.dom == other.dom and self.cod == other.cod \
            and np.all(np.isclose(self.array, other.array))

    def __ne__(self, other):
        return not self == other

    # backward predicate transformation
    def __lshift__(self, p):
        #print(p.dims, self.dom_dims, self.cod_dims)
        if p.dom != self.cod:
            raise Exception('Non-match in predicate transformation')
        m = p.dom.size # = self.cod.size
        n = self.dom.size
        mat = np.zeros((n,n)) + 0j
        # Perform a linear extension of the channel, encoding its
        # behaviour on basisvectors in a matrix to arbitrary matrices.
        for k in range(m):
            for l in range(m):
                mat = mat + p.array[k][l] * self.array[k][l]
        return Predicate(mat, self.dom)

    # forward state transformation
    def __rshift__(self, s):
        #print("rshift dims", s.dims, self.dom_dims)
        if s.dom != self.dom:
            raise Exception('Non-match in state transformation')
        n = s.dom.size # = self.dom.size
        m = self.cod.size
        mat = np.zeros((m, m)) + 0j
        for k in range(m):
            for l in range(m):
                # NB: the order of k,l must be different on the left
                # and right hand side, because in the Hilbert-Schmidt
                # inner product a transpose is used: <A,B> = tr(A*B).
                mat[k][l] = np.trace(np.dot(s.array, 
                                            self.array[l][k]))
        #print("rshift out", is_positive(mat), np.trace(mat), "\n", mat)
        return State(mat, self.cod)

    # parallel compositon
    def __matmul__(self, c):
        return Channel(np.kron(self.array, c.array), 
                       self.dom + c.dom,
                       self.cod + c.cod)

    # sequential composition
    def __mul__(self, c):
        if self.dom != c.cod:
            raise Exception('Non-matching dimensions in channel composition')
        #print(self.array, "\n", c.array)
        n = c.dom.size
        m = self.cod.size
        p = self.dom.size # = c.cod.size
        #print("channel composition dimensions, ( n =", n, ") -> ( p =", p, ") -> ( m =", m, ")")
        #
        # c.array is pxp of nxn, self.array = mxm of pxp
        # output mat must be mxm of nxn
        # 
        # these numbers must be double checked; everything is square so far
        mat = np.ndarray((m,m,n,n)) + 0j
        #print("shapes", c.array.shape, self.array.shape, mat.shape)
        for i in range(m):
            for j in range(m):
                mat[i][j] = sum([self.array[i][j][k][l] * c.array[k][l]
                                 for k in range(p) for l in range(p)])
        return Channel(mat, c.dom, self.cod)

    def as_operator(self):
        """ Operator from Channel """
        n = self.dom.size
        m = self.cod.size
        mat = np.zeros((n*m,n*m)) + 0j
        for i in range(n):
            for j in range(n):
                for k in range(m):
                    for l in range(m):
                        mat[m*i+k][m*j+l] = self.array[k][l][i][j]
        #print("positive operator", is_positive(mat))
        return Operator(mat, self.dom, self.cod)


    def as_kraus(self):
        return self.as_operator().as_kraus()


#
# Alternative representation of a channel n -> m, namely as a
# square n*m x n*m matrix, called (transition) operator
#
class Operator:
    def __init__(self, mat, dom, cod):
        #print(mat.shape, dom_dims, cod_dims)
        self.array = mat
        self.dom = dom
        self.cod = cod
        if mat.shape[0] != self.dom.size * self.cod.size \
           or mat.shape[1] != self.dom.size * self.cod.size:
            raise Exception('Non-matching matrix in channel creation')

    def __str__(self):
        return str(self.array)

    # backward predicate transformation
    def __lshift__(self, p):
        if p.dom != self.cod:
            raise Exception('Non-match in predicate transformation')
        m = p.dom.size # = self.cod.size
        n = self.dom.size
        # It is unclear why the transpose is needed here...
        out = tr1( np.dot(np.kron(np.eye(n), p.array), self.array), n ).T
        return Predicate(out, self.dom)

    # forward state transformation
    def __rshift__(self, s):
        #print("rshift dims", s.dims, self.dom_dims)
        if s.dom != self.dom:
            raise Exception('Non-match in state transformation')
        n = s.dom.size # = self.dom.size
        m = self.cod.size
        out = tr2( np.dot(np.kron(s.array, np.eye(m)), self.array), m ).T
        return State(out, self.cod)

    def as_channel(self):
        n = c.dom.size
        m = c.cod.size
        mat = np.zeros((m,m,n,n)) + 0j
        for i in range(n):
            for j in range(n):
                for k in range(m):
                    for l in range(m):
                        mat[k][l][i][j] = self.array[i*m+k][j*m+l]
        return Channel(mat, self.dom, self.cod)

    def as_kraus(self):
        n = self.dom.size
        m = self.cod.size
        SD = spectral_decomposition(self.array)
        # Turn each n*m vector e in SD into an nxm matrix kraus_e
        out = []
        for e in SD:
            kraus_e = np.zeros((n,m)) + 0j
            for i in range(n):
                for j in range(n):
                    kraus_e[i][j] = e[j*n+i]
        out.append(kraus_e)
        return Kraus(out, self.dom, self.cod)



#
# Kraus operators associated with a channel
#
# Kraus representation of a channel c : n -> m as a list of nxm
# matrices. The key propertie are:
#
#  c << p   equals
#     sum([np.dot(np.dot(e.T, p.array), e) for e in kraus(c)])
#
#  c >> s   equals
#     sum([np.dot(np.dot(e, s.array), e.T) for e in kraus(c)])
#
# Notice the occurrences of transpose .T in different places.
#
class Kraus:
    def __init__(self, mat_list, dom, cod):
        #print(mat.shape, dom_dims, cod_dims)
        self.array_list = mat_list
        self.dom = dom
        self.cod = cod

    # backward predicate transformation
    def __lshift__(self, p):
        if p.dom != self.cod:
            raise Exception('Non-match in predicate transformation')
        out = sum([np.dot(np.dot(e.T, p.array), e) 
                   for e in self.array_list])
        return Predicate(out, self.dom)

    # forward state transformation
    def __rshift__(self, s):
        if s.dom != self.dom:
            raise Exception('Non-match in state transformation')
        out = sum([np.dot(np.dot(e, s.array), e.T) 
                   for e in self.array_list])
        return State(out, self.cod)

    def as_channel(self):
        n = self.dom.size
        m = self.cod.size
        mat = np.zeros((m,m,n,n)) + 0j
        for k in range(m):
            for l in range(m):
                arg = np.zeros((m,m))
                arg[k][l] = 1
                mat[k][l] = sum([np.dot(np.dot(e, arg), e.T) 
                                 for e in self.array_list])
        return Channel(mat, self.dom, self.cod)

########################################################################
# 
# Functions for state, predicate, and channel
#
########################################################################

#
# Pure state from vector v via outer product |v><v|
#
def vector_state(*ls):
    if len(ls) == 0:
        raise Exception('Vector state creation requires a non-empty lsit')
    v = np.array(ls)
    s = np.linalg.norm(v)
    v = v / s
    mat = np.outer(v, v.conj())
    return State(mat, Dom([len(v)]))

#
# Computational unit state |i><i| of dimension n
#
def unit_state(n, i):
    if i < 0 or i >= n:
        raise Exception('Index out-of-range in unit state creation')
    ls = [0] * n
    ls[i] = 1
    return vector_state(*ls)

#
# ket state, taking 0's and 1's as input, as in ket(0,1,1) for |011>
#
def ket(*ls):
    n = len(ls)
    if n == 0:
        raise Exception('Empty ket is impossible')
    if n == 1:
        return unit_state(2, ls[0])
    return unit_state(2, ls[0]) @ ket(*ls[1:n])


def probabilistic_state(*ls):
    n = len(ls)
    s = sum(ls)
    mat = np.zeros((n,n))
    for i in range(n):
        mat[i,i] = ls[i]/s
    return State(mat, Dom([n]))

def diagonal_state(n):
    mat = np.zeros((n,n))
    for i in range(n):
        mat[i,i] = 1/n
    return State(mat, Dom([n]))

def random_state(n):
    # alternative use numpy.random and A*.A/trace(A*.A)
    # for predicates use A*.A / max (eigenvalue (A*.A))
    # A = np.random.rand(n,n)
    # B = np.random.rand(n,n)
    # C = np.zeros((n,n)) + 0j
    # for i in range(n):
    #     for j in range(n):
    #         C[i,j] = complex(A[i,j], B[i,j])
    # D = np.dot(C, conjugate_transpose(C))
    # D = (1/np.trace(D).real) * D
    # print(np.trace(D), is_positive(D))
    ls = [vector_state(*[complex(random.uniform(-10.0, 10.0),
                                 random.uniform(-10.0, 10.0))
                         for i in range(n)]) 
          for j in range(n)]
    amps = [random.uniform(0.0, 1.0) for i in range(n)]
    s = sum(amps)
    mat = sum([amps[i]/s * ls[i].array for i in range(n)])
    return State(mat, Dom([n]))

def random_probabilistic_state(n):
    ls = [random.uniform(0.0, 1.0) for i in range(n)]
    return probabilistic_state(*ls)


#
# Truth predicate, for arbitrary dims
#
def truth(*dims):
    if len(dims) == 0:
        return Predicate(np.eye(1), Dom([]))
    n = dims[0]
    p = Predicate(np.eye(n), Dom([n]))
    if len(dims) == 1:
        return p
    return p @ truth(*dims[1:])

def falsity(*dims):
    return ~truth(*dims)

def probabilistic_pred(*ls):
    n = len(ls)
    if n == 0:
        raise Exception('A non-empty list of numbers is required for a probabilistic predicate')
    if any([r < 0 or r > 1 for r in ls]):
        raise Exception('Probabilities cannot exceed 1 for a probabilistic predicate')
    return Predicate(np.diag(ls), Dom([n]))

def unit_pred(n, i):
    ls = [0] * n
    ls[i] = 1
    return probabilistic_pred(*ls)

#
# A random probabilitisc predicate of dimension n
#
def random_probabilistic_pred(n):
    ls = [random.uniform(0.0, 1.0) for i in range(n)]
    s = sum(ls)
    return probabilistic_pred(*[r/s for r in ls])

def random_pred(n):
    ls = [vector_state(*[complex(random.uniform(-10.0, 10.0),
                                 random.uniform(-10.0, 10.0))
                         for i in range(n)])
          for j in range(n)]
    amps = [random.uniform(0.0, 1.0) for i in range(n)]
    mat = sum([amps[i] * ls[i].array for i in range(n)])
    E = np.linalg.eigvals(mat)
    m = max([x.real for x in E])
    return Predicate(mat/m, Dom([n]))

#
# Choi matrix n x n of n x n matrices, obtained from n x n matrix u,
# by forming putting u * E_ij * u^*, where E_ij is |i><j|, at position
# (i,j). It satisfies choi(A @ B) = choi(A) @ choi(B), where @ is
# tensor
#
# http://mattleifer.info/2011/08/01/the-choi-jamiolkowski-isomorphism-youre-doing-it-wrong/
#
def choi(u):
    n = u.shape[0]
    mat = np.ndarray((n,n,n,n)) + 0j
    for i in range(n):
        for j in range(n):
            arg = np.zeros((n,n))
            arg[i][j] = 1
            out = np.dot(u, np.dot(arg, conjugate_transpose(u)))
            mat[i,j] = out
    return mat

    
#
# Channel obtained from a unitary matrix. The key properties are:
# 
#   chan(u) >> s  =  u * s.array * conj_trans(u)
#
#   chan(u) << p  =  u * p.array * conj_trans(u)
#
def channel_from_unitary(u, dom):
     if not is_unitary(u):
         raise Exception('Unitary matrix required for channel construction')
     return Channel(choi(u), dom, dom)



########################################################################
# 
# Concrete states
#
########################################################################

#
# Classical coin flip, for probability r in unit interval [0,1]
#
def cflip(r):
    if r < 0 or r > 1:
        raise Exception('Coin flip requires a number in the unit interval')
    return State(np.array([[r, 0],
                           [0, 1 - r]]), Dom([2]))

cfflip = cflip(0.5)

true = cflip(1)   # = ket(0)
false = cflip(0)  # = ket(1)


#
# Convex sum of states: the input list contains pairs (ri, si) where
# the ri are in [0,1] and add up to 1, and the si are states
#
def convex_state_sum(*ls):
    if len(ls) == 0:
        raise Exception('Convex sum cannot be empty')
    dom = ls[0][1].dom
    if any([s.dom != dom for r,s in ls]):
        raise Exception('Convex sum requires that states have the same dimensions')
    if any([r < 0 or r > 1 for (r,s) in ls]):
        raise Exception('Convex sum requires numbers in the unit interval')
    r = sum([r for r,s in ls])
    if not approx_eq_num(r, 1):
        raise Exception('Scalars must add up to 1 in convex sum')
    return State(sum([r * s.array for r,s in ls]), dom)


#
# identity channel dims -> dims
#
def idn(*dims):
    if len(dims) == 0:
        raise Exception('Identity channel requires non-empty list of dimensions')
    ch = channel_from_unitary( np.eye(dims[0]), Dom([dims[0]]) )
    if len(dims) == 1:
        return ch
    return ch @ idn(*dims[1:])

#
# unique channel discard : dims -> []
#
def discard(*dims):
    if len(dims) == 0:
        raise Exception('Discard channel requires non-empty list of dimensions')
    n = dims[0]
    mat = np.eye(n)
    mat.resize((1,1,n,n))
    ch = Channel(mat, Dom([n]), Dom([]))
    if len(dims) == 1:
        return ch
    return ch @ discard(*dims[1:])

#
# Channel dims -> dims that only keeps classical part, by measuring in
# the standard basis
#
def classic(*dims):
    if len(dims) == 0:
        raise Exception('Classic channel requires non-empty list of dimensions')
    n = dims[0]
    mat = np.zeros((n,n,n,n))
    for i in range(n):
        tmp = np.zeros((n,n))
        tmp[i][i] = 1
        mat[i][i] = tmp
    ch = Channel(mat, Dom([n]), Dom([n]))
    if len(dims) == 1:
        return ch
    return ch @ classic(*dims[1:])

#
# swap channel 2 @ 2 -> 2 @ 2
#
swap = channel_from_unitary(np.array([ [1, 0, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 0, 1] ]), 
                            Dom([2,2]))
# #
# # first projection channel 2 @ 2 -> 2
# #
# proj1 = Channel(np.array([[np.array([[1,0,0,0], 
#                                      [0,1,0,0],
#                                      [0,0,0,0],
#                                      [0,0,0,0]]), np.array([[0,0,0,0], 
#                                                             [0,0,0,0],
#                                                             [1,0,0,0],
#                                                             [0,1,0,0]])], 
#                           [np.array([[0,0,1,0], 
#                                      [0,0,0,1],
#                                      [0,0,0,0],
#                                      [0,0,0,0]]), np.array([[0,0,0,0], 
#                                                             [0,0,0,0],
#                                                             [0,0,1,0],
#                                                             [0,0,0,1]])]]),
#                Dom([2, 2]), Dom([2]))

# #
# # second projection channel 2 @ 2 -> 2
# #
# proj2 = Channel(np.array([[np.array([[1,0,0,0], 
#                                      [0,0,0,0],
#                                      [0,0,1,0],
#                                      [0,0,0,0]]), np.array([[0,0,0,0], 
#                                                             [1,0,0,0],
#                                                             [0,0,0,0],
#                                                             [0,0,1,0]])], 
#                           [np.array([[0,1,0,0], 
#                                      [0,0,0,0],
#                                      [0,0,0,1],
#                                      [0,0,0,0]]), np.array([[0,0,0,0], 
#                                                             [0,1,0,0],
#                                                             [0,0,0,0],
#                                                             [0,0,0,1]])]]),
#                Dom([2, 2]), Dom([2]))


#
# Kronecker channel n @ m -> n*m
#
def kron(n,m):
    return Channel(choi(np.eye(n*m)), Dom([n, m]), Dom([n*m]))

#
# Kronecker inverse channel n*m -> n @ m
#
def kron_inv(n,m):
    return Channel(choi(np.eye(n*m)), Dom([n*m]), Dom([n, m]))

#
# Auxiliary function, placing a matrix in the lower-right corner of a
# new matrix that is twice as big, with zeros everywhere else.
#
def lower_right(mat):
    # mat is assumed to be square
    n = mat.shape[0]
    out = np.zeros((2*n, 2*n)) + 0j
    for i in range(n):
        for j in range(n):
            out[n+i][n+j] = mat[i][j]
    return out

#
# Same as before, except that ones are put on the upper left diagonal;
# this is used for conditional channels from gates.
#
def lower_right_one(mat):
    # mat is assumed to be square
    n = mat.shape[0]
    out = np.zeros((2*n, 2*n)) + 0j
    for i in range(n):
        out[i,i] = 1
        for j in range(n):
            out[n+i][n+j] = mat[i][j]
    return out

x_matrix = np.array([[0,1],
                     [1,0]])

#
# Pauli-X channel 2 -> 2
#
x_chan = channel_from_unitary(x_matrix, Dom([2]))

#
# cnot channel 2 @ 2 -> 2 @ 2
#
# This should be the same as quantum-control(x_chan)
#
cnot = channel_from_unitary(lower_right_one(x_matrix), Dom([2, 2]))


y_matrix = np.array([[0,-complex(0, 1)],
                     [complex(0,1),0]])

#
# Pauli-Y channel 2 -> 2
#
y_chan = channel_from_unitary(y_matrix, Dom([2]))


z_matrix = np.array([[1,0],
                     [0,-1]])

#
# Pauli-Z channel 2 -> 2
#
z_chan = channel_from_unitary(z_matrix, Dom([2]))


hadamard_matrix = (1/math.sqrt(2)) * np.array([ [1, 1],
                                                [1, -1] ])

#
# Hadamard channel 2 -> 2
#
hadamard = channel_from_unitary(hadamard_matrix, Dom([2]))

#
# Basic states, commonly written as |+> and |->
#
plus = hadamard >> ket(0)
minus = hadamard >> ket(1)

hadamard_test = [plus.as_pred(),
                 minus.as_pred()]

#
# Controlled Hadamard 2 @ 2 -> 2 @ 2
#
chadamard = channel_from_unitary(lower_right_one(hadamard_matrix), 
                                 Dom([2, 2]))

#
# channel 2 @ 2 -> 2 @ 2 for producing Bell states
#
bell_chan = cnot * (hadamard @ idn(2))
# bell00 = bell_chan >> ket(0,0)
# bell01 = bell_chan >> ket(0,1)
# bell10 = bell_chan >> ket(1,0)
# bell11 = bell_chan >> ket(1,1)

bell00_vect = np.array([1,0,0,1])
bell01_vect = np.array([0,1,1,0])
bell10_vect = np.array([1,0,0,-1])
bell11_vect = np.array([0,1,-1,0])

bell00 = State(0.5 * np.outer(bell00_vect, bell00_vect), Dom([2,2]))
bell01 = State(0.5 * np.outer(bell01_vect, bell01_vect), Dom([2,2]))
bell10 = State(0.5 * np.outer(bell10_vect, bell10_vect), Dom([2,2]))
bell11 = State(0.5 * np.outer(bell11_vect, bell11_vect), Dom([2,2]))

bell_test = [bell00.as_pred(),
             bell01.as_pred(),
             bell10.as_pred(),
             bell11.as_pred()]

#
# Greenberger-Horne-Zeilinger states
#
ghz = (idn(2) @ cnot) >> ((bell_chan @ idn(2)) >> ket(0,0,0))

# The ghz states one by one

ghz_vect1 = np.array([1,0,0,0,0,0,0,1])
ghz_vect2 = np.array([1,0,0,0,0,0,0,-1])
ghz_vect3 = np.array([0,0,0,1,1,0,0,0])
ghz_vect4 = np.array([0,0,0,1,-1,0,0,0])
ghz_vect5 = np.array([0,0,1,0,0,1,0,0])
ghz_vect6 = np.array([0,0,1,0,0,-1,0,0])
ghz_vect7 = np.array([0,1,0,0,0,0,1,0])
ghz_vect8 = np.array([0,1,0,0,0,0,-1,0])

ghz1 = State(0.5 * np.outer(ghz_vect1, ghz_vect1), Dom([2,2,2]))
ghz2 = State(0.5 * np.outer(ghz_vect2, ghz_vect2), Dom([2,2,2]))
ghz3 = State(0.5 * np.outer(ghz_vect3, ghz_vect3), Dom([2,2,2]))
ghz4 = State(0.5 * np.outer(ghz_vect4, ghz_vect4), Dom([2,2,2]))
ghz5 = State(0.5 * np.outer(ghz_vect5, ghz_vect5), Dom([2,2,2]))
ghz6 = State(0.5 * np.outer(ghz_vect6, ghz_vect6), Dom([2,2,2]))
ghz7 = State(0.5 * np.outer(ghz_vect7, ghz_vect7), Dom([2,2,2]))
ghz8 = State(0.5 * np.outer(ghz_vect8, ghz_vect8), Dom([2,2,2]))

#
# The associated test
#
ghz_test = [ghz1.as_pred(),
            ghz2.as_pred(),
            ghz3.as_pred(),
            ghz4.as_pred(),
            ghz5.as_pred(),
            ghz6.as_pred(),
            ghz7.as_pred(),
            ghz8.as_pred()]

#
# W3 state
#
w3_vect = np.array([0,1,1,0,1,0,0,0])
w3 = State(1/3 * np.outer(w3_vect, w3_vect), Dom([2,2,2]))
           
def phase_shift_matrix(angle):
    return np.array([[1, 0],
                     [0, complex(math.cos(angle), math.sin(angle))]])

#
# Phase shift channel 2 -> 2, for angle between 0 and 2 pi
#
def phase_shift(angle):
    return channel_from_unitary(phase_shift_matrix(angle), 
                                Dom([2]))

#
# Controlled phase shift channel 2 @ 2 -> 2 @ 2, for angle between 0 and 2 pi
#
def cphase_shift(angle):
    return channel_from_unitary(lower_right_one(phase_shift_matrix(angle)),
                                Dom([2, 2]))



#
# toffoli channel 2 @ 2 @ 2 -> 2 @ 2 @ 2
#
toffoli = channel_from_unitary(np.array([ [1, 0, 0, 0, 0, 0, 0, 0],
                                          [0, 1, 0, 0, 0, 0, 0, 0],
                                          [0, 0, 1, 0, 0, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 1, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 1, 0, 0],
                                          [0, 0, 0, 0, 0, 0, 0, 1],
                                          [0, 0, 0, 0, 0, 0, 1, 0] ]),
                               Dom([2, 2, 2]))



#
# Convex sum of channels: the input list contains pairs (ri, ci) where
# the ri are in [0,1] and add up to 1, and the ci are channels
#
def convex_channel_sum(*ls):
    if len(ls) == 0:
        raise Exception('Convex sum cannot be empty')
    dom_dims = ls[0][1].dom_dims
    cod_dims = ls[0][1].cod_dims
    if any([c.dom_dims != dom_dims or c.cod_dims != cod_dims for r,c in ls]):
        raise Exception('Convex sum requires parallel channels')
    if any([r < 0 or r > 1 for (r,c) in ls]):
        raise Exception('Convex sum requires numbers in the unit interval')
    r = sum([r for r,c in ls])
    if not approx_eq_num(r, 1):
        raise Exception('Scalars must add up to 1 in convex sum')
    return Channel(sum([r * c.array for r,c in ls]), dom_dims, cod_dims)

#
# Channel constructed from a list of states.
#
# Let c = channel_from_states(s1, ..., sn). Then c >> t equals the
# convex sum of states, given by the pairs 
#
#        ( t >= unit_pred(n,i), si )
#
# This means that much information about t is lost.
#
# How this channel works on predicates is not clear.
#
def channel_from_states(*ls):
    n = len(ls)
    if n == 0:
        raise Exception('Non-empty list of state is need to form a channel')
    cod = ls[0].dom
    if any([s.dom != cod for s in ls]):
        raise Exception('States must all have the same type to form a channel')
    mat = np.zeros((cod.size, cod.size, n, n)) + 0j
    for j1 in range(cod.size):
        for j2 in range(cod.size):
            for i in range(n):
                mat[j1][j2][i][i] = ls[i].array[j1][j2]
    return Channel(mat, Dom([n]), cod)


########################################################################
# 
# Measurement and control
#
########################################################################


#
# Measurement channel dom -> 2, for a predicate p of type dom. This
# channel does not keep a record of the updated state.
#
# The key property is, for a state s of type dom:
#
#   meas_pred(p) >> s  =  s >= p
#
# where the right-hand-side must be interpreted as a classic state
# with domain [2].
#
def meas_pred(p):
    n = p.dom.size
    mat = np.zeros((2,2,n,n)) + 0j
    mat[0][0] = p.array
    mat[1][1] = (~p).array
    return Channel(mat, p.dom, Dom([2]))

meas0 = meas_pred(ket(0).as_pred())
meas1 = meas_pred(ket(1).as_pred())

#
# Measurement generalised from a predicate to a test, that is to a
# list of predicates that add up to truth. Measurement wrt. a
# predicate p is the same as measurement wrt. the test [p, ~p]
#
# The l predicates in the test must all have the same domain dom. The
# measurement channel then has time dom -> l
#
def meas_test(ts):
    l = len(ts)
    if l == 0:
        raise Exception('Test must have non-zero length in measurement')
    dom = ts[0].dom
    if any([t.dom != dom for t in ts]):
        raise Exception('Tests must have the same domain in measurement')
    t = ts[0]
    for i in range(l-1):
        t = t + ts[i+1]
    if not np.all(np.isclose(t.array, truth(*dom.dims).array)):
        raise Exception('The predicates in a test must add up to truth')
    mat = np.zeros((l,l,dom.size,dom.size)) + 0j
    for i in range(l):
        mat[i][i] = ts[i].array
    return Channel(mat, dom, Dom([l]))

#
# Measurements in some standard bases.
#
meas_hadamard = meas_test(hadamard_test)
meas_bell = meas_test(bell_test)
meas_ghz = meas_test(ghz_test)

#
# Instrument dom -> [2] @ dom , for a predicate p of type dom.
#
# The main properties are:
#
#   (instr(p) >> s) % [1,0]  =  meas_pred(p) >> s,
#
#   (instr(p) >> s) % [0,1]  =  convex_state_sum( (s >= p, s/p), 
#                                                 (s >= ~p, s/~p) )
#
#   instr(p) << truth(2) @ q  =  (p & q) + (~p & q) 
#
def instr(p):
    n = p.dom.size
    mat = np.zeros((2*n,2*n,n,n)) + 0j
    sqp = matrix_square_root(p.array)
    sqnp = matrix_square_root((~p).array)
    for i in range(n):
        for j in range(n):
            arg = np.zeros((n,n))
            arg[i][j] = 1
            out1 = np.dot(sqp, np.dot(arg, sqp))
            out2 = np.dot(sqnp, np.dot(arg, sqnp))
            mat[i][j] = out1
            mat[n+i][n+j] = out2
    return Channel(mat, p.dom, Dom([2]) + p.dom)


def pcase(p):
    def fun(*chan_pair):
        c = chan_pair[0]
        d = chan_pair[1]
        if c.dom != d.dom or c.cod != d.cod:
            raise Exception('channels must have equal domain and codomain in predicate case channel')
        return (discard(2) @ idn(*c.dom.dims)) * ccase(c,d) * instr(p)
    return fun


#
# classical control of a channel c : dom -> cod, giving a channel
# ccontrol(c) : [2] + dom -> [2] + cod
#
def ccontrol(c):
    cd = c.dom.size
    cc = c.cod.size
    shape = [2 + cc, 2 + cc, 2 + cd, 2 + cd]
    mat = np.zeros(shape) + 0j
    #print(cd, cc, c.array.shape, mat.shape)
    for i in range(cd):
        for j in range(cd):
            mat[i][j][i][j] = 1
            mat[cd+i][cd+j] = lower_right(c.array[i][j])
            #print("control submatrix", i, j)
            #print(c.array[i][j])
            #for k in range(cd):
                #print(i, j, k)
                # the next formulation is inexplicably close for hadamard
                # mat[i][cd+j][i][cd+k] = cmath.sqrt(c.array[i][j][i][j]) 
                # mat[cd+i][j][cd+k][j] = cmath.sqrt(c.array[i][j][i][j]) 
                ## mat[i][cd+j][i][cd+k] = 1-k
                # this one has no effect for cnot
                ## mat[cd+i][j][cd+k][j] = 1-k
            # this one works for cnot: 
            # mat[i][cd+j][i][2*cd - 1 - j] = 1
            # mat[cd+i][j][2*cd - 1 - i][j] = 1
    return Channel(mat, Dom([2]) + c.dom, Dom([2]) + c.cod)

#
# A list of channels c1, ..., ca all with the same domain dom and
# codomain cod gives a classical case channel [a]+dom -> [a]+cod
#
def ccase(*chans):
    a = len(chans)
    if a == 0:
        raise Exception('Non-empty channel list is required in control')
    dom = chans[0].dom
    cod = chans[0].cod
    if any([c.dom != dom or c.cod != cod for c in chans]):
        raise Exception('Channels all need to have the same (co)domain in control')
    mat = np.zeros((a*cod.size, a*cod.size, a*dom.size, a*dom.size)) + 0j
    for b in range(a):
        for k in range(cod.size):
            for l in range(cod.size):
                for i in range(dom.size):
                    for j in range(dom.size):
                        mat[b*cod.size + k][b*cod.size + l] \
                            [b*dom.size + i][b*dom.size + j] \
                            = chans[b].array[k][l][i][j]
    return Channel(mat, Dom([a]) + dom, Dom([a]) + cod)



########################################################################
# 
# Leifer-Spekkens style operations; experimental
#
########################################################################

#
# Turn channel n -> m into n -> n @ m
#
# Note: the definition is precisely the same as in the discrete case,
# but now we don't copy the state into the first coordinate, but
# measure it!
#
# In the second coordinate we keep the original channel:
#
#    (graph(c) >> t) % [0, 1]  =  c >> t
#
# But in the first coordinate:
#
#    (graph(c) >> t) % [1, 0]  = classic >> t
#
# Hence we have a probabilistic state with entries given by the validities:
#
#    t >= unit_pred(n,i)
#
def graph(c):
    if len(c.dom.dims) != 1:
        raise Exception('Tupling not defined for product input ')
    n = c.dom.dims[0]
    return channel_from_states(*[unit_state(n,i) @ (c >> unit_state(n,i)) 
                                 for i in range(n)])

#
# Turn a product state into a channel. We may assume that the first
# marginal of the product state is probabilistic.
#
def productstate2channel(s):
    if len(s.dom.dims) < 2:
        raise Exception('Product state required to form a channel')
    n = s.dom.dims[0]
    cod_dims = s.dom.dims[1:]
    ls = [s / (unit_pred(n,i) @ truth(*cod_dims)) % [0,1] 
          for i in range(n)]
    return channel_from_states(*ls)



########################################################################
# 
# Transition operator related stuff
#
########################################################################


#
# Transition operator n*n x n*n associated with unitary u of n x n.
#
# The main property is UAU^* = tr2((A@id)transition_operator(U), n).T
#
def transition_from_unitary(u):
    n = u.shape[0]
    mat = np.ndarray((n*n,n*n)) + 0j
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    mat[n*i+k][n*j+l] = u[i][k] * u[j][l].conjugate()
    return mat


def operator_from_unitary(u, dims):
     if not is_unitary(u):
         raise Exception('Unitary matrix required for channel construction')
     return Operator(transition_from_unitary(u), dims, dims)


#
# Own partial trace implementations tr1 and tr2
#
# mat has shape (n*m, n*m); tr1 goes to (n,n) and tr2 to (m,m)
#
def tr1(mat,n):
    k = mat.shape[0]
    m = int(k/n) # remove this many
    out = np.zeros((n,n)) + 0j
    for j in range(m):
        v = np.array([0]*m)
        v[j] = 1
        w = np.kron(np.eye(n), v)
        out = out + np.dot(np.dot(w, mat), w.T)
    return out

def tr2(mat,m):
    k = mat.shape[0]
    n = int(k/m)  # remove this many
    out = np.zeros((m,m)) + 0j
    for i in range(n):
        v = np.array([0]*n)
        v[i] = 1
        w = np.kron(v, np.eye(m))
        out = out + np.dot(np.dot(w, mat), w.T)
    return out
    

# # tests at: http://www.thphy.uni-duesseldorf.de/~ls3/teaching/1515-QOQI/Additional/partial_trace.pdf
# M = np.array([[1, 2, complex(0,3), 4],
#               [5, 6, 7, 8],
#               [9, 10,11,12],
#               [13,complex(0,-14),15,16]])
# print( tr1(M, 2) )
# print( M.reshape(2,2,2,2).trace(axis1 = 1, axis2 = 3)  )
# print( tr2(M, 2) )
# print( M.reshape(2,2,2,2).trace(axis1 = 0, axis2 = 2)  )



#
# Pauli-X channel 2 -> 2
#
x_oper = operator_from_unitary(np.array([[0,1],
                                         [1,0]]), Dom([2]))

#
# Pauli-Y channel 2 -> 2
#
y_oper = operator_from_unitary(np.array([[0,-complex(0, 1)],
                                         [complex(0,1),0]]), Dom([2]))


#
# Pauli-Z channel 2 -> 2
#
z_oper = operator_from_unitary(np.array([[1,0],
                                         [0,-1]]), Dom([2]))


#
# Hadamard channel 2 -> 2
#
hadamard_oper = operator_from_unitary((1/math.sqrt(2)) * np.array([ [1, 1],
                                                                  [1, -1] ]),
                                      Dom([2]))

#
# Basic states, commonly written as |+> and |->
#
plus_oper = hadamard_oper >> ket(0)
minus_oper = hadamard_oper >> ket(1)




########################################################################
# 
# Test functions
#
########################################################################


def validity():
    print("\nValidity tests")
    s1 = random_state(2)
    s2 = random_state(5)
    s3 = random_state(2)
    p1 = random_pred(2) 
    p2 = random_pred(5) 
    p3 = random_pred(2)
    print("* validity product difference test:", 
          (s1 @ s2 >= ~p1 @ (0.1 * p2)) - ((s1 >= 0.5 * ~p1) * (s2 >= 0.2 * p2)) )
    print("* transformation-validty difference test:", 
          (s1 @ s3 >= (chadamard << (p1 @ truth(2)))) \
          - ((chadamard >> s1 @ s3) >= (p1 @ truth(2))) )
    print("* weakening is the same as predicate transformation by a projection:", 
          p3 @ truth(2) == (idn(2) @ discard(2)) << p3 )

def marginals():
    print("\nMarginal tests")
    print("* third marginal of |000> is:\n", ket(0,0,0) % [0,0,1])
    a = random_state(2)
    b = random_state(2)
    print("* a random product state, and then several projection operations on that state:", 
          a == (a @ b) % [1,0],
          a == idn(2) >> a,
          a == idn(2) @ discard(2) >> (a @ b),
          a == discard(2) @ idn(2) >> (swap >> (a @ b)) )

def measurement():
    print("\nMeasurement and control tests")
    s = random_state(2)
    p = random_pred(2)
    print("* measurement channel applied to a state, with validity", 
           s >= p, "\n", meas_pred(p) >> s )
    r = random.uniform(0,1)
    print("* Classical control with classical control bit:")
    print( (ccontrol(x_chan) >> cflip(r) @ s) % [1,0] == 
           probabilistic_state(r, 1-r),
           (ccontrol(x_chan) >> cflip(r) @ s) % [0,1] == 
           convex_state_sum( (r, s), (1-r, x_chan >> s) ) )
    print("* Classical case with classical control bit:")
    print( (ccase(y_chan,x_chan) >> cflip(r) @ s) % [1,0] == 
           probabilistic_state(r, 1-r),
           (ccase(y_chan,x_chan) >> cflip(r) @ s) % [0,1] == 
           convex_state_sum( (r, y_chan >> s), (1-r, x_chan >> s) ) )

def instrument():
    # sometimes this fails because the matrix_square_root function fails
    print("\nInstrument tests")
    p = random_pred(2)
    q = random_pred(2)
    s = random_state(2)
    print( (instr(p) >> s) % [1,0] == meas_pred(p) >> s,
           (instr(p) >> s) % [0,1] == convex_state_sum( (s >= p, s/p), 
                                                        (s >= ~p, s/~p) ),
           instr(p) << truth(2) @ q == (p & q) + (~p & q) )
    print("channel equalities")
    print( (idn(2) @ discard(2)) * instr(p) == meas_pred(p) )


def conditioning():
    print("\nConditioning tests")
    print("Bayes")
    s = random_state(2)
    t = random_state(2)
    p = random_pred(2)
    q = random_pred(2)
    r = random_probabilistic_pred(2)
    t = random_probabilistic_pred(2)
    print( s/p >= q )
    print( (s >= p & q) / (s >= p) )
    print("cnot experiments")
    print( p )
    print( (cnot << unit_pred(2,0) @ q) == unit_pred(2,0) @ q )
    print( (cnot << unit_pred(2,1) @ r) == (unit_pred(2,1) @ ~r) )
    print( (cnot << unit_pred(2,1) @ q), "\n", (unit_pred(2,1) @ ~q) )
    print("cnot and instruments")
    print( cnot >> s @ t )
    print( meas_pred(s.as_pred()) >> t )


def channel():
    print("\nChannel tests")
    s1 = random_state(3)
    s2 = random_state(3)
    s3 = random_state(3)
    t = random_state(3)
    c = channel_from_states(s1, s2, s3)
    print("* channel from state state transformation as convex sum")
    print( convex_state_sum((t >= unit_pred(3, 0), s1),
                            (t >= unit_pred(3, 1), s2),
                            (t >= unit_pred(3, 2), s3)) )
    print( c >> t )
    print("* truth preservation by channels")
    print( truth(3) == c << truth(3), truth(3) == graph(c) << truth(3, 3) )
    print("* graph properties")
    c = x_chan * hadamard * phase_shift(math.pi/3)
    t = random_state(2)
    print( classic(2) >> t == (graph(c) >> t) % [1, 0] )
    print( c >> t )
    #print( graph(c) >> t )
    print( (graph(c) >> t) % [0, 1] )
    print("* first component of tuple is go-classic:",
          (idn(2) @ discard(2)) * graph(c) == classic(2) )
    print("* second component of tuple is the channel itself",
          (discard(2) @ idn(2)) * graph(c) == c )
    print( np.isclose((((discard(2) @ idn(2)) * graph(c)) >> t).array,
                      (c >> t).array.T) )
    print("* predicate as channel")
    p = random_pred(2)
    v = random_state(2)
    print( v >= p )
    print( p.as_chan() >> v )
    print("* discard channel; outcome is the identity matrix")
    print( discard(2) * hadamard )
    print("* from product to channel")
    w1 = random_probabilistic_state(2)
    w2 = random_state(2)
    w = chadamard >> (w1 @ w2)
    print( w1 ==  w % [1, 0] )
    dc = productstate2channel(w)
    print( dc >> ket(0) )
    print( w2 )
    print( w2 == (chadamard >> (ket(0) @ w2)) % [0,1] )
    # the next two states are also equal
    print( dc >> ket(1) )
    print( (chadamard >> (ket(1) @ w2)) % [0,1] )
    print("* next")
    print( dc >> v )
    print( (w / (v.as_pred() @ truth(2))) % [0,1] )
    dct = graph(productstate2channel(w))
    print("* product from channel form product: recover the original:", 
          np.allclose(w.array, (dct >> w1).array))
    u = random_state(2)
    print("* channel from product from channel")
    print( productstate2channel(graph(c) >> u) >> t )
    print( c >> t )
    print( np.allclose(c.array,
                       productstate2channel(graph(c) >> u).array) )

def transition():
    print("\nTransition tests")
    s = random_state(2)
    w = chadamard >> (s @ random_state(2))
    print("* transition state transformation is channel state transformation:")
    print( np.all(np.isclose((hadamard >> s).array, (hadamard_oper >> s).array)),
           np.all(np.isclose((x_chan >> s).array, (x_oper >> s).array)),
           np.all(np.isclose((y_chan >> s).array, (y_oper >> s).array)),
           np.all(np.isclose((z_chan >> s).array, (z_oper >> s).array)) )
    print( np.all(np.isclose((hadamard >> s).array, 
                             (hadamard.as_operator() >> s).array)),
           np.all(np.isclose((x_chan >> s).array, 
                             (x_chan.as_operator() >> s).array)),
           np.all(np.isclose((y_chan >> s).array, 
                             (y_chan.as_operator() >> s).array)),
           np.all(np.isclose((z_chan >> s).array, 
                             (z_chan.as_operator() >> s).array)),
           np.all(np.isclose(((hadamard @ x_chan) >> w ).array,
                             ((hadamard @ x_chan).as_operator() >> w).array)) )
    p = random_pred(2)
    c = x_chan * hadamard * y_chan * z_chan
    #print( c << p )
    #print( c.as_operator() << p )
    print( np.all(np.isclose(((c @ discard(2)) << p).array,
                             ((c @ discard(2)).as_operator() << p).array)) )
    c = chadamard * swap * cnot * swap
    p = cnot << (random_pred(2) @ random_pred(2))
    s = cnot >> (random_state(2) @ random_state(2))
    print("* Kraus test:",
          np.all(np.isclose( (c << p).array, (c.as_kraus() << p).array)),
          np.all(np.isclose( (c >> s).array, (c.as_kraus() >> s).array)) )
    d = hadamard * x_chan
    print( np.all(np.isclose(d.array,
                             d.as_kraus().as_channel().array)) )
    t = random_state(2) 
    #print( d >> t )
    #print( d.as_kraus().as_channel() >> t )

    

def experiment():
    c = hadamard * x_chan
    print( c )
    # same outcome for x_chan, y_chan
    print( chan2productpredicate(c) )
    print( chan2productstate(c) )
    # all marginals exist, and are the identity so far
    print("first marginal\n", chan2productpredicate(c) % [1, 0] )
    print("second marginal\n",  chan2productpredicate(c) % [0, 1] )
    print("shapes", chan2productpredicate(c).array.shape,
          chan2productstate(c).array.shape)
    p1 = random_pred(2)    
    t = random_state(2)
    # "conditional state" from Leiffer note: outcomes appear at strange points
    print( ((t.as_pred() @ truth(2)) & chan2productpredicate(c)) % [0, 1] )
    print("predicate transformer\n", c << t.as_pred(), "\n", c >> t )

    # The next two things look mysteriously similar...
    #print( ((s2p(t) @ truth(2)) & chan2productpredicate(c)) % [0, 1] )
    #print("state transformer\n", s2p(c >> t.conjugate()) )


def main():
    validity()
    marginals()
    measurement()
    instrument()
    #conditioning()
    #channel()
    #experiment()
    #transition()

if __name__ == "__main__":
    main()


