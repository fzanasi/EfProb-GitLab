#
# Notes for coding
#
# - Follow Python Style Guide "PEP 8"
#     https://www.python.org/dev/peps/pep-0008/
#   See also Google Python Style Guide
#     https://google.github.io/styleguide/pyguide.html
# - Use docstrings when appropriate
#   See: https://www.python.org/dev/peps/pep-0257/
#        https://google.github.io/styleguide/pyguide.html#Comments
#
# Naming conventions specific to this module (tentative)
#
# - Names of functions that return states and predicates start with
#   'stat' and 'pred', respectively.
#

from functools import reduce
import functools
import itertools
import operator
from math import inf
import math
import collections
import random

import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
import matplotlib.pyplot as plt

integrate_opts = {}
use_lru_cache = True
float_format_spec = ".3g"


def prod(iterable):
    """Product of elements in iterable."""
    return reduce(operator.mul, iterable, 1)


def kron2d(a, b, outer=np.outer):
    """Kronecker product of two 2d-arrays with given 'outer'."""
    if not (a.ndim == 2 and b.ndim == 2):
        raise ValueError("Invalid arguments")
    result = outer(a, b).reshape(a.shape + b.shape)
    result = np.concatenate(result, axis=1)
    result = np.concatenate(result, axis=1)
    return result


def nquad_wrapper(func, ranges, args=None, opts=None):
    if not ranges:
        return func()
    if supp_isempty(ranges):
        return 0.0
    if opts is None:
        opts = integrate_opts
    return integrate.nquad(func, ranges,
                           args=args, opts=opts)[0]


def _wrap(obj):
    array = np.empty((), dtype=object)
    array[()] = obj
    return array


class Interval(collections.namedtuple('R', 'l u')):
    """Real intervals"""
    __slots__ = ()

    def __str__(self):
        if self.isempty():
            return "{}"
        if self.l == -inf and self.u == inf:
            return "R"
        return "R({:{spec}}, {:{spec}})".format(self.l,
                                                self.u,
                                                spec=float_format_spec)

    def __contains__(self, item):
        return self.l <= item <= self.u

    def issubset(self, other):
        return other.l <= self.l and self.u <= other.u

    def intersect(self, other):
        return Interval(max(self.l, other.l), min(self.u, other.u))

    def union(self, other):
        return Interval(min(self.l, other.l), max(self.u, other.u))

    def isempty(self):
        return self.u < self.l


R = Interval
R_ = Interval(-inf, inf)
empty = Interval(inf, -inf)


def isR(obj):
    return obj is Interval


def _ensure_list(dom):
    if isinstance(dom, (tuple, range, Interval)) or isR(dom):
        return [dom]
    if not isinstance(dom, list):
        raise ValueError("Invalid domain or support")
    if dom and not (isinstance(dom[0],
                               (list, tuple, range, Interval))
                    or isR(dom[0])):
        return [dom]
    return dom


class Dom:
    """(Co)domains of states, predicates and channels"""
    def __init__(self, dom, disc=None, cont=None):
        dom = _ensure_list(dom)
        if disc is None or cont is None:
            dom_ = dom
            dom, disc, cont = [], [], []
            for d in dom_:
                if isR(d):
                    dom.append(R_)
                    cont.append(R_)
                elif isinstance(d, Interval):
                    dom.append(d)
                    cont.append(d)
                else:
                    dom.append(d)
                    disc.append(d)
        self._dom = dom
        self.disc = disc
        self.cont = cont
        self.iscont = bool(self.cont)

    def __iter__(self):
        return iter(self._dom)

    def __getitem__(self, key):
        return self._dom[key]

    def __add__(self, other):
        return Dom(self._dom + other._dom,
                   disc=self.disc + other.disc,
                   cont=self.cont + other.cont)

    def __mul__(self, n):
        return Dom(self._dom * n,
                   disc=self.disc * n,
                   cont=self.cont * n)

    def __bool__(self):
        return bool(self._dom)

    def __repr__(self):
        return repr(self._dom)

    def __str__(self):
        if not self:
            return "()"
        return " * ".join(str(d) for d in self)

    def __eq__(self, other):
        return self._dom == other._dom

    def __ne__(self, other):
        return not self == other

    def disc_get(self, index):
        return [self.disc[n][i] for n, i in enumerate(index)]

    def get_disc_indices(self, disc_args):
        return tuple(self.disc[n].index(a) for n, a
                     in enumerate(disc_args))

    def split(self, list_):
        disc_list, cont_list = [], []
        for d, x in zip(self, list_):
            if isinstance(d, Interval):
                cont_list.append(x)
            else:
                disc_list.append(x)
        return disc_list, cont_list

    def merge(self, disc_list, cont_list):
        di, ci = iter(disc_list), iter(cont_list)
        return [next(ci) if isinstance(d, Interval) else next(di)
                for d in self]


def asdom(dom):
    return dom if isinstance(dom, Dom) else Dom(dom)

def check_dom_match(dom1, dom2):
    if dom1 != dom2:
        raise ValueError("Domains do not match: {} and {}".format(dom1, dom2))

#
# Functions for supports
# (A support is just a list of intervals)
#

def supp_init(supp):
    supp = _ensure_list(supp)
    supp = [R_ if isR(s)
            else s if isinstance(s, Interval)
            else Interval(s[0], s[1])
            for s in supp]
    return supp

def supp_intersect(supp1, supp2):
    return [s1.intersect(s2) for s1, s2 in zip(supp1, supp2)]

def supp_union(supp1, supp2):
    return [s1.union(s2) for s1, s2 in zip(supp1, supp2)]

def supp_contains(supp, item):
    return all(i in s for s, i in zip(supp, item))

def supp_issubset(supp1, supp2):
    return all(s1.issubset(s2) for s1, s2 in zip(supp1, supp2))

def supp_isempty(supp):
    return any(s.isempty() for s in supp)

def supp_fun(supp, fun):
    def f(*xs):
        return fun(*xs) if supp_contains(supp, xs) else 0.0
    return f

def supp_fun2(supp1, supp2, fun):
    def f(xs, ys):
        return fun(xs, ys) if (supp_contains(supp1, xs)
                               and supp_contains(supp2, ys)) else 0.0
    return f


class Fun:
    """Functions with supports."""
    def __init__(self, fun, supp):
        supp = supp_init(supp)
        if use_lru_cache:
            fun = functools.lru_cache(maxsize=None)(fun)
        self.fun = supp_fun(supp, fun)
        self.supp = supp

    def __call__(self, *xs):
        return self.fun(*xs)

    def __add__(self, other):
        """Pointwise addition."""
        return Fun(lambda *xs: self(*xs) + other(*xs),
                   supp_union(self.supp, other.supp))

    def __mul__(self, other):
        """Pointwise multiplication."""
        return Fun(lambda *xs: self(*xs) * other(*xs),
                   supp_intersect(self.supp, other.supp))

    def integrate(self):
        return nquad_wrapper(self, self.supp)

    u_integrate = np.frompyfunc(lambda f: f.integrate(), 1, 1)

    @staticmethod
    def vect_integrate(array):
        """Integrates array of functions elementwise."""
        out = np.empty_like(array, dtype=float)
        Fun.u_integrate(array, out=out)
        return out

    def smul(self, scalar):
        return Fun(lambda *xs: self(*xs) * scalar,
                   self.supp)

    u_smul = np.frompyfunc(lambda f, s: f.smul(s), 2, 1)

    u_rsmul = np.frompyfunc(lambda s, f: f.smul(s), 2, 1)

    def sdiv(self, scalar):
        return Fun(lambda *xs: self(*xs) / scalar,
                   self.supp)

    u_sdiv = np.frompyfunc(lambda f, s: f.sdiv(s), 2, 1)

    def joint(self, other):
        n = len(self.supp)
        return Fun(lambda *xs:
                   self(*xs[:n]) * other(*xs[n:]),
                   self.supp + other.supp)

    u_joint = np.frompyfunc(lambda f, g: f.joint(g), 2, 1)

    def marginal(self, selectors):
        supp_int = [d for d, s in zip(self.supp, selectors) if not s]
        def fun(*xs):
            def integrand(*ys):
                xi, yi = iter(xs), iter(ys)
                args = [next(xi) if s else next(yi)
                        for s in selectors]
                return self(*args)
            return nquad_wrapper(integrand, supp_int)
        supp = [d for d, s in zip(self.supp, selectors) if s]
        return Fun(fun, supp)

    u_marginal = np.frompyfunc(lambda f, selectors: f.marginal(selectors), 2, 1)

    def asscalar(self):
        """Return a scalar, assuming 'self.cont_dim == 0'."""
        return self()

    _u_asscalar = np.frompyfunc(lambda f: f.asscalar(), 1, 1)

    @staticmethod
    def vect_asscalar(array):
        out = np.empty_like(array, dtype=float)
        Fun._u_asscalar(array, out=out)
        return out

    def ortho(self, dom):
        """Orthosupplement."""
        return Fun(lambda *xs: 1.0 - self(*xs), dom)

    u_ortho = np.frompyfunc(lambda f, dom: f.ortho(dom), 2, 1)

    def plot(self, preargs=[], interval=None, postargs=[], steps=256):
        axis = len(preargs)
        if interval:
            start = interval[0]
            stop = interval[1]
        else:
            start, stop = self.supp[axis]
        if stop < start:
            raise ValueError("Empty interval")
        if math.isinf(start) or math.isinf(stop):
            raise ValueError("Unbounded interval")
        fig, (ax) = plt.subplots(1, 1, figsize=(10,5))
        xs = np.linspace(start, stop, steps, endpoint=True)
        ys = [self(*(preargs+(x,)+postargs)) for x in xs]
        plt.interactive(True)
        ax.plot(xs, ys, color="blue", linewidth=2.0, linestyle="-")
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

    def asfun2(self):
        return Fun2(lambda _, ys: self(*ys), [], self.supp)

    u_asfun2 = np.frompyfunc(lambda f: f.asfun2(), 1, 1)


def asfun(fun, dom):
    if not isinstance(fun, Fun):
        return Fun(fun, dom)
    if not supp_issubset(fun.supp, dom):
        raise ValueError("Support must be a subset of the domain")
    return fun

u_asfun = np.frompyfunc(asfun, 2, 1)


class StateOrPredicate:
    """Common structures of states and predicates.

    Args:
    TODO

    Attributes:
    TODO
    """
    def __init__(self, array, dom):
        dom = asdom(dom)
        if dom.iscont:
            dtype = object
            array = u_asfun(array, _wrap(dom.cont))
        else:
            dtype = float
        self.shape = tuple(len(s) for s in dom.disc)
        array = np.asarray(array, dtype=dtype).reshape(self.shape)

        self.array = array
        self.dom = dom

    @staticmethod
    def _fromfun_getelm(fun, dom, disc_args):
        if not dom.iscont:
            return fun(*disc_args)
        def f(*cont_args):
            args = dom.merge(disc_args, cont_args)
            return fun(*args)
        return Fun(f, dom.cont)

    @classmethod
    def fromfun(cls, fun, dom):
        """Create a state/predicate from a function.

        Example:

           Predicate.fromfun(lambda x, y: x < y, [R, range(10)])

        """
        dom = asdom(dom)
        shape = tuple(len(d) for d in dom.disc)
        if dom.iscont:
            array = np.empty(shape, dtype=object)
        else:
            array = np.empty(shape, dtype=float)
        for index in np.ndindex(*shape):
            disc_args = dom.disc_get(index)
            array[index] = (
                StateOrPredicate._fromfun_getelm(fun, dom, disc_args))
        return cls(array, dom)

    def __add__(self, other):
        """Pointwise addition."""
        check_dom_match(self.dom, other.dom)
        return type(self)(self.array + other.array, self.dom)

    def __mul__(self, scalar):
        """Multiplies by a scalar"""
        if self.dom.iscont:
            return type(self)(Fun.u_smul(self.array, scalar), self.dom)
        return type(self)(self.array * scalar, self.dom)

    def __rmul__(self, scalar):
        return self * scalar

    def __matmul__(self, other):
        """Forms the joint state / predicate"""
        if self.dom.iscont:
            if other.dom.iscont:
                outer = Fun.u_joint.outer
            else:
                outer = Fun.u_smul.outer
        else:
            if other.dom.iscont:
                outer = Fun.u_rsmul.outer
            else:
                outer = np.outer
        return type(self)(outer(self.array, other.array),
                          self.dom + other.dom)

    def __pow__(self, n):
        if n == 0:
            raise ValueError("Power must be at least 1")
        return reduce(lambda s1, s2: s1 @ s2, [self] * n)

    def getvalue(self, *args, disc_args=None, cont_args=None):
        if disc_args is None or cont_args is None:
            disc_args, cont_args = self.dom.split(args)
        elm = self.array[self.dom.get_disc_indices(disc_args)]
        if self.dom.iscont:
            return elm(*cont_args)
        return elm

    @staticmethod
    def _plot_split(*args):
        preargs, i = [], 0
        for i, a in enumerate(args):
            if a is Ellipsis or isinstance(a, Interval):
                break
            elif isR(a):
                a = R_
                break
            preargs.append(a)
        return tuple(preargs), a, args[i+1:]

    def plot(self, *args, **kwargs):
        """Plots a certain axis of the state or predicate.

        Suppose self is a state/predicate on domain A * B * C and B is
        continuous type. The method can be called as

            self.plot(a, R(r1, r2), c)

        with arguments a, c and an interval R(r1, r2). It plots the
        function by varying the second argument from r1 to r2. We may
        use '...' (Ellipsis) instead of the interval as

            self.plot(a, ..., c)

        In this case the interval is set from its domain.
        """
        if not args:
            args = (...,)
        disc_args, cont_args = self.dom.split(args)
        fun = self.array[self.dom.get_disc_indices(disc_args)]
        (preargs,
         interval,
         postargs) = StateOrPredicate._plot_split(*cont_args)
        axis = len(preargs)
        if interval is Ellipsis:
            interval = self.dom.cont[axis]
        if not interval.issubset(self.dom.cont[axis]):
            raise ValueError("Interval must be a subset of domain")
        fun.plot(preargs=preargs, interval=interval,
                 postargs=postargs, **kwargs)


class State(StateOrPredicate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        if self.dom.iscont:
            return "State on " + str(self.dom)
        return " + ".join(
            "{:{spec}}|{}>".format(
                self.array[indices],
                ",".join([str(d) for d in self.dom.disc_get(indices)]),
                spec=float_format_spec)
            for indices in np.ndindex(*self.shape))

    def __ge__(self, pred):
        """Returns the validity."""
        check_dom_match(self.dom, pred.dom)
        if self.dom.iscont:
            return Fun.vect_integrate(self.array * pred.array).sum()
        return np.inner(self.array.ravel(), pred.array.ravel())

    def __truediv__(self, pred):
        """Returns the conditional state."""
        check_dom_match(self.dom, pred.dom)
        array = self.array * pred.array
        if self.dom.iscont:
            v = Fun.vect_integrate(array).sum()
        else:
            v = array.sum()
        if v == 0.0:
            raise ValueError("Zero validity making "
                            "conditioning impossible")
        if self.dom.iscont:
            return State(Fun.u_sdiv(array, v), self.dom)
        return State(array / v, self.dom)

    def __mod__(self, selectors):
        """Return a marginal state"""
        dom_marg, disc_sel, cont_sel = [], [], []
        for d, s in zip(self.dom, selectors):
            if s:
                dom_marg.append(d)
            if isinstance(d, Interval):
                cont_sel.append(s)
            else:
                disc_sel.append(s)
        array = self.array
        # First marginalise the continuous portion
        if not all(cont_sel):
            array = Fun.u_marginal(array, _wrap(cont_sel))
            if not any(cont_sel):
                array = Fun.vect_asscalar(array)
        # Marginalise the discrete portion
        if not all(disc_sel):
            axes = tuple(n for n, s in enumerate(disc_sel) if not s)
            array = array.sum(axes)
        return State(array, dom_marg)

    def as_pred(self):
        """
        Turning a state into a predicate. This works well in the discrete
        case but may not produce a predicate in the continuous case if
        the pdf becomes greater than 1
        """
        return Predicate(self.array, self.dom)

    def aschan(self):
        if self.dom.iscont:
            array = Fun.u_asfun2(self.array)
        else:
            array = self.array
        return Channel(array, [], self.dom)


class Predicate(StateOrPredicate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        if self.dom.iscont:
            return "Predicate on " + str(self.dom)
        return " | ".join(
            "{}: {:{spec}}".format(
                ",".join([str(d) for d in self.dom.disc_get(indices)]),
                self.array[indices],
                spec=float_format_spec)
            for indices in np.ndindex(*self.shape))

    def __and__(self, other):
        """Sequential conjunction (pointwise multiplication)."""
        check_dom_match(self.dom, other.dom)
        return Predicate(self.array * other.array, self.dom)

    def __or__(self, other):
        """De Morgan dual of sequential conjunction."""
        return ~(~self & ~other)

    def ortho(self):
        """Orthosupplement."""
        if self.dom.iscont:
            return Predicate(Fun.u_ortho(self.array,
                                         _wrap(self.dom.cont)),
                             self.dom)
        return Predicate(1.0 - self.array, self.dom)

    def __invert__(self):
        return self.ortho()


class Fun2:
    """Functions with two argments."""
    def __init__(self, fun, dom_supp, cod_supp):
        dom_supp = supp_init(dom_supp)
        cod_supp = supp_init(cod_supp)
        if use_lru_cache:
            fun = functools.lru_cache(maxsize=None)(fun)
        self.fun = supp_fun2(dom_supp, cod_supp, fun)
        self.dom_supp = dom_supp
        self.cod_supp = cod_supp

    def __call__(self, xs, ys):
        return self.fun(xs, ys)

    def __add__(self, other):
        """Pointwise addition."""
        return Fun2(lambda xs, ys: self(xs, ys) + other(xs, ys),
                    supp_union(self.dom_supp, other.dom_supp),
                    supp_union(self.cod_supp, other.cod_supp))

    def asscalar(self):
        return self((),())

    _u_asscalar = np.frompyfunc(lambda f: f.asscalar(), 1, 1)

    @staticmethod
    def vect_asscalar(array):
        out = np.empty_like(array, dtype=float)
        Fun2._u_asscalar(array, out=out)
        return out

    def stat_trans(self, stat_f):
        supp = supp_intersect(self.dom_supp, stat_f.supp)
        return Fun(lambda *ys:
                   nquad_wrapper(lambda *xs:
                                 self(xs, ys) * stat_f(*xs),
                                 supp),
                   self.cod_supp)

    u_stat_trans = np.frompyfunc(lambda f, s: f.stat_trans(s), 2, 1)

    def stat_trans_scalar(self, stat_s):
        return Fun(lambda *ys: self((), ys) * stat_s,
                   self.cod_supp)

    u_stat_trans_scalar = np.frompyfunc(lambda f, s:
                                        f.stat_trans_scalar(s), 2, 1)

    def pred_trans(self, pred_f):
        supp = supp_intersect(self.cod_supp, pred_f.supp)
        return Fun(lambda *xs:
                   nquad_wrapper(lambda *ys:
                                 pred_f(*ys) * self(xs,ys),
                                 supp),
                   self.dom_supp)

    u_pred_trans = np.frompyfunc(lambda f, p: f.pred_trans(p), 2, 1)

    def pred_trans_scalar(self, pred_s):
        return Fun(lambda *xs: pred_s * self(xs,()),
                   self.dom_supp)

    u_pred_trans_scalar = np.frompyfunc(lambda f, p:
                                        f.pred_trans_scalar(p), 2, 1)

    def fun_at(self, *xs):
        return Fun(lambda *ys: self(xs, ys), self.cod_supp)

    _u_fun_at = np.frompyfunc(lambda f, xs: f.fun_at(*xs), 2, 1)

    def vect_fun_at(array, xs):
        return Fun2._u_fun_at(array, _wrap(xs))

    def comp(self, other): # self o other
        def fun(xs, zs):
            def integrand(*ys):
                return self(ys, zs) * other(xs, ys)
            supp = supp_intersect(self.dom_supp, other.cod_supp)
            return nquad_wrapper(integrand, supp)
        return Fun2(fun, other.dom_supp, self.cod_supp)

    u_comp = np.frompyfunc(lambda f, g: f.comp(g), 2, 1)

    def comp_sc(self, other): # other is scalar
        def fun(xs, zs):
            def integrand(*ys):
                return self(ys, zs) * other
            supp = self.dom_supp
            return nquad_wrapper(integrand, supp)
        return Fun2(fun, [], self.cod_supp)

    u_comp_sc = np.frompyfunc(lambda f, g: f.comp_sc(g), 2, 1)

    def rcomp_sc(self, other):
        def fun(xs, zs):
            def integrand(*ys):
                return  other * self(xs, ys)
            supp = self.cod_supp
            return nquad_wrapper(integrand, supp)
        return Fun2(fun, self.dom_supp, [])

    u_rcomp_sc = np.frompyfunc(lambda f, g: g.rcomp_sc(f), 2, 1)

    def joint(self, other):
        def fun(xs, ys):
            lx = len(self.dom)
            ly = len(self.cod)
            return self(xs[:lx], ys[:ly]) * self(xs[lx:], ys[ly:])
        return Fun2(fun, self.dom_supp + other.dom_supp,
                    self.cod_supp + other.cod_supp)

    u_joint = np.frompyfunc(lambda f, g: f.joint(g), 2, 1)

    def smul(self, scalar):
        return Fun2(lambda xs, ys: self(xs, ys) * scalar,
                    self.dom_supp, self.cod_supp)

    u_smul = np.frompyfunc(lambda f, s: f.smul(s), 2, 1)

    u_rsmul = np.frompyfunc(lambda s, f: f.smul(s), 2, 1)


def asfun2(fun, dom, cod):
    if not isinstance(fun, Fun2):
        return Fun2(fun, dom, cod)
    if not (supp_issubset(fun.dom_supp, dom)
            and supp_issubset(fun.cod_supp, cod)):
        raise ValueError("Support must be a subset of the domain")
    return fun

u_asfun2 = np.frompyfunc(asfun2, 3, 1)


class Channel:
    """Channels.

    Attributes:
    TODO
    """
    def __init__(self, array, dom, cod):
        dom, cod = asdom(dom), asdom(cod)
        self.iscont = dom.iscont or cod.iscont
        if self.iscont:
            dtype = object
            array = u_asfun2(array, _wrap(dom.cont), _wrap(cod.cont))
        else:
            dtype = float
        self.dom_shape = tuple(len(s) for s in dom.disc)
        self.cod_shape = tuple(len(s) for s in cod.disc)
        self.shape = self.cod_shape + self.dom_shape
        array = np.asarray(array, dtype=dtype).reshape(self.shape)

        self.array = array
        self.dom = dom
        self.cod = cod
        self.dom_size = prod(self.dom_shape)
        self.cod_size = prod(self.cod_shape)

    @staticmethod
    def _fromklmap_getarray(klmap, cod, dom_disc_a):
        array = klmap(*dom_disc_a).array
        if cod.iscont:
            array = Fun.u_asfun2(array)
        return array

    @staticmethod
    def _fromklmap_getelm(klmap, dom, cod, dom_disc_a, cod_idx):
        if cod.cont:
            def f(xs, ys):
                args = dom.merge(dom_disc_a, xs)
                return klmap(*args).array[cod_idx](*ys)
        else:
            def f(xs, ys):
                args = dom.merge(dom_disc_a, xs)
                return klmap(*args).array[cod_idx]
        return Fun2(f, dom.cont, cod.cont)

    @classmethod
    def fromklmap(cls, klmap, dom, cod):
        dom, cod = asdom(dom), asdom(cod)
        iscont = dom.iscont or cod.iscont
        dom_shape = tuple(len(s) for s in dom.disc)
        cod_shape = tuple(len(s) for s in cod.disc)
        cod_ndim = len(cod.disc)
        shape = cod_shape + dom_shape
        if iscont:
            array = np.empty(shape, dtype=object)
        else:
            array = np.empty(shape, dtype=float)
        if not dom.iscont:
            for index in np.ndindex(*dom_shape):
                dom_disc_a = dom.disc_get(index)
                array[(...,)+index] = (
                    Channel._fromklmap_getarray(klmap, cod, dom_disc_a))
        else:
            for dom_idx in np.ndindex(*dom_shape):
                dom_disc_a = dom.disc_get(dom_idx)
                for cod_idx in np.ndindex(*cod_shape):
                    array[cod_idx+dom_idx] = (
                        Channel._fromklmap_getelm(klmap, dom, cod,
                                                  dom_disc_a, cod_idx))
        return cls(array, dom, cod)

    def __repr__(self):
        return "Channel of type: {} --> {}".format(self.dom, self.cod)

    def stat_trans(self, stat):
        check_dom_match(self.dom, stat.dom)
        dom_size = stat.array.size
        self_a = self.array.reshape(-1, dom_size)
        if stat.dom.iscont:
            stat_a = stat.array.reshape(1, dom_size)
            array = Fun2.u_stat_trans(self_a, stat_a).sum(1)
        else:
            if self.iscont:
                stat_a = stat.array.reshape(1, dom_size)
                array = Fun2.u_stat_trans_scalar(self_a, stat_a).sum(1)
            else:
                stat_a = stat.array.ravel()
                array = np.dot(self_a, stat_a)
        return State(array, self.cod)

    def __rshift__(self, stat):
        return self.stat_trans(stat)

    def pred_trans(self, pred):
        check_dom_match(self.cod, pred.dom)
        cod_size = pred.array.size
        self_a = self.array.reshape(cod_size, -1)
        if pred.dom.iscont:
            pred_a = pred.array.reshape(cod_size, 1)
            array = Fun2.u_pred_trans(self_a, pred_a).sum(0)
        else:
            if self.iscont:
                pred_a = pred.array.reshape(cod_size, 1)
                array = Fun2.u_pred_trans_scalar(self_a, pred_a).sum(0)
            else:
                pred_a = pred.array.ravel()
                array = np.dot(pred_a, self_a)
        return Predicate(array, self.dom)

    def __lshift__(self, pred):
        return self.pred_trans(pred)

    def getstate(self, *args, disc_args=None, cont_args=None):
        if disc_args is None or cont_args is None:
            disc_args, cont_args = self.dom.split(args)
        indices = self.dom.get_disc_indices(disc_args)
        array = self.array[(...,)+indices]
        if self.iscont:
            array = Fun2.vect_fun_at(array, cont_args)
            if not self.cod.iscont:
                array = Fun.vect_asscalar(array)
        return State(array, self.cod)

    def comp(self, other):
        """Computes the composition (self after other)."""
        check_dom_match(self.dom, other.cod)
        if self.iscont or other.iscont:
            if self.iscont and other.iscont:
                ufunc = Fun2.u_comp
            elif self.iscont:
                ufunc = Fun2.u_comp_sc
            else:
                ufunc = Fun2.u_rcomp_sc
            array = ufunc(self.array.reshape(self.cod_size,
                                             self.dom_size,
                                             1),
                          other.array.reshape(1,
                                              other.cod_size,
                                              other.dom_size)).sum(1)
            if not self.cod.iscont and not other.dom.iscont:
                array = Fun2.vect_asscalar(array)
        else:
            array = np.dot(self.array.reshape(self.cod_size,
                                              self.dom_size),
                           other.array.reshape(other.cod_size,
                                               other.dom_size))
        return Channel(array, other.dom, self.cod)

    def __mul__(self, other):
        return self.comp(other)

    def joint(self, other):
        selfa = self.array.reshape(self.cod_size,
                                   self.dom_size)
        othera = other.array.reshape(other.cod_size,
                                     other.dom_size)
        if self.iscont:
            if other.iscont:
                outer = Fun.u_joint.outer
            else:
                outer = Fun.u_smul.outer
        else:
            if other.iscont:
                outer = Fun.u_rsmul.outer
            else:
                outer = np.outer
        return Channel(kron2d(selfa, othera, outer=outer),
                       self.dom + other.dom,
                       self.cod + other.cod)

    def __matmul__(self, other):
        return self.joint(other)


def idn(dom):
    dom = asdom(dom)
    if dom.iscont:
        raise ValueError("Cannot make a continuous identity channel")
    dom_size = prod(len(s) for s in dom)
    return Channel(np.eye(dom_size), dom, dom)


def discard(dom):
    dom = asdom(dom)
    cod = Dom([])
    dom_shape = tuple(len(s) for s in dom.disc)
    if dom.iscont:
        array = np.full(dom_shape,
                        Fun2(lambda xs, _: 1.0, dom.cont, []),
                        dtype=object)
    else:
        array = np.ones(dom_shape)
    return Channel(array, dom, cod)


def swap(dom1, dom2):
    dom1 = asdom(dom1)
    dom2 = asdom(dom2)
    if dom1.iscont or dom2.iscont:
        raise ValueError("Cannot make a continuous swap channel")
    size1 = prod(len(s) for s in dom1)
    size2 = prod(len(s) for s in dom2)
    array = np.zeros((size2, size1, size1, size2))
    for n1 in range(size1):
        for n2 in range(size2):
            array[n2, n1, n1, n2] = 1.0
    return Channel(array, dom1+dom2, dom2+dom1)


def copy(dom):
    dom = asdom(dom)
    if dom.iscont:
        raise ValueError("Cannot make a continuous copy channnel")
    size = prod(len(s) for s in dom)
    array = np.zeros((size, size, size))
    for n in range(size):
        array[n, n, n] = 1.0
    return Channel(array, dom, dom+dom)


class DetChan:
    """Deterministic channels."""
    def __init__(self, funs, dom):
        self.funs = funs if isinstance(funs, (list, tuple)) else [funs]
        self.dom = asdom(dom)
        self.shape = tuple(len(s) for s in self.dom.disc)

    def _pred_trans_getelm(self, pred, disc_funs, cont_funs, disc_args):
        if not pred.dom.iscont:
            return pred.getvalue(
                disc_args=[f(*disc_args) for f in self.funs],
                cont_args=[])
        def fun(*cont_args):
            args = self.dom.merge(disc_args, cont_args)
            p_disc_args = [df(*args) for df in disc_funs]
            p_cont_args = [cf(*args) for cf in cont_funs]
            return pred.getvalue(disc_args=p_disc_args,
                                 cont_args=p_cont_args)
        return Fun(fun, self.dom.cont)

    def pred_trans(self, pred):
        array = np.empty(self.shape, dtype=object)
        disc_funs, cont_funs = pred.dom.split(self.funs)
        for index in np.ndindex(*self.shape):
            disc_args = self.dom.disc_get(index)
            array[index] = self._pred_trans_getelm(
                pred, disc_funs, cont_funs, disc_args)
        return Predicate(array, self.dom)

    def __lshift__(self, pred):
        return self.pred_trans(pred)

    def joint(self, other):
        d_len = len(self.dom.disc) + len(self.dom.cont)
        funs = ([lambda *args: f(*args[:d_len]) for f in self.funs]
                + [lambda *args: f(*args[d_len:]) for f in other.funs])
        return DetChan(funs, self.dom + other.dom)

    def __matmul__(self, other):
        return self.joint(other)

    def __pow__(self, n):
        if n == 0:
            raise ValueError("Power must be at least 1")
        return reduce(lambda s1, s2: s1 @ s2, [self] * n)


class RandVar(DetChan):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __mul__(self, other):
        if isinstance(other, DetChan):
            if not isinstance(other, RandVar):
                raise TypeError("Multiplicand must be a random variable")
            funs = [lambda *args: sf(*args) * of(*args)
                    for sf, of in zip(self.funs, other.funs)]
        else:
            if isinstance(other, list):
                funs = [lambda *args: sf(*args) * o
                        for sf, o in zip(self.funs, other)]
            else:
                funs = [lambda *args: sf(*args) * other
                        for sf in self.funs]
        return RandVar(funs, self.dom)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, DetChan):
            if not isinstance(other, RandVar):
                raise TypeError("Summand must be a random variable")
            funs = [lambda *args: sf(*args) + of(*args)
                    for sf, of in zip(self.funs, other.funs)]
        else:
            if isinstance(other, list):
                funs = [lambda *args: sf(*args) + o
                        for sf, o in zip(self.funs, other)]
            else:
                funs = [lambda *args: sf(*args) + other
                        for sf in self.funs]
        return RandVar(funs, self.dom)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        funs = [lambda *args: -sf(*args) for sf in self.funs]
        return RandVar(funs, self.dom)

    def __sub__(self, other):
        if isinstance(other, list):
            return self + [-o for o in other]
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def _exp_calc(self, elm, f, disc_args):
        if not self.dom.iscont:
            return elm * f(*disc_args)
        def fun(*cont_args):
            args = self.dom.merge(disc_args, cont_args)
            return f(*args)
        return (elm * Fun(fun, self.dom.cont)).integrate()

    def exp(self, stat):
        check_dom_match(self.dom, stat.dom)
        exp = [0.0] * len(self.funs)
        it = np.nditer(stat.array, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            disc_args = self.dom.disc_get(it.multi_index)
            for m, f in enumerate(self.funs):
                exp[m] += self._exp_calc(it[0], f, disc_args)
            it.iternext()
        if len(exp) == 1:
            return exp[0]
        return exp

    def _var_calc(self, elm, f, e, disc_args):
        if not self.dom.iscont:
            d = f(*disc_args) - e
            return elm * d * d
        def fun(*cont_args):
            args = self.dom.merge(disc_args, cont_args)
            v = e - f(*args)
            return v * v
        return (elm * Fun(fun, self.dom.cont)).integrate()

    def var(self, stat, exp=None):
        check_dom_match(self.dom, stat.dom)
        if exp is None:
            exp = self.exp(stat)
        if not isinstance(exp, list):
            exp = [exp]
        var = [0.0] * len(self.funs)
        it = np.nditer(stat.array, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            disc_args = self.dom.disc_get(it.multi_index)
            for m, (f, e) in enumerate(zip(self.funs, exp)):
                var[m] += self._var_calc(it[0], f, e, disc_args)
            it.iternext()
        if len(var) == 1:
            return var[0]
        return var

    def stdev(self, stat, exp=None):
        var = self.var(stat, exp=exp)
        if isinstance(var, list):
            return [math.sqrt(v) for v in var]
        return math.sqrt(var)

#
# Auxiliary functions to create states, predicates, channnels
#

# Aliases
state = State
predicate = Predicate
channel = Channel
detchan = DetChan
randvar = RandVar

state_fromfun = State.fromfun
pred_fromfun = Predicate.fromfun
chan_fromklmap = Channel.fromklmap


def flip(r, dom=[True, False]):
    return State([r, 1.0-r], [dom])


def const_state_or_pred(cls, value, subdom, dom=None):
    subdom = asdom(subdom)
    if dom is None:
        dom = subdom
    else:
        dom = asdom(dom)
    shape = tuple(len(s) for s in dom.disc)
    if dom.iscont:
        array = np.empty(shape, dtype=object)
        valelm = Fun(lambda *xs: value, subdom.cont)
        zeroelm = Fun(lambda *xs: 0.0, [empty]*len(dom.cont))
    else:
        array = np.empty(shape, dtype=float)
        valelm = value
        zeroelm = 0.0
    for index in np.ndindex(*shape):
        if all(dom.disc[n][i] in subdom.disc[n]
               for n, i in enumerate(index)):
            array[index] = valelm
        else:
            array[index] = zeroelm
    return cls(array, dom)


def uniform_state(subdom, dom=None):
    subdom = asdom(subdom)
    size = prod(len(s) for s in subdom.disc)
    vol = prod(u - l for l, u in subdom.cont)
    if not math.isfinite(vol):
        raise ValueError("Unbounded domain")
    value = 1.0 / (size * vol)
    return const_state_or_pred(State, value, subdom, dom)


def const_pred(value, subdom, dom=None):
    return const_state_or_pred(Predicate, value, subdom, dom)


def truth(subdom, dom=None):
    return const_pred(1.0, subdom, dom)


def falsity(dom):
    dom = asdom(dom)
    subdom = dom.merge(itertools.repeat([]), itertools.repeat(empty))
    return const_pred(0.0, subdom, dom)


#
# Uniform discrete state on {0,1,...,n-1}
#
def uniform_disc_state(n):
    return uniform_state(range(n))

#
# Unit discrete state 1|i> on {0,1,...,n-1}
#
def unit_disc_state(n, i):
    ls = [0] * n
    ls[i] = 1
    return State(ls, range(n))

#
# Random discrete state on {0,1,...,n-1}
#
def random_disc_state(n):
    ls = [random.uniform(0,1) for i in range(n)]
    s = sum(ls)
    return State([v/s for v in ls], range(n))


@functools.lru_cache(maxsize=None)
def gaussian_compensation(mu, sigma, lb, ub):
    return 1.0 - (stats.norm.cdf(lb, loc=mu, scale=sigma)
                  + stats.norm.sf(ub, loc=mu, scale=sigma))


def gaussian_fun(mu, sigma, supp=R):
    if isR(supp):
        def fun(x):
            return stats.norm.pdf(x, loc=mu, scale=sigma)
    else:
        s = gaussian_compensation(mu, sigma, supp[0], supp[1])
        def fun(x):
            return stats.norm.pdf(x, loc=mu, scale=sigma) / s
    return Fun(fun, [supp])


def gaussian_state(mu, sigma, supp=R):
    return State(gaussian_fun(mu, sigma, supp), [supp])


_sqrt2pi = math.sqrt(2 * math.pi)


def gaussian_pred(mu, sigma, supp=R, scaling=True):
    if scaling:
        s = _sqrt2pi * sigma
        def fun(x):
            return stats.norm.pdf(x, loc=mu, scale=sigma) * s
    else:
        def fun(x):
            return stats.norm.pdf(x, loc=mu, scale=sigma)
    return Predicate(Fun(fun, supp), [supp])

#
# Random discrete predicate on {0,1,...,n-1}
#
def random_disc_pred(n):
    return Predicate([random.uniform(0,1) for i in range(n)], range(n))

#
# Discrete that is 1 at i in {0,1,...,n-1}
#
def unit_pred(n, i):
    ls = [0] * n
    ls[i] = 1
    return Predicate(ls, range(n))


##############################################################
#
# Bart's additions (temporarily)
#
##############################################################

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

#
# Poisson distribution with rate parameter `lam' and upperbound
# ub. The distribution is restricted to the interval [0, ub-1]; hence
# the values have to adjusted so that they sum up to 1 on this
# interval.
#
def poisson(lam, ub):
    probabilities = [(lam ** k) * (math.e ** -lam) / factorial(k) 
                     for k in range(ub)]
    s = sum(probabilities)
    return State([p/s for p in probabilities], range(ub))


# Bayesian network auxiliaries

#
# The domain that is standardly used: bnd = Bayesian Network Domain
#
bnd = Dom(['t', 'f'])

#
# Basic (sharp) predicates on this domain
#
tt = Predicate([1,0], bnd)
ff = ~tt

#
# Function for modelling an initial node, as prior state
#
def bn_prior(r): 
    return State([r, 1-r], bnd)

#
# Conditional probability table converted into a channel. The input is
# a list of probabilities, of length 2^n, where n is the number of
# predecessor nodes.
#
def cpt(*ls):
    n = len(ls)
    if n == 0:
        raise Exception('Conditional probability table must have non-empty list of probabilities')
    log = math.log(n, 2)
    if log != math.floor(log):
        raise Exception('Conditional probability table must have 2^n elements')
    log = int(log)
    return Channel([ls, [1-r for r in ls]], bnd * log, bnd)


def channel_from_states(ls):
    n = len(ls)
    if n == 0:
        raise ValueError('Non-empty list required in channel formation')
    dom = ls[0].dom
    if any([s.dom != dom for s in ls]):
        raise ValueError('States must all have the same domain in channel formation')
    return None

#
# Convex sum of states: the input list contains pairs (ri, si) where
# the ri are in [0,1] and add up to 1, and the si are states
#
def convex_state_sum(*ls):
    if len(ls) == 0:
        raise ValueError('Convex sum cannot be empty')
    dom = ls[0][1].dom
    if any([s.dom != dom for r,s in ls[1:]]):
        raise ValueError('Convex sum requires that states have the same domain')
    if any([r < 0 or r > 1 for (r,s) in ls]):
        raise ValueError('Convex sum requires numbers in the unit interval')
    ar = np.array([r for r,s in ls])
    if not np.isclose(sum(ar), np.array([1])):
        raise ValueError('Scalars must add up to 1 in convex sum')
    return reduce(operator.add, [r * s for r,s in ls])


##############################################################
#
# Testing 
#
##############################################################



def test():
    stat = State([gaussian_fun(1, 1, R(-10, 10)).smul(0.8),
                  gaussian_fun(-1, 1, R(-10, 10)).smul(0.2)],
                 [[True, False], R(-10, 10)])
    pred = truth([[True, False], R(-10, 10)]) * 0.5
    print(stat >= pred)
    print(stat % [1, 0])
    (stat % [0, 1]).plot()


def sporter():
    sp1 = uniform_state(R(0, 1))
    sp2 = flip(1/3, ['g', '~g'])
    joint = sp1 @ sp2
    sp2win = Predicate([lambda x: 1 if 16*x < 11 else 0,
                        lambda x: 1 if 16*x < 1 else 0],
                       [R(0, 1), ['g', '~g']])
    print("Sporter 2 wins with prob.", joint >= sp2win)
    poster = joint / sp2win
    post_sp1 = poster % [1, 0]
    post_sp2 = poster % [0, 1]
    print("Posterior dist. of Sporter 2:", post_sp2)
    print("Plot posterior density of Sporter 1")
    post_sp1.plot()


def sporter2():
    sp1 = uniform_state(R(0, 1))
    sp2 = flip(1/3)
    joint = sp1 @ sp2
    perf1 = DetChan(lambda x: 16*x - 6, R(0, 1))
    perf2 = DetChan(lambda b: 5 if b else -5, [True, False])
    t = Predicate(lambda x,y: 1 if x < y else 0,
                  [R(-10, 10), R(-10, 10)])
    print("Sporter 2 wins with prob.",
          joint >= (perf1 @ perf2) << t)
    poster = joint / ((perf1 @ perf2) << t)
    post_sp1 = poster % [1, 0]
    post_sp2 = poster % [0, 1]
    print("Posterior dist. of Sporter 2:", post_sp2)
    print("Plot posterior density of Sporter 1")
    post_sp1.plot()


def dice():
    pips = [1,2,3,4,5,6]
    stat = uniform_state(pips)
    rv = RandVar(lambda x: x, pips)
    print("Exp.", rv.exp(stat))
    print("Var.", rv.var(stat))
    print("St.Dev.", rv.stdev(stat))
    exp = rv.exp(stat)
    rv2 = rv - exp
    print("Var. by formula 1:",
          (rv2 * rv2).exp(stat))
    print("Var. by formula 2:",
          (rv * rv).exp(stat) - rv.exp(stat) ** 2)

    rv_sum = RandVar(lambda x,y: x+y, [pips]*2)
    stat2 = stat @ stat
    print("Exp.", rv_sum.exp(stat2))
    print("Var.", rv_sum.var(stat2))
    print("St.Dev.", rv_sum.stdev(stat2))
    print("Var. by formula:",
          (rv_sum * rv_sum).exp(stat2) - rv_sum.exp(stat2) ** 2)


def test_chan():
    A = range(5)
    B = range(3)
    C = range(7)
    c = ((swap(B,A) @ idn(C))
         * swap(C, [B,A])
         * (idn(C) @ swap(A,B))
         * (swap(A,C) @ idn(B))
         * (idn(A) @ swap(B,C)))
    print(np.all(c.array == idn([A,B,C]).array))


def main():
    #sporter()
    print( convex_state_sum((0.2,flip(0.3)), (0.8, flip(0.8))))


if __name__ == "__main__":
    main()
