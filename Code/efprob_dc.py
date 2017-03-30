#
# Discrete and continuous probability library, prototype version
#
# Copyright: Bart Jacobs, Kenta Cho; 
# Radboud University Nijmegen
# efprob.cs.ru.nl
#
# Date: 2017-03-27
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


class NormalizationError(Exception):
    """Raised when normalization fails"""


def _prod(iterable):
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


def bar_plot(value_list, label_list, block=True):
    plt.subplots()
    x = range(len(value_list))
    y = value_list
    plt.xticks(x, label_list, rotation=45)
    plt.bar(x, y, align="center")
    plt.draw()
    plt.pause(0.001)
    if block:
        input("Press [enter] to continue.")


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
bool_dom = [True, False]


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

    def __len__(self):
        return len(self._dom)

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
        raise ValueError("Domains do not match: "
                         "{} and {}".format(dom1, dom2))

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
        if supp_contains(supp, xs):
            return fun(*xs)
        return 0.0
    return f

def supp_fun2(supp1, supp2, fun):
    def f(xs, ys):
        if supp_contains(supp1, xs) and supp_contains(supp2, ys):
            return fun(xs, ys)
        return 0.0
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

    def __sub__(self, other):
        """Pointwise subtraction."""
        return Fun(lambda *xs: self(*xs) - other(*xs),
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
        """Integrate functions in an array."""
        out = np.empty_like(array, dtype=float)
        Fun.u_integrate(array, out=out)
        return out

    def smul(self, scalar):
        return Fun(lambda *xs: self(*xs) * scalar, self.supp)

    u_smul = np.frompyfunc(lambda f, s: f.smul(s), 2, 1)

    u_rsmul = np.frompyfunc(lambda s, f: f.smul(s), 2, 1)

    def sdiv(self, scalar):
        return Fun(lambda *xs: self(*xs) / scalar, self.supp)

    u_sdiv = np.frompyfunc(lambda f, s: f.sdiv(s), 2, 1)

    def joint(self, other):
        n = len(self.supp)
        return Fun(lambda *xs: self(*xs[:n]) * other(*xs[n:]),
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
        """Return a scalar, assuming ``self.cont_dim == 0``."""
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

    def plot(self, preargs=(), interval=None,
             postargs=(), steps=256, block=True):
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
        if block:
            input("Press [enter] to continue.")

    def asfun1(self):
        return Fun2(lambda xs, _: self(*xs), self.supp, [])

    u_asfun1 = np.frompyfunc(lambda f: f.asfun1(), 1, 1)

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


class StateLike:
    """State-like objects.

    It is a superclass of states, random variables and predicates.
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
        """Create a state-like object from a function.

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
                StateLike._fromfun_getelm(fun, dom, disc_args))
        return cls(array, dom)

    def __call__(self, *args, **kwargs):
        return self.getvalue(*args, **kwargs)

    def __add__(self, other):
        """Pointwise addition."""
        check_dom_match(self.dom, other.dom)
        return type(self)(self.array + other.array, self.dom)

    def smul(self, scalar, cls=None):
        """Scalar multiplication."""
        if cls is None:
            cls = type(self)
        if self.dom.iscont:
            return cls(Fun.u_smul(self.array, scalar), self.dom)
        return cls(self.array * scalar, self.dom)

    def __mul__(self, scalar):
        return self.smul(scalar)

    def __rmul__(self, scalar):
        return self * scalar

    def joint(self, other):
        """Form a joint state-like."""
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

    def __matmul__(self, other):
        return self.joint(other)

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

    def _plot_getiter(self, *args):
        iters = []
        for n, a in enumerate(args):
            if a is Ellipsis:
                iters.append(range(len(self.dom.disc[n])))
            else:
                iters.append([self.dom.disc[n].index(a)])
        return itertools.product(*iters)

    @staticmethod
    def _plot_split(*args):
        i = 0
        interval = None
        for a in args:
            if a is Ellipsis or isinstance(a, Interval) or isR(a):
                interval = a
                break
            i += 1
        preargs = args[:i]
        interval = interval if not isR(interval) else R_
        postargs = args[i+1:]
        if any(a is Ellipsis or isinstance(a, Interval) or isR(a)
               for a in postargs):
            raise ValueError("Cannot plot dim > 1")
        return preargs, interval, postargs

    def plot(self, *args, **kwargs):
        """Plot some axes of the state-like.

        Let `s` be a state/predicate on domain A * B * C. If B is
        discrete type, then

            s.plot(a, ..., c)

        plots a bar chart by varying the second argument. If B is
        continuous type, then it plots a graph instead. We can
        restrict the plot to a certain interval R(r1, r2) by

            s.plot(a, R(r1, r2), c)

        This is necessary when the sencod axis has an unbounded
        domain. We can plot multiple axes as

            s.plot(a, ..., ...)

        only when both B and C are discrete. If we call the method
        without argument as `s.plot()`, it is interpreted as

            s.plot(..., ..., ...)

        """
        if not args:
            args = (...,) * len(self.dom)
        disc_args, cont_args = self.dom.split(args)
        discplot = any(a is Ellipsis for a in disc_args)
        contplot = any(a is Ellipsis
                       or isinstance(a, Interval) or isR(a)
                       for a in cont_args)
        if discplot and contplot:
            raise ValueError("Cannot plot discrete and continuous "
                             "axes at the same time")
        elif discplot:
            ellipsis_axes = [n for n, a in enumerate(disc_args)
                            if a is Ellipsis]
            values = []
            labels = []
            for idx in self._plot_getiter(*disc_args):
                if self.dom.iscont:
                    values.append(self.array[idx](*cont_args))
                else:
                    values.append(self.array[idx])
                label_args = [self.dom.disc[n][idx[n]]
                              for n in ellipsis_axes]
                labels.append(",".join(str(a) for a in label_args))
            bar_plot(values, labels, **kwargs)
        elif contplot:
            (preargs,
             interval,
             postargs) = StateLike._plot_split(*cont_args)
            fun = self.array[self.dom.get_disc_indices(disc_args)]
            axis = len(preargs)
            if interval is Ellipsis:
                interval = self.dom.cont[axis]
                if not interval.issubset(self.dom.cont[axis]):
                    raise ValueError("Interval must be a subset of domain")
            fun.plot(preargs=preargs, interval=interval,
                     postargs=postargs, **kwargs)
        else:
            raise ValueError("Nothing to plot")


class State(StateLike):
    """States."""
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

    def validity(self, pred):
        """Return the validity of a predicate."""
        return self.expectation(pred)

    def __ge__(self, pred):
        return self.validity(pred)

    def totalmass(self):
        """Total mass of the (improper) state"""
        if self.dom.iscont:
            return Fun.vect_integrate(self.array).sum()
        return self.array.sum()

    def normalize(self):
        """Normalize the improper state."""
        v = self.totalmass()
        if v == 0:
            raise NormalizationError("Total mass is zero")
        if math.isinf(v) and v > 0:
            raise NormalizationError("Total mass is infinite")
        if v < 0 or math.isnan(v):
            raise NormalizationError("Total mass is invalid (negative or NaN)")
        if self.dom.iscont:
            return State(Fun.u_sdiv(self.array, v), self.dom)
        return State(self.array / v, self.dom)

    def conditional(self, pred):
        """Return a conditional state."""
        check_dom_match(self.dom, pred.dom)
        try:
            return State(self.array * pred.array, self.dom).normalize()
        except NormalizationError as e:
            raise NormalizationError("Conditioning failed: {}".format(e)) from None

    def partial_conditional(self, pred):
        """Return a conditional state."""
        check_dom_match(self.dom, pred.dom)
        array = self.array * pred.array
        if self.dom.iscont:
            v = Fun.vect_integrate(array).sum()
        else:
            v = array.sum()
        v = v + (1 - self.totalmass())
        if v <= 0 or math.isinf(v) or math.isnan(v):
            raise ValueError("Denominator for conditioning is invalid")
        return State(array / v, self.dom)

    def __truediv__(self, pred):
        return self.conditional(pred)

    def marginal(self, selectors):
        """Return a marginal state."""
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

    def __mod__(self, selectors):
        return self.marginal(selectors)

    def as_pred(self):
        """Turn a state into a predicate.

        This works well in the discrete case but may not produce a
        predicate in the continuous case if the pdf becomes greater
        than 1.
        """
        return Predicate(self.array, self.dom)

    def as_chan(self):
        if self.dom.iscont:
            array = Fun.u_asfun2(self.array)
        else:
            array = self.array
        return Channel(array, [], self.dom)

    def expectation(self, randvar=None):
        if randvar is None:
            randvar = randvar_fromfun(lambda x: x, self.dom)
        check_dom_match(randvar.dom, self.dom)
        if self.dom.iscont:
            return Fun.vect_integrate(self.array * randvar.array).sum()
        return np.inner(self.array.ravel(), randvar.array.ravel())

    def variance(self, randvar=None, exp=None):
        if randvar is None:
            randvar = randvar_fromfun(lambda x: x, self.dom)
        check_dom_match(self.dom, randvar.dom)
        if exp is None:
            exp = self.expectation(randvar)
        if self.dom.iscont:
            a = np.empty_like(randvar.array, dtype=float)
            _var_integral_u(randvar.array, self.array, exp, out=a)
            return a.sum()
        a = randvar.array - exp
        a = a * a
        return np.inner(a.ravel(), self.array.ravel())

    def st_deviation(self, randvar=None, exp=None):
        if randvar is None:
            randvar = randvar_fromfun(lambda x: x, self.dom)
        return math.sqrt(self.variance(randvar, exp=exp))

    def covariance(self, randvar1 = None, randvar2 = None):
        if randvar1 is None or randvar2 is None:
            randvar1 = randvar_fromfun(lambda *x: x[0], self.dom)
            randvar2 = randvar_fromfun(lambda *x: x[1], self.dom)
        if self.dom != randvar1.dom or self.dom != randvar2.dom:
            raise Exception('Domain mismatch in covariance computation')
        e1 = self.expectation(randvar1)
        e2 = self.expectation(randvar2)
        rv = RandVar.fromfun(lambda *x: (randvar1.getvalue(*x) - e1) * 
                             (randvar2.getvalue(*x) - e2), 
                             self.dom)
        return self.expectation(rv)

    def correlation(self, randvar1 = None, randvar2 = None):
        if randvar1 is None or randvar2 is None:
            randvar1 = randvar_fromfun(lambda *x: x[0], self.dom)
            randvar2 = randvar_fromfun(lambda *x: x[1], self.dom)
        cov = self.covariance(randvar1, randvar2)
        sd1 = self.st_deviation(randvar1)
        sd2 = self.st_deviation(randvar2)
        return cov / (sd1 * sd2)

    def disintegration(self, selectors):
        """Disintegration of the joint state"""
        dom, cod = [], []
        for d, s in zip(self.dom, selectors):
            if s:
                dom.append(d)
            else:
                cod.append(d)
        dom = Dom(dom)
        cod = Dom(cod)
        s = self % selectors
        def f(*x):
            def g(*y):
                xi, yi = iter(x), iter(y)
                args = [next(xi) if s else next(yi) for s in selectors]
                return self.getvalue(*args) / s.getvalue(*x)
            return State.fromfun(g, cod)
        return Channel.fromklmap(f, dom, cod)


def _var_integral(rvfun, sfun, exp):
    def integrand(*xs):
        v = rvfun(*xs) - exp
        return v * v * sfun(*xs)
    return nquad_wrapper(integrand, sfun.supp)

_var_integral_u = np.frompyfunc(_var_integral, 3, 1)


class RandVar(StateLike):
    """Random Variables."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        if self.dom.iscont:
            return "Random variable on " + str(self.dom)
        return " | ".join(
            "{}: {:{spec}}".format(
                ",".join([str(d) for d in self.dom.disc_get(indices)]),
                self.array[indices],
                spec=float_format_spec)
            for indices in np.ndindex(*self.shape))

    def __and__(self, other):
        """Sequential conjunction (pointwise multiplication)."""
        check_dom_match(self.dom, other.dom)
        return type(self)(self.array * other.array, self.dom)

    def __sub__(self, other):
        """Subtraction."""
        check_dom_match(self.dom, other.dom)
        return type(self)(self.array - other.array, self.dom)

    # def exp(self, stat):
    #     check_dom_match(self.dom, stat.dom)
    #     if self.dom.iscont:
    #         return Fun.vect_integrate(self.array * stat.array).sum()
    #     return np.inner(self.array.ravel(), stat.array.ravel())

    # def var(self, stat, exp=None):
    #     check_dom_match(self.dom, stat.dom)
    #     if exp is None:
    #         exp = self.exp(stat)
    #     if self.dom.iscont:
    #         a = np.empty_like(self.array, dtype=float)
    #         _var_integral_u(self.array, stat.array, exp, out=a)
    #         return a.sum()
    #     a = self.array - exp
    #     a = a * a
    #     return np.inner(a.ravel(), stat.array.ravel())

    # def stdev(self, stat, exp=None):
    #     return math.sqrt(self.var(stat, exp=exp))


class Predicate(RandVar):
    """Predicates."""
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

    def __mul__(self, scalar):
        if 0.0 <= scalar <= 1.0:
            return self.smul(scalar)
        return self.smul(scalar, cls=RandVar)

    def ortho(self):
        """Orthosupplement."""
        if self.dom.iscont:
            return Predicate(Fun.u_ortho(self.array,
                                         _wrap(self.dom.cont)),
                             self.dom)
        return Predicate(1.0 - self.array, self.dom)

    def __invert__(self):
        return self.ortho()

    def __or__(self, other):
        """De Morgan dual of sequential conjunction."""
        return ~(~self & ~other)

    def as_chan(self, cod=bool_dom):
        cod = asdom(cod)
        if self.dom.iscont:
            array = np.empty((2,)+self.array.shape, dtype=object)
            array[0, ...] = Fun.u_asfun1(self.array)
            array[1, ...] = Fun.u_asfun1(
                Fun.u_ortho(self.array, _wrap(self.dom.cont)))
        else:
            array = np.empty((2,)+self.array.shape, dtype=float)
            array[0, ...] = self.array
            array[1, ...] = 1.0 - self.array
        return Channel(array, self.dom, cod)


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

    def fun_at1(self, *xs):
        return Fun(lambda *ys: self(xs, ys), self.cod_supp)

    _u_fun_at1 = np.frompyfunc(lambda f, xs: f.fun_at1(*xs), 2, 1)

    def vect_fun_at1(array, xs):
        return Fun2._u_fun_at1(array, _wrap(xs))

    def fun_at2(self, *ys):
        return Fun(lambda *xs: self(xs, ys), self.dom_supp)

    _u_fun_at2 = np.frompyfunc(lambda f, ys: f.fun_at2(*ys), 2, 1)

    def vect_fun_at2(array, ys):
        return Fun2._u_fun_at2(array, _wrap(ys))

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
            lx = len(self.dom_supp)
            ly = len(self.dom_supp)
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
    """Channels."""
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
        self.dom_size = _prod(self.dom_shape)
        self.cod_size = _prod(self.cod_shape)

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

    @classmethod
    def from_states(cls, states, dom=None):
        if dom is None:
            dom = range(len(states))
        dom = asdom(dom)
        dom_shape = (len(states),)
        cod = states[0].dom
        cod_shape = tuple(len(s) for s in cod.disc)
        shape = cod_shape + dom_shape
        if cod.iscont:
            array = np.empty(shape, dtype=object)
        else:
            array = np.empty(shape, dtype=float)
        for i, s in enumerate(states):
            a = s.array
            if cod.iscont:
                a = Fun.u_asfun2(a)
            array[(...,)+(i,)] = a
        return Channel(array, dom, cod)

    def __call__(self, *args, **kwargs):
        return self.get_state(*args, **kwargs)

    def __repr__(self):
        return "Channel of type: {} --> {}".format(self.dom, self.cod)

    def __add__(self, other):
        check_dom_match(self.dom, other.dom)
        check_dom_match(self.cod, other.cod)
        return Channel(self.array + other.array, self.dom, self.cod)

    def smul(self, scalar):
        """Scalar multiplication."""
        if self.iscont:
            return Channel(Fun2.u_smul(self.array, scalar),
                           self.dom, self.cod)
        return Channel(self.array * scalar, self.dom, self.cod)

    def stat_trans(self, stat):
        """State transformer."""
        check_dom_match(self.dom, stat.dom)
        dom_size = stat.array.size
        self_a = self.array.reshape(-1, dom_size)
        if stat.dom.iscont:
            stat_a = stat.array.reshape(1, dom_size)
            array = Fun2.u_stat_trans(self_a, stat_a).sum(1)
            if not self.cod.iscont:
                array = Fun.vect_asscalar(array)
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
        """Predicate transformer."""
        check_dom_match(self.cod, pred.dom)
        cod_size = pred.array.size
        self_a = self.array.reshape(cod_size, -1)
        if pred.dom.iscont:
            pred_a = pred.array.reshape(cod_size, 1)
            array = Fun2.u_pred_trans(self_a, pred_a).sum(0)
            if not self.dom.iscont:
                array = Fun.vect_asscalar(array)
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

    def get_state(self, *args, disc_args=None, cont_args=None):
        if disc_args is None or cont_args is None:
            disc_args, cont_args = self.dom.split(args)
        indices = self.dom.get_disc_indices(disc_args)
        array = self.array[(...,)+indices]
        if self.iscont:
            array = Fun2.vect_fun_at1(array, cont_args)
            if not self.cod.iscont:
                array = Fun.vect_asscalar(array)
        return State(array, self.cod)

    def get_likelihood(self, *args, disc_args=None, cont_args=None):
        if disc_args is None or cont_args is None:
            disc_args, cont_args = self.cod.split(args)
        indices = self.cod.get_disc_indices(disc_args)
        array = self.array[indices+(...,)]
        if self.iscont:
            array = Fun2.vect_fun_at2(array, cont_args)
            if not self.dom.iscont:
                array = Fun.vect_asscalar(array)
        return RandVar(array, self.dom)

    def comp(self, other):
        """Compute the composition (self after other)."""
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
    dom_size = _prod(len(s) for s in dom)
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


def abort(dom, cod):
    dom = asdom(dom)
    cod = asdom(cod)
    dom_shape = tuple(len(s) for s in dom.disc)
    cod_shape = tuple(len(s) for s in cod.disc)
    shape = cod_shape + dom_shape
    if dom.iscont or cod.iscont:
        array = np.full(shape,
                        Fun2(lambda xs, ys: 0.0,
                             [empty]*len(dom.cont),
                             [empty]*len(cod.cont)),
                        dtype=object)
    else:
        array = np.zeros(shape)
    return Channel(array, dom, cod)


def swap(dom1, dom2):
    dom1 = asdom(dom1)
    dom2 = asdom(dom2)
    if dom1.iscont or dom2.iscont:
        raise ValueError("Cannot make a continuous swap channel")
    size1 = _prod(len(s) for s in dom1)
    size2 = _prod(len(s) for s in dom2)
    array = np.zeros((size2, size1, size1, size2))
    np.einsum('ijji->ij', array)[:] = 1.0
    # This is same as
    # for j in range(size1):
    #     for i in range(size2):
    #         array[i,j,j,i] = 1.0
    return Channel(array, dom1+dom2, dom2+dom1)


def copy(dom):
    dom = asdom(dom)
    if dom.iscont:
        raise ValueError("Cannot make a continuous copy channnel")
    size = _prod(len(s) for s in dom)
    array = np.zeros((size, size, size))
    for n in range(size):
        array[n, n, n] = 1.0
    return Channel(array, dom, dom+dom)


def graph(c):
    return (idn(c.dom) @ c) * copy(c.dom)


def instr(p):
    return (p.as_chan() @ idn(p.dom)) * copy(p.dom)


def asrt(p):
    if p.dom.iscont:
        raise ValueError("Cannot make a continuous assert channel")
    return Channel(np.diagflat(p.array), p.dom, p.dom)


def case_channel(*channels, case_dom=None):
    if not channels:
        raise ValueError("Number of channels must be > 0")
    n = len(channels)
    if case_dom is None:
        case_dom = range(n)
    case_dom = asdom(case_dom)
    dom = channels[0].dom
    cod = channels[0].cod
    for c in channels[1:]:
        if c.dom != dom or c.cod != cod:
            raise ValueError("Domains and codomains must be the same")
    dom_shape = tuple(len(s) for s in dom.disc)
    cod_shape = tuple(len(s) for s in cod.disc)
    array = np.empty(cod_shape + (n,) + dom_shape)
    for i, c in enumerate(channels):
        array[(slice(None),)*len(cod_shape)
              + (i,) + (...,)] = c.array
    return Channel(array, case_dom + dom, cod)


def tuple_channel(chan1, chan2):
    if chan1.dom != chan2.dom:
        raise ValueError("Domains must be the same")
    dom = chan1.dom
    if not chan1.iscont and not chan2.iscont:
        # `outer` for each column
        array = np.einsum('ik,jk->ijk',
                          chan1.array.reshape(chan1.cod_size,
                                              chan1.dom_size),
                          chan2.array.reshape(chan2.cod_size,
                                              chan2.dom_size))
        return Channel(array, dom, chan1.cod + chan2.cod)
    # In other cases, we use `Channel.fromklmap`
    return Channel.fromklmap(lambda *args: chan1(*args) @ chan2(*args),
                             dom, chan1.cod + chan2.cod)


def ifthenelse(pred, chan1, chan2):
    return case_channel(chan1, chan2, case_dom=bool_dom) * instr(pred)


def predicates_from_channel(c):
    if c.cod.iscont:
        return ValueError('Codomain must be discrete')
    if len(c.cod) != 1:
        return ValueError('Codomain must be one-dimensional')
    if c.dom.iscont:
        return [Predicate(Fun2.vect_fun_at2(c.array[i, ...], ()),
                          c.dom)
                for i in range(len(c.cod[0]))]
    return [Predicate(c.array[i, ...], c.dom)
            for i in range(len(c.cod[0]))]

def channel_denotation(c, s):
    return [(s >= p, s/p) for p in predicates_from_channel(c)]


class DetChan:
    """Deterministic channels."""
    def __init__(self, funs, dom):
        self.funs = funs if isinstance(funs, (list, tuple)) else [funs]
        self.dom = asdom(dom)
        self.shape = tuple(len(s) for s in self.dom.disc)

    def _pred_trans_getelm(self, pred, disc_funs, cont_funs, disc_args):
        if not self.dom.iscont:
            return pred.getvalue(*[f(*disc_args) for f in self.funs])
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
        return type(self)(funs, self.dom + other.dom)

    def __matmul__(self, other):
        return self.joint(other)

    def __pow__(self, n):
        if n == 0:
            raise ValueError("Power must be at least 1")
        return reduce(lambda s1, s2: s1 @ s2, [self] * n)


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
randvar_fromfun = RandVar.fromfun
chan_fromklmap = Channel.fromklmap
chan_from_states = Channel.from_states

#
# Functions analogous to sum, prod
#

def joint(iterable):
    return reduce(operator.matmul, iterable)

def convex_sum(iterable):
    return reduce(operator.add, (s.smul(r) for r, s in iterable))

def andthen(iterable):
    return reduce(operator.and_, iterable)

def flip(r, dom=bool_dom):
    return State([r, 1.0-r], [dom])


def const_state_or_pred(cls, value, subdom, dom=None):
    subdom = asdom(subdom)
    if dom is None:
        dom = subdom
    else:
        dom = asdom(dom)
    if len(dom) == 1 and len(subdom) == 0:
        subdom = Dom([[]])
    elif len(dom) != len(subdom):
        raise ValueError("Length of subdom and dom differs")
    for s, d in zip(subdom, dom):
        if isinstance(s, Interval) != isinstance(d, Interval):
            raise ValueError("Type of subdom and dom differs")
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
    size = _prod(len(s) for s in subdom.disc)
    vol = _prod(u - l for l, u in subdom.cont)
    if not math.isfinite(vol):
        raise ValueError("Unbounded domain")
    value = 1.0 / (size * vol)
    return const_state_or_pred(State, value, subdom, dom)


def const_pred(value, subdom, dom=None):
    return const_state_or_pred(Predicate, value, subdom, dom)


def event(subdom, dom):
    return const_pred(1.0, subdom, dom)


def point_state(point, dom):
    dom = asdom(dom)
    if dom.iscont:
        raise ValueError("Cannot create a continuous point state")
    if isinstance(point, tuple):
        return const_state_or_pred(State, 1.0,
                                   [[p] for p in point], dom)
    return const_state_or_pred(State, 1.0, [point], dom)

def point_pred(point, dom):
    dom = asdom(dom)
    if dom.iscont:
        raise ValueError("Cannot create a continuous point predicate")
    if isinstance(point, tuple):
        return event([[p] for p in point], dom)
    return event([point], dom)

yes_pred = point_pred(True, bool_dom)
no_pred = point_pred(False, bool_dom)
or_pred = Predicate([1,1,1,0], [bool_dom,bool_dom])
and_pred = Predicate([1,0,0,0], [bool_dom,bool_dom])

def truth(dom):
    return event(dom, dom)

def falsity(dom):
    dom = asdom(dom)
    subdom = dom.merge(itertools.repeat([]), itertools.repeat(empty))
    return const_pred(0.0, subdom, dom)


def random_state(dom):
    dom = asdom(dom)
    if dom.iscont:
        raise ValueError("Cannot create a continuous random state")
    shape = tuple(len(s) for s in dom.disc)
    array = np.random.random_sample(shape)
    array = array / array.sum()
    return State(array, dom)

def random_pred(dom):
    dom = asdom(dom)
    if dom.iscont:
        raise ValueError("Cannot create a continuous random predicate")
    shape = tuple(len(s) for s in dom.disc)
    array = np.random.random_sample(shape)
    return Predicate(array, dom)


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


or_chan = chan_from_states([flip(1), flip(1), flip(1), flip(0)], 
                           [bool_dom, bool_dom])
and_chan = chan_from_states([flip(1), flip(0), flip(0), flip(0)], 
                            [bool_dom, bool_dom])
ortho_chan = chan_from_states([flip(0), flip(1)], bool_dom)

def id_rv(dom):
    return RandVar(lambda x: x, dom)

def proj1_rv(joint_dom):
    return randvar_fromfun(lambda *x: x[0], joint_dom)

def proj2_rv(joint_dom):
    return randvar_fromfun(lambda *x: x[1], joint_dom)



##############################################################
#
# Bart's additions (temporarily)
#
##############################################################

#
# Total variation distance between discrete states
#
def tvdist(s, t):
    if s.dom != t.dom or s.dom.iscont or t.dom.iscont:
        raise Exception('Distance requires equal, discrete domains')
    return 0.5 * sum(abs(np.ndarray.flatten(s.array) \
                         - np.ndarray.flatten(t.array)))

#
# Direct influence of a predicate on a state
#
def dir_infl(pred, state):
    return tvdist(state, state/pred)

#
# Crossover influence
#
def cross_infl(pred, joint_state):
    marg2 = joint_state % [0,1]
    dom2 = marg2.dom
    return tvdist(marg2, (joint_state / (pred @ truth(dom2))) % [0,1])



#
# Poisson distribution with rate parameter `lam' and upperbound
# ub. The distribution is restricted to the interval [0, ub-1]; hence
# the values have to adjusted so that they sum up to 1 on this
# interval.
#
def poisson(lam, ub):
    probabilities = [(lam ** k) * (math.e ** -lam) / math.factorial(k) 
                     for k in range(ub)]
    s = sum(probabilities)
    return State([p/s for p in probabilities], range(ub))

#
# Binomial distribution on {0,1,2,...,N} with probability p in [0,1]
#
def binomial(N, p):
    Nfac = math.factorial(N)
    def binom_coeff(k):
        return Nfac / (math.factorial(k) * math.factorial(N-k))
    return State([binom_coeff(k) * (p ** k) * ((1-p) ** (N-k)) 
                  for k in range(N+1)],
                 range(N+1))


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
# Function for a predicat on a state, in a Bayesian network
#
def bn_pred(r,s):
    return Predicate([r,s], bnd)

bn_pos_pred = bn_pred(1,0)
bn_neg_pred = bn_pred(0,1)

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


def _shannon_ic(x):
    return -math.log2(x) if x != 0 else 0

_shannon_ic = np.vectorize(_shannon_ic, otypes=[float])

def _shannon_ic_cont(fun):
    def f(x):
        v = fun(x)
        return -math.log2(v) if v != 0 else 0
    return Fun(f, fun.supp)

_shannon_ic_cont = np.vectorize(_shannon_ic_cont, otypes=[object])

def shannon_entropy(s):
    """Shannon entropy"""
    if s.dom.iscont:
        rv = RandVar(_shannon_ic_cont(s.array), s.dom)
    else:
        rv = RandVar(_shannon_ic(s.array), s.dom)
    return s >= rv

def mutual_information(js):
    n = len(js.dom)
    if n < 2:
        raise Exception('Mutual information is defined only for joint states')
    selectors = []
    for i in range(n):
        ls = [0] * n
        ls[i] = 1
        selectors = selectors + [ls]
    marginals = [ js % sel for sel in selectors ]
    return sum(np.vectorize(shannon_entropy)(marginals)) - shannon_entropy(js)


##############################################################
#
# Testing 
#
##############################################################



def test():
    stat = State([gaussian_fun(1, 1, R(-10, 10)).smul(0.8),
                  gaussian_fun(-1, 1, R(-10, 10)).smul(0.2)],
                 [bool_dom, R(-10, 10)])
    pred = truth([bool_dom, R(-10, 10)]) * 0.5
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
    perf2 = DetChan(lambda b: 5 if b else -5, bool_dom)
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
    rv = RandVar.fromfun(lambda x: x, pips)
    print("Exp.", stat.expectation(rv))
    print("Var.", stat.variance(rv))
    print("St.Dev.", stat.st_deviation(rv))
    print("Var. by formula:",
          stat.expectation(rv & rv) - stat.expectation(rv) ** 2)
    # exp = rv.exp(stat)
    # rv2 = rv - exp
    # print("Var. by formula 2:",
    #       (rv2 * rv2).exp(stat))

    rv_sum = RandVar.fromfun(lambda x,y: x+y, [pips]*2)
    stat2 = stat @ stat
    print("Exp.", stat2.expectation(rv_sum))
    print("Var.", stat2.variance(rv_sum))
    print("St.Dev.", stat2.st_deviation(rv_sum))
    print("Var. by formula:",
          stat2.expectation(rv_sum & rv_sum) - stat2.expectation(rv_sum) ** 2)


def disintegration_test():
    X = ['x', 'y']
    A = ['a', 'b']
    s = State([0.1, 0.2, 0.3, 0.4], [X, A])
    d0 = s.disintegration([1, 0]) # X --> A
    d1 = s.disintegration([0, 1]) # A --> X
    print(((idn(X) @ d0) * copy(X)) >> (s % [1, 0]))
    print(((d1 @ idn(A)) * copy(A)) >> (s % [0, 1]))

    prior = flip(0.5)
    c = chan_from_states([gaussian_state(2,1,R(-10,10)),
                          gaussian_state(-2,1,R(-10,10))],
                         bool_dom)
    d = (((idn(bool_dom) @ c) * copy(bool_dom))
         >> prior).disintegration([0, 1])
    print(d.get_state(2))
    print(d.get_state(0))
    print(d.get_state(-1))


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
    #dice()
    #sporter2()
    print( convex_sum([(0.2,flip(0.3)), (0.8, flip(0.8))]) )


if __name__ == "__main__":
    main()
