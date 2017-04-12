#
# See paper Dan Ghica https://arxiv.org/abs/1702.01695
#
from efprob_dc import *

t = random_state(range(2))
q = random_pred(range(2))

# print("\nInstrument conditioning test")
# print( ((instr(q) >> t) / (yes_pred @ truth(range(2)))) % [0,1],
#        " equals ", t / q )
# print( ((instr(q) >> t) / (no_pred @ truth(range(2)))) % [0,1],
#        " equals ", t / ~q )

class CC:
    # CC stands for Channel-with-Condition
    def __init__(self, chan_with_cond):
        self.chan = chan_with_cond

    @classmethod
    def observe(self, pred):
        return CC( instr(pred) )

    @classmethod
    def unit(self, chan_without_cond):
        return CC( point_state(True, bool_dom).as_chan() @ chan_without_cond )

    @classmethod
    def idn(self, dom):
        return CC.unit(idn(dom))

    @classmethod
    def discard(self, dom):
        return CC.unit(discard(dom))

    @classmethod
    def new(self, state):
        return CC.unit(state.as_chan())

    def __repr__(self):
        return repr(self.chan)

    def __mul__(self, other):
        # Compute the Kleisli composition (other ; self).
        # NOTE: this is the reversed order of functional composition *
        # of channels.
        return CC( (and_chan @ idn(other.chan.cod[1:])) \
                   * (idn(bool_dom) @ other.chan) \
                   * self.chan )

    def __matmul__(self, other):
        # parallel composition self @ other
        return CC( (and_chan @ \
                    idn(self.chan.cod[1:]) @ \
                    idn(other.chan.cod[1:])) \
                   * \
                   (idn(bool_dom) @ \
                    swap(self.chan.cod[1:], bool_dom) @ \
                    idn(other.chan.cod[1:])) \
                   * \
                   (self.chan @ other.chan) )

    def smul(self, scalar):
        return CC(self.chan.smul(scalar))

    def __add__(self, other):
        return CC(self.chan + other.chan)

    def run(self, state):
        out = self.chan >> state
        out_dom = out.dom[1:]
        return (discard(bool_dom) @ idn(out_dom)) \
            >> (out / (yes_pred @ truth(out_dom)))


IDN = CC.idn
UNIT = CC.unit
CONVEX_SUM = convex_sum
DISCARD = CC.discard
OBSERVE = CC.observe
NEW = CC.new

def assign_val(point, dom):
    dom = asdom(dom)
    if dom.iscont:
        raise Exception("Cannot assing for continuous domains")
    return chan_fromklmap(lambda x: point_state(point, dom), dom, dom)

def ASSIGN_VAL(point, dom):
    return CC.unit(assign_val(point, dom))


def ASSING_STATE(stat):
    dom = asdom(stat.dom)
    if dom.iscont:
        raise Exception("Cannot assing for continuous domains")
    return CC.unit(chan_fromklmap(lambda x: stat, dom, dom))

bool_to_num = Channel.from_states([State([1,0], range(2)), 
                                   State([0,1], range(2))],
                                  bool_dom)

def IFTHENELSE(pred, cond_chan1, cond_chan2):
    if pred.dom != cond_chan1.chan.dom \
       or pred.dom != cond_chan2.chan.dom \
        or cond_chan1.chan.cod != cond_chan2.chan.cod:
        return Exception('Domain mismatch in if-then-else')
    return CC(case_channel(cond_chan1.chan, cond_chan2.chan) \
              * (bool_to_num @ idn(pred.dom)) \
              * instr(pred))

def REPEAT(n):
    def rep(cc, m):
        if m < 1:
            raise Exception('Repeating must be done at least once')
        if m == 1:
            return cc
        return cc * rep(cc, m-1)
    return lambda cond_chan : rep(cond_chan, n)
        

def REPEATED_OBSERVE(preds, obs_to_pred_fun=None, pre_chan=None, post_chan=None):
    n = len(preds)
    if n < 1:
        raise Exception('Repeated observation must be done at least once')
    pred = preds[0] if obs_to_pred_fun is None else obs_to_pred_fun(preds[0])
    cr = OBSERVE(pred)
    if not pre_chan is None:
        cr = pre_chan * cr
    if not post_chan is None:
        cr = cr * post_chan
    if n == 1:
        return cr
    return cr * REPEATED_OBSERVE(preds[1:], 
                                 obs_to_pred_fun = obs_to_pred_fun,
                                 pre_chan = pre_chan,
                                 post_chan = post_chan)


#
# random channels, states and predicates
#
c = cpt(0.2, 0.5, 0.7, 1)
d = cpt(0.3, 0.9)
s1 = random_state(bnd)
s2 = random_state(bnd)
p1 = random_pred(bnd)
p2 = random_pred(bnd)
print("\nSuccessive observation and transformation test")

print( d >> ((c >> ((s1 / p1) @ s2)) / p2) )

prog = OBSERVE(p1 @ truth(bnd)) * \
       UNIT(c) * \
       OBSERVE(p2) * \
       UNIT(d) 

print( prog.run(s1 @ s2) )

def asrt(p):
    return chan_fromklmap(lambda x: p(x) * point_state(x, p.dom),
                          p.dom,
                          p.dom)

subprog = d * asrt(p2) * c * (asrt(p1) @ idn(bnd))

print( (subprog >> (s1 @ s2)) / truth(bnd) )


def abort(dom):
    return chan_fromklmap(lambda x,y: 0 * uniform_state(dom), dom, dom)

def ABORT(dom):
    return CC(point_state(False, bool_dom).as_chan() @ idn(dom))

dom = [range(2), range(2)]

DIVERGE = CONVEX_SUM([(0.5,
                       ABORT(dom)),
                      (0.5,
                       CONVEX_SUM([(0.5,
                                    ASSIGN_VAL(0, range(2)) @ IDN(range(2))),
                                   (0.5,
                                    ASSIGN_VAL(1, range(2)) @ IDN(range(2)))]) \
                       * \
                       CONVEX_SUM([(0.5,
                                    IDN(range(2)) @ ASSIGN_VAL(0, range(2))),
                                   (0.5,
                                    IDN(range(2)) @ ASSIGN_VAL(1, range(2)))]) \
                       * \
                       OBSERVE( (point_pred(0, range(2)) @ ~point_pred(0, range(2))) \
                                + (~point_pred(0, range(2)) @ point_pred(0, range(2))) )
                      )])

t = random_state(range(2))

print( DIVERGE.run(t @ t) )

pred = truth(range(2)) @ point_pred(0,range(2))

print( DIVERGE.chan << (yes_pred @ pred) )
print( DIVERGE.chan << (yes_pred @ truth(dom)) )
                      


print("\nExamples from JKKOGM\'15")

s0 = random_state(range(2))

prog1 = CONVEX_SUM([(0.4, 
                     ASSIGN_VAL(0, range(2))),
                    (0.6, 
                     ASSIGN_VAL(1, range(2)))]) \
        * \
        OBSERVE(point_pred(1, range(2)))

print( prog1.run(s0) )

prog2 = CONVEX_SUM([(0.4, 
                     ASSIGN_VAL(0, range(2)) \
                     * \
                     OBSERVE(point_pred(1, range(2)))),
                    (0.6, 
                     ASSIGN_VAL(1, range(2)) \
                     * \
                     OBSERVE(point_pred(1, range(2))))])

print( prog2.run(s0) )


print("\nDisease-mood example")

disease_dom = ['D', '~D']
mood_dom = ['M', '~M']
w = State([0.05, 0.5, 0.4, 0.05], [disease_dom, mood_dom])
print(w)


p3 = IFTHENELSE(point_pred('D', disease_dom) @ truth(mood_dom),
                IDN(disease_dom) @ IDN(mood_dom) @ NEW(flip(9/10)),
                IDN(disease_dom) @ IDN(mood_dom) @ NEW(flip(1/20))) \
        * \
        OBSERVE(truth(disease_dom) @ truth(mood_dom) @ yes_pred) \
        * \
        (DISCARD(disease_dom) @ IDN(mood_dom) @ DISCARD(bool_dom))

print( p3.run(w) )


print("\nCoin-bias learning in discrete form")
N = 20
precision = 3
bias_dom = [math.floor((10 ** precision) * (i+1)/(N+1) + 0.5) / (10 ** precision)
            for i in range(N)]
prior = uniform_state(bias_dom)
chan = chan_fromklmap(lambda r: flip(r), bias_dom, bool_dom)

#
# listing all the predicates is possible, but this can be done easier,
# see below
#
p4 = REPEATED_OBSERVE([chan << no_pred, 
                       chan << yes_pred, 
                       chan << yes_pred, 
                       chan << yes_pred, 
                       chan << no_pred, 
                       chan << no_pred, 
                       chan << yes_pred, 
                       chan << yes_pred])

posterior = p4.run(prior)
#posterior.plot()

print( posterior.expectation() )

def pf(b):
    return chan << yes_pred if b == 1 else chan << no_pred

p5 = REPEATED_OBSERVE([0,1,1,1,0,0,1,1], 
                      obs_to_pred_fun = pf,
                      pre_chan = IDN(bias_dom),
                      post_chan = IDN(bias_dom))

print( p5.run(prior).expectation() )


print("\nMarkov chain model")

ACGT = ['A', 'C', 'G', 'T']
s0 = State([0.3, 0.2, 0.1, 0.4], ACGT)
A = Predicate([1,0,0,0], ACGT)
C = Predicate([0,1,0,0], ACGT)
G = Predicate([0,0,1,0], ACGT)
T = Predicate([0,0,0,1], ACGT)

trs = Channel([[0.1, 0.3, 0.3, 0.3],
               [0.3, 0.1, 0.3, 0.3],
               [0.3, 0.3, 0.1, 0.3],
               [0.3, 0.3, 0.3, 0.1]], ACGT, ACGT)
obs = Channel([[0.85, 0.05, 0.05, 0.05],
               [0.05, 0.85, 0.05, 0.05],
               [0.05, 0.05, 0.85, 0.05],
               [0.05, 0.05, 0.05, 0.85]], ACGT, ACGT)

# s1 = trs >> (s0 / (obs << C))
# print( s1 )
# s2 = trs >> (s1 / (obs << A))
# print( s2 )
# s3 = trs >> (s2 / (obs << A))
# print( s3 )
# s4 = trs >> (s3 / (obs << A))
# print( s4  )
# s5 = trs >> (s4 / (obs << G))
# print( s5 )

p6 = REPEATED_OBSERVE([C,A,A,A,G], 
                      obs_to_pred_fun = lambda x: obs << x,
                      post_chan = UNIT(trs))

print( p6.run(s0) )

print("\nFabio")

s = random_state(range(5))
p = random_pred(range(5))
q = random_pred(range(5))

print( s / p / q )

print( REPEATED_OBSERVE([p,q]).run(s) )


