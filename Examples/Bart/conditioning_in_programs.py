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

    def __repr__(self):
        return repr(self.chan)

    def __mul__(self, other):
        # Compute the Kleisli composition (self after other).
        return CC( (and_chan @ idn(self.chan.cod[1:])) \
                   * (idn(bool_dom) @ self.chan) \
                   * other.chan )

    def __add__(self, other):
        return CC(self.chan + other.chan)

    def smul(self, scalar):
        return CC(self.chan.smul(scalar))

    def enforce(self, state):
        out = self.chan >> state
        out_dom = out.dom[1:]
        return (discard(bool_dom) @ idn(out_dom)) \
            >> (out / (yes_pred @ truth(out_dom)))


def assign_val(point, dom):
    dom = asdom(dom)
    if dom.iscont:
        raise Exception("Cannot assing for continuous domains")
    return chan_fromklmap(lambda x: point_state(point, dom), dom, dom)

def assign_state(point, stat):
    dom = asdom(stat.dom)
    if dom.iscont:
        raise Exception("Cannot assing for continuous domains")
    return chan_fromklmap(lambda x: stat, dom, dom)

bool_to_num = Channel.from_states([State([1,0], range(2)), 
                                   State([0,1], range(2))],
                                  bool_dom)

def ifthenelse(pred, cond_chan1, cond_chan2):
    if pred.dom != cond_chan1.chan.dom \
       or pred.dom != cond_chan2.chan.dom \
        or cond_chan1.chan.cod != cond_chan2.chan.cod:
        return Exception('Domain mismatch in if-then-else')
    return CC(case_channel(cond_chan1.chan, cond_chan2.chan) \
              * (bool_to_num @ idn(pred.dom)) \
              * instr(pred))

# print( (instr(Predicate([0,1], range(2))) \
#                >> uniform_state(range(2))) )

# print( case_channel(assign_val(0, range(2)), assign_val(1, range(2))) \
#        >> ((bool_to_num @ idn(range(2))) \
#            >> (instr(Predicate([0,1], range(2))) \
#                >> uniform_state(range(2)))) )

# print( ifthenelse(Predicate([0,0], range(2)),
#                   CC.unit(assign_val(0, range(2))),
#                   CC.unit(assign_val(1, range(2)))).chan >> uniform_state(range(2)) )


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

prog = CC.unit(d) \
       * CC.observe(p2) \
       * CC.unit(c) \
       * CC.observe(p1 @ truth(bnd))

print( prog.enforce(s1 @ s2) )


print("\nExamples from JKKOGM\'15")

s0 = random_state(range(2))

prog1 = CC.observe(point_pred(1, range(2))) \
        * convex_sum([(0.4, 
                       CC.unit(assign_val(0, range(2)))),
                      (0.6, 
                       CC.unit(assign_val(1, range(2))))])

print( prog1.enforce(s0) )

prog2 = convex_sum([(0.4, 
                     CC.observe(point_pred(1, range(2))) \
                     * CC.unit(assign_val(0, range(2)))),
                    (0.6, 
                     CC.observe(point_pred(1, range(2))) \
                     * CC.unit(assign_val(1, range(2))))])

print( prog2.enforce(s0) )


print("\nDisease-mood example")

disease_dom = ['D', '~D']
mood_dom = ['M', '~M']
w = State([0.05, 0.5, 0.4, 0.05], [disease_dom, mood_dom])

prog3 = CC.unit(discard(disease_dom) @ idn(mood_dom) @ discard(bool_dom)) \
        * CC.observe(truth(disease_dom) @ truth(mood_dom) @ yes_pred) \
        * ifthenelse(point_pred('D', disease_dom) @ truth(mood_dom),
                     CC.unit(idn(disease_dom) @ idn(mood_dom) @ flip(9/10).as_chan()),
                     CC.unit(idn(disease_dom) @ idn(mood_dom) @ flip(1/20).as_chan()))

print( prog3.enforce(w) )
