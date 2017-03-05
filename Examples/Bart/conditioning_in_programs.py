from efprob_dc import *

t = random_state(range(2))
q = random_pred(range(2))

print("\nInstrument conditioning test")
print( ((instr(q) >> t) / (yes_pred @ truth(range(2)))) % [0,1],
       " equals ", t / q )
print( ((instr(q) >> t) / (no_pred @ truth(range(2)))) % [0,1],
       " equals ", t / ~q )

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

    def enforce(self, state):
        out = self.chan >> state
        out_dom = out.dom[1:]
        return (discard(bool_dom) @ idn(out_dom)) \
            >> (out / (yes_pred @ truth(out_dom)))


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
       * CC.unit(c) * \
       CC.observe(p1 @ truth(bnd))
print( prog.enforce(s1 @ s2) )

