from functools import reduce
import efprob_dc as ep
from efprob_dc import (bool_dom,
                       truth, falsity,
                       State, Predicate, Channel,
                       chan_fromklmap,
                       state_fromfun, pred_fromfun,
                       point_state, point_pred,
                       uniform_state,
                       random_state, random_pred,
                       flip)

class ChanCond:
    """Channel with Condition"""
    def __init__(self, chan, dom, cod):
        self.chan = chan
        self.dom = ep.asdom(dom)
        self.cod = ep.asdom(cod)

    def __repr__(self):
        return "Channel with condition of type: {} --> {}".format(self.dom, self.cod)

    def __mul__(self, other):
        """Sequential composition g o f"""
        chan = ((ep.and_chan @ ep.idn(self.cod))
                * (ep.idn(bool_dom) @ self.chan)
                * other.chan)
        return ChanCond(chan, other.dom, self.cod)

    def __matmul__(self, other):
        """Parallel composition f (x) g"""
        chan = ((ep.and_chan @ ep.idn(self.cod) @ ep.idn(other.cod))
                * (ep.idn(bool_dom) @ ep.swap(self.cod, bool_dom)
                   @ ep.idn(other.cod))
                * (self.chan @ other.chan))
        return ChanCond(chan, self.dom + other.dom, self.cod + other.cod)

    def run(self, state=ep.State(1, [])):
        s = self.chan >> state
        s = s.totalmass() * (s / (ep.yes_pred @ truth(self.cod)))
        return s % ([0] + [1] * len(self.cod))

    def run2(self, state=ep.State(1, [])):
        s = self.chan >> state
        sp = ep.asrt(ep.yes_pred @ truth(self.cod)) >> s
        sp = (1 /(sp.totalmass() + 1 - s.totalmass())) * sp
        return sp % ([0] + [1] * len(self.cod))


def embed(chan):
    chancond = ep.point_state(True, bool_dom).as_chan() @ chan
    return ChanCond(chancond, chan.dom, chan.cod)

def idn(dom):
    return embed(ep.idn(dom))

def discard(dom):
    return embed(ep.discard(dom))

def copy(dom):
    return embed(ep.copy(dom))

def observe(pred):
    return ChanCond(ep.instr(pred), pred.dom, pred.dom)

def new(state):
    return embed(state.as_chan())

def ifthenelse(pred, cc1, cc2):
    dom = cc1.dom
    cod = cc1.cod
    chan = ep.ifthenelse(pred, cc1.chan, cc2.chan)
    return ChanCond(chan, dom, cod)

def seq(*ccs):
    return reduce(lambda f, g: g * f, ccs)

def newsample(chan):
    return embed(ep.tuple_channel(ep.idn(chan.dom), chan))

def prob_choice(r, cc1, cc2):
    dom = cc1.dom
    return ifthenelse(r * truth(dom), cc1, cc2)

def abort(dom, cod):
    return embed(ep.abort(dom, cod))



def examples():
    bnd = ep.bnd
    c = ep.cpt(0.2, 0.5, 0.7, 1)
    d = ep.cpt(0.3, 0.9)
    s1 = random_state(bnd)
    s2 = random_state(bnd)
    p1 = random_pred(bnd)
    p2 = random_pred(bnd)

    print( d >> ((c >> ((s1 / p1) @ s2)) / p2) )

    print(
        seq(observe(p1 @ truth(bnd)),
            embed(c),
            observe(p2),
            embed(d)
        ).run(s1 @ s2)
    )

    print(
        seq(prob_choice(0.4,
                        new(point_state(0, range(2))),
                        new(point_state(1, range(2)))),
            observe(point_pred(0, range(2)))
        ).run()
    )

    print(
        seq(prob_choice(0.4,
                        seq(new(point_state(0, range(2))),
                            observe(point_pred(0, range(2)))),
                        seq(new(point_state(1, range(2))),
                            observe(point_pred(0, range(2))))),
        ).run()
    )

    disease_dom = ['D', '~D']
    mood_dom = ['M', '~M']
    inital_state = State([0.05, 0.5, 0.4, 0.05], [disease_dom, mood_dom])

    print(
        seq(
            ifthenelse(point_pred('D', disease_dom) @ truth(mood_dom),
                       idn([disease_dom, mood_dom]) @ new(flip(9/10)),
                       idn([disease_dom, mood_dom]) @ new(flip(1/20))),
            idn([disease_dom, mood_dom]) @ observe(point_pred(True, bool_dom)),
            discard(disease_dom) @ idn(mood_dom) @ discard(bool_dom)
        ).run(inital_state)
    )

    # Use prob_choice and point_state instead

    # Syntax sugar
    def newflip(r):
        return prob_choice(r,
                           new(point_state(True, bool_dom)),
                           new(point_state(False, bool_dom)))

    print(
        seq(
            ifthenelse(point_pred('D', disease_dom) @ truth(mood_dom),
                       idn([disease_dom, mood_dom]) @ newflip(9/10),
                       idn([disease_dom, mood_dom]) @ newflip(1/20)),
            idn([disease_dom, mood_dom]) @ observe(point_pred(True, bool_dom)),
            discard(disease_dom) @ idn(mood_dom) @ discard(bool_dom)
        ).run(inital_state)
    )

    # Fish example

    fish_dom = [20 * (i+1) for i in range(15)]

    seq(
        new(uniform_state(fish_dom)),
        newsample(chan_fromklmap(lambda fish_num:
                                 ep.binomial(20, 20 / fish_num),
                                 fish_dom, range(21))),
        idn(fish_dom) @ observe(point_pred(5, range(21))),
        idn(fish_dom) @ discard(range(21))
    ).run().plot()


# Example from Katoen's paper / slide

cc = prob_choice(0.5,
                 abort([], [0,1]),
                 seq(prob_choice(0.5,
                                 new(point_state(0, [0,1])),
                                 new(point_state(1, [0,1]))),
                     prob_choice(0.5,
                                 idn([0,1]) @ new(point_state(0, [0,1])),
                                 idn([0,1]) @ new(point_state(1, [0,1]))),
                     observe(pred_fromfun(lambda x, y: x == 0 or y == 0,
                                          [[0,1], [0,1]])),
                     discard([0,1]) @ idn([0,1])))

# >>> cc.chan()
# 0.25|True,0> + 0.125|True,1> + 0|False,0> + 0.125|False,1>

print(cc.run()) # gives 0.333|0> + 0.167|1>
print(cc.run2()) # gives 0.286|0> + 0.143|1>
