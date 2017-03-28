from functools import reduce
import efprob_dc as ep


class Program:
    def __init__(self, chan, pre, post):
        self.chan = chan
        self.pre = pre
        self.post = post

    def run(self, state=ep.State(1, [])):
        return (self.chan >> state).normalize()


def skip(pre):
    post = pre
    chan = ep.idn(pre)
    return Program(chan, pre, post)


def newassign(fun, dom, pre):
    post = pre + [dom]
    chan = ep.chan_fromklmap(lambda *args:
                             ep.point_state(args + (fun(*args),), post),
                             pre,
                             post)
    return Program(chan, pre, post)


def discard(index, pre):
    post = pre[:index] + pre[index+1:]
    chan = ep.idn(pre[:index]) @ ep.discard(pre[index]) @ ep.idn(pre[index+1:])
    return Program(chan, pre, post)


def assign(fun, index, pre):
    post = pre
    chan = ep.chan_fromklmap(lambda *args:
                             ep.point_state(args[:index]
                                            + (fun(*args),)
                                            + args[index+1:], post),
                             pre,
                             post)
    return Program(chan, pre, post)


def seq(prog1, prog2):
    return Program(prog2.chan * prog1.chan,
                   prog1.pre,
                   prog2.post)


def do(*programs):
    return reduce(seq, programs)


def ifthenelse(fun, prog1, prog2):
    pre = prog1.pre
    post = prog1.pre
    pred = ep.pred_fromfun(fun, pre)
    chan = (ep.case_channel(prog1.chan, prog2.chan, case_dom=ep.bool_dom)
            * ep.instr(pred))
    return Program(chan, pre, post)


def observe(fun, pre):
    post = pre
    pred = ep.pred_fromfun(fun, pre)
    chan = ep.asrt(pred)
    return Program(chan, pre, post)


def prob_choice(r, prog1, prog2):
    return ifthenelse(lambda *_: r, prog1, prog2)


test1 = do(
    prob_choice(0.4,
                newassign(lambda: 0, range(2), []),
                newassign(lambda: 1, range(2), [])),
    observe(lambda x: x == 0, [range(2)])
).run()


test2 = do(
    prob_choice(0.4,
                do(newassign(lambda: 0, range(2), []),
                   observe(lambda x: x == 0, [range(2)])),
                do(newassign(lambda: 1, range(2), []),
                   observe(lambda x: x == 0, [range(2)])))
).run()


def newflip(r, pre):
    return prob_choice(r,
                       newassign(lambda *_: True,
                                 [True, False],
                                 pre),
                       newassign(lambda *_: False,
                                 [True, False],
                                 pre))

disease_dom = ['D', '~D']
mood_dom = ['M', '~M']

test3 = do(
    ifthenelse(lambda d,m: d == 'D',
               newflip(9/10, [disease_dom, mood_dom]),
               newflip(1/20, [disease_dom, mood_dom])),
    observe(lambda d,m,t: t == True, [disease_dom, mood_dom, [True, False]]),
    discard(0, [disease_dom, mood_dom, [True, False]]),
    discard(1, [mood_dom, [True, False]])
).run(ep.State([0.05, 0.5, 0.4, 0.05], [disease_dom, mood_dom]))
