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


def newsample(klmap, dom, pre):
    post = pre + [dom]
    chan = ep.chan_fromklmap(lambda *args:
                             ep.point_state(args, pre) @ klmap(*args),
                             pre,
                             post)
    return Program(chan, pre, post)


def newassign(fun, dom, pre):
    return newsample(lambda *args: ep.point_state(fun(*args), [dom]),
                     dom, pre)


def sample(klmap, index, pre):
    post = pre
    chan = ep.chan_fromklmap(lambda *args:
                             ep.point_state(args, pre[:index])
                             @ klmap(*args)
                             @ ep.point_state(args, pre[index+1:]),
                             pre,
                             post)
    return Program(chan, pre, post)


def assign(fun, index, pre):
    return sample(lambda *args: ep.point_state(fun(*args), [dom]),
                  index, pre)


def discard(index, pre):
    post = pre[:index] + pre[index+1:]
    chan = ep.idn(pre[:index]) @ ep.discard(pre[index]) @ ep.idn(pre[index+1:])
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


# Syntax sugar
def prob_choice(r, prog1, prog2):
    return ifthenelse(lambda *_: r, prog1, prog2)


# Examples

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


disease_dom = ['D', '~D']
mood_dom = ['M', '~M']
bool_dom = [True, False]
inital_state = ep.State([0.05, 0.5, 0.4, 0.05], [disease_dom, mood_dom])

test3 = do(
    ifthenelse(lambda d,m: d == 'D',
               newsample(lambda *_: ep.flip(9/10), bool_dom, [disease_dom, mood_dom]),
               newsample(lambda *_: ep.flip(1/20), bool_dom, [disease_dom, mood_dom])),
    observe(lambda d,m,t: t == True, [disease_dom, mood_dom, bool_dom]),
    discard(0, [disease_dom, mood_dom, bool_dom]),
    discard(1, [mood_dom, bool_dom])
).run(inital_state)


# Use prob_choice and assign instead

# Syntax sugar
def newflip(r, pre):
    return prob_choice(r,
                       newassign(lambda *_: True,
                                 bool_dom,
                                 pre),
                       newassign(lambda *_: False,
                                 bool_dom,
                                 pre))

test4 = do(
    ifthenelse(lambda d,m: d == 'D',
               newflip(9/10, [disease_dom, mood_dom]),
               newflip(1/20, [disease_dom, mood_dom])),
    observe(lambda d,m,t: t == True, [disease_dom, mood_dom, bool_dom]),
    discard(0, [disease_dom, mood_dom, bool_dom]),
    discard(1, [mood_dom, bool_dom])
).run(inital_state)


# Fish example

fish_dom = [20 * (i+1) for i in range(15)]

test5 = do(
    newsample(lambda *_: ep.uniform_state(fish_dom), fish_dom, []),
    newsample(lambda fish_num: ep.binomial(20, 20 / fish_num),
              range(21), [fish_dom]),
    observe(lambda fish_num, marked: marked == 5,
            [fish_dom, range(21)]),
    discard(1, [fish_dom, range(21)])
).run()
