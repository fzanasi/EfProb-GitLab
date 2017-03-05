from efprob_dc import *

#
# Examples from:
#
# https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/fose-icse2014.pdf
#

#
# Domains for booleans and numbers
#
maxnum = 10
ndom = range(maxnum)


#
# Basic random variables; same as id_rv(ndom)
#
nrv = randvar_fromfun(lambda i: i, ndom)

#
# initial state
#
init = State([1], [])

#
# Random variable turning Booleans into numbers
#
bool_as_num = RandVar([1,0], bool_dom)

#
# Example 1a
#
print("\n* Example 1(a)")

c1 = flip(0.5).as_chan()
c2 = flip(0.5).as_chan()
ex1a = c1 @ c2 >> init

print("Output distribution: ", ex1a )
print("Output expectations: ", 
      ex1a.expectation(bool_as_num @ truth(bool_dom)),
      ex1a.expectation(truth(bool_dom) @ bool_as_num) )


#
# Example 1b
#
print("\n* Example 1(b)")

#
# This function is not what we want, since it applies conditioning
# internally. We need to apply it externally. It works in the examples
# below since the channel c has empty domain [] in each case.
#
# def observe(p, c):
#     return chan_fromklmap(lambda *args: c.get_state(*args) / p,
#                           c.dom,
#                           c.cod)

ex1b = ((c1 @ c2) >> init) / or_pred

print("Distribution: ", ex1b )
print("Expectations: ", 
      ex1b.expectation(bool_as_num @ truth(bool_dom)),
      ex1b.expectation(truth(bool_dom) @ bool_as_num) )

print("\nAlternative construction")
ex1b_alt = (instr(or_pred) * (c1 @ c2)) >> init
#print( ((ex1b_alt >> init) / (yes_pred @ truth([bool_dom, bool_dom]))) % [0,1,1] )
print( enforce(ex1b_alt) )


"""

#
# Redefinition of a channel; result is equal to the original
#
def rechan(c):
    return chan_fromklmap(lambda *args: c.get_state(*args), c.dom, c.cod)


#
# Example 2
#
print("\n* Example 2")

bton = Channel.from_states([State([1,0], range(2)), 
                            State([0,1], range(2))],
                           bool_dom)


def ifthenelse(pred, chan1, chan2):
    if pred.dom != chan1.dom or pred.dom != chan2.dom or chan1.cod != chan2.cod:
        return Exception('Domain mismatch in if-then-else')
    return case_channel(chan1, chan2) * (bton @ idn(pred.dom)) * instr(pred)

inc_chan = chan_fromklmap(lambda i: point_state(i+1, ndom) if i+1 < maxnum
                          else point_state(i, ndom), ndom, ndom)

ex2 = (observe(truth(ndom) @ or_pred,
               ifthenelse(truth(ndom) @ truth(bool_dom) @ yes_pred,
                          inc_chan @ idn(bool_dom) @ idn(bool_dom),
                          idn(ndom) @ idn(bool_dom) @ idn(bool_dom)) \
               * (ifthenelse(truth(ndom) @ yes_pred, 
                             inc_chan @ idn(bool_dom),
                             idn(ndom) @ idn(bool_dom)) @ flip(0.5).as_chan()) \
               * (point_state(0, ndom).as_chan() @ flip(0.5).as_chan())) \
       >> init) % [1, 0, 0]


print("Distribution: ", ex2 )
print("Expectation: ", ex2.expectation(nrv) )

print("\n* Example 4")

def iterate_while(pred, chan, upper):
    if upper <= 0:
        return idn(pred.dom)
    return ifthenelse(pred, 
                      iterate_while(pred, chan, upper-1) * chan,
                      idn(pred.dom))

ex4 = iterate_while(truth(bool_dom) @ yes_pred,
                    ortho_chan @ idn(bool_dom),
                    3)
#
# It is unclear what is happening here...
#
print( (ex4 >> flip(1) @ flip(0.5)) % [1,0] )



#
# Example 7
#
print("\n* Example 7: Discrete Time Markov Chain")

#
# Mimic a dice by using a fair coin multiple times. The dice
# probabilities appear in the last six states, after several
# iterations (at least 12, it seems)
#

dtmc_states = range(13)

dtmc_chan = Channel.from_states([
    State([0,0.5,0.5,0, 0,0,0,0, 0,0,0,0,0], dtmc_states),
    State([0,0,0,0.5, 0.5,0,0,0, 0,0,0,0,0], dtmc_states),
    State([0,0,0,0, 0,0.5,0.5,0, 0,0,0,0,0], dtmc_states),
    State([0,0.5,0,0, 0,0,0,0.5, 0,0,0,0,0], dtmc_states),
    State([0,0,0,0, 0,0,0,0, 0.5,0.5,0,0,0], dtmc_states),
    State([0,0,0,0, 0,0,0,0, 0,0,0.5,0.5,0], dtmc_states),
    State([0,0,0.5,0, 0,0,0,0, 0,0,0,0,0.5], dtmc_states),
    State([0,0,0,0, 0,0,0,1, 0,0,0,0,0], dtmc_states),
    State([0,0,0,0, 0,0,0,0, 1,0,0,0,0], dtmc_states),
    State([0,0,0,0, 0,0,0,0, 0,1,0,0,0], dtmc_states),
    State([0,0,0,0, 0,0,0,0, 0,0,1,0,0], dtmc_states),
    State([0,0,0,0, 0,0,0,0, 0,0,0,1,0], dtmc_states),
    State([0,0,0,0, 0,0,0,0, 0,0,0,0,1], dtmc_states)
    ])

N=20
s = point_state(0, dtmc_states)
for i in range(N):
    #s.plot()
    s = dtmc_chan >> s
print(s)


# http://users.ices.utexas.edu/~njansen/files/publications/katoen-et-al-olderog-2015.pdf


"""
