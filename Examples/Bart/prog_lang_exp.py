from efprob_dc import *

#
# Examples from:
#
# https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/fose-icse2014.pdf
#

#
# Domains for booleans and numbers
#
bdom = [True,False]
maxnum = 10
ndom = range(maxnum)

#
# Basic "logical" predicates
#
fst_pred = Predicate([1,0], bdom)
snd_pred = ~fst_pred
or_pred = Predicate([1,1,1,0], [bdom,bdom])
and_pred = Predicate([1,0,0,0], [bdom,bdom])

#
# Basic random variables
#
nrv = randvar_fromfun(lambda i: i, ndom)

#
# initial state
#
init = State([1], [])

#
# Random variable turning Booleans into numbers
#
bool_as_num = RandVar([1,0], bdom)

#
# Example 1a
#
print("\n* Example 1(a)")

c1 = flip(0.5).as_chan()
c2 = flip(0.5).as_chan()
ex1a = c1 @ c2 >> init

print("Output distribution: ", ex1a )
print("Output expectations: ", 
      (bool_as_num @ truth(bdom)).exp(ex1a),
      (truth(bdom) @ bool_as_num).exp(ex1a) )


#
# Example 1b
#
print("\n* Example 1(b)")

#
# This function is not what we want, since it applies conditioning
# internally. We need to apply it externally. It works in the examples
# below since the channel c has empty domain [] in each case.
#
def observe(p, c):
    return chan_fromklmap(lambda *args: c.get_state(*args) / p,
                          c.dom,
                          c.cod)

ex1b = observe(or_pred, c1 @ c2) >> init

print("Distirbution: ", ex1b )
print("Expectations: ", 
      (bool_as_num @ truth([True, False])).exp(ex1b),
      (truth([True, False]) @ bool_as_num).exp(ex1b) )


#
# Example 2
#
print("\n* Example 2")

def instr(p):
    return (p.as_chan() @ idn(p.dom)) * copy(p.dom)

bton = Channel.from_states([State([1,0], range(2)), 
                            State([0,1], range(2))],
                           bdom)

def ifthenelse(pred, chan1, chan2):
    if pred.dom != chan1.dom or pred.dom != chan2.dom or chan1.cod != chan2.cod:
        return Exception('Domain mismatch in if-then-else')
    return case_channel(chan1, chan2) * (bton @ idn(pred.dom)) * instr(pred)

inc_chan = chan_fromklmap(lambda i: point_state(i+1, ndom) if i+1 < maxnum
                          else point_state(i, ndom), ndom, ndom)

ex2 = (observe(truth(ndom) @ or_pred,
               ifthenelse(truth(ndom) @ truth(bdom) @ fst_pred,
                          inc_chan @ idn(bdom) @ idn(bdom),
                          idn(ndom) @ idn(bdom) @ idn(bdom)) \
               * (ifthenelse(truth(ndom) @ fst_pred, 
                             inc_chan @ idn(bdom),
                             idn(ndom) @ idn(bdom)) @ flip(0.5).as_chan()) \
               * (point_state(0, ndom).as_chan() @ flip(0.5).as_chan())) \
       >> init) % [1, 0, 0]


print("Distribution: ", ex2 )
print("Expectation: ", nrv.exp(ex2) )


print("\n* Example 4")

def iterate_while(pred, chan, upper):
    if upper <= 0:
        return idn(pred.dom)
    return ifthenelse(pred, 
                      iterate_while(pred, chan, upper-1) * chan,
                      idn(pred.dom))

ortho_chan = Channel.from_states([flip(0), flip(1)], bdom)

ex4 = iterate_while(truth(bdom) @ fst_pred,
                    ortho_chan @ idn(bdom),
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

