from efprob_dc import *

#
# Examples from:
#
# https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/fose-icse2014.pdf

#
# initial state
#
init = State([1], [])

#
# Random variable turning Booleans into numbers
#
bool_as_num = RandVar([1,0], [True,False])

#
# Example 1
#
print("\nExample 1")

c1 = flip(0.5).as_chan()
c2 = flip(0.5).as_chan()
ex1 = c1 @ c2 >> init

print( ex1 )
#
# Kenta: why multiply the expected outcomes in the @ case? I would
# expect a tuple as return.
#
print( (bool_as_num @ bool_as_num).exp(ex1) )



def instr(p):
    return (p.as_chan() @ idn(p.dom)) * copy(p.dom)

or_pred = Predicate([1,1,1,0], [[True,False],[True,False]])

#
# Example 2
#
print("\nExample 2")

ex2 = (instr(or_pred) * (c1 @ c2)) >> init

print( ex2 )

print( (ex2 / (Predicate([1,0], [True,False]) @ truth([True,False]) @ truth([True,False]))) % [0,1,1] )

#
# Kenta: in this example the right state is extracted by conditioning
# with Predicate([1,0], [True,False]) @ ... and then taking the
# marginal. (The same happens in the quantum case in the last
# illustration of Section 3.4 in the Manual.) 
#
# Question: can we also do this conditioning "inside" a channel, like
# in observe below?
#

print("\nExample 2 alternative")

def observe(p, c):
    return chan_fromklmap(lambda *args: c.get_state(*args) / p,
                          c.dom,
                          c.cod)

c = Channel.from_states([flip(0.2), flip(0.5), flip(1)])

print( observe(Predicate([0.3, 0.8], [True,False]), c) 
       >> uniform_state(range(3)) )

print( observe( Predicate([1,0], [True,False]) @ 
                truth([True,False]) @ 
                truth([True,False]), 
                instr(or_pred) * (c1 @ c2) ) 
       >> init )
