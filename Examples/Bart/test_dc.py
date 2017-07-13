from efprob_dc import *

#
# Experiments of turning predicates into states.
#

D = [range(3), bool_dom]

p = random_pred(D)

print("\nA random predicate, turned into a state via normalisation")
print( p )
print( uniform_state(D) / p )

print("\nThe normalisation factor arises from the validity in the uniform state")
r = (uniform_state(D) >= p) * reduce(operator.mul, [len(s) for s in p.dom], 1)
print( r )
print( 1/r * p )

s = random_state(D)

print("\nEach state is a conditioning of the uniform state")
print( s )
print( uniform_state(D) / s.as_pred() )

#
# A cup channel [] -> dom * dom
#

def cup_chan(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    if dom.iscont:
        raise Exception('cap only exists for discrete domains')
    dom_size = reduce(operator.mul, [len(s) for s in dom], 1)
    ls = [1]
    for i in range(dom_size-1):
        ls = ls + dom_size*[0] + [1]
    return Channel(1/dom_size * np.array(ls), [], dom + dom)

def cup_state(dom):
    return cup_chan(dom) >> init_state

print("\nA cup channel example")

print( cup_state(D) )
print("The cup channel preserves truth: ", cup_chan(D) << truth(D + D) )

print("\nA second `crossover' way to turn a predicate into a state, with same outcome as before")
print( cup_state(D) / (truth(D) @ p) % [1,1,0,0] )

print("\nAlso a state can be obtained from such `crossover'")
print( cup_state(D) / (truth(D) @ s.as_pred()) % [1,1,0,0] )

print("\nDiscard/truth after cup is uniform state:")
print( (idn(D) @ discard(D)) >> cup_state(D) )
print("Hence validity of p @ truth in cup state is validity of p in uniform state")
print( uniform_state(D) >= p )
print( cup_state(D) >= (p @ truth(D)) )

print("\nPredicate p @ idn after cup is state 1/n * p")
print( (p.as_chan() @ idn(D)) >> cup_state(D) )

print("\nValidity of p1 @ p2 in cup state is inner product of probabilities")
p1 = random_pred(D)
p2 = random_pred(D)
print(p1)
print(p2)
print( cup_state(D) >= (p1 @ p2) )

print("\nSemi-cartesian closed structure")

# 
# chan : A x B -> C
#
def Lambda(chan):
    A = chan.dom[0]
    B = chan.dom[1]
    C = chan.cod
    return lambda a: ((idn(B) @ chan_fromklmap(lambda b: chan(a,b), B, C)) * copy(B)) >> uniform_state(B)

def Ev(stat, b):
    return (stat // [1,0])(b)

A = bool_dom
B = range(2)
C = range(7)

ch = chan_from_states([random_state(C), random_state(C), 
                       random_state(C), random_state(C)], [A,B])

a = True
b = 1

print( ch(a, b) )
print( Ev(Lambda(ch)(a), b) )
