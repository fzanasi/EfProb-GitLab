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

def cup(dom):
    return reduce(operator.matmul, [len(s) for s in dom.disc]) * (cup_chan(dom) >> init_state)

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
C = range(3)

ch = chan_from_states([random_state(C), random_state(C), 
                       random_state(C), random_state(C)], [A,B])

a = True
b = 1

print( ch(a, b) )
print( Ev(Lambda(ch)(a), b) )


print("\nParallel channels from states")

c1 = chan_from_states([random_state(range(2)),random_state(range(2))], range(2))
c2 = chan_from_states([random_state(range(2)),random_state(range(2))], range(2))

w = (c1 @ c2) >> cup_state(range(2))

print( w )

# a00 = 0.5 * ( c1(0)(0) * c2(0)(0) + c1(1)(0) * c2(1)(0) ) 
# print( a00 )
# a01 = 0.5 * ( c1(0)(0) * c2(0)(1) + c1(1)(0) * c2(1)(1) )
# print( a01,
#        0.5 * ( c1(0)(0) + c1(1)(0) ) - a00 )
# a10 = 0.5 * ( c1(0)(1) * c2(0)(0) + c1(1)(1) * c2(1)(0) )
# print( a10,
#        0.5 * ( c2(0)(0) + c2(1)(0) ) - a00 )
# a11 = 0.5 * ( c1(0)(1) * c2(0)(1) + c1(1)(1) * c2(1)(1) )
# print( a11, 1 - a01 - a10 + a00 )

q = random_pred(range(2))

# Nice "Bayesian" relation
print( (w / (truth(range(2)) @ q)) % [1,0] )
print( c1 >> (uniform_state(range(2)) / (c2 << q)) )



print("\nFrom state to channels")

w = random_state([range(2), range(2)])

print( w )

r1 = random.uniform(0, min(2 * (w(0,1) + w(0,0)),
                           (2 * (w(1,0) + w(0,0))*(w(0,1) + w(0,0)) - w(0,0)) / (w(1,0) + w(0,0)) ) )
r0 = 2 * (w(0,1) + w(0,0)) - r1

print( r0, r1 )

s0 = (w(0,0) - r1*(w(1,0) + w(0,0))) / (w(0,1) + w(0,0) - r1)
s1 = 2 * (w(1,0) + w(0,0)) - s0

print( s0, s1 )

c = chan_from_states([flip(r0, range(2)),flip(r1, range(2))], range(2))
d = chan_from_states([flip(s0, range(2)),flip(s1, range(2))], range(2))

print( (c @ d) >> cup_state(range(2)) )

print("\nNon-entwined case")

w1 = random_state(range(2))
w2 = random_state(range(2))

w = w1 @ w2

print( w )

c1 = chan_from_states([w1,w1], range(2))
c2 = chan_from_states([w2,w2], range(2))

print( (c1 @ c2) >> cup_state(range(2)) )


print("\nVia disintegration and assert")

v = random_state([A,C])

c = v // [1,0]
a = asrt((v % [1,0]).as_pred())

d = v // [0,1]
b = asrt((v % [0,1]).as_pred())

#print( v % [1,0] )
#print( a << truth(A) )

print("\nstate reconstruction")
print( v )
v1 = ((a @ c) >> cup(Dom(A)))
v2 = ((d @ b) >> cup(Dom(C)))
print ( v1 )
print ( v2 )

p = random_pred(A)
q = random_pred(C)

print("\nvalidities")
print( v >= p @ truth(C), 2 * (uniform_state(A) >= (a << p)) )
print( v >= truth(A) @ q, uniform_state(A) >= (c << q) )


print("")
print( (v1 / (truth(A) @ q)) % [1,0] )
print( a >> (uniform_state(A) / (c << q)) )
print( (v1 / (p @ truth(C))) % [0,1] )
print( c >> (uniform_state(A) / (a << p)) )

# print("\n")
# print( b << truth(C) )
# print( (v2 / (p @ truth(C))) % [0,1] )
# print( b >> (uniform_state(C) / (d << p)) )


w = random_state([range(2), range(3)])
c = w // [1,0]
s = random_state([range(2)])
print( graph(c) >> s )
print( (idn([range(2)]) @ c) >> (s @ s) )
