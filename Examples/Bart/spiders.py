#
# 
#
from efprob_qu import *

def ket_zero_vector(n):
    zeros = np.zeros((2 ** n, 1))
    zeros[0,0] = 1
    return zeros

def ket_zero_matrix(n, m):
    return np.dot(ket_zero_vector(n), np.transpose(ket_zero_vector(m)))

def ket_one_vector(n):
    zeros = np.zeros((2 ** n, 1))
    zeros[(2 ** n)-1, 0] = 1
    return zeros

def ket_one_matrix(n, m):
    return np.dot(ket_one_vector(n), np.transpose(ket_one_vector(m)))

def white_spider_matrix(n,m):
    return ket_zero_matrix(n,m) + ket_one_matrix(n,m)

print("\nSpider compositions")

#print( white_spider_matrix(1,2) )

print( np.dot(np.kron(np.eye(2), white_spider_matrix(1,2)),
              white_spider_matrix(3,1)) )
              
print( np.dot(white_spider_matrix(1,3),
              np.kron(np.eye(2), white_spider_matrix(2,1))) )

print( np.dot(white_spider_matrix(2,10),
              white_spider_matrix(10,3)) )

print("Multiplication spider 2 -> 1")
print( white_spider_matrix(1,2) )

print("\nIsometry, only for (white) spiders 1 -> n")

print( is_isometry(white_spider_matrix(8,1)) )
print( is_isometry(white_spider_matrix(3,2)) )

def plus_vector(n):
    v = np.eye(1)
    for i in range(n):
        v = np.kron(v, np.array([[-1/math.sqrt(2)], [1/math.sqrt(2)]]))
    return v

def plus_matrix(n, m):
    return np.dot(plus_vector(n), np.transpose(plus_vector(m)))

def min_vector(n):
    v = np.eye(1)
    for i in range(n):
        v = np.kron(v, np.array([[1/math.sqrt(2)], [1/math.sqrt(2)]]))
    return v

def min_matrix(n, m):
    return np.dot(min_vector(n), np.transpose(min_vector(m)))

def black_spider_matrix(n,m):
    return plus_matrix(n,m) + min_matrix(n,m)


print( np.all(np.isclose(np.dot(black_spider_matrix(3,10),
                                black_spider_matrix(10,2)),
                         black_spider_matrix(3,2))) )

print("\nIsometry, only for (black) spiders 1 -> n")
print( is_isometry(black_spider_matrix(8,1)) )
print( is_isometry(black_spider_matrix(3,2)) )

print("\nExperiments\n")

# From: https://arxiv.org/abs/1102.2368

a = complex(random.uniform(-10.0, 10.0),
            random.uniform(-10.0, 10.0))
b = complex(random.uniform(-10.0, 10.0),
            random.uniform(-10.0, 10.0))

n =  a * a.conjugate() + b * b.conjugate()

# state
s = np.array([[a/n], [b/n]])
# with modifier
f = np.dot(white_spider_matrix(1,2), np.kron(np.eye(2), s))

print("Random state")
print( s )
print("modifier")
print( f )
print("state can be reconstructed")
print( np.dot(f, white_spider_matrix(1,0)) )
print("modifier is self-transposed")
print( np.dot(np.kron(np.dot(white_spider_matrix(0,2), 
                             np.kron(np.eye(2), f)),
                      np.eye(2)),
              np.kron(np.eye(2), white_spider_matrix(2,0))) )
print("inverse modifier")
g = np.linalg.inv(f)
print( g )


print("\nInstruments")

def modifier(array, dom):
    n = array.shape[0]
    s = matrix_square_root(array)
    mat = np.zeros((n,n,n,n)) + 0j
    for i in range(n):
        for j in range(n):
            mat[i][j] = array[i][j].conjugate() * np.eye(n)
    return Channel(mat, dom, dom)


N = 2

v = random_state([N])
#v = random_probabilistic_state(N)

print("")
#print( v )
mv = modifier(v.array, v.dom)
#mvi = modifier(np.linalg.inv(v.array), v.dom)
#print( mv )
print("Important equality: reconstruct state from quotient")
print( v == uniform_probabilistic_state(v.dom) / v.as_pred() )
print( truth(v.dom) == mv << truth(v.dom) )
print( v == mv >> uniform_probabilistic_state(v.dom) )
#print( mvi >> v )
print( is_positive(mv.as_operator().array) )

print("\nSquare root modifier: a subunital channel: 1/(N*N) missing")

def sqr_modifier(array, dom):
    n = array.shape[0]
    s = matrix_square_root(array)
    mat = np.zeros((n,n,n,n)) + 0j
    for i in range(n):
        for j in range(n):
            mat[i][j] = np.dot(s, np.dot(matrix_base(i,j,n), s))
    return Channel(mat, dom, dom)


sqrmv = sqr_modifier(v.array, v.dom)
sqrmvi = sqr_modifier(np.linalg.inv(v.array), v.dom)

print( np.all(np.isclose(v.array, 
                         N * N * (sqrmv >> uniform_probabilistic_state(v.dom)).array)) )
print( is_positive(sqrmv.as_operator().array) )

# Next outcomes are equal, up to a rather larger error
#print( np.isclose((sqrmv * sqrmvi).array, idn(N).array) )
#print( np.isclose((sqrmvi * sqrmv).array, idn(N).array) )
# not a unitary map
#print( sqrmv << truth(N) )

print("\nLeifer-Spekkens style\n")

def cup(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    return State(1/dom.size * idn(dom).as_operator().array, dom * 2)

print( cup([2,3]) >= truth([2,3,2,3]) )

# v2 = np.array([1,0,0,1])
# print( np.all(2 * cup([2]).array == np.outer(v2.transpose(),v2)) )

# v3 = np.array([1,0,0,0,1,0,0,0,1])
# print( np.all(3 * cup([3]).array == np.outer(v3.transpose(),v3)) )

# v4 = np.array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])
# print( np.all(4 * cup([4]).array == np.outer(v4.transpose(),v4)) )

def cup_chan(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    ls = [1]
    for i in range(n-1):
        ls = ls + n*[0] + [1]
    v = np.array(ls)
    mat = np.zeros((n*n,n*n,1,1))
    mat[...,0,0] = np.outer(v.transpose(), v)
    return Channel(1/n * mat, [], dom + dom)

print("Basic cup checks")
print( cup_chan([2,5]) == cup([2,5]).as_chan() )
print( truth([]) == cup_chan([3,4]) << truth([3,4,3,4]) )

def cup_state(dom):
    return cup_chan(dom) >> init_state

def cap_pred(dom):
    return cap_chan(dom) << truth([])

dom=Dom([2,2])

print( cup_state(dom) >= truth(dom+dom) )


#
# |v> = sum_{i} |ii>, giving matrix |v><v|, as in Leifer-Spekkens
#
def cap_chan(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    n = dom.size
    ls = [1]
    for i in range(n-1):
        ls = ls + n*[0] + [1]
    v = np.array(ls)
    mat = np.zeros((1,1,n*n,n*n))
    mat[0][0] = np.outer(v.transpose(), v)
    return Channel(n * mat, dom * 2, [])
    #return Channel(mat, dom * 2, [])

print("\nCap-Cup\n")

print("\nGround after cup is uniform: ",
      uniform_probabilistic_state(dom) ==
      (idn(dom) @ discard(dom)) >> cup_state(dom) )

q = random_pred(dom)

print("Predicate p after cup is substate 1/n * p.conj: ",
      np.all(np.isclose(1/q.dom.size * q.array.conjugate(),
                        ((q.as_subchan() @ idn(dom)) >> cup_state(dom)).array)) )

print("\nAs a result, using snake, cap after uniform @ idn is truth: ",
      truth(dom).as_subchan() == 
      cap_chan(dom) * (uniform_probabilistic_state(dom).as_chan() @ idn(dom)) )

s = random_state(dom)

print("Cap after state s is random variable n * s.conj: ",
      np.all(np.isclose(dom.size * s.array.conjugate(),
                        (cap_chan(dom) * (s.as_chan() @ idn(dom))).array)) )

print("\nPredicate p is cap after assert_p @ uniform: ",
      q == (sqr_modifier(q.array, dom) @ uniform_probabilistic_state(dom).as_chan()) << cap_pred(dom) )

print("Cap after assert_p @ idn is cap after idn @ assert_p.conj: ",
      (sqr_modifier(q.array, dom) @ idn(dom)) << cap_pred(dom) ==
      (idn(dom) @ sqr_modifier(q.array.conjugate(), dom)) << cap_pred(dom) )

print("")

print( np.all(np.isclose((((uniform_probabilistic_state(dom) / q).as_chan() @ idn(dom)) << cap_pred(dom)).array,
                         1 / (uniform_probabilistic_state(dom) >= q) * q.array.conjugate())) )


# Cap is not unital!
#print( cap_chan([2]) << truth([]) )
print("\nPositivity of cap:", 
      is_positive(cap_chan([9]).as_operator().array) )

snake1 = (cap_chan(dom) @ idn(dom)) * (idn(dom) @ cup_chan(dom))
snake2 = (idn(dom) @ cap_chan(dom)) * (cup_chan(dom) @ idn(dom))

print("Snake equations: ", snake1 == idn(dom), snake2 == idn(dom) )

print("\nChannel-state duality\n")

#ph = 0.5
#c =  z_chan * x_chan * phase_shift(ph)
c = hadamard

def channel_to_state(chan):
    return (idn(chan.dom) @ chan) * cup_chan(chan.dom) >> init_state

print("Validity of truth: ", channel_to_state(c) >= truth([2,2]) )
#e = random_pred(c.dom + c.cod)
#print( (idn(c.dom) @ c) << e )

def state_to_channel(stat):
    n = stat.dom.dims[0]
    m = stat.dom.dims[1]
    return (cap_chan([n]) @ idn([m])) * (idn([n]) @ stat.as_chan())

bs = random_state([2,2])

print("\nPositivity of channel: ", 
      is_positive(state_to_channel(bs).as_operator().array) )

print("Starting from fixed channel: ", 
      c == state_to_channel(channel_to_state(c)) )
print("Starting from random state: ", 
      bs == channel_to_state(state_to_channel(bs)) )
    
#print("\nchannel differences in conjugates??")
# print( channel_to_state(state_to_channel(bs)) )
# print("")
# print( channel_to_state1(state_to_channel(bs)) )



print("\nExperiments\n")

#p = random_probabilistic_pred((bs % [1,0]).dom)
p = random_pred((bs % [1,0]).dom)
#q = random_probabilistic_pred((bs % [0,1]).dom)
q = random_pred((bs % [0,1]).dom)

# The following outcomes are different

#print( bs >= (truth((bs % [1,0]).dom) @ q) )
#print( (bs % [1,0]) >= (state_to_channel(bs) << q) )

#print( (bs / (truth((bs % [1,0]).dom) @ q)) % [1,0] )
#print( (bs % [1,0]) / (state_to_channel(bs) << q) )


def extract(stat):
    sqr_chan = sqr_modifier(1/(stat % [1,0]).dom.size * 
                            # adding transpose here yields a unital map
                            np.linalg.inv((stat % [1,0]).array).transpose(), 
                            (stat % [1,0]).dom)
    return state_to_channel(stat) * sqr_chan

# extract does not produce a unital channel!
print( truth((bs % [1,0]).dom) == extract(bs) << truth((bs % [0,1]).dom) )
#print( extract(bs) << truth((bs % [0,1]).dom) )

# Validities are the same for probabilistic predicate
print("First coordinate validities: ",
      bs >= (p @ truth((bs % [0,1]).dom)),
      (bs % [0,1]) >= (extract(swap >> bs) << p) )
print("Second coordinate validities: ",
      bs >= (truth((bs % [1,0]).dom) @ q),
      (bs % [1,0]) >= (extract(bs) << q) )

# force one coordinate to be classical; "hybrid operator" by Leifer-Spekkens
# This one works, second coord. classical, extraction from quantum to classical

# First coordinate forced to be uniform:
#bs = convex_state_sum((0.5, (meas0 @ idn((bs % [0,1]).dom)) >> bs), (0.5, (meas1 @ idn((bs % [0,1]).dom)) >> bs))

# Second coordinate forced to be uniform:
#bs = convex_state_sum((0.5, (idn((bs % [1,0]).dom) @ meas0) >> bs), (0.5, (idn((bs % [1,0]).dom) @ meas1) >> bs))

# force first component to be probabilistic
bs = (meas0 @ idn((bs % [0,1]).dom)) >> bs


print("First and second marginals")
print( bs % [1,0] )
print( bs % [0,1] )

print("\nConditioning, in first coordinate: ",
      (bs / (p @ truth((bs % [0,1]).dom))) % [0,1] ==
      extract(bs) >> (bs % [1,0] / p),
      (bs / (p @ truth((bs % [0,1]).dom))) % [0,1] ==
      (bs % [0,1]) / (extract(swap >> bs) << p),
      extract(bs) >> (bs % [1,0] / p) ==
      (bs % [0,1]) / (extract(swap >> bs) << p) )

# Equal when first marginal is uniform
print( (bs / (p @ truth((bs % [0,1]).dom))) % [0,1] )
print( extract(bs) >> (bs % [1,0] / p) )
print( (bs % [0,1]) / (extract(swap >> bs) << p) )

print("\nConditioning, in second coordinate: ",
      (bs / (truth((bs % [1,0]).dom) @ q)) % [1,0] == 
      extract(swap >> bs) >> (bs % [0,1] / q),
      (bs / (truth((bs % [1,0]).dom) @ q)) % [1,0] ==
      (bs % [1,0]) / (extract(bs) << q),
      extract(swap >> bs) >> (bs % [0,1] / q) ==
      (bs % [1,0]) / (extract(bs) << q) )

# Equal when second marginal is uniform
print( (bs / (truth((bs % [1,0]).dom) @ q)) % [1,0] )
print( extract(swap >> bs) >> (bs % [0,1] / q) )
print( (bs % [1,0]) / (extract(bs) << q) )



# print( np.all(np.isclose((bs / (p @ truth((bs % [0,1]).dom)) % [0,1]).array,
#                          1 / (bs >= (p @ truth((bs % [0,1]).dom))) * 
#                          ((p.as_subchan() @ idn((bs % [0,1]).dom)) >> bs).array)) )



print("\nAssert experiments: succesful")

D = Dom([2])

s = random_state(D)

print( uniform_probabilistic_state(D) == 
       sqr_modifier(1/D.size * np.linalg.inv(s.array), D) >> s )

print( np.all(np.isclose(uniform_probabilistic_state(D).array,
                         (sqr_modifier(1/D.size * np.linalg.inv(s.array), D) >> s).array)) )


p = random_pred(D)
q = random_pred(D)

print( p & q == sqr_modifier(p.array, D) << q )
print( s / p == sqr_modifier(1/(s >= p) * p.array, D) >> s )
print("")
#print( np.dot(p.array, q.array) )
#print( modifier(p.array, D) << q )



"""
print("\nPredicate experiments")

qs = q.as_subchan()
print( q )
print( qs, qs.dom, qs.cod, ket(0) >= q )
# recover q
print( qs << truth([]) )

print("\nPredicates and channels")

def predicate_to_channel(pred):
    n = pred.dom.dims[0]
    m = pred.dom.dims[1]
    return (pred.as_subchan() @ idn([m])) * (idn([n]) @ cup_chan([m]))

e = random_pred(dom)

print( e.as_subchan() * (idn([2]) @ uniform_probabilistic_state([2]).as_chan()) )
print( predicate_to_channel(e) << truth([dom.dims[1]]) )

print("")


print("")
print(q)
print( np.isclose(q.array.conjugate(),
                  2 * ((q.as_subchan() @ idn([2])) >> cup_state([2])).array) )
print( np.isclose(q.array.conjugate(),
                  2 * ((idn([2]) @ q.as_subchan()) >> cup_state([2])).array) )
#print( uniform_probabilistic_state(q.dom) / q )

    
"""

"""

print("\nProbabilistic example works appropriately\n")

# Probabilistic disease-mood example

dm = kron_inv(2,2) >> probabilistic_state(0.05, 0.5, 0.4, 0.05)
pt = probabilistic_pred(0.9, 0.05)

print("A priori test likelihood: ", dm % [1,0] >= pt)

print("A posterior state, in three different ways")
print( (dm / (pt @ truth([2]))) % [0,1] )
m2d = extract(dm)
print( m2d >> (dm % [1,0] / pt) )
d2m = extract( (swap >> dm) )
print( dm % [0,1] / (d2m << pt) )

"""
