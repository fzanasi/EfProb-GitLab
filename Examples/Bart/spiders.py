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

def modifier(stat):
    n = stat.dom.size
    mat = np.zeros((n,n,n,n)) + 0j
    for i in range(n):
        for j in range(n):
            mat[i][j] = stat.array[i][j].conjugate() * np.eye(n)
    return Channel(mat, stat.dom, stat.dom)

N = 3

v = random_state([N,N])
#v = random_probabilistic_state(N)

print("")
#print( v )
mv = modifier(v)
#print( mv )
print("")
print( v == uniform_probabilistic_state(v.dom) / v.as_pred() )
print( truth(v.dom) == mv << truth(v.dom) )
print( v == mv >> uniform_probabilistic_state(v.dom) )
print( is_positive(mv.as_operator().array) )

print("\nSquare root modifier")

def sqr_modifier(stat):
    n = stat.dom.size
    s = matrix_square_root(stat.array)
    mat = np.zeros((n,n,n,n)) + 0j
    for i in range(n):
        for j in range(n):
            mat[i][j] = np.dot(s, np.dot(matrix_base(i,j,n), s))
    return Channel(mat, stat.dom, stat.dom)


sqrmv = sqr_modifier(v)
sqrmvi = sqr_modifier(State(np.linalg.inv(v.array), v.dom))

print( np.all(np.isclose(v.array, 
                         N * (sqrmv >> uniform_probabilistic_state(v.dom)).array)) )
print( is_positive(sqrmv.as_operator().array) )
# Next outcomes are equal, up to a rather larger error
#print( np.isclose((sqrmv * sqrmvi).array, idn(N).array) )
#print( np.isclose((sqrmvi * sqrmv).array, idn(N).array) )
# not a unitary map
#print( sqrmv << truth(N) )

print("\nLeifer-Spekkens style\n")

def cap(dom):
    dom = dom if isinstance(dom, Dom) else Dom(dom)
    return State(1/dom.size * idn(dom).as_operator().array, dom * 2)

print( cap([2]) )

print( classic([2]) @ classic([4]) == classic([2,4]) )

ph = 1.5
c = x_chan * phase_shift(ph) * z_chan

# from channel to binary state
bs = (idn([2]) @ c) >> cap([2])
#print( bs )

#print( c.as_operator() )

def extract(w):
    n = w.dom.dims[0]
    m = w.dom.dims[1]
    v = Operator(n * w.array, Dom([n]), Dom([m]))
    return v.as_channel()

#print( extract(bs) )
#print( c )




"""

c = (hadamard @ discard(2))

#print( c.dom, c.cod )

#print( c.as_operator() )

unif22 = kron_inv(2,2) >> uniform_probabilistic_state(4)

print("\nCaps and cups")

cap2 = bell00.as_chan()
# fails:
cup2 = bell00.as_pred().as_chan()


#print( cup2 << Predicate(np.eye(1), []) )

w = kron_inv(2,2) >> random_state(4)

print( pair_extract(w)[1] )

print( (idn(2) @ w.as_chan()).cod )

print( (cap2 @ idn(2)) * (idn(2) @ w.as_chan()) )

snake1 = (unif22.as_pred().as_chan() @ idn(2)) * (idn(2) @ unif22.as_chan())
snake2 = (bell00.as_pred().as_chan() @ idn(2)) * (idn(2) @ bell00.as_chan())

print("\nsnake")
#print( snake2 )
#print( snake2 << truth(2) )

unif2222 = (kron_inv(2,2) @ kron_inv(2,2)) >> (kron_inv(4,4) >> uniform_probabilistic_state(16))

#print( (c @ idn(2,2)) >> unif2222 )

ground22 = unif22.as_pred().as_chan()

print("\nGround types: ", ground22.dom, ground22.cod )

cpred = ground22 * (c @ idn(2))

#print( cpred.dom )




"""




# mvi = mv.inversion(uniform_probabilistic_state(N))
# print( mvi * mv )
# print( mv * mvi )
# print( idn(2) )
# v_inv = State(np.linalg.inv(v.array), Dom([N]))
# mv1 = modifier(v_inv)
# print( mv1 )
# print( mv1 * mv )
# print( mv * mv1 )

#print( modifier(v) >> v_inv )
