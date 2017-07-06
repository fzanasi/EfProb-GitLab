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

N = 2

v = random_state(N)
#v = random_probabilistic_state(N)

print("")
print( v )
mv = modifier(v)
#print( mv )
print("")
print( truth(N) == mv << truth(N) )
print( v == mv >> uniform_probabilistic_state(N) )
print( is_positive(mv.as_operator().array) )

print("\nSquare root modifier")

def sq_modifier(stat):
    n = stat.dom.size
    mat = np.zeros((n,n,n,n)) + 0j
    c = instr(stat.as_pred())
    for i in range(n):
        for j in range(n):
            mat[i][j] = 2 * (c << (point_pred(0,2) @ Predicate(matrix_base(i,j,n),
                                                           stat.dom))).array
    return Channel(mat, stat.dom, stat.dom)

# kr = np.kron(matrix_square_root(v.array), np.eye(N))

# vop = Operator(np.dot(kr,
#                       np.dot(idn(N).as_operator().array,
#                              kr)),
#                Dom([N]), Dom([N]))

sqmv = sq_modifier(v)

print( sqmv >> uniform_probabilistic_state(N) )

#mvi = pair_extract(graph_pair(v, idn(N)))[1]









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
