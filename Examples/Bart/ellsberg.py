#
# From: Identifying Quantum Structures in the Ellsberg Paradox
# Diederik Aerts, Sandro Sozzo, Jocelyn Tapia
# International Journal of Theoretical Physics, 53, pp. 3666-3682, 2014 
# https://arxiv.org/abs/1302.3850
#
# Alternatively, from: http://www.quantum-cognition.de/texts/Blutner_beimGraben_Synthese_final.pdf
#
# Summarizing, the probability of risky events is calculated by the
# Born rule; the probability of events under ignorance is calculated
# by mixing. Quantum probabilities correspond to the normal case of
# judging risks. The case of ignorance (no explicit probabilities are
# provided by the geometry of projections) are handled by the mixed
# case of the density matrix (Franco 2007c). 14
#
#
from efprob_qu import *
from math import *
import efprob_dc as dc

print("\nEllsberg Paradox")
print("================\n")

# Approximation of the underlying state. Really what is needed is
# "dependent" version.

E1 = dc.flip(1/3) @ dc.uniform_state(dc.R(0,60))

R = dc.Predicate([1,0], [True,False]) @ dc.truth(dc.R(0,60))
Y = dc.truth([True,False]) @ dc.Predicate(lambda x: x/90, dc.R(0,60))
B = dc.truth([True,False]) @ dc.Predicate(lambda x: 2/3 - x/90, dc.R(0,60))

#
# Sidenote: alternatively one can take B = ~(R + Y)
# But taking B = ~(R | Y) yields the wrong outcome 0.44444
#

print("Validity of truth ", 
      E1 >= dc.truth([True,False]) @ dc.truth(dc.R(0,60)) )

print("E1 probabilities of Red, Yellow, Black ", 
      E1 >= R, E1 >= Y, E1 >= B )

RV1 = dc.RandVar(lambda *x: (12 if x[0] else 0), E1.dom)
RV2 = dc.RandVar(lambda *x: 12 * (2/3 - x[1]/90), E1.dom)
RV3 = dc.RandVar(lambda *x: (12 if x[0] else 0) + 12 * x[1]/90, E1.dom)
RV4 = dc.RandVar(lambda *x: 12 * x[1]/90 + 12 * (2/3 - x[1]/90), E1.dom)

print("Expected values of 4 experiments: ",
      RV1.exp(E1), RV2.exp(E1), RV3.exp(E1), RV4.exp(E1) )

print("Given not Red, probability of Yellow: ", (E1 / ~R) >= Y)


E2 = dc.State(dc.Fun(lambda *x: 0 if x[0] < 2/3 else 1/50, 
                    [dc.R(0,1), dc.R(0,60)]), 
             [dc.R(0,1), dc.R(0,60)])

ER = dc.Predicate(lambda *x: x[0], [dc.R(0,1), dc.R(0,60)])
EY = dc.Predicate(lambda *x: x[1]/36, [dc.R(0,1), dc.R(0,60)])
EB = dc.Predicate(lambda *x: 1 - x[1]/36, [dc.R(0,1), dc.R(0,60)])

print( E2 >= ER, E2 >= EY, E2 >= EB, E2/ER >= EY )

ch = dc.Channel.from_states([dc.uniform_state(dc.R(0,60)), dc.uniform_state(dc.R(0,60))])


#((E1 / R) % [1,0]).plot()
#((E1 / Y) % [0,1]).plot()


print("\nThird attempt")

E3 = dc.uniform_state(dc.R(0,60))

Y3 = dc.Predicate(lambda x: x/90, dc.R(0,60))
B3 = dc.Predicate(lambda x: 2/3 - x/90, dc.R(0,60))
R3 = ~(Y3 + B3)

RV31 = 12 * R3
RV32 = 12 * B3
RV33 = 12 * R3 + 12 * Y3


print("Probabilities: ", E3 >= R3, E3 >= Y3, E3 >= B3 )
print("Pay off: ", E3/Y3 >= B3)


print("\nBayesian approach")
print("=================\n")

#
# Number of Yellow and Black balls
# Number of Red balls is N/2
#
N=20 

options1 = ['A', 'B']

A_bet = dc.Predicate([1,0], options1)
B_bet = dc.Predicate([0,1], options1)

bet1 = dc.chan_fromklmap(#lambda i, x: N/(2*i+N) if x=='A' else  2*i/(2*i+N),
    lambda i: dc.State([N/(2*i+N), 2*i/(2*i+N)], options1),
    range(N+1),
    options1)

prior1 = dc.uniform_disc_state(N+1)

print("Half black prior distribution: ", 
      bet1 >> dc.unit_disc_state(N+1, int(N/2)) )
print("Uniform prior distribution: ", bet1 >> prior1 )

#
# 0 = A_bet, 1 = B_bet
#
bet1_list = [0]*40 + [1]*19

post1 = prior1
for i in bet1_list:
    bet_pred = A_bet if i==0 else B_bet
    post1 = post1 / (bet1 << bet_pred)

print("Posterior distribution: ", bet1 >> post1 )
post1.plot()




"""

print("\nQuantum version")
print("===============\n")

v = 1/sqrt(3) * np.array([complex(1,0), complex(1,0), complex(1,0)])
vc = vector_state(*v)

# states from (the end of) section 2)

v12 = vector_state(1/(2*sqrt(3)) * complex(1, sqrt(3)),
                   1/(2*sqrt(3)) * complex(0, sqrt(6)),
                   1/(2*sqrt(3)) * complex(sqrt(2),0))

v34 = vector_state(1/(2*sqrt(3)) * complex(sqrt(3),1),
                   1/(2*sqrt(3)) * complex(0, sqrt(2)),
                   1/(2*sqrt(3)) * complex(sqrt(6),0))

# color predicates: red, yellow, black

R = unit_pred(3,0)
Y = unit_pred(3,1)
B = unit_pred(3,2)

print("R, Y, B in v12 ", v12 >= R, v12 >= Y, v12 >= B)
print("R, Y, B in v34 ", v34 >= R, v34 >= Y, v34 >= B)

print("")

# random variables

F1 = 12 * R
F2 = 12 * B
F3 = 12 * R + 12 * Y
F4 = 12 * R + 12 * B

print("F1, F2, F3, F4 in v12 ", v12 >= F1, v12 >= F2, v12 >= F3, v12 >= F4)
print("F1, F2, F3, F4 in v34 ", v34 >= F1, v34 >= F2, v34 >= F3, v34 >= F4)

print("")

e1 = np.array([0.38 * e ** complex(0, radians(61.2)),
               0.13 * e ** complex(0, radians(248.4)),
               0.92 * e ** complex(0, radians(194.4))])

e3 = np.array([0.25 * e ** complex(0, radians(251.27)),
               0.55 * e ** complex(0, radians(246.85)),
               0.90 * e ** complex(0, radians(218.8))])

P4 = vector_state(*e1)
P2 = vector_state(*e3)

O12 = truth(3) - 2 * P2
O34 = truth(3) - 2 * P4

print("vc >= P2, should be 0.5, and is ", vc >= P2 )
print("vc >= P4, should be 0.5, and is  ", vc >= P4 )
print("v12 >= P2, should be 0.32, and is  ", v12 >= P2 )
print("v34 >= P4, should be 0.69, and is  ", v34 >= P4 )
print("v12 >= O12 ", v12 >= O12 )
print("v34 >= O34 ", v34 >= O34 )

print("\nWell-formedness tests")
print( np.inner(e1, e3) )
print( P2 >= P2 )
print( P4 >= P4 )
print( P2 >= P4 )


"""

"""

# No solution!!

s1 = 1
s2 = 1
s3 = 1
s4 = 1

from sympy import Symbol, conjugate, I, solve
z1 = Symbol("z1")
z2 = Symbol("z2")
z3 = Symbol("z3")
w1 = Symbol("w1")
w2 = Symbol("w2")
w3 = Symbol("w3")
print( solve( [ 
    #z1 + z2 + z3 - s1 * sqrt(0.5 * sqrt(3)),
    w1 + w2 + w3 - s2 * sqrt(0.5 * sqrt(3)),
    #z1 * (sqrt(3) + I) + z2 * sqrt(2) * I + z3 * sqrt(6) - s2 * sqrt(2 * sqrt(3) * 0.69),
    w1 * (1 + sqrt(3)*I) + w2 * sqrt(6) * I + w3 * sqrt(2) - s4 * sqrt(2 * sqrt(3) * 0.32),
    #z1 * conjugate(z1) + z2 * conjugate(z2) + z3 * conjugate(z3) - 1,
    w1 * conjugate(w1) + w2 * conjugate(w2) + w3 * conjugate(w3) - 1
    ],
              [z1, z2, z3, w1, w2, w3] ) )

"""