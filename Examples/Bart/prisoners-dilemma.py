#
# Based on: 
#
# http://rspb.royalsocietypublishing.org/content/early/2009/03/23/rspb.2009.0121
#
# see also:
#
# http://mypage.iu.edu/~jbusemey/quantum/QDT.pdf
#
from efprob_qu import *
from math import *
import efprob_dc as dc

#
# Two versions are presented, namely:
#
# Classical probabilistic, labeled with `M' for Markov
#
# Quantum, labeled with `Q'
#

print("\nPrisoner's dilemma")
print("==================\n")

#
# Prior uniform states
#
Mpsi = dc.uniform_disc_state(4)

Qpsi_vect = 0.5 * np.array([1,1,1,1])
Qpsi = vector_state(*Qpsi_vect)

#
# Basic predicates, where D = defect, C = cooperate. In a combination
# the opponent is mentioned first, so that:
#   DC = opponent defects, you cooperate
#
MDD = dc.unit_pred(4,0)
MDC = dc.unit_pred(4,1)
MCD = dc.unit_pred(4,2)
MCC = dc.unit_pred(4,3)

QDD = unit_pred(4,0)
QDC = unit_pred(4,1)
QCD = unit_pred(4,2)
QCC = unit_pred(4,3)

print("Markov state where opponent defects:")
print( Mpsi / (MDD | MDC) )
print("Quantum state where opponent defects:")
print( Qpsi / (QDD | QDC) )

#
# "We assume that the time evolution of the initial state to the final
# state corresponds to the thought process leading to a decision."
#

def K(mu_d, mu_c):
    return np.array([[-1, mu_d,  0,  0    ],
                     [1,  -mu_d, 0,  0    ],
                     [0,  0,     -1, mu_c ],
                     [0,  0,     1,  -mu_c]])

def MT(t, mu_d, mu_c):
    return scipy.linalg.expm(t * K(mu_d, mu_c))

def Mch(t, mu_d, mu_c):
    return dc.Channel(MT(t, mu_d, mu_c), range(4), range(4))

# No prior knowledge about opponent
#plot( lambda t: (Mch(t, 5, 10) >> Mpsi) >= (MDD + MCD), 0, 2 )
# Known defection by opponent
#plot( lambda t: (Mch(t, 5, 10) >> (Mpsi / (MDD + MDC))) >= (MDD + MCD), 0, 2 )
# Known cooperation by opponent
#plot( lambda t: (Mch(t, 5, 10) >> (Mpsi / (MCD + MCC))) >= (MDD + MCD), 0, 2 )

#
# Markov line from Fig. 9.2 from Busemeyer-Bruza book 
#
#plot( lambda k: (Mch(pi/2, k, k) >> Mpsi) >= (MDD + MCD), 0, 45 )


def HA(mu_d, mu_c):
    f_d = 1 / sqrt(1 + (mu_d ** 2))
    f_c = 1 / sqrt(1 + (mu_c ** 2))
    return RandVar(np.array([[f_d * mu_d, f_d, 0, 0],
                             [f_d, -f_d * mu_d, 0, 0],
                             [0, 0, f_c * mu_c, f_c],
                             [0, 0, f_c,-f_c * mu_c]]), Dom([4]))

def HB(gam):
    return RandVar(-(gam/sqrt(2)) * np.array([[1, 0, 1, 0],
                                              [0, -1, 0, 1],
                                              [1, 0, -1, 0],
                                              [0, 1, 0, 1]]), Dom([4]))
def K(delta):
    return RandVar(np.array([[-1, 0, delta, 0],
                             [0, -delta, 0, 1],
                             [1, 0, -delta, 0],
                             [0, delta, 0, -1]]), Dom([4]))

def HC(mu_d, mu_c, gam):
    return HA(mu_d, mu_c) + HB(gam)

def HK(gam, delta):
    return HB(gam) + K(delta)

s1 = HB(pi/2).evolution(Qpsi)(1)
print("\nHB example", s1 >= QDD, s1 >= QDC, s1 >= QCD, s1 >= QCC )

# No prior knowledge about opponent
#HA(0.5,0.5).plot_evolution(Qpsi, QDD + QCD, 0, 10)

# Known defection by opponent
#HA(5,5).plot_evolution(Qpsi / (QDD | QDC), QDD + QCD, 0, 10)

# Known cooperation by opponent
#HA(5,5).plot_evolution(Qpsi / (QCD | QCC), QDD + QCD, 0, 10)

# No prior knowledge about opponent
#HC(0.5,0.5,1).plot_evolution(Qpsi, QDD + QCD, 0, 10)


#
# Fig. 9.2 from Busemeyer-Bruza book for the quantum case. The formula
# below describes what is called QD in the book. Is the plot QD'*QD ??
# Does that explain the mismatch.
#
#plot( lambda k: HA(k,k).evolution(Qpsi)(pi/2) >= (QDD + QCD), 0, 20 )

# Book bottom p. 279
h = 0.5263
c = 2.2469

post = HK(c,1).evolution(Qpsi)(pi/2)

# last row of Table 9.4
print("must be 0.72 ", post / (QDD | QDC) >= QDD + QCD )
print("must be 0.84 ",  post / (QCD | QCC) >= QDD + QCD )

print("\nQuantum Encryption test\n")

M = random_randvar(4).array

U = scipy.linalg.expm(complex(0, -1) * M)

chU = channel_from_unitary(U, Dom([2,2]), Dom([2,2]))

chV = channel_from_unitary(conjugate_transpose(U), Dom([2,2]), Dom([2,2]))

v = kron_inv(2,2) >> random_state(4)
#print( v )
print( v >= unit_pred(2,0) @ truth(2) )
print( v >= truth(2) @ unit_pred(2,1) )

# encryption via ChU and encoding ob 2 bits 0,1 via conditioning

w = (chU >> v) / (chV << ~unit_pred(2,0) @ unit_pred(2,1))

#w = (idn(2) @ hadamard) >> w

print("measuring encrypted version")
#print( w )
print( w >= unit_pred(2,0) @ truth(2) )
print( w >= truth(2) @ unit_pred(2,1) )
print("decoding")
print( (chV >> w) >= unit_pred(2,0) @ truth(2) )
print( (chV >> w) >= truth(2) @ unit_pred(2,1) )

# print("\nMarkov Encryption test\n")

# v = dc.State(dc.random_disc_state(4).array, dc.Dom(range(2)) * 2)

# print(v >= dc.unit_pred(2,0) @ dc.truth(range(2)) )
# print(v >= dc.truth(range(2)) @ dc.unit_pred(2,1) )

# t = 4

# chU = dc.Channel(MT(t, 5, 5), dc.Dom(range(2)) * 2, dc.Dom(range(2)) * 2)
# chV = dc.Channel(np.linalg.inv(MT(t, 5, 5)), dc.Dom(range(2)) * 2, dc.Dom(range(2)) * 2)

# print( chV >> (chU >> v) )
# print( v )

# w = (chU >> v) / (chV << dc.unit_pred(2,0) @ ~dc.unit_pred(2,1))


# print( w >= dc.unit_pred(2,0) @ dc.truth(range(2)) )
# print( w >= dc.truth(range(2)) @ dc.unit_pred(2,1) )
# print("decoding")
# print( (chV >> w) >= dc.unit_pred(2,0) @ dc.truth(range(2)) )
# print( (chV >> w) >= dc.truth(range(2)) @ dc.unit_pred(2,1) )




