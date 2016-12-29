#
# Examples from J. Busemeyer, P. Buza: Quantum Models of Cognition and
# Decisision, CUP, 2012
#
# Legenda
#
#  ~p        negation (orthocomplement) of predicate p
#  p + q     sum of predicates p,q, partially defined
#  p & q     sequential conjunction ("andthen") of predicate p,q
#  p | q     sequential disjuntion, De Morgan dual of &
#  s >= p    validity of predicate p in state s  
#            (read >= as entailment |= or \models in LaTeX)
#  s / p     conditioning (revision) of state s with predicate p,
#            to be read as: s, given p
#  c >> s    forward transformation of state s along channel c
#  c << p    backward transformation of predicate p along channel c
#
from efprob_qu import *

import efprob_dc as dc


print("\nExamples from book: Quantum Models of Cognition and Decisision")
print("==============================================================")

print("\n2.1.2.1 Analysis of a single categorical variable\n")

#
# Initial / prior state in which various validities >= will be computed.
#
s = vector_state(-0.6963, 0.6963, 0.1741)

#
# Basic predicates for "man", with A = Audi, B = BMW, C = Cadillac
#
B = unit_pred(3, 0)
A = unit_pred(3, 1)
C = unit_pred(3, 2)

print("Probability of man choosing Audi: ", s >= A )
print("Probability of man choosing BMW: ", s >= B )
print("Probability of man choosing Cadillac: ", s >= C )
print("Probability of man choosing Audi or BMW: ", s >= A+B)
print("(since predicates are sharp this is the same as: ", s >= A | B, ")" )

print("\n2.1.2.3 Incompatible representation\n")

#
# States from 2.1.2.5
#
u = vector_state(1/math.sqrt(2), 1/math.sqrt(2), 0)
v = vector_state(1/2, -1/2, 1/math.sqrt(2))
w = vector_state(-1/2, 1/2, 1/math.sqrt(2))

#
# Associated predicates for "wife", where:
#
# U = wife wants BMW
# V = wife wants Audi
# W = wife wants Cadillac
#
U = u.as_pred()
W = w.as_pred()
V = v.as_pred()

print("Wife wants these cars with different probabilities:")
print("BMW: ", s >= U)
print("Audi: ", s >= V)
print("Cadillac: ", s >= W)

print("\n2.1.2.5 Uncertainty principle\n")

# The unitary matrix involved is made explicit, and turned into a
# channel

Uni = np.array([[1/math.sqrt(2), 1/2, -1/2], 
                [1/math.sqrt(2), -1/2, 1/2], 
                [0, 1/math.sqrt(2), 1/math.sqrt(2)]])

print("The transition matrix is unitary:", is_unitary(Uni))
print("It has inverse / conjugate-transpose:\n")
print( conjugate_transpose(Uni) )

ch = channel_from_unitary(Uni, Dom([3]), Dom([3]))

print("\nVia predicate transformation, the wife's probabilities can also be computed as:")
print("BMW: ", s >= ch << B)
print("Audi: ", s >= ch << A)
print("Cadillac: ", s >= ch << C)

print("\n2.1.2.6 Order effects\n")

print("Wife prefers Cadillac, then man BMW:", s >= (ch << C) & B )
print("Man prefers BMW, then wife Cadillac:", s >= B & (ch << C) )

print("\n2.1.2.7 Interference effect\n")

#
# Failure of the law of total probability
#
print("Total probability formulation:",
      (s / A >= U) * (s >= A) + 
      (s / B >= U) * (s >= B) + 
      (s / C >= U) * (s >= C) )

#
# Via Bayes' rule this is equal to:
#
print("Equivalently, via Bayes' rule:", 
      (s >= A & U) + (s >= B & U) + (s >= C & U) )

#
# But not equal to:
#
print("But this is not equal to:", s >= U )

print("\n2.1.2.9 Coarse and complete measurements, pure and mixed states\n")

sporty = A+B
luxurious = C
print("Given that wife wants luxurious, man wants sporty: ",  
      s / (ch << luxurious) >= sporty )
print("Given that man wants sporty, wife wants luxurious: ",   
      s / sporty >= (ch << luxurious) )
print("Given that man wants Audi, wife wants luxurious: ",    
      s / A >= (ch << luxurious) )

print("\n2.1.2.10 Compatible representation\n")

#
# Problem: what does conjunction /\ used in the book mean? It seems
# that parallel conjunction @ is intended, moving to a product
# state. But then, how is the three dimensional vector S be
# represented in this 9 dimensional product?
#
#

AA = A @ (ch << A)
BB = B @ (ch << B)
CC = C @ (ch << C)

agree_state = (s @ s) / (AA + BB + CC)

print( s @ u >= BB )

print( agree_state >= BB, "??" )


print("\n3.1.1 Analysis of first question\n")

s = vector_state(0.8367, 0.5477)

#
# Basic predicates: 
#
# Cy = Clinton yes, Cn = Clinton no
# Gy = Gore yes, Gn = Gore no
#
Cy = unit_pred(2,0)
Cn = ~Cy
Gy = vector_state(1,1).as_pred()
Gn = ~Gy

print("Probability of honesty of Clinton")
print("Yes: ",  s >= Cy )
print("No: ",  s >= Cn )

print("Probability of honesty of Gore")
print("Yes: ",  s >= Gy )
print("No: ",  s >= Gn )


print("\n3.1.2 Analysis of second question\n")

print("Probability of yes to Gore and then yes to Clinton plus no to Gore and then yes to Clinton: ", s >= (Gy & Cy) + (Gn & Cy) )

print("Probability of yes to Clinton and then yes to Gore plus no to Clinton and then yes to Gore: ", s >= (Cy & Gy) + (Cn & Gy) )


print("\n4.1.1.1 Probability for a single question\n")

#
# state and predicates
#
s = vector_state(0.987, -0.1564)
fem = unit_pred(2,0)
bate = vector_state(math.cos(0.4 * math.pi), math.sin(0.4 * math.pi)).as_pred()

print("Probability of being feminist: ", s >= fem )
print("Probability of being bank teller: ", s >= bate )

print("\n4.1.1.2 Probability for conjunction\n")

print("Feminist and then bank teller: ", s >= fem & bate )

print("\n4.1.1.3 Probability for disjunction\n")

print("Bank teller or then feminist: ", s >= bate | fem )


print("\n4.2.4.2 Naive perspective\n")

# from 4.2.4.1. Note, there is a typo in the definition of H1 !!!  It
# is corrected below, and then gives the vector of probabilities at
# the top of p.138.

H1 = np.array([[1, 1, 0, 0],
               [1, -1, 0, 0],
               [0, 0, 1, 1],
               [0, 0, 1, -1]])

H2 = np.array([[1, 0, 1, 0],
               [0, -1, 0, 1],
               [1, 0, -1, 0],
               [0, 1, 0, 1]])

H = H1 + H2

print("Matrix H is Hermitian: ", is_hermitian(H), "\n" )

n = np.array([math.sqrt(0.459/2),
              math.sqrt(0.459/2),
              math.sqrt(0.541/2),
              math.sqrt(0.541/2)])

n_state = vector_state(*n)

#
# Predicates:
#
# pGpE = positive Guilt, positive Evidence
# pGnE = positive Guilt, negative Evidence
# nGpE = negative Guilt, positive Evidence
# nGnE = negative Guilt, negative Evidence
#
pGpE = unit_pred(4,0)
pGnE = unit_pred(4,1)
nGpE = unit_pred(4,2)
nGnE = unit_pred(4,3)

print("Guilt for the first judgement, without evidence: ", 
      n_state >= pGpE + pGnE )

print("\n4.2.4.3 Prosecution perspective\n")

#
# Shift to the prosecution's perspective
#
xp = 1.2393
Upn = scipy.linalg.expm(complex(0,-1) * xp * conjugate_transpose(H))

print("Matrix Upn is unitary: ", is_unitary(Upn) )

chpn = channel_from_unitary(conjugate_transpose(Upn), Dom([4]), Dom([4]))

p = np.dot(Upn, n)

#
# state incorporating prosecution's perspective
#
p_state = vector_state(*p)

print("State p can also be obtained from state transformation: ",
      p_state == chpn >> n_state )

print("\nSquared magnitudes of vector p: ", [ abs(x) ** 2 for x in p] )

print("\nThis is the same as the validities of the 4 basic predicates: ",
      p_state >= pGpE,
      p_state >= pGnE,
      p_state >= nGpE,
      p_state >= nGnE )

#
# State update after presentation of positive evidence
#
p_plus = p_state / (pGpE + nGpE)

print("\nJudged probability of guilt for the second judgement, after the first piece of (positive) evidence: ", p_plus >= pGpE )

print("\n4.2.4.4 Defense perspective\n")

#
# Shift to defensive perspective, from neutral perspective
#
xd = -3.8324
Udn = scipy.linalg.expm(complex(0,-1) * xd * conjugate_transpose(H))

#
# Shift to defense perspective, starting from prosection's perspective
#
Udp = np.dot(Udn, conjugate_transpose(Upn))

print("Matrices Udn and Udp are unitary: ", is_unitary(Udn), is_unitary(Udp) )

chdn = channel_from_unitary(conjugate_transpose(Udn), Dom([4]), Dom([4]))
chdp = channel_from_unitary(conjugate_transpose(Udp), Dom([4]), Dom([4]))

d = np.dot(Udn, n)

#
# state incorporating defense perspective
#
d_state = chdn >> n_state

print("State d can also be obtained from state transformations: ",
      d_state == vector_state(*d), d_state == chdp >> p_state)

#
# state incorporating defense perspective after seeing prosecution's
# (positive) evidence
#
d_plus = chdp >> p_plus

print("\nValidities of the 4 basic predicates in state d+: ", 
      d_plus >= pGpE, 
      d_plus >= pGnE, 
      d_plus >= nGpE, 
      d_plus >= nGnE )

#
# State update after presentation of negative evidence, after positive
# evidence
#
d_plus_minus = d_plus / (pGnE + nGnE)

print("\nJudged probability of guilt for the second judgement, after both pieces of evidence: ",  d_plus_minus >= pGnE )

print("\nOwn addition, as comparison: the probability of guilt given only the negative evidence is: ", 
      d_state / (pGnE + nGnE) >= pGnE )


print("\n5.2 Non-compositional models of concept combinations based in quantum interference\n")

#
# states
#
food_vector = np.array([0.9354, 0, 0.3536])
food_state = vector_state(*food_vector)
scalar = math.e ** complex(0,1.5214)
plant_vector = scalar * np.array([0.2358, 0.752, -0.6159])
plant_state = vector_state(*plant_vector)
conjunction_vector = 1/math.sqrt(2) * (food_vector + plant_vector)
conjunction_state = vector_state(*conjunction_vector)

#
# predicates
#
M1 = unit_pred(3, 2)
M2 = unit_pred(3, 0) + unit_pred(3, 1)

print("mu_i(A) is: ", food_state >= M2 )
print("mu_i(B) is: ", plant_state >= M2 )
print("Conjunction of states gives: ", conjunction_state >= M2 )


print("\n7.3 An analysis of spooky-activation-at-a-distance in terms of a composite quantum system\n")

pt = 0.7
pa1 = 0.2
pa2 = 0.35

t_vector = np.array([math.sqrt(1 - pt), math.sqrt(pt)])
a1_vector = np.array([math.sqrt(1 - pa1), math.sqrt(pa1)])
a2_vector = np.array([math.sqrt(1 - pa2), math.sqrt(pa2)])

t = vector_state(*t_vector)
a1 = vector_state(*a1_vector)
a2 = vector_state(*a2_vector)

psit_vector = np.kron(t_vector, np.kron(a1_vector, a2_vector))

psit = t @ a1 @ a2

#print( np.all(np.isclose(psit.array, vector_state(*psit_vector).array)) )


print("\n8.1.1.4 Numerical example\n")

domain = ['+', '-']

Mprior = dc.state([1, 0], domain)

min_pred = dc.Predicate([0,1], domain)

K = np.array([[-1, 1],
              [1, -1]])

def T(t):
    return scipy.linalg.expm(t * K)

def chT(t):
    return dc.Channel(T(t), domain, domain)

print("Markov transition probabilities from plus to minus at time t, see the dashed line in Fig. 8.4")
for i in range(25):
    print("t =", i/8, " ", (chT(i/8) >> Mprior) >= min_pred )

print("\nRevised (conditioned) state after observing min at time t=1:")
print( (chT(1) >> Mprior) / min_pred )

print("\n8.1.2.4 Numerical example\n")

H = np.array([[0, 2],
              [2, 0]])

def U(t):
    return scipy.linalg.expm(t * complex(0, 1) * H)

def chU(t):
    return channel_from_unitary(U(t), Dom([2]), Dom([2]))

Qprior = ket(0)

Mmin = unit_pred(2,1)
Mplus = unit_pred(2,0)

print("Quantum transition probabilities from plus to minus at time t, see the dashed line in Fig. 8.4")
for i in range(25):
    print("t =", i/8, " ", (chU(i/8) >> Qprior) >= Mmin )

print("\nRevised (conditioned) state after observing min at time t=1:")
print( (chU(1) >> Qprior) / Mmin )
