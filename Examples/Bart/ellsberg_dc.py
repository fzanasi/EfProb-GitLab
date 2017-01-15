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
# http://economics.stackexchange.com/questions/5304/question-about-the-ellsberg-paradox-in-expected-utility-theory
#
from efprob_dc import *
from math import *

print("\nEllsberg Paradox")
print("================\n")

#
# N is a constant for the number of Red balls in an urn. The sum of
# the numbers of Yellow and Black balls is 2*N.
# 
# We use the variable y for the (unknown) number of Yellow
# balls. Notice that 0 <= y <= 2*N. Hence the domain of the state for
# y has size 2*N+1
#
N=30
yellow_domain = range(2*N+1)
unif_prior = uniform_disc_state(2*N+1)
halfway_prior = unit_disc_state(2*N+1, N)
#
# Own addition: compensation for risc aversion; puttig them at 0 means
# no compensation
#
a=0.344
b=0.377

#
# Domains for Red and Yellow balls
#
RYB_dom = ['R', 'Y', 'B']

ch = chan_fromklmap(lambda y: 
                     State([1/3, 2/3 * y/(2*N), 2/3 * (2*N - y)/(2*N)], 
                           RYB_dom),
                     yellow_domain,
                     RYB_dom)

rv1 = RandVar(lambda x: 12 if x=='R' else 0, RYB_dom)
rv2 = RandVar(lambda x: 12 if x == 'B' else 0, RYB_dom)

print("Uniform Yellow: ", ch >> unif_prior )
print("Half Yellow, for N =", N, "is: ", ch >> halfway_prior )
print("Random Yellow: ", ch >> random_disc_state(2*N+1) )


#
# Finding of Diederik Aerts et al: 40 out 59 choose Red

bet1_list = [1] * 40 + [2] * 19

print("\n* Bet 1")
print("Prior expected gain for option 1: ", rv1.exp(ch >> unif_prior) )
print("Prior expected gain for option 2: ", rv2.exp(ch >> unif_prior) )

post1 = unif_prior
for i in bet1_list:
    post1 = post1 / (ch << (Predicate([0,1,a], RYB_dom) if i==1 else
                             Predicate([0,0,1], RYB_dom)))

print("Posterior distribution: ", ch >> post1 )
print("Posterior expected number of Yellow: ",
      RandVar(lambda x: x, yellow_domain).exp(post1) )
print("Posterior expected gain for option 1: ", rv1.exp(ch >> post1) )
print("Posterior expected gain for option 2: ", rv2.exp(ch >> post1) )
#print( (ch >> post1) / Predicate([1,0,1], RYB_dom) )
#post1.plot()

rv3 = RandVar(lambda x: 12 if x=='R' or x == 'Y' else 0, RYB_dom)
rv4 = RandVar(lambda x: 12 if x=='Y' or x == 'B' else 0, RYB_dom)

print("\n* Bet 2")
print("Prior expected gain for option 3: ", rv3.exp(ch >> unif_prior) )
print("Prior expected gain for option 4: ", rv4.exp(ch >> unif_prior) )

#
# Finding of Diederik Aerts et al: 18 out 59 choose Red or Yellow
#
bet2_list = [3] * 18 + [4] * 41

post2 = unif_prior
for i in bet2_list:
    post2 = post2 / (ch << (Predicate([0,1,0], RYB_dom) if i==3 else
                            Predicate([0,b,1], RYB_dom)))

print("Posterior distribution: ", ch >> post2 )
print("Posterior expected number of Yellow: ",
      RandVar(lambda x: x, yellow_domain).exp(post2) )
print("Posterior expected gain for option 3: ", rv3.exp(ch >> post2) )
print("Posterior expected gain for option 4: ", rv4.exp(ch >> post2) )
#post2.plot()


# print("\n* Combined updating")
#
# post3 = unif_prior
# for i in bet1_list + bet2_list:
#     post3 = post3 / (ch << Predicate([0,1,0,0], [R_dom, B_dom]) if i==0 else 
#                      ch << Predicate([0,0,1,0], [R_dom, B_dom]) if i==1 else
#                      ch2 << Predicate([0,1,0,0], [RY_dom, YB_dom]) if i==2 else
#                      ch2 << Predicate([0,0,1,0], [RY_dom, YB_dom]))

#post3.plot()




"""

# Approximation of the underlying state. Really what is needed is
# "dependent" version.

E1 = flip(1/3) @ uniform_state(R(0,60))

R = Predicate([1,0], [True,False]) @ truth(R(0,60))
Y = truth([True,False]) @ Predicate(lambda x: x/90, R(0,60))
B = truth([True,False]) @ Predicate(lambda x: 2/3 - x/90, R(0,60))

#
# Sidenote: alternatively one can take B = ~(R + Y)
# But taking B = ~(R | Y) yields the wrong outcome 0.44444
#

print("Validity of truth ", 
      E1 >= truth([True,False]) @ truth(R(0,60)) )

print("E1 probabilities of Red, Yellow, Black ", 
      E1 >= R, E1 >= Y, E1 >= B )

RV1 = RandVar(lambda *x: (12 if x[0] else 0), E1.dom)
RV2 = RandVar(lambda *x: 12 * (2/3 - x[1]/90), E1.dom)
RV3 = RandVar(lambda *x: (12 if x[0] else 0) + 12 * x[1]/90, E1.dom)
RV4 = RandVar(lambda *x: 12 * x[1]/90 + 12 * (2/3 - x[1]/90), E1.dom)

print("Expected values of 4 experiments: ",
      RV1.exp(E1), RV2.exp(E1), RV3.exp(E1), RV4.exp(E1) )

print("Given not Red, probability of Yellow: ", (E1 / ~R) >= Y)


E2 = State(Fun(lambda *x: 0 if x[0] < 2/3 else 1/50, 
                    [R(0,1), R(0,60)]), 
             [R(0,1), R(0,60)])

ER = Predicate(lambda *x: x[0], [R(0,1), R(0,60)])
EY = Predicate(lambda *x: x[1]/36, [R(0,1), R(0,60)])
EB = Predicate(lambda *x: 1 - x[1]/36, [R(0,1), R(0,60)])

print( E2 >= ER, E2 >= EY, E2 >= EB, E2/ER >= EY )

ch = Channel.from_states([uniform_state(R(0,60)), uniform_state(R(0,60))])


#((E1 / R) % [1,0]).plot()
#((E1 / Y) % [0,1]).plot()


print("\nThird attempt")

E3 = uniform_state(R(0,60))

Y3 = Predicate(lambda x: x/90, R(0,60))
B3 = Predicate(lambda x: 2/3 - x/90, R(0,60))
R3 = ~(Y3 + B3)

RV31 = 12 * R3
RV32 = 12 * B3
RV33 = 12 * R3 + 12 * Y3


print("Probabilities: ", E3 >= R3, E3 >= Y3, E3 >= B3 )
print("Pay off: ", E3/Y3 >= B3)


print("\nBayesian approach")
print("=================\n")

#
# N is a constant for the number of Red balls in an urn. The sum of
# the numbers of Yellow and Black balls is 2*N.
# 
# We use the variable y for the (unknown) number of Yellow
# balls. Notice that 0 <= y <= 2*N. Hence the domain of the state for
# y has size 2*N+1
#
N=30 
yellow_domain = range(2*N+1)
unif_prior = uniform_disc_state(2*N+1)
halfway_prior = unit_disc_state(2*N+1, N)

#
# A : Red ball
# B : Black ball
#
bet1_domain = ['A', 'B']
A_bet = Predicate([1,0], bet1_domain)
B_bet = Predicate([0,1], bet1_domain)

#
# Probability of A: 1/3
# Probability of B: 2/3 * (2*N-y)/(2*N)
# Normalisation yields the probability N/(3*N-y) for A used below.
#
bet1 = chan_fromklmap(lambda y: flip(N/(3*N-y), bet1_domain),
                         yellow_domain,
                         bet1_domain)

print("* First bet")
print("Half black prior distribution: ", bet1 >> halfway_prior )
print("Uniform prior distribution: ", bet1 >> unif_prior )

#
# 0 = A_bet, 1 = B_bet
#
bet1_list = [0]*40 + [1]*19

post1 = unif_prior
for i in bet1_list:
    bet_pred = A_bet if i==0 else B_bet
    post1 = post1 / (bet1 << bet_pred)

print("Posterior distribution: ", bet1 >> post1 )
exp1 = RandVar(lambda x: x, yellow_domain).exp(post1)
print("Expected Yellow value: ", exp1 )
print("Distribution for y =", floor(exp1+0.5), " is ",
      bet1 >> unit_disc_state(2*N+1, floor(exp1+0.5)))
#post1.plot()


print("\n* Second bet")
#
# C : Red or Yellow
# D : Yellow or Black
#
bet2_domain = ['C', 'D']
C_bet = Predicate([1,0], bet2_domain)
D_bet = Predicate([0,1], bet2_domain)

#
# Probability of C: 1/3 + 2/3 * y/(2*N)
# Probability of D: 2/3
# Normalisation gives the probability used below for C.
#
bet2 = chan_fromklmap(lambda y: flip((N+y)/(3*N+y), bet2_domain),
                         yellow_domain,
                         bet2_domain)

print("Half black prior distribution: ", bet2 >> halfway_prior )
print("Uniform prior distribution: ", bet2 >> unif_prior )

#
# 0 = C_bet, 1 = D_bet
#
bet2_list = [0]*18 + [1]*41

post2 = unif_prior
for i in bet2_list:
    bet_pred = C_bet if i==0 else D_bet
    post2 = post2 / (bet2 << bet_pred)

print("Posterior distribution: ", bet2 >> post2 )
exp2 = RandVar(lambda x: x, yellow_domain).exp(post2)
print("Expected Yellow value: ", exp2)
print("Distribution for y =", floor(exp2+0.5), " is ",
      bet2 >> unit_disc_state(2*N+1, floor(exp2+0.5)))
#post2.plot()


print("\nMachina paradox")
print("===============\n")

machina_domain = [range(51), range(52)]

mach1 = chan_fromklmap(lambda x,y: 
                          flip(772650/(5151*(x+50) + 5050*(y+51) + 772650), 
                                  [1,2]),
                          machina_domain,
                          [1,2])

half_min_mach_prior = unit_disc_state(51,25) @ unit_disc_state(52,25)
half_plus_mach_prior = unit_disc_state(51,25) @ unit_disc_state(52,26)
unif_mach_prior = uniform_disc_state(51) @ uniform_disc_state(52)

print("* First bet")
print("Just below half: ", mach1 >> half_min_mach_prior )
print("Just above half: ", mach1 >> half_plus_mach_prior )
print("Uniform: ", mach1 >> unif_mach_prior )

mach_bet1_list = [1,2,1,1,1,2]

mach1_post = unif_mach_prior
for i in mach_bet1_list:
    pred = Predicate([1,0], [1,2]) if i==1 else Predicate([0,1], [1,2])
    mach1_post = mach1_post / (mach1 << pred)

print("Posterior: ",  mach1 >> mach1_post )
print("Expected values of 1 and 3: ",
      RandVar(lambda *x: x, machina_domain).exp(mach1_post))

print("\n* Second bet")

mach2 = chan_fromklmap(lambda x,y: 
                          flip((5151*x + 515100 + 5050*y) / 
                                  (15453*x + 772650 + 15150*y), 
                                  [3,4]),
                          machina_domain,
                          [3,4])

print("Just below half: ", mach2 >> half_min_mach_prior )
print("Just above half: ", mach2 >> half_plus_mach_prior )
print("Uniform: ", mach2 >> unif_mach_prior )

mach_bet2_list = [3,4,3,3,3,4]

mach2_post = unif_mach_prior

for i in mach_bet2_list:
    pred = Predicate([1,0], [3,4]) if i==3 else Predicate([0,1], [3,4])
    mach2_post = mach2_post / (mach2 << pred)

print("Posterior: ",  mach2 >> mach2_post )
print("Expected values of 1 and 3: ",
      RandVar(lambda *x: x, machina_domain).exp(mach2_post))



"""
