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
# From: https://arxiv.org/abs/1208.2354
#
# Ambiguity in economics is typically considered as a situation
# without a unique probability model describing it as opposed to risk,
# which is defined as a situation with such a probability model
# describing it.
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



print("\nMachina paradox")
print("===============\n")


machina_domain = [range(51), range(52)]
ball_domain = [1, 2, 3, 4]

mach1 = chan_fromklmap(lambda x,y: State([x/50, 1-x/50, y/51, 1 - y/51],
                                         ball_domain),                       
                       machina_domain,
                       ball_domain)

half_min_mach_prior = unit_disc_state(51,25) @ unit_disc_state(52,25)
half_plus_mach_prior = unit_disc_state(51,25) @ unit_disc_state(52,26)
unif_mach_prior = uniform_disc_state(51) @ uniform_disc_state(52)

print("* First bet")
print("Just below half: ", mach1 >> half_min_mach_prior )
print("Just above half: ", mach1 >> half_plus_mach_prior )
print("Uniform: ", mach1 >> unif_mach_prior )

f1 = RandVar(lambda x: 202 if x==1 or x==2 else 101, ball_domain)
f2 = RandVar(lambda x: 202 if x==1 or x==3 else 101, ball_domain)
f3 = RandVar(lambda x: 303 if x==1 or x==2 else 
             202 if x==2 else
             101 if x==1 else 0, ball_domain)
f4 = RandVar(lambda x: 303 if x==1 or x==2 else 
             101 if x==2 else
             202 if x==1 else 0, ball_domain)

print("Expected values of bets 1, 2, 3, 4: ",
      f1.exp(mach1 >> unif_mach_prior),
      f2.exp(mach1 >> unif_mach_prior),
      f3.exp(mach1 >> unif_mach_prior),
      f4.exp(mach1 >> unif_mach_prior) )


print("\nMonty Hall problem")
print("==================\n")

prize_domain = ['G', 'C']


#
# Number of doors to choose from
#
N=100

#
# You pick door 0; all other options are combined
#

opendoors = chan_fromklmap(lambda x: flip(1) if x==0 else flip(0),
                           range(N),
                           [True, False])

print( opendoors >> uniform_disc_state(100) )

#print( prior / (pick << Predicate([1,1,0], [1,2,3])) )
