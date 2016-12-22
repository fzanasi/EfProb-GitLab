from efprob_dc import *

#
# Urn example from the BLOG icjai 2005 paper:
#
# https://people.eecs.berkeley.edu/~russell/papers/ijcai05-blog.pdf
#

# 
# Domain of colours: blue and green, with associated predicates bleu, green
#
col = Dom(['B', 'G'])
blue = Predicate([1,0], col)
green = Predicate([0,1], col)

#
# State capturing urn with one ball, same likelihood of blue and green
#
s1 = State([0.5, 0.5], col)

#
# Random variable on domain of n balls, counting the number of blue
# balls; its codomain is range(n+1)
#
def blues_rv(n):
    return RandVar(lambda *xs: sum([1 if xs[i] == 'B' else 0 
                                    for i in range(len(xs))]),  col * n)

#
# Predicate for likelihood of blue on state of n balls
#
def blues(n):
    return blues_rv(n) << Predicate([i/n for i in range(n+1)], range(n+1))

print("\nLikelihood of drawing a blue ball from an urn with n balls")
for i in range(10):
    print(i+1, ": ", s1 ** (i+1) >= blues(i+1) )

#
# Predicate for noisy obsersation of the colour: it's wrong with
# probability 0.2
#
def noisy_blues(n):
    return blues_rv(n) << Predicate([0.8 * i/n + 0.2 * (n-i)/n
                                     for i in range(n+1)], range(n+1))

print("\nLikelihood of (noisy) observing a blue ball drawn from an urn with n balls")
for i in range(10):
    print(i+1, ": ", s1 ** (i+1) >= noisy_blues(i+1) )

#
# Auxiliary function for probability of iterated observation of
# predicate p in state s, via Bayes' rule.
#
def iter_prob(s, p, n):
    r = s >= p
    if n == 1:
        return r 
    return r * iter_prob(s/p, p, n-1)

#
# Predicate for likelihood of observing successively m blue drawn
# balls in urn with n balls; the drawn balls are returned into the
# urn.
#
def iter_noisy_blues(n, m):
    return iter_prob(s1 ** n, noisy_blues(n), m)

print("\nLikelihood of observing 5 consecutive blue balls in urn with n balls")
for i in range(10):
    print(i+1, ": ", iter_noisy_blues(i+1, 5) )

#
# Constant M, describing the number of urns, consecutively with 1, 2,
# ..., M balls
#
M = 8

#
# Uniform prior
#
uprior = State([1/M for i in range(M)], [i+1 for i in range(M)])

#
# Predicate on uniform prior, giving probability of observing 10
# successive blue balls in urn with i balls
#
upred = Predicate([iter_noisy_blues(i+1, 10) for i in range(M)], 
                  [i+1 for i in range(M)])

#
# See Figure 4 of paper mentioned at beginning
#
(uprior / upred).plot()

M=20

#
# Poisson prior
#
pprior = poisson(6, M)

#
# Predicate on poisson prior, giving probability of observing 10
# successive blue balls in urn with i balls
#
ppred = Predicate([0] + [iter_noisy_blues(i+1, 10) for 
                         i in range(M-1)], range(M))

#
# See Figure 4 of paper mentioned at beginning
#
pprior.plot()
(pprior / ppred).plot()

