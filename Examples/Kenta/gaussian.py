#
# Gaussian posterior estimation
# from an Anglican worksheet
#

import math
from efprob_dc import *

# We know the standard deviation
sigma = math.sqrt(2)

# Prior distribution of the mean
mu = gaussian_state(1, math.sqrt(5), R)

chan = chan_fromklmap(lambda x: gaussian_state(x, sigma, R),
                      R, R)
## same as:
# import scipy.stats as stats
# chan = channel(lambda xs, ys:
#                stats.norm.pdf(ys[0], loc=xs[0], scale=sigma),
#                R, R)

data = [9, 8]

# Plot the prior
mu.plot(R(-20, 20))

# Conditioning using 'Gaussian' predicate
pred = chan << gaussian_pred(data[0], 0.1, R)
for d in data[1:]:
    pred = pred & (chan << gaussian_pred(d, 0.1, R))
mu_post = mu / pred

## This is the same as iterated conditioning below, which is slower.
# mu_post = mu
# for d in data:
#     mu_post = mu_post / (chan << gaussian_pred(d, 0.1, R))

mu_post.plot(R(-20, 20))

# Use likelihood (it's precise and faster)
lik = chan.get_likelihood(data[0])
for d in data[1:]:
    lik = lik & chan.get_likelihood(d)
mu_post2 = mu / lik

mu_post2.plot(R(-20, 20))

# not implemented yet...

# # Compute the posterior
# post_mu = chan.update(mu, [(d,) for d in data])
# post_mu.plot()

# # Compute the posterior by iteration (slow)
# post_mu2 = mu
# for d in data:
#     post_mu2 = chan.inversion(post_mu2).getstate(d)

# post_mu2.plot()
