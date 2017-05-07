from efprob_dc import *

#
# Example based on Russell-Norvig book, chapter 15
#
# See also: http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

# General idea: there are channels trs_n : X_n -> X_{n+1} and obs_n :
# X_{n+1} -> Y_n with an initial state w_0.
# There is a list of observations y_i in Y_i, leading to successive states
# w_{i+1} = obs_{i}.inversion(trs_i >> w_i)(y_i)
#
# Possibly, the transition maps have an additional "time" argument,
# and the observations are timestamped, of the form (t_i, y_i), where
# t_i is the time difference (Delta) between y_{i-1} and y_i. Then:
# w_{i+1} = obs_{i}.inversion(trs_i(t_i) >> w_i)(y_i)

mu0 = 0
sigma0 = 2

x0 = gaussian_state(mu0, sigma0, R(-10,10))

def trs(sigma):
    return chan_fromklmap(lambda x: gaussian_state(x, sigma, R(-10,10)),
                          R(-10,10), R(-10,10))

noise = chan_fromklmap(lambda x: gaussian_state(x, 1, R(-10,10)),
                     R(-10,10), R(-10,10))

x0trs = trs(sigma0) >> x0
x1 = noise.inversion(x0trs)(2.5)

print("\nRussel-Norving, Table 15.8, approximately")
#x0.plot()
#x0trs.plot()
x1.plot()

mu1 = x1.expectation()
sigma1 = x1.st_deviation()

print(mu1, sigma1)

x1trs = trs(sigma1) >> x1
x2 = noise.inversion(x1trs)(4)

x2.plot()

mu2 = x2.expectation()
sigma2 = x2.st_deviation()

print(mu2, sigma2)


