from efprob_dc import *

print("\nWaste example")

# From: Barry R. Cobb, Prakash P. Shenoy, Inference in Hybrid Bayesian
# Networks with Mixtures of Truncated Exponentials
# http://www.sciencedirect.com/science/article/pii/S0888613X0500040X
#
# Simplified version of: Probabilistic Networks and Expert Systems, 7.7.1
# (See the end of this file)

Rdom = R(-5,25)

# Burning regime: Stable or Unstable
B = flip(0.2)

# Filter state: Intact or Defective
F = flip(0.1)

# Type of waste: Industrial or Household
W = flip(0.8)

prior = B @ F @ W

E = chan_fromklmap(lambda f, w:
                   gaussian_state(1, 1, Rdom) if f and w else
                   gaussian_state(0, 1, Rdom) if f and not w else
                   gaussian_state(8, 1, Rdom) if not f and w else
                   gaussian_state(5, 1, Rdom), 
                   [bool_dom, bool_dom], Rdom)

# Figure 12 via:
#(E >> (F @ point_state(True, bool_dom))).plot()
#(E >> (F @ point_state(False, bool_dom))).plot()

D = chan_fromklmap(lambda b, e, w:
                   gaussian_state(8 + e, 4, Rdom) if b and w else
                   gaussian_state(7 + e, 4, Rdom) if b and not w else
                   gaussian_state(6 + e, 4, Rdom) if not b and w else
                   gaussian_state(5 + e, 4, Rdom), 
                   [bool_dom, Rdom, bool_dom], Rdom)

bn = D \
     * (idn(bool_dom) @ E @ idn(bool_dom)) \
     * (idn(bool_dom) @ idn(bool_dom) @ copy(bool_dom))

efficiency = E >> (F @ W)
dust = bn >> prior

# Figure 14 via:
#dust.plot(R(0,25))
#efficiency.plot(R(-5,20))

# Value in Cobb & Shenoy: 12.9401, 11.876. Last number is wrong
print("\nExpected dust value and variance: ", 
      dust.expectation(), 
      dust.variance() )

# Value in Cobb & Shenoy: 6.74014, 6.22196.
print("\nExpected efficiency value and variance: ", 
      efficiency.expectation(), 
      efficiency.variance() )

dis = bn.inversion(prior)(10)

print("\nDisintegrations:")
print("For B:", dis % [1,0,0] )
# Value in Cobb & Shenoy: 0.13
print("For F:", dis % [0,1,0] )
# Value in Cobb & Shenoy: 0.4836. Substantial difference!
print("For W:", dis % [0,0,1] )


# From: https://arxiv.org/abs/1301.6724
# See also: Barry R. Cobb, Prakash P. Shenoy, Inference in Hybrid Bayesian
# Networks with Mixtures of Truncated Exponentials

print("\nCrop-Subsidy-Price-Buy example")

# Subsidy

s = flip(0.3)

# Crop

dom = R(0,20)
c = gaussian_state(5,1,dom)

prior = s @ c

# Channels, for `price' and `buy'

p = chan_fromklmap(lambda x, y: gaussian_state(10-y, 1) if x 
                   else gaussian_state(20-y, 1),
                   [bool_dom, dom], dom)

b = chan_fromklmap(lambda x: flip(1 - 1/(1+math.exp(x-5))), dom, bool_dom)

# Computed states

price = p >> prior

# Values in Cobb & Shenoy: 11.9902, 22.9373

print("\nPrice expectation and variance: ",
      price.expectation(), price.variance() )

#price.plot()

buy = b >> price

# Value in Cobb & Shenoy: 0.849586

print( buy >= yes_pred )

#buy.plot()



"""

# Burning regime: Stable or Unstable
burning_dom = ['S', 'U']
burning = flip(0.85, burning_dom)

# Filter state: Intact or Defective
filter_dom = ['I', 'D']
filter = flip(0.95, filter_dom)

# Type of waste: Industrial or Household
waste_dom = ['I', 'H']
waste = flip(2/7, waste_dom)

prior = waste @ filter @ burning

# Filter efficiency: waste_dom @ filter_dom -> Rdom
Rdom = R(-10,10)
filt_eff = chan_fromklmap(lambda w, f:
                          gaussian_state(-3.2, 00.2, Rdom)
                          if w == 'H' and f == 'I' else
                          gaussian_state(-0.5, 0.1, Rdom)
                          if w == 'H' and f == 'D' else
                          gaussian_state(-3.9, 00.2, Rdom)
                          if w == 'I' and f == 'I' else
                          gaussian_state(-0.4, 0.1, Rdom),
                          [waste_dom, filter_dom], Rdom)

#(filt_eff >> (waste @ filter)).plot()

# Emission of CO2: burning_dom -> Rdom

co2_dom = chan_from_states([gaussian_state(-2, 0.1, Rdom),
                            gaussian_state(-1, 0.3, Rdom)], burning_dom)

print("Expected CO2 when burning is stable: ", 
      (co2_dom >> point_state('S', burning_dom)).expectation() )

# Emission of dust: burning_dom @ waste_dom @ filter_efficience -> Rdom

dust_em = chan_fromklmap(lambda w, e, b:
                         gaussian_state(6.5 + e, 0.03, Rdom) 
                         if b == 'S' and w == 'I' else
                         gaussian_state(6.0 + e, 0.04, Rdom)
                         if b == 'S' and w == 'H' else
                         gaussian_state(7.5 + e, 0.1, Rdom)
                         if b == 'U' and w == 'I' else
                         gaussian_state(7.0 + e, 0.1, Rdom),
                         [waste_dom, Rdom, burning_dom], Rdom)

bn = dust_em \
     * (idn(waste_dom) @ filt_eff @ idn(burning_dom)) \
     * (copy(waste_dom) @ idn(filter_dom) @ idn(burning_dom))

print(bn >> prior)

#(bn >> prior).plot()

"""
