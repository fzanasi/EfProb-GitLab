from efprob_dc import *

# Example from slide 16 in:
# http://users.math.yale.edu/users/gw289/CpSc-445-545/Slides/CPSC445%20-%20Topic%2006%20-%20Bayesian%20Classification.pdf

#
# To be used as "dependent" Bayesian classifier
#

Exercise = bn_prior(0.7)
Diet = bn_prior(0.25)

HeartDisease = cpt(0.25, 0.45, 0.55, 0.75)
HeartBurn = cpt(0.2, 0.85)

BloodPressure = cpt(0.85, 0.2)
ChestPain = cpt(0.8, 0.6, 0.4, 0.1)

network = (BloodPressure @ ChestPain) \
          * (copy(bnd) @ idn(bnd)) \
          * (HeartDisease @ HeartBurn) \
          * (idn(bnd) @ copy(bnd))

prior = (Exercise @ Diet)
PressurePain = network >> prior

print("\nPrior pressure and pain probabilities")
print( PressurePain % [1,0] )
print( PressurePain % [0,1] )

print("\nBelief update")

inv = network.inversion(prior)

print("\nObserving pressure and pain gives updated exercise and diet distributions:")
print( inv('t','t') )
print( prior / (network << tt @ tt) )

print("\nObserving pressure and no pain gives updated exercise and diet distributions:")
print( inv('t','f') )
print( prior / (network << tt @ ff) )

print("\nObserving only pain, via adapted network")
print( prior / (network << tt @ truth(bnd)) )
print( ((idn(bnd) @ discard(bnd)) * network).inversion(prior)('t') )

