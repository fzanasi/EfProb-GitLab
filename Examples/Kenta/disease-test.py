from efprob_dc import *

# Very simple bayesian network of the form: D --> P
D = ['D', '~D']
P = ['P', '~P']
disease = state([0.01, 0.99], D)
sensitivity = [0.9, 0.05]
test = channel([sensitivity, [1.0 - v for v in sensitivity]],
               D, P)

positive = predicate([1, 0], P)

print("Prior state on disease:")
print(" ", disease)

print("Prior state on test result:")
print(" ", test >> disease)

print("Predicate on disease that test result is positive:")
print(" ", test << positive)

print("Probability that the test is positive:")
print(" ", test >> disease >= positive)

print("Equivalently:")
print(" ", disease >= test << positive)

print("Posterior state on disease given the test result is poisitve:")
print(" ", disease / (test << positive))
