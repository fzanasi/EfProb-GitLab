#
# Example from:
# https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Sex_classification
#
from math import sqrt
import statistics as st

from efprob_dc import *


male_data = [[6, 5.92, 5.58, 5.92],
             [180, 190, 170, 165],
             [12, 11, 12, 10]]
female_data = [[5, 5.5, 5.42, 5.75],
               [100, 150, 130, 150],
               [6, 8, 7, 9]]

male_means = [st.mean(d) for d in male_data]
male_vars = [st.variance(d) for d in male_data]
print(male_means, male_vars)

female_means = [st.mean(d) for d in female_data]
female_vars = [st.variance(d) for d in female_data]
print(female_means, female_vars)

sample = [6, 130, 8]

MF = ['M', 'F']
prior = uniform_state(MF)
# Gaussian assumption
chans = [chan_from_states([gaussian_state(male_means[i],
                                          sqrt(male_vars[i])),
                           gaussian_state(female_means[i],
                                          sqrt(female_vars[i]))],
                          MF)
         for i in range(3)]

likelihood = (chans[0].get_likelihood(sample[0])
              & chans[1].get_likelihood(sample[1])
              & chans[2].get_likelihood(sample[2]))
print(likelihood)

post = prior / likelihood
print(post)
