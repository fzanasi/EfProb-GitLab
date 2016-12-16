# Example from:
# https://probmods.org/conditioning.html#example-causal-inference-in-medical-diagnosis

from functools import reduce
from efprob_dc import *

bdom = [True, False]

cold = flip (0.2)
stomach_flu = flip(0.1)
lung_cancer = flip(0.01)
TB = flip(0.005)
other = flip(0.1)

prior = cold @ stomach_flu @ lung_cancer @ TB @ other

def flipif_chan(r):
    return chan_fromklmap(lambda x: flip(r) if x else flip(0.0),
                          bdom, bdom)

def any_chan(n):
    return chan_fromklmap(lambda *xs: flip(1.0) if any(xs) else flip(0.0),
                          [bdom] * n, bdom)

cough = (any_chan(4)
         * (flipif_chan(0.5) # cold
            @ flipif_chan(0.3) # lung_cancer
            @ flipif_chan(0.7) # TB
            @ flipif_chan(0.01))) # other

fever = (any_chan(4)
         * (flipif_chan(0.3) # cold
            @ flipif_chan(0.5) # stomach_flu
            @ flipif_chan(0.2) # TB
            @ flipif_chan(0.01))) # other

chest_pain = (any_chan(3)
              * (flipif_chan(0.4) # lung_cancer
                 @ flipif_chan(0.5) # TB
                 @ flipif_chan(0.01))) # other

shortness_of_breath = (any_chan(3)
                       * (flipif_chan(0.4) # lung_cancer
                          @ flipif_chan(0.5) # TB
                          @ flipif_chan(0.01))) # other

def select_chan(*selectors):
    return reduce(lambda x, y: x @ y,
                  [idn(bdom) if s else discard(bdom) for s in selectors])

predT = predicate([1, 0], bdom)

## This will create huge array, which doesn't work!!
# pred = ((((cough * select_chan(1, 0, 1, 1, 1))
#           @ (fever * select_chan(1, 1, 0, 1, 1))
#           @ (chest_pain * select_chan(0, 0, 1, 1, 1))
#           @ (shortness_of_breath * select_chan(0, 0, 1, 1, 1)))
#          * (copy([bdom]*5) @ copy([bdom]*5))
#          * copy([bdom]*5))
#         << predT ** 4)

pred = ((cough * select_chan(1, 0, 1, 1, 1) << predT)
        & (fever * select_chan(1, 1, 0, 1, 1) << predT)
        & (chest_pain * select_chan(0, 0, 1, 1, 1) << predT)
        & (shortness_of_breath * select_chan(0, 0, 1, 1, 1) << predT))

posterior = prior / pred

lung_cancer_TB_post = posterior % [0, 0, 1, 1, 0]

print(lung_cancer_TB_post)
