# Example from:
# https://probmods.org/conditioning.html#example-causal-inference-in-medical-diagnosis

from efprob_dc import *

bdom = [True, False]

breast_cancer = flip(0.01)
benign_cyst = flip(0.2)
prior = breast_cancer @ benign_cyst

test_cancer = chan_fromklmap(lambda x: flip(0.8) if x else flip(0),
                             bdom, bdom)
## same as:
# test_cancer = channel([0.8, 0.0,
#                        0.2, 1.0], bdom, bdom)
test_cyst = chan_fromklmap(lambda x: flip(0.5) if x else flip(0),
                           bdom, bdom)
## same as:
# test_cyst = channel([0.5, 0.0,
#                      0.5, 1.0], bdom, bdom)
positive_mammogram = pred_fromfun(lambda x, y: x or y, [bdom, bdom])

## same as:
# positive_mammogram = predicate([1, 1, 1, 0], [bdom, bdom])

posterior = prior / ((test_cancer @ test_cyst) << positive_mammogram)
breast_cancer_post = posterior % [1, 0]

print(breast_cancer_post)
