from efprob_dc import *

A_dom = ['a', '~a']
B_dom = ['b', '~b']

diseases = State([1/6, 1/4, 1/3, 1/4], [A_dom, B_dom])

A_test = chan_fromklmap(lambda x: flip(0.9) if x=='a' else flip(0.05),
                        A_dom,
                        [True, False])

pos_test = Predicate([1,0], [True, False])

print("\nJoint diseases example, with cross-over influence")
print("-------------------------------------------------")

print("\n* Original joint A-B diseases, with B marginal")
print( diseases )
print( diseases % [0,1] )

print("\n* Updated B probability after positive A test")
#
# Exact outcome: 40/97 = 0.41237113402061853
#
print( (diseases / ((A_test << pos_test) @ truth(B_dom))) % [0,1] )


print("\nBayesian disease network")
print("------------------------")

genetic_heredity = bn_prior(1/50)
environmental_factors = bn_prior(1/10)

disease = cpt(9/10, 8/10, 4/10, 0)
test = cpt(9/10, 1/20)

symptoms = cpt(9/10, 1/15)
healt_care = cpt(4/5, 1/10)

print("\n* Initial disease probability")
print( disease >> (genetic_heredity @ environmental_factors) )

print("\n* Positive test probability")
print( (disease >> (genetic_heredity @ environmental_factors)) 
       >= (test << bn_pos_pred) )

print("\n* Disease probability after positive test")
print( (disease >> (genetic_heredity @ environmental_factors)) 
       / (test << bn_pos_pred) )

transition_pred = bn_pred(0.88, 0.1)

print("\nTO BE CHECKED: the next three points do not coincide with the outcomes in the paper")

print("\n* Initial state, updated with genetic transition predicate")
print( (genetic_heredity @ environmental_factors) 
       / (transition_pred @ truth(bnd)) )

print("\n* Initial state transformed by disease and test")
print( (test * disease) >> (genetic_heredity @ environmental_factors) )

print("\n* Updated initial state transformed by disease and test")
print( (test * disease) >> ((genetic_heredity @ environmental_factors)
                            / (transition_pred @ truth(bnd))) )


print("\n* Conditioning creates entwinedness")
p_create = bn_pred(0.85, 0.2)
w = (genetic_heredity @ environmental_factors) / (disease << p_create)
print("conditioned joint state: ", w )
print("first marginal: ", w % [1,0] )
print("second marginal: ", w % [0,1] )
print("product of marginals: ", (w % [1,0]) @ (w % [0,1]) )


print("\n* Influence between marginals of joint states")
sigma = State([1/3, 1/4, 1/6, 1/4], bnd + bnd)
print(sigma)
print("second marginal of sigma: ", 
      sigma % [0,1] )
print("sigma updated with q: ", 
      sigma / (bn_pos_pred @ truth(bnd)) )
print("second marginal of updated sigma: ", 
      (sigma / (bn_pos_pred @ truth(bnd))) % [0,1] )

print("\n* Direct influences")
omega = bn_prior(4/5)
rho = bn_prior(1/2)
sigma = bn_prior(1/5)
print("influence on omega: ", dir_infl(test << bn_pos_pred, omega) )
print("influence on rho: ", dir_infl(test << bn_pos_pred, rho) )
print("influence on sigma: ", dir_infl(test << bn_pos_pred, sigma) )

print("\n* Serial connections")
omega = bn_prior(1/100)
#
# Exact outcome: 18/117 = 0.15384615384615385
#
print("omega updated with positive test: ", omega / (test << bn_pos_pred) )
r = random.uniform(0,1)
q = bn_pred(r, 1-r)
print("with arbitrary predicate added: ",
      omega / (test << (bn_pos_pred & (healt_care << q))) )

print("\n* Non-blocking with non-sharp predicates")
p = bn_pred(1/3, 1/4)
q = bn_pred(1/5, 1)
print( omega / (test << p) )
print( omega / (test << (p & (healt_care << q))) )

print("\n* Collider connection, with crossover influences:")
print( cross_infl(bn_neg_pred, 
                  genetic_heredity @ environmental_factors) )
print( cross_infl(bn_neg_pred, 
                  (genetic_heredity @ environmental_factors) 
                  / (disease << p_create)) )
print( cross_infl(bn_neg_pred, 
                  (genetic_heredity @ environmental_factors) 
                  / (disease << bn_pos_pred)) )


