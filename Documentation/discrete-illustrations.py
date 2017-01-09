from efprob_dc import *

def states():

    print("\nSection: States\n")

    print("* fair flip state, with domain True, False")
    fflip = State([0.5, 0.5], [True, False])
    print( fflip )

    print("\n===\n")

    print("* fair flip state, with domain H, T")
    fflipHT = State([0.5, 0.5], ['H', 'T'])
    print( fflipHT )

    print("\n===\n")

    print("* flip function")
    def flip(r): return State([r, 1-r], [True,False])
    print( flip(0.2) )

    print("\n===\n")

    print("* fair dice state")
    fdice = State([1/6,1/6,1/6,1/6,1/6,1/6], [1,2,3,4,5,6])
    print(fdice)

    print("\n===\n")

    print("* fair dice uniform state")
    fdice = uniform_state([1,2,3,4,5,6])
    print(fdice)

    print("\n===\n")

    print("* fair dice uniform state, dots range")
    fdice = uniform_state(['*','**','***','****','*****','******'])
    print(fdice)

    print("\n===\n")

    print("* discrete state example, length 5")
    print( uniform_disc_state(5) )

    print("\n===\n")

    print("* unit state example")
    print( unit_disc_state(4,2) )

    print("\n===\n")

    print("* random discrete state example, twice length 5")
    print( random_disc_state(5) )
    print( random_disc_state(5) )

    print("\n===\n")

    print("* poisson state example, length 20")
    print( poisson(3,20) )


def operations_on_states():

    print("\nSubsection: Operations on states\n")

    print("* two flips in parallel")
    def flip(r): return State([r, 1-r], [True,False])
    print( flip(0.2) )
    print( flip(0.7) )
    print( flip(0.2) @ flip(0.7) )

    print("\n===\n")

    print("* flip and uniform in parallel")
    print( flip(0.2) @ uniform_disc_state(4) )

    print("\n===\n")

    print("* flip and uniform and flip in parallel")
    print( flip(0.2) @ uniform_disc_state(4) @ flip(0.8) )

    print("\n===\n")

    print("* iterated flip-product, 3 times")
    print( flip(0.2) ** 3 )

    print("\n===\n")

    print("* convex sum")
    print( convex_state_sum((0.2,flip(0.3)), (0.5,flip(0.8)), (0.3,flip(1))) )

    print("\n===\n")

    print("* marginalisation")
    s = random_disc_state(3)
    t = random_disc_state(2)
    print( s )
    print( t )
    print( s @ t )
    print( (s @ t) % [1,0] )
    print( (s @ t) % [0,1] )

    print("\n===\n")

    print("* two-out-of-three marginalisation")
    print( (random_disc_state(100) @ flip(0.5) @ uniform_disc_state(2)) % [0,1,1] )

def excursion():

    print("\nSubsection: An excursion on domains of states\n")

    print(" * domain of flip")
    print( flip(0.8).dom )

    print("\n===\n")

    print(" * domain of product")
    print( (flip(0.8) @ uniform_disc_state(5)).dom )

    print("\n===\n")

    print("* disc of domain")
    print( (flip(0.8) @ uniform_disc_state(5)).dom.disc )

    print("\n===\n")

    print("* constructed product state")
    s = State([0.1,0.1,0.1,0.7], [[True,False], [0,1]])
    print( s )


def predicates():

    print("\nSection: Predicates\n")


def conditioning():

    print("\nSubsection: Conditioning\n")

    print("* Dice conditionings")
    fdice = uniform_state([1,2,3,4,5,6])
    even = Predicate([0,1,0,1,0,1], [1,2,3,4,5,6])
    atmost4 = Predicate([1,1,1,1,0,0], [1,2,3,4,5,6])
    print("Validity of even and atmost4 ", 
          fdice >= even, fdice >= atmost4 )

    print("\n===\n")

    print("* even, given atmost4")
    print( fdice / atmost4 >= even  )

    print("\n===\n")

    print("* dice, given atmost4")
    print( fdice / atmost4 )

    print("\n===\n")

    print("* dice, given even")
    print( fdice / even )
    print( fdice / even >= atmost4 )

    print("\n===\n")

    print("* local conditioning check")
    s = random_disc_state(2)
    t = random_disc_state(3)
    p = random_disc_pred(2)
    print( (s @ t) / (p @ truth(t.dom)) )
    print( (s / p) @ t )

    print("\n===\n")

    print("* Church medical example")
    prior = flip(0.01) @ flip(0.005) @ flip(0.2) @ flip(0.1) @ flip(0.1)
    up = Predicate([1,0], [True,False])
    W = truth([True,False])
    LC = up @ W @ W @ W @ W
    TB = W @ up @ W @ W @ W
    CO = W @ W @ up @ W @ W
    SF = W @ W @ W @ up @ W
    OT = W @ W @ W @ W @ up
    cough = 0.5 * CO | 0.3 * LC | 0.7 * TB | 0.01 * OT
    fever = 0.3 * CO | 0.5 * SF | 0.2 * TB | 0.01 * TB
    chest_pain = 0.4 * LC | 0.5 * TB | 0.01 * OT
    short_breath = 0.4 * LC | 0.5 * TB | 0.01 * OT
    post = prior / (cough & fever & chest_pain & short_breath)
    print( post % [1, 1, 0, 0, 0] )


    print("\n===\n")

    print("* Conditioning introduced entwinedness")
    s = flip(0.6) @ flip(0.8)
    print( s )
    p = Predicate([1, 1, 1, 0], s.dom)
    t = s / p
    print( t )
    print( (t % [1,0]) @ (t % [0,1]) )

    print("\n===\n")

    print("* Law of total probability")
    s = random_disc_state(4)
    p = random_disc_pred(4)
    # scaling is used to make sure these predicates are summable
    q1 = 0.5 * random_disc_pred(4)
    q2 = 0.5 * random_disc_pred(4)
    q3 = ~(q1 + q2)
    print("total probalility formula:", 
          (s / q1 >= p) * (s >= q1) + 
          (s / q2 >= p) * (s >= q2) + 
          (s / q3 >= p) * (s >= q3) )
    print("equivalently, by Bayes:",
          (s >= q1 & p) + (s >= q2 & p) + (s >= q3 & p) )
    print("the predicat's validity, directly:", s >= p )



def random_variables():

    print("\nSection: Random Variables\n")

    print("* umbrella sales expectation")
    rain_state = flip(0.3)
    umbrella_sales_rv = RandVar(lambda x: 100 if x else -20, [True,False])
    print( umbrella_sales_rv.exp(rain_state) )

    print("\n===\n")

    print("* two-dice expectation")
    fdice = uniform_state([1,2,3,4,5,6])
    twodice = fdice @ fdice
    sum_rv = RandVar(lambda x,y: x+y, twodice.dom)
    print( sum_rv.exp(twodice) )

    print("\n===\n")

    print("* Even-odd expectation")
    even = Predicate([0,1,0,1,0,1], [1,2,3,4,5,6])
    odd = ~even
    print( sum_rv.exp( twodice / (even @ even) ) )
    print( sum_rv.exp( twodice / (odd @ odd) ) )

    print("\n===\n")

    print("* Dice sums expectation")
    def sums_exp(n): return RandVar(lambda *xs: sum(xs),
                                    (fdice ** n).dom).exp(fdice ** n)
    print( sums_exp(1) )
    print( sums_exp(2) )
    print( sums_exp(3) )
    #print( sums_exp(8) )

    print("\n===\n")

    print("* Dice even sums expectation")
    def even_sums_exp(n): return RandVar(lambda *xs: sum(xs),
                                         (fdice ** n).dom).exp( (fdice ** n) / (even ** n) )
    print( even_sums_exp(1) )
    print( even_sums_exp(2) )
    #print( even_sums_exp(8) )


    print("\n===\n")

#     print("* Conditional expectation")
#     # https://www.ma.utexas.edu/users/gordanz/notes/conditional_expectation.pdf
#     domain = ['a', 'b', 'c', 'd', 'e', 'f']
#     unif = uniform_state(domain)
#     X = RandVar(lambda x: 1 if x == 'a' else 
#                 3 if x == 'b' or x == 'c' else
#                 5 if x == 'd' or x == 'e' else
#                 7,
#                 domain)
#     Y = RandVar([1,3,3,5,5,7], domain)
#     print( X.exp(unif) )
#     print( Y.exp(unif) )
#     Y = RandVar(lambda x: 2 if x == 'a' or x == 'b' else 
#                 1 if x == 'c' or x == 'd' else
#                 7,
#                 domain)
#     Z = RandVar([3,3,3,3,2,2], domain)
#     X = RandVar([1,3,3,5,5,7], domain)
#     Y = RandVar([2,2,1,1,7,7], domain)
#     #print("Expected values of X,Y: ", X.exp(unif), Y.exp(unif) )
#     #print( Y.funs[0]('f') )
#     print( Y.funs[5] )
#     print( Predicate([1 if Y.funs[x]==Y.funs[3] else 0 for x in range(len(domain))], 
#                          domain) )
#     for i in range(len(domain)):
#         print(domain[i], 
# #              X.exp(unif / Predicate([1 if Y.funs[0](x)==Y.funs[0](domain[i]) else 0 for x in domain], 
#               X.exp(unif / Predicate([1 if Y.funs[x]==Y.funs[i] else 0 for x in range(len(domain))], 
#                          domain)) )






def channels():

    print("\nSection: Channels\n")

    c = cpt(0.2, 0.5)
    s = bn_prior(random.uniform(0,1))
    print( s )
    print( (graph(c) >> s) % [1,0] )
    print( c >> s )
    print( (graph(c) >> s) % [0,1] )



def state_pred_transformation():

    print("\nSubsection: State and predicate transformation\n")

    print("* Disease-test")
    disease_domain = ['D', '~D']
    prior_disease_state = State([1/100, 99/100], disease_domain)
    disease_pred = Predicate([1,0], disease_domain)

    test_domain = ['T', '~T']
    test_pred = Predicate([1,0], test_domain)
    sensitivity = Channel([[9/10, 1/20], 
                           [1/10, 19/20]], disease_domain, test_domain)
    print( sensitivity.dom )
    print( sensitivity.cod )
    print( sensitivity.array )
    print( sensitivity << test_pred )
    print( prior_disease_state >= sensitivity << test_pred )
    print( sensitivity >> prior_disease_state >= test_pred )
    print( prior_disease_state / (sensitivity << test_pred) )

    print("\n===\n")

    print("* Genetic hidden Markov model")
    ACGT = ['A', 'C', 'G', 'T']
    s0 = State([0.3, 0.2, 0.1, 0.4], ACGT)
    A = Predicate([1,0,0,0], ACGT)
    C = Predicate([0,1,0,0], ACGT)
    G = Predicate([0,0,1,0], ACGT)
    T = Predicate([0,0,0,1], ACGT)
    print( s0 >= A )

    trs = Channel([[0.1, 0.3, 0.3, 0.3],
                   [0.3, 0.1, 0.3, 0.3],
                   [0.3, 0.3, 0.1, 0.3],
                   [0.3, 0.3, 0.3, 0.1]], ACGT, ACGT)
    obs = Channel([[0.85, 0.05, 0.05, 0.05],
                   [0.05, 0.85, 0.05, 0.05],
                   [0.05, 0.05, 0.85, 0.05],
                   [0.05, 0.05, 0.05, 0.85]], ACGT, ACGT)

    s1 = trs >> (s0 / (obs << C))
    print( s1 )
    s2 = trs >> (s1 / (obs << A))
    print( s2 )
    s3 = trs >> (s2 / (obs << A))
    print( s3 )
    s4 = trs >> (s3 / (obs << A))
    print( s4  )
    s5 = trs >> (s4 / (obs << G))
    print( s5 )


def seq_par_composition():

    print("\nSubsection: Sequential and parallel composition\n")

    # print("* Disease-mood state")
    # disease_domain = ['D', '~D']
    # mood_domain = ['M', '~M']
    # prior = State([0.05, 0.5, 0.4, 0.05], [disease_domain, mood_domain])
    # print( prior )
    # print( prior % [0,1] )

    # print("\n===\n")

    # print("* Updated disease-mood state")
    # test_domain = ['T', '~T']
    # test_pred = Predicate([1,0], test_domain)
    # sensitivity = Channel([[9/10, 1/20], 
    #                        [1/10, 19/20]],
    #                       disease_domain, 
    #                       test_domain)
    # post = prior / ((sensitivity << test_pred) @ truth(mood_domain))
    # print( post % [0,1] )

    # print("\n===\n")

    # print("* Updated disease-mood state, via channel")


def bayesian_networks():

    print("\nSection: Bayesian networks\n")

    print("* Paper acceptance illustration")
    time = bn_prior(0.4)
    skill = bn_prior(0.7)
    prior = time @ skill
    well_written = cpt(0.8, 0.3)
    strong_results = cpt(0.9, 0.6, 0.4, 0.1)
    positive_reviews = cpt(0.8, 0.5, 0.6, 0.1)
    member_champion = cpt(0.4, 0.1, 0.3, 0.0)
    acceptance = cpt(1.0, 0.7, 0.8, 0.1)
    print( well_written.dom )
    print( strong_results.dom )

    bn = acceptance \
         * (positive_reviews @ member_champion) \
         * (idn(bnd) @ swap(bnd,bnd) @ idn(bnd)) \
         * (copy(bnd) @ copy(bnd)) \
         * (well_written @ strong_results) \
         * (copy(bnd) @ idn(bnd))
    print( prior >= (bn << tt) )
    E = Predicate([0.2, 0.4, 0.3, 0.6], prior.dom)
    print( bn >> prior/E )
    print( bn >> prior/E >= tt )
    print( prior/E >= bn << tt )

    print("\n===\n")

    print("* Channel versus logic")

    prior = bn_prior(0.25)
    B_chan = cpt(0.4, 0.1)
    C_chan = cpt(0.2, 0.3)

    A_form = tt
    B_form = 0.4 * A_form | 0.1 * ~A_form
    C_form = 0.2 * B_form | 0.3 * ~B_form

    print( prior >= tt )
    print( prior >= A_form )

    print("\n===\n")

    print("* Event B")
    print( prior >= B_chan << tt )
    print( prior >= B_form )

    print("\n===\n")

    print("* Event C")
    print( prior >= B_chan << (C_chan << tt) )
    print( prior >= C_form )

    print("\n===\n")

    print("* B and C example")
    print( prior >= B_chan << (tt & C_chan << tt) )
    print( prior >= (B_chan << tt) & (B_chan << (C_chan << tt)) )
    print( prior >= B_form & C_form )





def main():
    #states()
    #operations_on_states()
    #excursion()
    #predicates()
    #operations_on_predicates()
    #validity()
    #conditioning()
    #random_variables()
    channels()
    #state_pred_transformation()
    #seq_par_composition()
    #bayesian_networks()

if __name__ == "__main__":
    main()

