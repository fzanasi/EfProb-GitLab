from efprob_dc import *

def states():

    print("\nSection: States\n")

    print("* print uniform state")
    u = uniform_state(R(0,5))
    print( u )
    #u.plot()

    print("\n===\n")

    print("* print product of states")
    s = uniform_state(R(-1,1)) @ uniform_state(R(5,10))
    print( s )
    t = flip(0.3) @ s
    print( t )



def operations_on_states():

    print("\nSubsection: Operations on states\n")


def state_pred_transformation():

    print("\nSubsection: State and predicate transformation\n")

    print("\n===\n")

    print("* Coin parameter learning")
    bias_dom = R(0,1)
    prior = uniform_state(bias_dom)
    chan = chan_fromklmap(lambda r: flip(r), bias_dom, bool_dom)
    print( chan >> prior )
    observations = [0,1,1,1,0,0,1,1]
    s = prior
    # s.plot()
    for ob in observations:
        pred = yes_pred if ob==1 else no_pred
        s = s / (chan << pred)
        # s.plot()
    print( chan >> s )
    print( randvar_fromfun(lambda r: r, bias_dom).exp(s) )




def main():
    #states()
    #operations_on_states()
    #basic_states()
    #predicates()
    #operations_on_predicates()
    #validity()
    #conditioning()
    state_pred_transformation()


if __name__ == "__main__":
    main()

