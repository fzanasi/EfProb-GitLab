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




def main():
    states()
    operations_on_states()
    #basic_states()
    #predicates()
    #operations_on_predicates()
    #validity()
    #conditioning()
    #weakening()
    #state_transformation()
    #predicate_transformation()

if __name__ == "__main__":
    main()

