from unifiedprob import *

def states():

    print("\nSection: States\n")

    u = uniform_state(R(0,5))
    print( u )
    #u.plot()

    print("\n===\n")



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

