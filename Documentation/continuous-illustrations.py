from efprob_dc import *
from math import *

def states():

    print("\nSection: States\n")

    print("* print uniform state")
    u = uniform_state(R(0,5))
    print( u )
    #u.plot()

    print("\n===\n")

    print("* Validity of truth predicate")
    s = State(lambda x: 0.5 * x, R(0,2))
    print( s >= truth(R(0,2)) )

    print("\n===\n")

    print("* Infinite domain")
    u = State(lambda x: e ** -x, R(0, inf))
    print( u >= truth(u.dom) )
    v = State(lambda x: e ** x, R(-inf, 0))
    print( v >= truth(v.dom) )
    w = State(lambda x: 1/sqrt(pi) * e ** (-x*x), R)
    print( w >= truth(w.dom) )

    print("\n===\n")

    print("* Product domains")
    s = State(lambda x,y: 4 * x * y, [R(0,1), R(0,1)])
    print( s >= truth(s.dom) )
    t = State(lambda x,y,z: 8/15 * x * y * z, [R(0,1), R(1,2), R(2,3)])
    print( t >= truth(t.dom) )

    print("\n===\n")

    print("* Gaussian distributions")
    s = gaussian_state(2, 1)
    print( s )
    t = gaussian_state(2, 1, R(-10,10))
    print( t >= truth(t.dom) )
    #t.plot()


def predicates():

    print("\nSection: Predicates\n")

    print("* Predicate definition")
    q = Predicate(lambda x: 1 if x < 0 else 0.5, R(-1,1))
    u = uniform_state(R(-1,1))
    print( u >= q )
    s = u @ flip(0.4)
    print( s >= ~q @ yes_pred )

def conditioning():

    print("\nSubsection: Conditioning\n")

    print("* Archeological tomb")
    dom = R(0,100)
    p1 = Predicate(lambda x: 0.6 * e ** (-1/2000 * (x - 20) ** 2), dom)
    p2 = Predicate(lambda x: 0.9 * e ** (-1/1500 * (x - 50) ** 2), dom)
    p3 = Predicate(lambda x: 0.8 * e ** (-1/1000 * (x - 90) ** 2), dom)
    u = uniform_state(dom)
    #u.plot()
    findings = [1, 3, 2, 1, 1]
    def pred(f): return p1 if f==1 else p2 if f==2 else p3
    for f in findings: u = u / pred(f)
        #u.plot()
    print("Expected age: ", u.expectation() )

def operations_on_states():

    print("\nSubsection: Operations on states\n")

    print("* print product of states")
    s = uniform_state(R(-1,1)) @ uniform_state(R(5,10))
    print( s )
    t = flip(0.3) @ s
    print( t )


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
    print( s.expectation(randvar_fromfun(lambda r: r, bias_dom)) )


def all():
    states()
    operations_on_states()
    predicates()
    conditioning()
    state_pred_transformation()



def main():
    #all()
    #states()
    #operations_on_states()
    #predicates()
    conditioning()
    #state_pred_transformation()


if __name__ == "__main__":
    main()

