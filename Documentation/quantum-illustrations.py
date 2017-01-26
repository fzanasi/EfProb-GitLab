#
# All illustration used in the Quantum chapters, organised per
# subsection, see the main method at the end.
#
#
from efprob_qu import *
from math import *

def states():
    print("\nSection: States\n")
    print("* printing basic states")
    print( ket(0) )
    print( ket(1) )
    print( plus )
    print( minus )



def operations_on_states():
    print("\nSubsection: Operations on states\n")

    print("* Printing product states")
    print( ket(0) @ ket(1) )
    
    print( ket(0) @ plus )

    print("\n===\n")

    print("* printing product states")
    print( ket(0,1,1) )

    print("\n===\n")

    print("* Printing product states")
    print( ket(0) ** 3 )

    print("\n===\n")

    print("* Printing product states")
    print( (plus @ ket(0)) ** 4 )

    print("\n===\n")

    print("* Printing convex sum example")
    print( convex_state_sum((0.2,ket(0)),(0.3,plus),(0.5,minus)) )

    print("\n===\n")

    # print("* Alternatively written convex sum")
    # print( (ket(0) + ket(1))(0.2) )

    print("\n===\n")

    print("* Marginalisation")
    print( ket(0) @ ket(1) @ plus @ minus % [1,0,1,0] )




def basic_states():
    print("\nSubsection: Basic states\n")

    print("* Probabilistic state, normalised")
    print( probabilistic_state(3,4,1) )

    print("\n===\n")

    print("* Classical flip")
    print( cflip(0.3) )

    print("\n===\n")

    print("* Product of these")
    print( cfflip @ probabilistic_state(2,2,1) )

    print("\n===\n")

    print("* Vector state")
    print( vector_state(1, complex(1, 2), -2) )

    print("\n===\n")

    print( vector_state(0.5 * sqrt(3), complex(0, 0.5)) )

    print("\n===\n")

    print("* Unit state")
    print( unit_state(3,2) )

    print("\n===\n")

    print("* Random state")
    print( random_state(3) )

    print("\n===\n")

    print("* Random probabilistic state")
    print( random_probabilistic_state(2) )




def predicates():

    print("\nSection: Predicates\n")

    print("* Falsity and truth")
    print( falsity(2) )

    print( truth(2,3) )

    print("\n===\n")

    print("* Probabilistic predicate, of length 4")
    print( probabilistic_pred(0.1,1,0.25,0.0) )

    print("\n===\n")

    print("* Unit predicate, of length 4")
    print( unit_pred(4,2) )

    print("\n===\n")



def operations_on_predicates():

    print("\nSubsection: Operations on predicates\n")

    print("* Random predicate and its orthocomplement")
    p = random_pred(2)
    print(p)
    print(~p)

    print("\n===\n")

    print("* Scaled probabilistic predicate")
    print( 0.3 * probabilistic_pred(0.2,0.5) )

    print("\n===\n")

    print("* Sum of unit predicates")
    print( unit_pred(4,2) + unit_pred(4,0) )

    print("\n===\n")

    print("* Bell states form a test")
    print( bell00.as_pred() + bell01.as_pred() + 
           bell10.as_pred() + bell11.as_pred() )

    print("\n===\n")

    print("* Also GHZ states form a test")
    print( ghz1.as_pred() + ghz2.as_pred() + ghz3.as_pred() + ghz4.as_pred() +
           ghz5.as_pred() + ghz6.as_pred() + ghz7.as_pred() + ghz8.as_pred() )


    print("\n===\n")

    print("* Parallel conjunction of truth and falsity")
    print( truth(2) @ falsity(3) )

    print("\n===\n")

    print("* Sequential conjunction of two probabilistic predicates")
    print( probabilistic_pred(0.2,0.8) & probabilistic_pred(0.4, 0.6) )

    print("\n===\n")

    print("* Weakening illustration")
    s = cflip(0.4)
    t = ket(0,0)
    p = unit_pred(2,1)
    print( s >= p )
    print( s @ t >= p @ truth(2,2) )



def validity():

    print("\nSubsection: Validity\n")

    print("* Validity example")
    v = vector_pred(0.5 * sqrt(3), complex(0, 0.5)) 
    print( v )
    print( ket(0) >= v )
    print( ket(1) >= v )

    print("\n===\n")

    print("* Validity preserves scaling example")
    # s = random_state(100)
    # p = random_pred(100)
    # print( s >= 0.3 * p )
    # print( 0.3 * (s >= p) )

    print("\n===\n")

    print("* Commutativity of sequential conjunction")
    p = unit_pred(2,0)
    q = plus.as_pred()
    print( ket(0) >= p & q )
    print( ket(0) >= q & p )

    print("\n===\n")

    print("* Validity of states")
    s = random_state(8)
    t = random_state(8)
    print( s >= t.as_pred() )
    print( t >= s.as_pred() )

    print("\n===\n")

    print("* Validity of vector states")
    v1 = np.random.rand(5)
    v2 = v1/np.linalg.norm(v1)
    w1 = np.random.rand(5)
    w2 = w1/np.linalg.norm(w1)
    s = vector_state(*v2)
    t = vector_state(*w2)
    print( s >= t.as_pred() )
    print( np.inner(v2, w2) ** 2 )

    print("\n===\n")

    print("* Bell table")
    v1 = vector_pred(1/sqrt(2), 1/sqrt(2))
    A1 = v1
    B1 = v1
    v2 = vector_pred(1/sqrt(2), 0.5/sqrt(2) * complex(1, sqrt(3)))
    A2 = v2
    B2 = v2
    print( A1 )
    print( A2 )
    print("")
    print( bell00 >= A1 @ B1, bell00 >= A1 @ ~B1, bell00 >= ~A1 @ B1, bell00 >= ~A1 @ ~B1 )
    print("")
    print( bell00 >= A1 @ B2, bell00 >= A1 @ ~B2, bell00 >= ~A1 @ B2, bell00 >= ~A1 @ ~B2 )
    print("")
    print( bell00 >= A2 @ B1, bell00 >= A2 @ ~B1, bell00 >= ~A2 @ B1, bell00 >= ~A2 @ ~B1 )
    print("")
    print( bell00 >= A2 @ B2, bell00 >= A2 @ ~B2, bell00 >= ~A2 @ B2, bell00 >= ~A2 @ ~B2 )



def conditioning():

    print("\nSubsection: Conditioning\n")

    print("* Bayes' rule illustration")
    s = random_state(2)
    p = random_pred(2)
    q = random_pred(2)
    print( s / p >= q )
    print( (s >= p & q) / (s >= p) )

    print("\n===\n")

    print("* polarisation illustration, in several steps")
    s = random_state(2)
    vert_filt = unit_pred(2,0)
    hor_filt = unit_pred(2,1)
    diag_filt = plus.as_pred()
    print( s >= vert_filt )
    s_vert = s / vert_filt

    print("\n===\n")

    print( s_vert >= vert_filt )
    print( s_vert >= hor_filt )

    print("\n===\n")

    print( s_vert >= diag_filt & hor_filt )

    print("\n===\n")

    print( s_vert >= hor_filt & diag_filt )

    print("\n===\n")

    print( s_vert / diag_filt >= hor_filt )

    print("\n===\n")

    print("* Failure of the law of total probability")

    print( s >= hor_filt )
    print( (s / diag_filt >= hor_filt) * (s >= diag_filt) + 
           (s / ~diag_filt >= hor_filt) * (s >= ~diag_filt) )
    print( (s >= diag_filt & hor_filt) + (s >= ~diag_filt & hor_filt) )


def random_variables():

    print("\nSubsection: Operations on random variables\n")

    print("* Types of scalar multiplications")
    print( type( ket(0) ) )
    print( type( 0.5 * ket(0).as_pred() ) )
    print( type( 5 * ket(0).as_pred() ) )

    print("\n===\n")

    print("* Types of scalar additions")
    print( type( unit_pred(2,0) ) )
    print( type( unit_pred(2,0) + unit_pred(2,0) ) )

    print("\n===\n")

    print("* Dice variance, quantum style")
    dice = uniform_probabilistic_state(6)
    points = 1 * unit_pred(6,0) + 2 * unit_pred(6,1) + 3 * unit_pred(6,2) \
             + 4 * unit_pred(6,3) + 5 * unit_pred(6,4) + 6 * unit_pred(6,5)
    print( dice >= points )
    print( points.variance(dice) )


def state_transformation():

    print("\nSubsection: State transformation\n")

    print("* plus and minus states via state transformation")
    plus = hadamard >> ket(0)
    minus = hadamard >> ket(1)
    print( plus )
    print( minus )

    print("\n===\n")

    print("* cnot channel applied to various ket combinations")
    print( cnot >> ket(0,0) )
    print( cnot >> ket(0,1) )
    print( cnot >> ket(1,0) )
    print( cnot >> ket(1,1) )


def predicate_transformation():

    print("\nSubsection: Predicate transformation\n")

    print("* Predicates used in Bell table, via predicate transformation")
    A1 = hadamard << unit_pred(2,0)
    print( A1 )
    angle = pi / 3
    A2 = phase_shift(angle) << A1
    print( A2 )

    print("\n===\n")

    print("* Transformations validity")
    p = random_pred(2)
    s = random_state(2)
    print( x_chan >> (hadamard >> s) >= p )
    print( (x_chan * hadamard) >> s >= p )
    print( s >= hadamard << (x_chan << p) )
    print( s >= (x_chan * hadamard) << p )


def structural_channels():
    
    print("\nSubsection: Structural channels\n")

    print("* Plus state via hadamard channel")
    c = (hadamard @ idn(2)) * cnot
    print( c >> ket(0,0) )

    print("\n===\n")

    print("* Outcome of discarding")
    print( discard(2) >> ket(0) )
    print( discard(2) >> ket(1) )

    print("\n===\n")

    print("* Projection, in various forms")
    s = random_state(2)
    print( s )
    t = random_state(2)
    #print( (proj1 * cnot) >> (s @ t) )
    print( ((idn(2) @ discard(2)) * cnot) >> (s @ t) )
    print( (cnot >> (s @ t)) % [1,0] )
    print( ((idn(2) @ discard(2)) * cnot) >> (s @ t) == (cnot >> (s @ t)) % [1,0] )

    print("\n===\n")

    print("* Ancilla added to a channel")
    c = hadamard
    s = random_state(2)
    print( (c @ ket(1).as_chan()) >> s )
    print( (c >> s) @ ket(1) )

    print("\n===\n")

    print("* Discard and ancilla")
    s = random_state(2)
    print( (discard(2) @ ket(0).as_chan()) >> s )
    print( (ket(0).as_chan() @ discard(2)) >> s )

def measurement():
    
    print("\nSubsection: Measurement, control and instruments\n")

    print("* Probabilistic outcome of measurement")
    p = random_pred(5)
    s = random_state(5)
    print( s >= p )
    print( meas_pred(p) >> s )

    print("\n===\n")

    print("* Classic channel taking out the diagonal")
    s = random_state(3)
    print( s )
    print( classic(3) >> s )

    print("\n===\n")

    print("* Measurement in the Bell basis")
    bell_test = [bell00.as_pred(), bell01.as_pred(), bell10.as_pred(), bell11.as_pred()]
    meas_bell = meas_test(bell_test)
    w = cnot >> (random_state(2) @ random_state(2))
    print( w )
    print( meas_bell >> w )
    print( w >= bell00.as_pred() )
    print( w >= bell01.as_pred() )
    print( w >= bell10.as_pred() )
    print( w >= bell11.as_pred() )

    print("\n===\n")

    print("* Equality of channels")
    print( cnot * (classic(2) @ idn(2)) == ccontrol(x_chan) )

    print("\n===\n")

    print("* ccase illustration")
    s = probabilistic_state(0.2, 0.3, 0.5)
    t = random_state(2)
    w = ccase(x_chan, hadamard, idn(2)) >> (s @ t)
    print( w % [1, 0] )
    print( w % [0, 1] )
    print( convex_state_sum((0.2, x_chan >> t), \
                            (0.3, hadamard >> t), \
                            (0.5, idn(2) >> t)) )

    print("\n===\n")

    print("* First projection of instrument is measurement")
    p = random_pred(5)
    print( (idn(2) @ discard(5)) * instr(p) == meas_pred(p) )

    print("\n===\n")

    print("* Second projection of instrument is convex sum")
    s = random_state(5)
    print( discard(2) @ idn(5) * instr(p) >> s ==
           convex_state_sum( (s >= p, s / p), (s >= ~p, s / ~p) ) )

    print("\n===\n")

    print("* Intrument predicate transformation")
    p = random_pred(10)
    q = random_pred(10)
    print( instr(p) << truth(2) @ q == (p & q) + (~p & q) )
    print( instr(p) << unit_pred(2,0) @ q == p & q )
    print( instr(p) << unit_pred(2,1) @ q == ~p & q )

    print("\n===\n")

    print("* Predicate case")
    p = random_pred(2)
    s = random_state(2)
    print( pcase(p)(x_chan, y_chan) >> s ==
           convex_state_sum( (s >= p, x_chan >> s/p), (s >= ~p, y_chan >> s/~p) ) )
    q = random_pred(2)
    print( pcase(p)(x_chan, y_chan) << q ==
           (p & x_chan << q) + (~p & y_chan << q) )


def teleportation_and_superdensecoding():
    
    print("\nSection: Measurement, control and instruments\n")

    print("* Teleportation, with the Bell state")
    alice = (meas0 @ meas0) * (hadamard @ idn(2)) * cnot
    bob = (discard(2) @ idn(2)) \
          * ccontrol(z_chan) \
          * (idn(2) @ discard(2) @ idn(2)) \
          * (idn(2) @ ccontrol(x_chan))
    teleportation = bob * (alice @ idn(2)) * (idn(2) @ bell00.as_chan())
    s = random_state(2)
    print( s )
    print( teleportation >> s )

    print("\n===\n")

    print("* Equality of states")
    print( s == (teleportation >> s) )

    print("\n===\n")

    print("* Equality of channels")
    print( np.isclose(teleportation.array, idn(2).array) )
    print( np.all(np.isclose(teleportation.array, idn(2).array)) )
    print( teleportation == idn(2) )

    print("\n===\n")

    print("* Superdense coding, with the Bell state")
    alice = (discard(2) @ idn(2)) * ccontrol(x_chan) \
            * (idn(2) @ discard(2) @ idn(2)) \
            * (idn(2) @ ccontrol(z_chan)) * (swap @ idn(2))
    bob = (meas0 @ meas0) * (hadamard @ idn(2)) * cnot
    def superdense_coding(r, s):
        return bob >> ((alice @ idn(2)) >> (cflip(r) @ cflip(s) @ bell00))
    r = random.uniform(0,1)
    s = random.uniform(0,1)
    print( r )
    print( s )
    print( superdense_coding(r,s) % [1,0] )
    print( superdense_coding(r,s) % [0,1] )

    print("\n===\n")

    print("* Superdense coding, with the Bell state, as channel")
    sdc = bob * (alice @ idn(2)) * (idn(2,2) @ bell00.as_chan())
    sdck = kron(2,2) * sdc * kron_inv(2,2)
    print("channel types", sdc.dom, sdc.cod )
    print( sdc == classic(2) @ classic(2), sdck == classic(4),  )

    print("\n===\n")

    print("* Teleportation, with the GHZ state")
    bell_test = [bell00.as_pred(), bell01.as_pred(), bell10.as_pred(), bell11.as_pred()]
    meas_bell = meas_test(bell_test)
    print( meas_bell.dom, meas_bell.cod )
    alice = (meas_bell @ idn(2,2)) * (idn(2) @ ghz.as_chan())
    print( alice.dom, alice.cod )

    print("\n===\n")

    print("* GHZ-teleportation, Bob's side")
    hadamard_test = [plus.as_pred(), minus.as_pred()]
    meas_hadamard = meas_test(hadamard_test)
    bob = ( discard(8) @ idn(2) ) \
          * ccase(idn(2), z_chan, x_chan, x_chan * z_chan, \
                  z_chan, idn(2), x_chan * z_chan, x_chan) \
        * ( kron(4, 2) @ idn(2) ) \
        * ( idn(4) @ meas_hadamard @ idn(2) )
    print( bob.dom, bob.cod )
    ghz_teleporation = bob * alice

    print("\n===\n")

    print("* GHZ-teleportation, test")
    s = random_state(2)
    print( s )
    print( ghz_teleporation >> s )
    print( ghz_teleporation == idn(2) )

    print("\n===\n")

    print("* GHZ-superdense coding")
    iy_matrix = np.array([[0,1],
                          [-1,0]])
    iy_chan = channel_from_unitary(iy_matrix, [2], [2])
    alice = (discard(8) @ idn(2,2)) * ccase(idn(2) @ idn(2), 
                                            idn(2) @ x_chan, 
                                            x_chan @ idn(2), 
                                            x_chan @ x_chan, 
                                            z_chan @ idn(2), 
                                            z_chan @ x_chan, 
                                            iy_chan @ idn(2), 
                                            iy_chan @ x_chan) 
    print( alice.dom, alice.cod )

    print("\n===\n")

    print("* GHZ-superdense coding, Bob's side")
    bob = meas_ghz
    ghz_sdc = bob * (alice @ idn(2)) * (idn(8) @ ghz.as_chan())
    print( ghz_sdc.dom, ghz_sdc.cod )

    print("\n===\n")

    print("* GHZ-superdense coding 8-test")
    s = random_probabilistic_state(8)
    print( s )
    print( ghz_sdc >> s )

    print("\n===\n")

    print("* GHZ-superdense coding classical bit test")
    k1 = kron(4,2) * (kron(2,2) @ idn(2))
    k2 = (kron_inv(2,2) @ idn(2)) * kron_inv(4,2)
    r1 = random_probabilistic_state(2)
    r2 = random_probabilistic_state(2)
    r3 = random_probabilistic_state(2)
    print( r1 )
    print( r2 )
    print( r3 )
    r = k1 >> (r1 @ r2 @ r3)
    w = k2 >> (ghz_sdc >> r)
    print( w % [1,0,0] )
    print( w % [0,1,0] )
    print( w % [0,0,1] )

def order_inference():

    print("\nSection: Order inference effect example\n")    

    print("* States for the three perspectives")
    H = np.array([[2, 1, 1, 0],
                  [1, -2, 0, 1],
                  [1, 0, 0, 1],
                  [0, 1, 1, 0]])
    Ujp = scipy.linalg.expm(complex(0,-1) * 1.2393 * H)
    Ujd = scipy.linalg.expm(complex(0,-1) * -3.8324 * H)
    Upd = np.dot(Ujd, conjugate_transpose(Ujp))
    Udp = np.dot(Ujp, conjugate_transpose(Ujd))
    print("Self-adjoint and unitaries: ",
          is_hermitian(H), is_unitary(Ujp), is_unitary(Ujd), is_unitary(Upd) )
    chjp = channel_from_unitary(Ujp, [4], [4])
    chjd = channel_from_unitary(Ujd, [4], [4])
    chpd = channel_from_unitary(Upd, [4], [4])
    chdp = channel_from_unitary(Udp, [4], [4])
    J = vector_state(sqrt(0.459/2),sqrt(0.459/2),sqrt(0.541/2),sqrt(0.541/2))
    P = chjp >> J
    D = chjd >> J
    print("Equality of states: ", D == chpd >> P, P == chdp >> D )
    print("Alternative descriptions via evolution: ",
          P == RandVar(H, [4]).evolution(J)(1.2393),
          D == RandVar(H, [4]).evolution(J)(-3.8324) )

    print("\n===\n")

    print("* Predicates for the four kinds of information, and naieve perspective")
    pGpE = unit_pred(4,0)
    pGnE = unit_pred(4,1)
    nGpE = unit_pred(4,2)
    nGnE = unit_pred(4,3)
    print( J >= pGpE | pGnE )
    print( J >= nGpE | nGnE )

    print("\n===\n")

    print("* Prosecutor's perspective")
    print("Guilt and non-guilt: ", P >= pGpE | pGnE, P >= nGpE | nGnE )
    print("Similarly, via predicate transformation: ",
          J >= chjp << (pGpE | pGnE), J >= chjp << (nGpE | nGnE) )
    print( P / (pGpE | nGpE) >= (pGpE | pGnE) )
    print("This can also be obtained by moving back and forth, since the channel is MIU")
    print( chjp >> (J / (chjp << (pGpE | nGpE))) >= (pGpE | pGnE) )

    print("\n===\n")

    print("* Defense perspective")
    print( D >= pGpE | pGnE )
    print( D / (chdp << (pGpE | nGpE)) >= pGpE, 
           D / (chdp << (pGpE | nGpE)) >= pGnE,
           D / (chdp << (pGpE | nGpE)) >= nGpE,
           D / (chdp << (pGpE | nGpE)) >= nGnE )
    print( D / (chdp << (pGpE | nGpE)) >= pGpE | pGnE )

    print("\n===\n")

    print("* Judge's perspective")
    P_ev = chjp << (pGpE | nGpE)
    D_ev = chjd << (pGnE | nGnE)
    print("Positive evidence of prosector first: ",
          J / P_ev / D_ev >= pGpE | pGnE )
    print("Negative evidence of defense first: ",
          J / D_ev / P_ev >= pGpE | pGnE )


def main():
    # states()
    # operations_on_states()
    # basic_states()
    # predicates()
    # operations_on_predicates()
    # validity()
    # conditioning()
    random_variables()
    # state_transformation()
    # predicate_transformation()
    # structural_channels()
    # measurement()
    # teleportation_and_superdensecoding()
    # order_inference()

if __name__ == "__main__":
    main()
